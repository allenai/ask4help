from typing import Tuple, Dict, Optional, List
from allenact.utils.system import get_logger
from collections import OrderedDict

import os
import gym
import torch
import torch.nn as nn
from gym.spaces.dict import Dict as SpaceDict

from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    LinearCriticHead,
    LinearActorHead,
    ObservationType,
    DistributionType,
)
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.utils.model_utils import FeatureEmbedding
from allenact.embodiedai.models.basic_models import RNNStateEncoder
from allenact.embodiedai.models.aux_models import AuxiliaryModel
from allenact.embodiedai.aux_losses.losses import MultiAuxTaskNegEntropyLoss
from allenact.base_abstractions.distributions import CategoricalDistr

from typing import TypeVar
from allenact.embodiedai.models.fusion_models import Fusion
from allenact.base_abstractions.distributions import Distr


FusionType = TypeVar("FusionType", bound=Fusion)


class succ_pred_model(nn.Module):
    def __init__(self, input_size):
        super(succ_pred_model, self).__init__()

        self.rnn_unit = RNNStateEncoder(input_size=input_size, hidden_size=512)
        self.linear_layer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x, hidden_states, masks):

        out, rnn_hidden_state = self.rnn_unit(x, hidden_states, masks)
        out = self.linear_layer(out)

        return out, rnn_hidden_state


# succ_pred_model= succ_pred_model(512)#.load_state_dict('./')
# succ_pred_model.load_state_dict('./')


class MultiDimActionDistr(Distr):
    """
    Takes two categorical distributions and outputs a joint multidimensional distributions
    """

    def __init__(self, actor_distr, label_distr):
        super().__init__()
        self.actor_distr = actor_distr
        self.label_distr = label_distr

    def sample(self):
        actor_out = self.actor_distr.sample()
        label_out = self.label_distr.sample()
        return {"nav_action": actor_out, "ask_action": label_out}

    def log_prob(self, value):
        return self.label_distr.log_prob(
            value["ask_action"]
        )  # + self.actor_distr.log_prob(value["nav_action"])

    def entropy(self):
        return self.label_distr.entropy()  # + self.actor_distr.entropy()

    def mode(self):
        return {
            "nav_action": self.actor_distr.mode(),
            "ask_action": self.label_distr.mode(),
        }


class VisualNavActorCritic(ActorCriticModel[CategoricalDistr]):
    """Base class of visual navigation / manipulation (or broadly, embodied AI)
    model.

    `forward_encoder` function requires implementation.
    """

    goal_dims: int
    add_target_to_residual: bool
    goal_visual_encoder: torch.nn.Module

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        hidden_size=512,
        multiple_beliefs=False,
        beliefs_fusion: Optional[FusionType] = None,
        auxiliary_uuids: Optional[List[str]] = None,
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)
        self._hidden_size = hidden_size
        assert multiple_beliefs == (beliefs_fusion is not None)
        self.multiple_beliefs = multiple_beliefs
        self.beliefs_fusion = beliefs_fusion
        self.auxiliary_uuids = auxiliary_uuids
        if isinstance(self.auxiliary_uuids, list) and len(self.auxiliary_uuids) == 0:
            self.auxiliary_uuids = None

        self.nav_action_space = action_space["nav_action"]
        self.ask_action_space = action_space["ask_action"]
        self.succ_pred_rnn_hidden_state = None
        self.succ_pred_model = None

        """
        mlp_input_size = self._hidden_size+48+6 ## 6 for prev action embedding
        
        
        self.ask_actor_head = LinearActorHead(mlp_input_size,self.ask_action_space.n) ## concatenating frozen beliefs with success prediction output
        self.ask_critic_head = LinearCriticHead(mlp_input_size) ## 6 for prev action embedding
        

        self.ask_policy_mlp = nn.Sequential(
            nn.Linear(mlp_input_size,mlp_input_size//2),
            nn.ReLU(),
            nn.Linear(mlp_input_size//2,mlp_input_size//4),
        )

        self.ask_policy_gru = RNNStateEncoder(mlp_input_size//4,128,
                    num_layers=1,
                    rnn_type="GRU",
                    trainable_masked_hidden_state=False,) ##restore to run gru variant
        self.ask_policy_gru_hidden_state = None

        self.ask_actor_head = LinearActorHead(128,self.ask_action_space.n) ## concatenating frozen beliefs with success prediction output
        self.ask_critic_head = LinearCriticHead(128) 
        """

        self.end_action_idx = 3

        # self.succ_pred_model = succ_pred_model(512)#.load_state_dict('./')

        # Define the placeholders in init function
        self.state_encoders: nn.ModuleDict
        self.aux_models: nn.ModuleDict
        self.actor: LinearActorHead
        self.critic: LinearCriticHead

    def create_state_encoders(
        self,
        obs_embed_size: int,
        prev_action_embed_size: int,
        num_rnn_layers: int,
        rnn_type: str,
        add_prev_actions: bool,
        trainable_masked_hidden_state=False,
    ):
        rnn_input_size = obs_embed_size

        self.prev_action_embedder = FeatureEmbedding(
            input_size=self.nav_action_space.n,
            output_size=prev_action_embed_size if add_prev_actions else 0,
        )
        self.prev_action_embedder.requires_grad_(False)

        if add_prev_actions:
            rnn_input_size += prev_action_embed_size

        state_encoders = OrderedDict()  # perserve insertion order in py3.6
        if self.multiple_beliefs:  # multiple belief model
            for aux_uuid in self.auxiliary_uuids:
                state_encoders[aux_uuid] = RNNStateEncoder(
                    rnn_input_size,
                    self._hidden_size,
                    num_layers=num_rnn_layers,
                    rnn_type=rnn_type,
                    trainable_masked_hidden_state=trainable_masked_hidden_state,
                )
            # create fusion model
            self.fusion_model = self.beliefs_fusion(
                hidden_size=self._hidden_size,
                obs_embed_size=obs_embed_size,
                num_tasks=len(self.auxiliary_uuids),
            )
            self.fusion_model.requires_grad_(False)

        else:  # single belief model
            state_encoders["single_belief"] = RNNStateEncoder(
                rnn_input_size,
                self._hidden_size,
                num_layers=num_rnn_layers,
                rnn_type=rnn_type,
                trainable_masked_hidden_state=trainable_masked_hidden_state,
            )

        print("state_encoders", list(state_encoders.keys()))
        self.state_encoders = nn.ModuleDict(state_encoders)
        self.state_encoders.requires_grad_(False)

        self.belief_names = list(self.state_encoders.keys())

        get_logger().info(
            "there are {} belief models: {}".format(
                len(self.belief_names), self.belief_names
            )
        )

    def create_expert_encoder(
        self,
        input_size: int,
        prev_action_embed_size: int,
        num_rnn_layers: int,
        rnn_type: str,
        trainable_masked_hidden_state=False,
    ):

        self.prev_expert_action_embedder = FeatureEmbedding(
            input_size=self.nav_action_space.n, output_size=prev_action_embed_size,
        )

        self.expert_encoder = RNNStateEncoder(
            input_size
            + prev_action_embed_size
            + (0 if not self.add_target_to_residual else self.goal_dims),
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
        )

    def create_ask4_help_module(
        self,
        prev_action_embed_size: int,
        num_rnn_layers: int,
        rnn_type: str,
        ask_gru_hidden_size=128,
        trainable_masked_hidden_state=False,
        adaptive_reward=False,
    ):

        self.prev_ask_action_embedder = FeatureEmbedding(
            input_size=self.ask_action_space.n, output_size=prev_action_embed_size,
        )

        self.expert_mask_embedder = FeatureEmbedding(
            input_size=2, output_size=prev_action_embed_size
        )

        if adaptive_reward:
            self.reward_function_embedder = FeatureEmbedding(
                input_size=30, output_size=prev_action_embed_size * 2
            )
        else:
            self.reward_function_embedder = None

        if adaptive_reward:
            mlp_input_size = (
                self._hidden_size
                + 48
                + prev_action_embed_size
                + prev_action_embed_size
                + prev_action_embed_size * 2
            )
            ## ask_action + expert_action embedding + reward_function_embed
        else:
            mlp_input_size = (
                self._hidden_size + 48 + prev_action_embed_size + prev_action_embed_size
            )

        self.ask_policy_mlp = nn.Sequential(
            nn.Linear(mlp_input_size, mlp_input_size // 2),
            nn.ReLU(),
            nn.Linear(mlp_input_size // 2, mlp_input_size // 4),
        )

        self.ask_policy_gru = RNNStateEncoder(
            mlp_input_size // 4,
            ask_gru_hidden_size,
            num_layers=num_rnn_layers,
            rnn_type="GRU",
            trainable_masked_hidden_state=False,
        )

        self.ask_actor_head = LinearActorHead(
            ask_gru_hidden_size, self.ask_action_space.n
        )  ## concatenating frozen beliefs with success prediction output
        self.ask_critic_head = LinearCriticHead(ask_gru_hidden_size)

    def load_state_dict(self, state_dict):
        new_state_dict = OrderedDict()
        for key in state_dict.keys():
            if "state_encoder." in key:  # old key name
                new_key = key.replace("state_encoder.", "state_encoders.single_belief.")
            else:
                new_key = key
            new_state_dict[new_key] = state_dict[key]

        return super().load_state_dict(
            new_state_dict, strict=False
        )  # compatible in keys

    def create_actorcritic_head(self):
        self.actor = LinearActorHead(self._hidden_size, self.nav_action_space.n)
        self.actor.requires_grad_(False)
        self.critic = LinearCriticHead(self._hidden_size)
        self.critic.requires_grad_(False)

    def create_aux_models(self, obs_embed_size: int, action_embed_size: int):
        if self.auxiliary_uuids is None:
            return
        aux_models = OrderedDict()
        for aux_uuid in self.auxiliary_uuids:
            aux_models[aux_uuid] = AuxiliaryModel(
                aux_uuid=aux_uuid,
                action_dim=self.nav_action_space.n,
                obs_embed_dim=obs_embed_size,
                belief_dim=self._hidden_size,
                action_embed_size=action_embed_size,
            )

        self.aux_models = nn.ModuleDict(aux_models)
        self.aux_models.requires_grad_(False)

    @property
    def num_recurrent_layers(self):
        """Number of recurrent hidden layers."""
        return list(self.state_encoders.values())[0].num_recurrent_layers

    @property
    def recurrent_hidden_state_size(self):
        """The recurrent hidden state size of a single model."""

        return {
            "single_belief": self._hidden_size,
            "residual_gru": self._hidden_size,
            "ask4help_gru": 128,
            "succ_pred_gru": 512,
        }
        # return self._hidden_size

    def _recurrent_memory_specification(self):

        if self.is_finetuned:
            self.belief_names.append("ask4help_gru")
            self.belief_names.append("succ_pred_gru")

        if self.adapt_belief:
            self.belief_names.append("residual_gru")
        return {
            memory_key: (
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size[memory_key]),
                ),
                torch.float32,
            )
            for memory_key in self.belief_names
        }

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        raise NotImplementedError("Obs Encoder Not Implemented")

    def fuse_beliefs(
        self, beliefs_dict: Dict[str, torch.FloatTensor], obs_embeds: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        all_beliefs = torch.stack(list(beliefs_dict.values()), dim=-1)  # (T, N, H, k)

        if self.multiple_beliefs:  # call the fusion model
            return self.fusion_model(all_beliefs=all_beliefs, obs_embeds=obs_embeds)
        # single belief
        beliefs = all_beliefs.squeeze(-1)  # (T,N,H)
        return beliefs, None

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        """Processes input batched observations to produce new actor and critic
        values. Processes input batched observations (along with prior hidden
        states, previous actions, and masks denoting which recurrent hidden
        states should be masked) and returns an `ActorCriticOutput` object
        containing the model's policy (distribution over actions) and
        evaluation of the current state (value).

        # Parameters
        observations : Batched input observations.
        memory : `Memory` containing the hidden states from initial timepoints.
        prev_actions : Tensor of previous actions taken.
        masks : Masks applied to hidden states. See `RNNStateEncoder`.
        # Returns
        Tuple of the `ActorCriticOutput` and recurrent hidden state.
        """

        assert self.is_finetuned

        expert_action_obs = observations["expert_action"]
        expert_action = expert_action_obs[:, :, 0]
        expert_action_mask = expert_action_obs[:, :, 1]

        nactions = self.nav_action_space.n

        with torch.no_grad():
            # 1.1 use perception model (i.e. encoder) to get observation embeddings
            obs_embeds = self.forward_encoder(observations)
            if self.add_target_to_residual:
                goal_emb = self.goal_visual_encoder.get_object_type_encoding(
                    observations
                )  # used for residual GRU

            nsteps, nsamplers, _ = obs_embeds.shape

            prev_actions_embeds = self.prev_action_embedder(prev_actions["nav_action"])
            joint_embeds = torch.cat(
                (obs_embeds, prev_actions_embeds), dim=-1
            )  # (T, N, *)

            beliefs_dict = {}
            for key, model in self.state_encoders.items():
                beliefs_dict[key], rnn_hidden_states = model(
                    joint_embeds, memory.tensor(key), masks
                )
                memory.set_tensor(key, rnn_hidden_states)  # update memory here

            # 3. fuse beliefs for multiple belief models
            beliefs, task_weights = self.fuse_beliefs(beliefs_dict, obs_embeds)

        if self.adapt_belief:
            # Only done when adaptation is switched on

            # Design:
            # 1. Keep the underlying agent's beliefs intact, so no influence of the residual belief at all.
            # 2. Only use residual beliefs for the input to the linear head.
            # 3. The residual GRU should observe the current agent's belief to decide how to modify it to predict the
            #    expert behavior.
            # 4. Ensure gradients flow through the actor head (as we want the residual belief to help in combination
            #    with the actor linear head).
            # 5. Since the agent's beliefs might get messed up, we can also add a target object type input to the
            #    residual GRU.
            # 6. If the imitation loss seems to go down over time, we should augment the ask4help module with 3
            #    possible modes: agent, expert, predicted/overridden expert (overridden expert incurs no penalty)
            # 7. Training with 3 helper modes:
            #    - phase 1: only PPO loss for agent/expert choices
            #    - phase 2: only imitation from expert
            #    - phase 3: only PPO for agent/expert/override choices
            #    - iterate phase 2 and phase 3 for ever?

            new_beliefs = []

            residual_hidden_states = memory.tensor("residual_gru")

            for step in range(nsteps):
                cur_beliefs = beliefs[step : step + 1, ...]
                masks_step = masks[step : step + 1, ...]
                expert_action_embedding = self.prev_expert_action_embedder(
                    expert_action[step, :].unsqueeze(0)
                )

                if self.add_target_to_residual:
                    current_goal_emb = goal_emb[step : step + 1, ...]
                    res_input = torch.cat(
                        (cur_beliefs, expert_action_embedding, current_goal_emb), dim=-1
                    )
                else:
                    res_input = torch.cat(
                        (cur_beliefs, expert_action_embedding), dim=-1
                    )

                beliefs_residual, residual_hidden_states = self.expert_encoder(
                    res_input, residual_hidden_states, masks_step
                )

                cur_expert_action_mask = expert_action_mask[
                    step : step + 1, ...
                ].unsqueeze(-1)

                new_beliefs.append(
                    cur_beliefs + beliefs_residual * cur_expert_action_mask
                )

            memory.set_tensor("residual_gru", residual_hidden_states)

            new_beliefs = torch.cat(new_beliefs, dim=0)
        else:
            new_beliefs = beliefs

        # Ensure gradients flow through the actor head
        actor_pred_distr = self.actor(new_beliefs)

        if self.end_action_in_ask:
            # Make logits of end so small that it's never picked by the agent
            actor_pred_distr.logits[:, :, self.end_action_idx] -= 999

        if self.succ_pred_model is None:
            self.succ_pred_model = succ_pred_model(512).to(beliefs.device)
            self.succ_pred_model.load_state_dict(
                torch.load(
                    "./storage/best_auc_clip_run_belief_480_rollout_len.pt",
                    map_location=beliefs.device,
                )
            )

        succ_pred_out, succ_rnn_hidden_states = self.succ_pred_model(
            beliefs, memory.tensor("succ_pred_gru"), masks
        )
        memory.set_tensor("succ_pred_gru", succ_rnn_hidden_states)
        succ_prob = torch.sigmoid(succ_pred_out)

        succ_prob_inp = succ_prob.repeat(1, 1, 48)

        ask_policy_input = torch.cat((beliefs, succ_prob_inp), dim=-1)

        prev_ask_action_embed = self.prev_ask_action_embedder(
            prev_actions["ask_action"]
        )
        expert_mask_embed = self.expert_mask_embedder(expert_action_mask)

        if self.adaptive_reward:
            reward_config_embed = self.reward_function_embedder(
                observations["reward_config_sensor"]
            )
            ask_policy_input = torch.cat(
                (
                    ask_policy_input,
                    prev_ask_action_embed,
                    expert_mask_embed,
                    reward_config_embed,
                ),
                dim=-1,
            )
        else:
            ask_policy_input = torch.cat(
                (ask_policy_input, prev_ask_action_embed, expert_mask_embed), dim=-1
            )

        assert self.ask_actor_head is not None, "initialisation error"

        ask_policy_input = self.ask_policy_mlp(ask_policy_input)

        ask_policy_input, ask_hidden_states = self.ask_policy_gru(
            ask_policy_input, memory.tensor("ask4help_gru"), masks
        )
        memory.set_tensor("ask4help_gru", ask_hidden_states)

        ask_pred_distr = self.ask_actor_head(ask_policy_input)
        ask_pred_value = self.ask_critic_head(ask_policy_input)

        expert_logits = (torch.zeros(nsteps, nsamplers, nactions) + 1e-3).to(
            beliefs.device
        )

        for step in range(nsteps):
            for samp in range(nsamplers):
                expert_action_idx = expert_action[step, samp].item()
                expert_logits[step, samp, expert_action_idx] = 999

        expert_action_mask = expert_action_mask.unsqueeze(-1)

        action_logits = (
            expert_logits * expert_action_mask
            + (1 - expert_action_mask) * actor_pred_distr.logits
        )

        actor_distr = CategoricalDistr(logits=action_logits)

        output_distr = MultiDimActionDistr(actor_distr, ask_pred_distr)

        # 4. prepare output
        extras = (
            {
                aux_uuid: {
                    "beliefs": (
                        beliefs_dict[aux_uuid] if self.multiple_beliefs else beliefs
                    ),
                    "obs_embeds": obs_embeds,
                    "ask_action_logits": ask_pred_distr.logits,
                    "model_action_logits": actor_pred_distr.logits,
                    "expert_actions": observations["expert_action"],
                    "prev_actions": prev_actions,
                    "aux_model": (
                        self.aux_models[aux_uuid]
                        if aux_uuid in self.aux_models
                        else None
                    ),
                }
                for aux_uuid in self.auxiliary_uuids
            }
            if self.auxiliary_uuids is not None
            else {"model_action_logits": actor_pred_distr.logits}
        )

        if self.multiple_beliefs:
            extras[MultiAuxTaskNegEntropyLoss.UUID] = task_weights

        actor_critic_output = ActorCriticOutput(
            distributions=output_distr, values=ask_pred_value, extras=extras,
        )

        return actor_critic_output, memory
