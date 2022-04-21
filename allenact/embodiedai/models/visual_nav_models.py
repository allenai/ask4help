from typing import Tuple, Dict, Optional, List
from allenact.utils.system import get_logger
from collections import OrderedDict

import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self,input_size):
        super(succ_pred_model,self).__init__()

        self.rnn_unit = RNNStateEncoder(input_size=input_size,hidden_size=512)
        self.linear_layer = nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Linear(128,32),nn.ReLU(),nn.Linear(32,8),nn.ReLU(),nn.Linear(8,1))

    def forward(self,x,hidden_states,masks):

        out,rnn_hidden_state = self.rnn_unit(x,hidden_states,masks)
        out = self.linear_layer(out)

        return out,rnn_hidden_state

# succ_pred_model= succ_pred_model(512)#.load_state_dict('./')
# succ_pred_model.load_state_dict('./')

class MultiDimActionDistr(Distr):
    '''
    Takes two categorical distributions and outputs a joint multidimensional distributions
    '''

    def __init__(self,actor_distr,label_distr,done_prob,tethered_done_score):
        super().__init__()
        self.actor_distr = actor_distr
        self.label_distr = label_distr
        self.done_prob = done_prob
        self.tethered_done = tethered_done_score

    def sample(self):
         actor_out = self.actor_distr.sample()
         label_out = self.label_distr.sample()
         return {"nav_action": actor_out, "ask_action": label_out,"done_prob":self.done_prob,"tethered_done":self.tethered_done}

    def log_prob(self,value):
        return self.label_distr.log_prob(value["ask_action"]) #+ self.actor_distr.log_prob(value["nav_action"]) 

    def entropy(self):
        return self.label_distr.entropy() #+ self.actor_distr.entropy() 
    def mode(self):
        return {"nav_action":self.actor_distr.mode(),"ask_action":self.label_distr.mode(),"done_prob":self.done_prob,"tethered_done":self.tethered_done}

class VisualNavActorCritic(ActorCriticModel[CategoricalDistr]):
    """Base class of visual navigation / manipulation (or broadly, embodied AI)
    model.

    `forward_encoder` function requires implementation.
    """

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

        self.nav_action_space = action_space['nav_action']
        self.ask_action_space = action_space['ask_action']
        self.succ_pred_rnn_hidden_state = None
        self.succ_pred_model = None

        '''
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
        '''

        self.end_action_idx = 3
        self.visual_offset = None

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

        else:  # single belief model
            state_encoders["single_belief"] = RNNStateEncoder(
                rnn_input_size,
                self._hidden_size,
                num_layers=num_rnn_layers,
                rnn_type=rnn_type,
                trainable_masked_hidden_state=trainable_masked_hidden_state,
            )

        self.state_encoders = nn.ModuleDict(state_encoders)

        self.belief_names = list(self.state_encoders.keys())

        get_logger().info(
            "there are {} belief models: {}".format(
                len(self.belief_names), self.belief_names
            )
        )

    def create_visual_residual_model (self,input_size,prev_action_embed_size: int,):

        self.prev_expert_action_embedder = FeatureEmbedding(
            input_size=self.nav_action_space.n,
            output_size=prev_action_embed_size,
        )

        self.prev_expert_mask_embedder = FeatureEmbedding(
            input_size=2,
            output_size=prev_action_embed_size,
        )
        input_size = input_size + prev_action_embed_size*2

        self.vis_mlp = nn.Sequential(nn.Linear(input_size,input_size//4),nn.ReLU(),nn.Linear(input_size//4,20))

        self.offset_embed = nn.Embedding(20,2048) #FeatureEmbedding(input_size=20,output_size=2048)

        self.offset_embed.weight.data *= 0.1 
    
    def create_action_residual_model (self,
        input_size: int,
        prev_action_embed_size: int,
        num_rnn_layers: int,
        rnn_type: str,
        trainable_masked_hidden_state=False,):
        self.prev_expert_action_embedder = FeatureEmbedding(
            input_size=self.nav_action_space.n,
            output_size=prev_action_embed_size,
        )

        self.prev_expert_mask_embedder = FeatureEmbedding(
            input_size=2,
            output_size=prev_action_embed_size,
        )

        self.expert_encoder = RNNStateEncoder(input_size+prev_action_embed_size + prev_action_embed_size, ## one for action + one for mask
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
            )
        self.residual_mlp =  nn.Sequential(nn.Linear(self._hidden_size,self._hidden_size//2),nn.ReLU(),nn.Linear(self._hidden_size//2,self.nav_action_space.n))    

        self.fusion_mlp = nn.Sequential(nn.Linear(self.nav_action_space.n*2,self.nav_action_space.n),nn.ReLU(),nn.Linear(self.nav_action_space.n,self.nav_action_space.n))

        self.prev_expert_action = None
            
    def create_tethered_policy(self):

        self.shared_mlp = nn.Sequential(nn.Linear(1568,1568//2),nn.ReLU(),nn.Linear(1568//2,512))
        # self.obs_mlp = nn.Sequential(nn.Linear(1568,1568//2),nn.ReLU(),nn.Linear(1568//2,512))
        # self.target_mlp = nn.Sequential(nn.Linear(1568,1568//2),nn.ReLU(),nn.Linear(1568//2,512))
        # inp_size = num_steps*num_processes
        self.cosine_similarity = nn.CosineSimilarity(dim=1,eps=1e-6)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.out_mlp = nn.Linear(1,1)
    
    def create_expert_encoder(self,
        input_size: int,
        prev_action_embed_size: int,
        num_rnn_layers: int,
        rnn_type: str,
        trainable_masked_hidden_state=False,):

        self.prev_expert_action_embedder = FeatureEmbedding(
            input_size=self.nav_action_space.n,
            output_size=prev_action_embed_size,
        )

        self.prev_expert_mask_embedder = FeatureEmbedding(
            input_size=2,
            output_size=prev_action_embed_size,
        )

        self.expert_encoder = RNNStateEncoder(input_size+prev_action_embed_size + prev_action_embed_size, ## one for action + one for mask
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
            )

        self.belief_fusion_mlp = nn.Sequential(nn.Linear(self._hidden_size*2,self._hidden_size*2),nn.ReLU(),nn.Linear(self._hidden_size*2,self._hidden_size))    

        # self.expert_encoder.rnn.weight_hh_l0.data *= 0.1
        # self.expert_encoder.rnn.bias_hh_l0.data *= 0    

    def create_ask4_help_module(self,
        prev_action_embed_size: int,
        num_rnn_layers: int,
        rnn_type:str,
        ask_gru_hidden_size=128,
        trainable_masked_hidden_state=False,
        adaptive_reward=False,
        ):

        self.prev_ask_action_embedder = FeatureEmbedding(
            input_size=self.ask_action_space.n,
            output_size=prev_action_embed_size,
        )

        self.expert_mask_embedder = FeatureEmbedding(input_size=2,output_size=prev_action_embed_size)

        if adaptive_reward:
            self.reward_function_embedder = FeatureEmbedding(input_size=13,output_size=prev_action_embed_size*2)
        else:
            self.reward_function_embedder = None     

        if adaptive_reward:
            mlp_input_size = self._hidden_size + 48 + prev_action_embed_size + prev_action_embed_size + prev_action_embed_size*2 
            ## ask_action + expert_action embedding + reward_function_embed
        else:
            mlp_input_size = self._hidden_size + 48 + prev_action_embed_size + prev_action_embed_size 
                
        self.ask_policy_mlp = nn.Sequential(
            nn.Linear(mlp_input_size,mlp_input_size//2),
            nn.ReLU(),
            nn.Linear(mlp_input_size//2,mlp_input_size//4),
        )

        self.ask_policy_gru = RNNStateEncoder(mlp_input_size//4,
                    ask_gru_hidden_size,
                    num_layers=num_rnn_layers,
                    rnn_type="GRU",
                    trainable_masked_hidden_state=False,)

        self.ask_actor_head = LinearActorHead(ask_gru_hidden_size,self.ask_action_space.n) ## concatenating frozen beliefs with success prediction output
        self.ask_critic_head = LinearCriticHead(ask_gru_hidden_size)            


    def load_state_dict(self, state_dict):
        new_state_dict = OrderedDict()
        for key in state_dict.keys():
            if "state_encoder." in key:  # old key name
                new_key = key.replace("state_encoder.", "state_encoders.single_belief.")
            else:
                new_key = key
            new_state_dict[new_key] = state_dict[key]

        return super().load_state_dict(new_state_dict,strict=False)  # compatible in keys

    def create_actorcritic_head(self):
        self.actor = LinearActorHead(self._hidden_size, self.nav_action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

    def create_aux_models(self, obs_embed_size: int, action_embed_size: int,num_steps: int, num_processes: int):
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
                num_steps = num_steps,
                num_processes = num_processes,
            )

        self.aux_models = nn.ModuleDict(aux_models)

    def set_grad_false(self,param_list):
        ## setting underlying objectnav model's grad to zero
        for name,W in self.named_parameters():
            if name in param_list:
                W.requires_grad = False

    @property
    def num_recurrent_layers(self):
        """Number of recurrent hidden layers."""
        return list(self.state_encoders.values())[0].num_recurrent_layers

    @property
    def recurrent_hidden_state_size(self):
        """The recurrent hidden state size of a single model."""

        return {'single_belief':self._hidden_size,'residual_gru':self._hidden_size,'ask4help_gru':128,'succ_pred_gru':512}
        # return self._hidden_size

    def _recurrent_memory_specification(self):

        if self.is_finetuned:
            self.belief_names.append('ask4help_gru')
            self.belief_names.append('succ_pred_gru')

        if self.adapt_belief:
            self.belief_names.append('residual_gru')   
    

        if self.adapt_policy:
            self.belief_names.append('residual_gru')

        out_dict1 = {
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
            
        if self.tethered_policy_memory:
            self.belief_names += self.scenes_list

            num_objects = len(self.objects_list)
            embedding_dim = 2048#1568 #512 #2048#1568
            num_steps = 5

            ## add dimension for steps 
            ## change embedding_dim to 512
            ## modify action space for tethered policy 

            out_dict2 = {
                memory_key: (
                    (
                        ("layer", self.num_recurrent_layers),
                        ("sampler", None),
                        # ("expert_step",num_steps),
                        ("object_dim",num_objects),
                        ("hidden", embedding_dim),
                        ("h",7),
                        ("w",7)
                    ),
                    torch.float32,
                )
                for memory_key in self.scenes_list
            }

            merged_dict = {**out_dict1,**out_dict2}

            return merged_dict

        else:      
            return out_dict1


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

        if self.is_finetuned:
            

            # obs_zero_dim = observations['rgb_clip_resnet'].shape[0] 

            # if self.visual_offset is not None:
            #     offset_zero_dim = self.visual_offset.shape[0]

            #     if obs_zero_dim==offset_zero_dim:
            #         nsteps, nsamplers = observations['rgb_clip_resnet'].shape[:2]

            #         self.visual_offset = self.visual_offset.view(nsteps,nsamplers,2048,7,7)
            #         print (self.visual_offset.requires_grad,'visual offset')
            #         observations['rgb_clip_resnet'] = observations['rgb_clip_resnet'] + self.visual_offset

            expert_action_obs = observations['expert_action']
            expert_action = expert_action_obs[:,:,0]
            expert_action_mask = expert_action_obs[:,:,1]

            

            nactions = self.nav_action_space.n

            expert_mask_embed = self.expert_mask_embedder(expert_action_mask)

            # 1.1 use perception model (i.e. encoder) to get observation embeddings
            obs_embeds = self.forward_encoder(observations)

            obs_emb_size = obs_embeds.size(-1)
            nsteps,nsamplers,_ = obs_embeds.shape

            

            clip_resnet_obs =  observations['rgb_clip_resnet'].view(nsteps,nsamplers,2048,7,7)

            # print (clip_resnet_obs.shape)
            # exit()

            # clip_resnet_obs = F.avg_pool2d(clip_resnet_obs,4).unsqueeze(-1).unsqueeze(-1)

            # clip_resnet_obs  = clip_resnet_obs.view(nsteps,nsteps,-1)

            prev_actions_embeds = self.prev_action_embedder(prev_actions['nav_action'])
            joint_embeds = torch.cat((obs_embeds, prev_actions_embeds), dim=-1)  # (T, N, *)
            
            if not self.adapt_belief:
                with torch.no_grad():

                    beliefs_dict = {}
                    for key, model in self.state_encoders.items():
                        beliefs_dict[key], rnn_hidden_states = model(
                            joint_embeds, memory.tensor(key), masks
                        )
                        memory.set_tensor(key, rnn_hidden_states)  # update memory here
                
                    # 3. fuse beliefs for multiple belief models
                    beliefs, task_weights = self.fuse_beliefs(
                        beliefs_dict, obs_embeds
                    )
                    beliefs_combined = beliefs


            if self.adapt_belief:
                ### only done when adaptation is switched on###
                joint_embeds_all = torch.cat((obs_embeds, prev_actions_embeds), dim=-1)
                beliefs_combined = None
                for step in range(nsteps):    
                    # 1.2 use embedding model to get prev_action embeddings
                    joint_embeds = joint_embeds_all[step,:,:].unsqueeze(0)
                    masks_step = masks[step,:,:].unsqueeze(0)
                    
                    # 2. use RNNs to get single/multiple beliefs
                    # with torch.no_grad():
                    beliefs_dict = {}
                    for key, model in self.state_encoders.items():
                        beliefs_dict[key], rnn_hidden_states = model(
                            joint_embeds, memory.tensor(key), masks_step
                        )
                        
                        memory.set_tensor(key, rnn_hidden_states)  # update memory here
                
                    # 3. fuse beliefs for multiple belief models
                    beliefs, task_weights = self.fuse_beliefs(
                        beliefs_dict, obs_embeds
                    )  # fused beliefs

                    if beliefs_combined is None:
                        beliefs_combined = beliefs
                    else:
                        beliefs_combined = torch.cat((beliefs_combined,beliefs),dim=0)
                    
                    expert_action_embedding = self.prev_expert_action_embedder(expert_action[step,:].unsqueeze(0))
                    expert_mask_embedding = self.prev_expert_mask_embedder(expert_action_mask[step,:].unsqueeze(0))
                    
                    res_input = torch.cat((beliefs,expert_action_embedding,expert_mask_embedding),dim=-1)
                    beliefs_residual,residual_hidden_states = self.expert_encoder(res_input,memory.tensor('residual_gru'),masks_step)
                    memory.set_tensor('residual_gru',residual_hidden_states)

                    beliefs_residual = beliefs_residual * expert_action_mask.unsqueeze(-1)[step,:,:].unsqueeze(0)

                    beliefs_updated = self.belief_fusion_mlp(torch.cat((beliefs,beliefs_residual),dim=-1)) #beliefs + beliefs_residualß

                    memory.set_tensor('single_belief',beliefs_updated)
                 
            

            if self.tethered_policy_memory:
                goal_object_idx = observations['goal_object_type_ind']

                scene_name_obs = observations['scene_name_sensor']
                scene_obj_count = observations['scene_object_count']

                # memory_inp = torch.zeros(nsteps,nsamplers,obs_emb_size).to(obs_embeds.device)
                memory_inp = torch.zeros(nsteps,nsamplers,2048,7,7).to(obs_embeds.device)

                ## filling up memory 
                for i,step in enumerate(range(nsteps)):
                    for samp in range(nsamplers):

                        train_flag,scn_idx1,scn_idx2 = scene_name_obs[step,samp]

                        scene_name = 'FloorPlan'
                        if train_flag:
                            scene_name+='_Train' + str(scn_idx1.item()) + '_' + str(scn_idx2.item())
                        else:
                            scene_name+='_Val' + str(scn_idx1.item()) + '_' + str(scn_idx2.item())
                        obj_idx = goal_object_idx[step,samp]
                        scn_obj_count = scene_obj_count[step,samp].item()

                        ## pop the first and the append the last, as soon as appended the end action,save it
                        ## move the list to the left by one index

                        # print (masks[step,samp])

                        if scn_obj_count==1 and not (masks[step,samp].item()):
                            a = memory.tensor(scene_name).clone()

                            # print (a.shape)

                            # a[:,:,samp,obj_idx.item(),:] *= 0.0
                            # print (a.shape,'memory tensor shape')
                            # exit()

                            a[:,samp,obj_idx.item(),:,:,:] *= 0.0 ##use this for end only version  
                            memory.set_tensor(scene_name,a)
                            # exit()

                        memory_inp[step,samp,:,:,:] = memory.tensor(scene_name)[0,samp,obj_idx.item()]

                        if scn_obj_count==1:
                            if expert_action_mask[step,samp]:
                                if expert_action[step,samp].item() == self.end_action_idx:

                                    a = memory.tensor(scene_name).clone() ## a is only one dimensional since its a per step thing 

                                    if a[:,samp,obj_idx.item(),:,:,:].sum()==0:
                                        print ('memory updated')
                                        print (clip_resnet_obs[step,samp,:].shape)
                                        # a[:,samp,obj_idx.item(),:] = obs_embeds[step,samp,:]
                                        a[:,samp,obj_idx.item(),:,:,:] = clip_resnet_obs[step,samp,:,:,:]
                                        # a[:,samp,obj_idx.item(),:] = joint_embeds[step,samp,:]
                                        # exit()

                                    memory.set_tensor(scene_name,a)   
                                else:
                                    continue  
                            else:
                                continue

            obs_embeds = obs_embeds.view(nsteps*nsamplers,-1)
            # memory_inp = memory_inp.view(nsteps*nsamplers,-1)
            beliefs_inp = beliefs.view(nsteps*nsamplers,-1)

            joint_embeds_inp = joint_embeds.view(nsteps*nsamplers,-1)

            # out_1 = self.shared_mlp(obs_embeds) #self.obs_mlp(obs_embeds)
            # out_2 = self.shared_mlp(memory_inp) #self.target_mlp(memory_inp)

            # print (out_1.shape,out_2.shape)

            # clip_resnet_obs = clip_resnet_obs.view(nsteps*nsamplers,-1)

            # print (clip_resnet_obs.shape)
            # print (memory_inp.shape)

            clip_resnet_obs = clip_resnet_obs.view(nsteps*nsamplers,-1)
            memory_inp = memory_inp.view(nsteps*nsamplers,-1)

            print (clip_resnet_obs.sum())
            print (memory_inp.sum())

            # cos_sim = self.mse_loss(obs_embeds,memory_inp).mean(-1).unsqueeze(-1)

            # print (clip_resnet_obs.shape)
            # print (memory_inp.shape)

            cos_sim = torch.mm(clip_resnet_obs,memory_inp.T)
            print (cos_sim.shape)
            exit()
            # exit()

            # cos_sim = self.mse_loss(clip_resnet_obs,clip_resnet_obs).mean(-1).unsqueeze(-1)
            # cos_sim = self.cosine_similarity(clip_resnet_obs,memory_inp).unsqueeze(-1)

            # print (cos_sim)
            # exit()
            # cos_sim = self.mse_loss(joint_embeds_inp,memory_inp).mean(-1).unsqueeze(-1)

            # print (cos_sim,'mse')
            # print (observations['rgb_clip_resnet'].shape)
            # exit()

            # l2_norm = torch.norm(out_1-out_2,p=2,dim=-1)
            # exit()

            # print (l2_norm,'l2 norm')

            # cos_sim = self.cosine_similarity(memory_inp,beliefs_inp).unsqueeze(-1)

            # print (cos_sim.shape,'cosine')
            # exit()
            
            # tethered_out = self.out_mlp(cos_sim)

            tethered_out = cos_sim

            # print (tethered_out.shape)

            obs_embeds = obs_embeds.view(nsteps,nsamplers,-1)
            tethered_out = tethered_out.view(nsteps,nsamplers,-1)

            beliefs = beliefs_combined

            # actor_pred_distr = self.actor(beliefs)

            if self.adapt_visual:

                ## do step by step processing when nsteps is non zero

                expert_action_embedding = self.prev_expert_action_embedder(expert_action)
                expert_mask_embedding = self.prev_expert_mask_embedder(expert_action_mask)

                out = self.vis_mlp(torch.cat((beliefs,expert_action_embedding,expert_mask_embedding),dim=-1))

                out = out.view(nsteps*nsamplers,20)
                offset = out @ self.offset_embed.weight.clone() 

                offset = offset * expert_action_mask.view(nsteps*nsamplers).unsqueeze(-1)

                offset = offset.unsqueeze(-1).unsqueeze(-1)

                offset = offset.repeat(1,1,7,7)

                self.visual_offset = offset 
            
            if self.adapt_policy:
                actor_distr = self.actor(beliefs)
                done_prob = torch.softmax(actor_distr.logits,dim=-1)[:,:,self.end_action_idx].unsqueeze(-1)

                expert_action_embedding = self.prev_expert_action_embedder(expert_action)
                expert_mask_embedding = self.prev_expert_mask_embedder(expert_action_mask)
                
                res_input = torch.cat((beliefs,expert_action_embedding,expert_mask_embedding),dim=-1)
                policy_residual,residual_hidden_states = self.expert_encoder(res_input,memory.tensor('residual_gru'),masks)
                memory.set_tensor('residual_gru',residual_hidden_states)

                out = self.residual_mlp(policy_residual)

                out_logits = self.fusion_mlp(torch.cat((out,actor_distr.logits),dim=-1))
                actor_pred_distr  = CategoricalDistr(logits=out_logits) 

            else:
                actor_pred_distr = self.actor(beliefs)
                done_prob = torch.softmax(actor_pred_distr.logits,dim=-1)[:,:,self.end_action_idx].unsqueeze(-1)
                done_prob_updated = done_prob

                '''
                ## only for thresholding 
                logits = actor_pred_distr.logits

                
                ## thresholding end at 0.70
                for step in range(nsteps):
                    for samp in range(nsamplers):
                        if done_prob[step,samp]<0.70:
                            # sum_before = logits[step,samp].sum()
                            # print (logits[step,samp],'before')
                            # print (logits[step,samp].sum(),'before')
                            # print (logits[step,samp].shape,'before')
                            logits[step,samp,self.end_action_idx] = -999
                            # logits[step,samp]*=(sum_before/logits[step,samp].sum())
                            # logits[step,samp] = torch.softmax(logits[step,samp],dim=-1)
                            # print (logits[step,samp])
                            # exit()

                done_prob_updated = torch.softmax(logits,dim=-1)[:,:,self.end_action_idx].unsqueeze(-1)
                actor_pred_distr = CategoricalDistr(logits=logits) 
                '''           
                                        
        
            ## restore for grad check
            '''
            if nsteps == 4:
                for name,W in self.named_parameters():
                    if W.requires_grad is True:
                        # if W.grad is not None:
                        if name == 'expert_encoder.rnn.weight_ih_l0':
                            print (W.grad,'gradient check')
                            # print (W.grad.shape,name) 
                            # exit()
            '''                    
                            
            with torch.no_grad():

                # if self.end_action_in_ask:
                #     ## making logits of end so small that it's never picked by the agent.
                #     actor_pred_distr.logits[:,:,self.end_action_idx] -= 999

                if self.succ_pred_model is None:
                    self.succ_pred_model = succ_pred_model(512).to(beliefs.device)
                    self.succ_pred_model.load_state_dict(
                        torch.load('./storage/best_auc_clip_run_belief_480_rollout_len.pt',
                                    map_location=beliefs.device))
               
                succ_pred_out, succ_rnn_hidden_states = self.succ_pred_model(beliefs, memory.tensor('succ_pred_gru'),masks)
                memory.set_tensor('succ_pred_gru', succ_rnn_hidden_states)
                succ_prob = torch.sigmoid(succ_pred_out)

                succ_prob_inp = succ_prob.repeat(1,1,48)                

            beliefs_ask_policy = beliefs.clone()
            beliefs_ask_policy = beliefs_ask_policy.detach()
            ask_policy_input = torch.cat((beliefs_ask_policy,succ_prob_inp),dim=-1)
            prev_ask_action_embed = self.prev_ask_action_embedder(prev_actions['ask_action'])
            # expert_mask_embed = self.expert_mask_embedder(expert_action_mask)

            if self.adaptive_reward:
                reward_config_embed = self.reward_function_embedder(observations['reward_config_sensor'])                
                ask_policy_input = torch.cat((ask_policy_input,prev_ask_action_embed,expert_mask_embed,reward_config_embed),dim=-1)
            else:
                ask_policy_input = torch.cat((ask_policy_input,prev_ask_action_embed,expert_mask_embed),dim=-1)

            if self.ask_actor_head is None:
                print ('initialisation error')
                exit()
                self.ask_actor_head = LinearActorHead(self._hidden_size+48,self.ask_action_space.n).to(beliefs.device) ## concatenating frozen beliefs with success prediction output
                self.ask_critic_head = LinearCriticHead(self._hidden_size+48).to(beliefs.device)            
            
            ask_policy_input = self.ask_policy_mlp(ask_policy_input)

            ask_policy_input,ask_hidden_states = self.ask_policy_gru(ask_policy_input,memory.tensor('ask4help_gru'),masks)

            memory.set_tensor('ask4help_gru', ask_hidden_states)

            ask_pred_distr = self.ask_actor_head(ask_policy_input)
            ask_pred_value = self.ask_critic_head(ask_policy_input)

            expert_logits = (torch.zeros(nsteps, nsamplers, nactions) + 1e-3).to(beliefs.device)

            for step in range(nsteps):
                for samp in range(nsamplers):
                    expert_action_idx = expert_action[step,samp].item()
                    expert_logits[step,samp,expert_action_idx] = 999
            
            expert_action_mask = expert_action_mask.unsqueeze(-1)
            
            action_logits = expert_logits * expert_action_mask + (1 - expert_action_mask) * actor_pred_distr.logits
            
            actor_distr = CategoricalDistr(logits=action_logits)

            output_distr = MultiDimActionDistr(actor_distr, ask_pred_distr,done_prob_updated,torch.sigmoid(tethered_out)) ##sigmoid to change to probability

            # 4. prepare output
            extras = (
                {
                    aux_uuid: {
                        "beliefs": (
                            beliefs_dict[aux_uuid] if self.multiple_beliefs else beliefs_combined
                        ),
                        "obs_embeds": obs_embeds,
                        "ask_action_logits":ask_pred_distr.logits,
                        "model_action_logits":actor_pred_distr.logits,
                        "expert_actions":observations['expert_action'],
                        "prev_actions":prev_actions,
                        "tethered_output": tethered_out if self.tethered_policy_memory else None,
                        "memory_input": memory_inp if self.tethered_policy_memory else None,
                        "aux_model": (
                            self.aux_models[aux_uuid]
                            if aux_uuid in self.aux_models
                            else None
                        ),
                    }
                    for aux_uuid in self.auxiliary_uuids
                }
                if self.auxiliary_uuids is not None
                else {}
            )

            if self.multiple_beliefs:
                extras[MultiAuxTaskNegEntropyLoss.UUID] = task_weights


            actor_critic_output = ActorCriticOutput(
                distributions=output_distr,
                values=ask_pred_value,
                extras=extras,
            )

            return actor_critic_output,memory

        
        print ('logic error model')
        exit()
        # 1.1 use perception model (i.e. encoder) to get observation embeddings
        obs_embeds = self.forward_encoder(observations)
        # 1.2 use embedding model to get prev_action embeddings
        prev_actions_embeds = self.prev_action_embedder(prev_actions)
        joint_embeds = torch.cat((obs_embeds, prev_actions_embeds), dim=-1)  # (T, N, *)

        # 2. use RNNs to get single/multiple beliefs
        beliefs_dict = {}
        for key, model in self.state_encoders.items():
            beliefs_dict[key], rnn_hidden_states = model(
                joint_embeds, memory.tensor(key), masks
            )
            memory.set_tensor(key, rnn_hidden_states)  # update memory here

        # 3. fuse beliefs for multiple belief models
        beliefs, task_weights = self.fuse_beliefs(
            beliefs_dict, obs_embeds
        )  # fused beliefs

        # 4. prepare output
        extras = (
            {
                aux_uuid: {
                    "beliefs": (
                        beliefs_dict[aux_uuid] if self.multiple_beliefs else beliefs
                    ),
                    "obs_embeds": obs_embeds,
                    "aux_model": (
                        self.aux_models[aux_uuid]
                        if aux_uuid in self.aux_models
                        else None
                    ),
                }
                for aux_uuid in self.auxiliary_uuids
            }
            if self.auxiliary_uuids is not None
            else {}
        )

        if self.multiple_beliefs:
            extras[MultiAuxTaskNegEntropyLoss.UUID] = task_weights

        '''
        expert_action_obs = observations['expert_action'].squeeze()
        expert_action = expert_action_obs[0]
        expert_action_mask = expert_action_obs[1]

        nsteps,nsamplers,_ = beliefs.shape
        nactions = self.nav_action_space.n

        actor_pred_distr = self.actor(beliefs)


        with torch.no_grad():
            if self.succ_pred_model is None:
                self.succ_pred_model = succ_pred_model(512).to(beliefs.device)
                self.succ_pred_model.load_state_dict(torch.load('./storage/best_auc_clip_run_belief_480_rollout_len.pt',map_location=beliefs.device))
            if self.succ_pred_rnn_hidden_state is None:
                self.succ_pred_rnn_hidden_state = torch.zeros(1,1,512).to(beliefs.device)
            succ_pred_out,rnn_hidden_states = self.succ_pred_model(beliefs,self.succ_pred_rnn_hidden_state,masks)
            self.succ_pred_rnn_hidden_state = rnn_hidden_states
            succ_prob = torch.sigmoid(succ_pred_out).squeeze()

        expert_logits = torch.zeros(nsteps,nsamplers,nactions) + 1e-3
        expert_logits[:,:,expert_action.item()] = 999
        action_logits = expert_logits*expert_action_mask + (1-expert_action_mask) * actor_pred_distr.logits
        actor_distr = CategoricalDistr(logits=action_logits)

        threshold = 0.2 ## To be updated

        if succ_prob<threshold:
            succ_pred_logit = torch.tensor([0.001,999]).unsqueeze(0).unsqueeze(0).to(beliefs.device)
        else:
            succ_pred_logit = torch.tensor([999,0.001]).unsqueeze(0).unsqueeze(0).to(beliefs.device)

        # succ_pred_logit = torch.tensor([0.001,999]).unsqueeze(0).unsqueeze(0).to(beliefs.device)

        output_distr = MultiDimActionDistr(actor_distr,CategoricalDistr(logits=succ_pred_logit))
        '''

        actor_critic_output = ActorCriticOutput(
            distributions=self.actor(beliefs),
            values=self.critic(beliefs),
            extras=extras,
        )

        return actor_critic_output, memory
