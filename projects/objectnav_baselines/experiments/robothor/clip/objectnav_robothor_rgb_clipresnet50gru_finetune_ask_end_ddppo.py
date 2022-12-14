from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor
from allenact_plugins.ithor_plugin.ithor_sensors import (
    RGBSensorThor,
    GoalObjectTypeThorSensor,
)
from allenact.base_abstractions.sensor import ExpertActionSensor

from projects.objectnav_baselines.experiments.clip.objectnav_mixin_clipresnetgru import (
    ObjectNavMixInClipResNetGRUConfig,
)
from projects.objectnav_baselines.experiments.objectnav_mixin_ddppo import (
    ObjectNavMixInPPOConfig,
)
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig

from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    LinearDecay,
)

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import gym
import torch.nn as nn
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from projects.objectnav_baselines.models.object_nav_models import (
    ResnetTensorObjectNavActorCritic,
)



class ObjectNavRoboThorClipRGBPPOExperimentConfig(
    ObjectNavRoboThorBaseConfig,
    ObjectNavMixInPPOConfig,
    ObjectNavMixInClipResNetGRUConfig,
):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input."""

    CLIP_MODEL_TYPE = "RN50"

    SENSORS = [
        RGBSensorThor(
            height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
        ),
        GoalObjectTypeThorSensor(
            object_types=ObjectNavRoboThorBaseConfig.TARGET_TYPES,
        ),
        ExpertActionSensor(nactions=len(ObjectNavTask.class_action_names()), ),
    ]

    def __init__(self):

        super().__init__()

        # self.REWARD_CONFIG = {
        # "step_penalty": -0.01,
        # "goal_success_reward": 10.0,
        # "failed_stop_reward": -8.0, ##double this maybe?  ##change this
        # "shaping_weight": 1.0,
        # "penalty_for_ask": -0.3,
        # }

        self.REWARD_CONFIG = {
        "step_penalty":            -0.01,
        "goal_success_reward":      0.00,
        "failed_stop_reward":      -10.00,
        "shaping_weight":           0.00,
        "penalty_for_init_ask":    -1.00, ##decreasing this as well
        "penalty_for_ask_recurring": -0.0,##removing recurring cost
        "penalty_for_step_ask":    -0.0,
        }

    @classmethod
    def tag(cls):
        return "Objectnav-RoboTHOR-RGB-ClipResNet50GRU-FINETUNE-DDPPO"


    def training_pipeline(self, **kwargs):
        # PPO
        ppo_steps = int(15000000)
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 4
        num_steps = 128//2
        save_interval = 2500000
        log_interval = 10000 if torch.cuda.is_available() else 1
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5

        named_losses = {"ppo_loss": (PPO(**PPOConfig), 1.0)}
        named_losses = self._update_with_auxiliary_losses(named_losses)

        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={key: val[0] for key, val in named_losses.items()},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(
                    loss_names=list(named_losses.keys()),
                    max_stage_steps=ppo_steps,
                    loss_weights=[val[1] for val in named_losses.values()],
                )
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )    

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:

        has_rgb = any(isinstance(s, RGBSensor) for s in cls.SENSORS)
        has_depth = any(isinstance(s, DepthSensor) for s in cls.SENSORS)

        goal_sensor_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, GoalObjectTypeThorSensor)),
            None,
        )

        end_action_in_ask = True

        if end_action_in_ask:
            action_space = gym.spaces.Dict({"nav_action": gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
                                     "ask_action": gym.spaces.Discrete(4)})  ## 3 means take END action, 2 means stop asking, 1 means start asking, 0 means do nothing
        else:
            action_space = gym.spaces.Dict({"nav_action": gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
                                     "ask_action": gym.spaces.Discrete(3)})
                                    # 2 means stop asking, 1 means start asking, 0 means do nothing


        return ResnetTensorObjectNavActorCritic(
            action_space=action_space,  # gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_resnet_preprocessor_uuid="rgb_clip_resnet" if has_rgb else None,
            depth_resnet_preprocessor_uuid="depth_clip_resnet" if has_depth else None,
            hidden_size=512,
            goal_dims=32,
            is_finetuned=True,
            end_action_in_ask=end_action_in_ask,
        )





