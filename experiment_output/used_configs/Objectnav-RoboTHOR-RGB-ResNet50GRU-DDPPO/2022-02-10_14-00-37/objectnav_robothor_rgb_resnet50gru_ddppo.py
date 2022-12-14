### THIS FILE ORIGINALLY LOCATED AT '/home/kunals/eai_proj/clip_model/allenact/projects/objectnav_baselines/experiments/robothor/objectnav_robothor_rgb_resnet50gru_ddppo.py'

from allenact_plugins.ithor_plugin.ithor_sensors import (
    RGBSensorThor,
    GoalObjectTypeThorSensor,
)
from projects.objectnav_baselines.experiments.objectnav_mixin_ddppo import (
    ObjectNavMixInPPOConfig,
)
from projects.objectnav_baselines.experiments.objectnav_mixin_resnetgru import (
    ObjectNavMixInResNetGRUConfig,
)
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)


class ObjectNavRoboThorRGBPPOExperimentConfig(
    ObjectNavRoboThorBaseConfig, ObjectNavMixInPPOConfig, ObjectNavMixInResNetGRUConfig,
):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input."""

    RESNET_TYPE = "RN50"

    SENSORS = [
        RGBSensorThor(
            height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        GoalObjectTypeThorSensor(
            object_types=ObjectNavRoboThorBaseConfig.TARGET_TYPES,
        ),
    ]

    @classmethod
    def tag(cls):
        return "Objectnav-RoboTHOR-RGB-ResNet50GRU-DDPPO"
