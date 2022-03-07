from allenact_plugins.ithor_plugin.ithor_sensors import (
    RGBSensorThor,
    GoalObjectTypeThorSensor,
)
from projects.objectnav_baselines.experiments.objectnav_mixin_resnet18gru import (
    ObjectNavMixInResNet18GRUConfig,
)
from projects.objectnav_baselines.experiments.objectnav_thor_mixin_ddppo_and_gbc import (
    ObjectNavThorMixInPPOAndGBCConfig,
)
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)


class ObjectNaviThorRGBPPOExperimentConfig(
    ObjectNavRoboThorBaseConfig,
    ObjectNavThorMixInPPOAndGBCConfig,
    ObjectNavMixInResNet18GRUConfig,
):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input."""

    SENSORS = ObjectNavThorMixInPPOAndGBCConfig.SENSORS + (  # type:ignore
        RGBSensorThor(
            height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        GoalObjectTypeThorSensor(
            object_types=ObjectNavRoboThorBaseConfig.TARGET_TYPES,
        ),
    )

    @classmethod
    def tag(cls):
        return "Objectnav-RoboTHOR-RGB-ResNetGRU-DDPPOAndGBC"
