from typing import Any

from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor
from allenact.utils.system import get_logger


class ResnetPreProcessorHabitat(ResNetPreprocessor):
    """Preprocess RGB or depth image using a ResNet model."""

    def __init__(self, *args, **kwargs: Any):
        super().__init__(*args, **kwargs)
        get_logger().warning(
            "`ResnetPreProcessorHabitat` is deprecated, use `ResNetPreprocessor` instead."
        )
