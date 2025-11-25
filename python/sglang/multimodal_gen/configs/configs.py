# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from enum import Enum

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class DatasetType(str, Enum):
    """
    Enumeration for different dataset types.
    """

    HF = "hf"
    MERGED = "merged"

    @classmethod
    def from_string(cls, value: str) -> "DatasetType":
        """Convert string to DatasetType enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid dataset type: {value}. Must be one of: {', '.join([m.value for m in cls])}"
            ) from None

    @classmethod
    def choices(cls) -> list[str]:
        """Get all available choices as strings for argparse."""
        return [dataset_type.value for dataset_type in cls]


class VideoLoaderType(str, Enum):
    """
    Enumeration for different video loaders.
    """

    TORCHCODEC = "torchcodec"
    TORCHVISION = "torchvision"

    @classmethod
    def from_string(cls, value: str) -> "VideoLoaderType":
        """Convert string to VideoLoader enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid video loader: {value}. Must be one of: {', '.join([m.value for m in cls])}"
            ) from None

    @classmethod
    def choices(cls) -> list[str]:
        """Get all available choices as strings for argparse."""
        return [video_loader.value for video_loader in cls]
