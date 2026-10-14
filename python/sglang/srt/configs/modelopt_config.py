# Configuration for NVIDIA ModelOpt quantization integration
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelOptConfig:
    """Configuration for NVIDIA ModelOpt quantization operations.

    This configuration class holds parameters for ModelOpt quantization,
    checkpoint management, and model export operations.

    Args:
        quant: Quantization method/type (e.g., "fp8", "fp4")
        checkpoint_restore_path: Path to restore ModelOpt checkpoint from
        checkpoint_save_path: Path to save ModelOpt checkpoint to
        export_path: Path to export quantized model in HuggingFace format
        quantize_and_serve: Whether to quantize and serve in one step
    """

    quant: Optional[str] = None
    checkpoint_restore_path: Optional[str] = None
    checkpoint_save_path: Optional[str] = None
    export_path: Optional[str] = None
    quantize_and_serve: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Add any validation logic if needed
        pass
