"""https://arxiv.org/abs/2510.14624: Efficient Video Sampling: Pruning Temporally Redundant Tokens for Faster VLM Inference"""

from .evs_module import EVS, EVSConfig, EVSEmbeddingResult
from .evs_processor import EVSProcessor

__all__ = [
    "EVS",
    "EVSConfig",
    "EVSEmbeddingResult",
    "EVSProcessor",
]
