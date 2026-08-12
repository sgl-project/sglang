"""Canonical string→dtype map for diffusion precision configs."""

import torch

PRECISION_TO_TYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

__all__ = ["PRECISION_TO_TYPE"]
