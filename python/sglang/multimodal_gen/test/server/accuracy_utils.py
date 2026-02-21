from __future__ import annotations

import os
from typing import Any, Dict

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def seed_and_broadcast(seed: int, tensor: torch.Tensor) -> torch.Tensor:
    """Seed and broadcast tensor across ranks for determinism."""
    torch.manual_seed(seed)
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        torch.distributed.broadcast(tensor, src=0)
    return tensor


def log_tensor_stats(name: str, tensor: torch.Tensor) -> None:
    if tensor is None:
        return
    if os.environ.get("SGLANG_DIFFUSION_ACC_DEBUG", "0") != "1":
        return
    t = tensor.detach().float().cpu()
    logger.info(
        "[%s] stats: shape=%s mean=%.6f std=%.6f min=%.6f max=%.6f",
        name,
        list(t.shape),
        t.mean().item(),
        t.std().item(),
        t.min().item(),
        t.max().item(),
    )


def log_inputs(prefix: str, inputs: Dict[str, Any]) -> None:
    if os.environ.get("SGLANG_DIFFUSION_ACC_DEBUG", "0") != "1":
        return
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            log_tensor_stats(f"{prefix}.{k}", v)


def extract_output_tensor(output: Any) -> torch.Tensor:
    """Best-effort extraction of a tensor from model outputs."""
    if isinstance(output, torch.Tensor):
        return output
    if getattr(output, "last_hidden_state", None) is not None:
        return output.last_hidden_state
    if getattr(output, "hidden_states", None):
        return output.hidden_states[-1]
    if getattr(output, "pooler_output", None) is not None:
        return output.pooler_output
    if getattr(output, "logits", None) is not None:
        return output.logits
    if (
        isinstance(output, (list, tuple))
        and output
        and isinstance(output[0], torch.Tensor)
    ):
        return output[0]
    raise ValueError(f"Could not extract tensor from output of type {type(output)}")
