from __future__ import annotations

import os
from typing import Any, Dict

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.accuracy_config import (
    DEFAULT_TEXT_ENCODER_VOCAB_SIZE,
    TEXT_ENCODER_INPUT_SEED,
    TEXT_ENCODER_TOKEN_LENGTH,
    TEXT_ENCODER_TOKEN_MAX,
    TEXT_ENCODER_TOKEN_MIN,
)

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


def _config_to_dict(config: Any) -> Dict[str, Any]:
    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict):
        return {}
    config_dict = to_dict()
    return config_dict if isinstance(config_dict, dict) else {}


def resolve_text_encoder_vocab_size(config: Any) -> int:
    config_dict = _config_to_dict(config)
    vocab_size = config_dict.get("vocab_size")
    if isinstance(vocab_size, int) and vocab_size > 0:
        return vocab_size

    text_config = config_dict.get("text_config")
    if isinstance(text_config, dict):
        nested_vocab_size = text_config.get("vocab_size")
        if isinstance(nested_vocab_size, int) and nested_vocab_size > 0:
            return nested_vocab_size

    return DEFAULT_TEXT_ENCODER_VOCAB_SIZE


def build_deterministic_text_encoder_inputs(
    config: Any, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    vocab_size = resolve_text_encoder_vocab_size(config)
    max_token_id = max(
        TEXT_ENCODER_TOKEN_MIN + 1, min(vocab_size, TEXT_ENCODER_TOKEN_MAX)
    )

    torch.manual_seed(TEXT_ENCODER_INPUT_SEED)
    input_ids = torch.randint(
        TEXT_ENCODER_TOKEN_MIN,
        max_token_id,
        (1, TEXT_ENCODER_TOKEN_LENGTH),
        device="cpu",
        dtype=torch.long,
    ).to(device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def resolve_text_encoder_forward_module(model: nn.Module) -> nn.Module:
    get_encoder = getattr(model, "get_encoder", None)
    return get_encoder() if callable(get_encoder) else model


def extract_output_tensor(output: Any) -> torch.Tensor:
    """Best-effort extraction of a tensor from model outputs."""
    if isinstance(output, torch.Tensor):
        return output

    last_hidden_state = getattr(output, "last_hidden_state", None)
    if last_hidden_state is not None:
        return last_hidden_state

    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states:
        return hidden_states[-1]

    pooler_output = getattr(output, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output

    logits = getattr(output, "logits", None)
    if logits is not None:
        return logits

    if (
        isinstance(output, (list, tuple))
        and output
        and isinstance(output[0], torch.Tensor)
    ):
        return output[0]
    raise ValueError(f"Could not extract tensor from output of type {type(output)}")
