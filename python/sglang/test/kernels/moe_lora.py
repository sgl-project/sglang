"""Shared helpers for the MoE LoRA runner numerics tests."""

import torch


def cutedsl_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0):
        return False
    try:
        import cuda.bindings.driver  # noqa: F401
        import cutlass  # noqa: F401
    except Exception:
        return False
    return True


def make_token_lora_mapping(traffic: str, num_tokens: int) -> torch.Tensor:
    if traffic == "active":
        return torch.zeros(num_tokens, dtype=torch.int32)
    if traffic == "mixed":
        # Batch preparation normalizes every inactive resident assignment
        # to -1 before any MoE layer consumes the mapping.
        pattern = torch.tensor([0, -1, -1, 0, -1, -1], dtype=torch.int32)
        return pattern.repeat((num_tokens + pattern.numel() - 1) // pattern.numel())[
            :num_tokens
        ].contiguous()
    if traffic == "base_only":
        # Config choice and graph shape stay fixed while every token follows
        # the base sentinel; one topology serves every batch.
        return torch.full((num_tokens,), -1, dtype=torch.int32)
    raise AssertionError(f"unknown traffic pattern {traffic}")


def standalone_output_allocation(runner, *, num_tokens, dtype, device):
    """Match eager output geometry without requiring a serving TP group."""
    return torch.empty((num_tokens, runner.hidden_size), dtype=dtype, device=device)


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).norm() / b.norm().clamp(min=1e-12)
