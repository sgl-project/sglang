"""Shared utilities for heter_moe tests."""

from types import SimpleNamespace

import torch

CUDA_AVAILABLE = torch.cuda.is_available()


def init_mock_server_args():
    """Initialize minimal ServerArgs so Triton kernel config lookup works.

    ServerArgs.__post_init__ tries to download model config, so we create
    a bare object bypassing __init__ and set only the fields the kernel
    config path needs.
    """
    try:
        import sglang.srt.server_args as sa

        try:
            sa.get_global_server_args()
        except ValueError:
            mock = object.__new__(sa.ServerArgs)
            mock.enable_deterministic_inference = False
            mock.disable_moe_autotuning = False
            sa._global_server_args = mock
    except Exception:
        pass


def make_topk_output(topk_weights, topk_ids, router_logits=None):
    """Construct a TopKOutput-shaped shim for HeterFusedMoE.forward() tests."""
    return SimpleNamespace(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=router_logits,
    )


if CUDA_AVAILABLE:
    init_mock_server_args()
