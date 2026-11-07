# SPDX-License-Identifier: Apache-2.0
import math
import sys
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import torch

from sglang.multimodal_gen import envs

DEEPSYMM_DIST_NAME = "deep-symm"
DEEPSYMM_PACKAGE = "applications.ai.gpu.deepsymm"
DEEPSYMM_RING_PRECISIONS = ("bf16", "fp8", "fp8_mixed", "mxfp4")


@lru_cache(maxsize=1)
def _import_ring_attn_ipc_forward():
    try:
        deepsymm_version = version(DEEPSYMM_DIST_NAME)
    except PackageNotFoundError as e:
        raise ImportError(
            f"XPU DeepSymm ring attention requires package {DEEPSYMM_PACKAGE!r} "
            f"(distribution {DEEPSYMM_DIST_NAME!r}), which is not installed; "
            "build it from source into the same environment as SGLang."
        ) from e
    try:
        from deep_symm.ring_attn import ring_attn_ipc_forward
    except (ImportError, OSError) as e:
        raise ImportError(
            f"Package {DEEPSYMM_PACKAGE!r} ({DEEPSYMM_DIST_NAME} {deepsymm_version}) "
            f"is installed but deep_symm.ring_attn failed to load: {e!r}."
        ) from e
    return ring_attn_ipc_forward


@torch.compiler.disable
def deepsymm_ring_precision(
    query: torch.Tensor, attn_impl, *, causal: bool, is_cross_attention: bool
) -> str | None:
    precision = envs.SGLANG_DIFFUSION_XPU_DEEPSYMM_RING_PRECISION.strip().lower()
    if precision == "off":
        return None
    if precision not in DEEPSYMM_RING_PRECISIONS:
        raise ValueError(
            "SGLANG_DIFFUSION_XPU_DEEPSYMM_RING_PRECISION must be 'off' or one of "
            f"{DEEPSYMM_RING_PRECISIONS}, got {precision!r}"
        )
    if query.device.type != "xpu" or query.dtype != torch.bfloat16:
        return None
    from sglang.multimodal_gen.runtime.layers.attention.backends.xpu_backend import (
        XPUAttentionImpl,
    )

    if not isinstance(attn_impl, XPUAttentionImpl):
        return None
    head_size = attn_impl.head_size
    if (
        causal
        or is_cross_attention
        or head_size not in (64, 128)
        or not math.isclose(attn_impl.softmax_scale, head_size**-0.5, rel_tol=1e-6)
    ):
        return None
    num_kv_heads = attn_impl.num_kv_heads
    if head_size != 128 or num_kv_heads not in (None, attn_impl.num_heads):
        return "bf16"
    return precision


@torch.compiler.disable
def deepsymm_ring_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    ring_group: torch.distributed.ProcessGroup,
    precision: str,
) -> torch.Tensor:
    return _import_ring_attn_ipc_forward()(
        process_group=ring_group,
        q=query.contiguous(),
        k=key.contiguous(),
        v=value.contiguous(),
        return_softmax_lse=False,
        precision=precision,
    )


def clear_deepsymm_ring_cache() -> None:
    if "deep_symm.ring_attn" not in sys.modules:
        return
    from deep_symm.ring_attn import ring_attn_ipc_clear_cache

    ring_attn_ipc_clear_cache()
