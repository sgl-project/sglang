# SPDX-License-Identifier: Apache-2.0
import math
import socket
import sys
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import torch
import torch.distributed as dist

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

DEEPSYMM_DIST_NAME = "deep-symm"
DEEPSYMM_PACKAGE = "applications.ai.gpu.deepsymm"
DEEPSYMM_RING_PRECISIONS = ("bf16", "fp8", "fp8_mixed", "mxfp4")
# deep_symm runs these only for B == 1.
_MX_PRECISIONS = ("fp8_mixed", "mxfp4")


def _parse_ring_precision(value: str) -> str | None:
    precision = value.strip().lower()
    if precision == "off":
        return None
    if precision not in DEEPSYMM_RING_PRECISIONS:
        raise ValueError(
            "SGLANG_DIFFUSION_XPU_DEEPSYMM_RING_PRECISION must be 'off' or one of "
            f"{DEEPSYMM_RING_PRECISIONS}, got {precision!r}"
        )
    return precision


# Read once so the default "off" path never reaches a torch.compiler.disable call.
DEEPSYMM_RING_PRECISION = _parse_ring_precision(
    envs.SGLANG_DIFFUSION_XPU_DEEPSYMM_RING_PRECISION
)

_ring_group_on_one_host_cache: dict[int, bool] = {}
# (query shape, key shape, precision) of the arenas deep_symm holds, and of those
# the current request has used (see begin_deepsymm_ring_request).
_held_ring_shapes: set[tuple] = set()
_request_ring_shapes: set[tuple] = set()
# A failed ring pass leaves a poisoned relay cached in deep_symm; only a clear drops it.
_clear_before_next_ring_call = False


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


def _ring_group_on_one_host(ring_group: dist.ProcessGroup) -> bool:
    """deep_symm's shm control block is node-local, so a ring spanning hosts hangs."""
    key = id(ring_group)
    if key not in _ring_group_on_one_host_cache:
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_sp_group,
        )

        # Gather over the SP group's gloo group: an object gather on the ring's
        # device (oneCCL) group is avoidable, and every SP rank reaches the first
        # eligible ring layer together.
        cpu_group = get_sp_group().cpu_group
        sp_hosts = [None] * dist.get_world_size(group=cpu_group)
        dist.all_gather_object(sp_hosts, socket.gethostname(), group=cpu_group)
        host_of = dict(zip(dist.get_process_group_ranks(cpu_group), sp_hosts))
        hosts = [host_of[r] for r in dist.get_process_group_ranks(ring_group)]
        _ring_group_on_one_host_cache[key] = len(set(hosts)) == 1
        if not _ring_group_on_one_host_cache[key]:
            logger.warning(
                "SGLANG_DIFFUSION_XPU_DEEPSYMM_RING_PRECISION is set, but the ring "
                "group spans hosts %s; using the torch ring instead",
                sorted(set(hosts)),
            )
    return _ring_group_on_one_host_cache[key]


@lru_cache(maxsize=None)
def _warn_mx_batch_fallback(precision: str) -> None:
    logger.warning(
        "DeepSymm ring %s runs only for batch 1; batch > 1 requests use the bf16 ring",
        precision,
    )


@torch.compiler.disable
def deepsymm_ring_precision(
    query: torch.Tensor,
    attn_impl,
    *,
    ring_group: dist.ProcessGroup,
    causal: bool,
    is_cross_attention: bool,
    dropout_p: float = 0.0,
) -> str | None:
    precision = DEEPSYMM_RING_PRECISION
    if precision is None:
        return None
    if query.device.type != "xpu" or query.dtype != torch.bfloat16 or dropout_p != 0.0:
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
    if not _ring_group_on_one_host(ring_group):
        return None
    # Fail on the first eligible layer, before any ring collective, when the
    # package is missing.
    _import_ring_attn_ipc_forward()
    num_kv_heads = attn_impl.num_kv_heads
    if head_size != 128 or num_kv_heads not in (None, attn_impl.num_heads):
        return "bf16"
    if precision in _MX_PRECISIONS and query.shape[0] != 1:
        _warn_mx_batch_fallback(precision)
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
    global _clear_before_next_ring_call
    shape = (tuple(query.shape), tuple(key.shape), precision)
    if _clear_before_next_ring_call:
        clear_deepsymm_ring_cache()
    elif shape not in _held_ring_shapes and _held_ring_shapes - _request_ring_shapes:
        # deep_symm is about to allocate arenas for a new shape while it still
        # holds a previous request's: free those first (all ring ranks see the
        # same shapes in the same order, so they all clear here together).
        clear_deepsymm_ring_cache()
    _held_ring_shapes.add(shape)
    _request_ring_shapes.add(shape)
    try:
        return _import_ring_attn_ipc_forward()(
            process_group=ring_group,
            q=query.contiguous(),
            k=key.contiguous(),
            v=value.contiguous(),
            return_softmax_lse=False,
            precision=precision,
        )
    except Exception:
        # Without a clear, every later request with this shape would hit the
        # poisoned relay; the ranks whose pass failed clear at their next ring call.
        _clear_before_next_ring_call = True
        raise


def clear_deepsymm_ring_cache(*, forget_groups: bool = False) -> None:
    """Free deep_symm's IPC arenas; a collective, so every ring rank calls it together.

    ``forget_groups`` (process-group teardown) also drops the per-group host check,
    keyed by ``id(ring_group)``: a recreated group may reuse a stale id on some
    ranks only, which would then skip the check's all_gather and hang the rest.
    """
    global _clear_before_next_ring_call
    _held_ring_shapes.clear()
    _clear_before_next_ring_call = False
    if forget_groups:
        _ring_group_on_one_host_cache.clear()
        _request_ring_shapes.clear()
    if "deep_symm.ring_attn" not in sys.modules:
        return
    from deep_symm.ring_attn import ring_attn_ipc_clear_cache

    ring_attn_ipc_clear_cache()


def begin_deepsymm_ring_request() -> None:
    """Before each forward, on every rank. deep_symm keys its arenas on the K/V
    shape and never evicts them; the first ring call of a later request that needs
    a shape it does not hold frees the arenas this request has not used, before
    allocating (a new resolution, frame count or batch would otherwise pin another
    set next to the old one). Shapes are taken at the ring call, so they are the
    resolved ones (e.g. from an I2V condition image)."""
    _request_ring_shapes.clear()
