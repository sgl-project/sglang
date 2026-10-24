"""Cake (DeepGEMM-port) MegaMoE prepared pipelines on SM100a / SM103a via FlashInfer.

FlashInfer entries (experimental, catalog-driven):

* ``flashinfer.mega_moe_v3``: ``prepare_pipeline(inputs) -> V3Plan``,
  ``prepare_grouped_l2(A, B, SFA, SFB, per_expert_M, *, out=None, packed_a=None,
  packed_b=None)``, ``prepare_grouped_fused(bindings, per_expert_M)``,
  ``prepare_grouped_l1(bindings, per_expert_M)``, ``bind_prepared(...)``,
  ``V3Plan`` (implementation ``flashinfer.experimental.mega_moe_v3.runtime``,
  catalog ``mega_moe_v3.v2`` = ``catalog.json``, 9 programs per arch).
* ``flashinfer.source_mega_moe``: ``prepare_mega_moe(...) = MegaMoEPlan(...)``
  (implementation ``flashinfer.experimental.source_mega_moe.runtime``, catalog
  ``source_mega_moe.v2``).

Contract at FlashInfer ``46340689a5ab``: SM100 (sm_100a, 148 SMs) and SM103
(sm_103a, 152 SMs) only; the route is selected by compute capability AND the
physical SM count (no override). Exported routes: pipeline ``E=384, top_k=6,
H=5120, I=2304`` with ``T in {1, 16, 128, 512}`` FP4 and ``T=16`` FP8;
grouped_l2 / grouped_fused ``E=4, per_expert_M=[1024]*4, H=7168, I=3072``;
grouped_l1 ``E=4, per_expert_M=[256]*4, N=256, K=512``; source MegaMoE
``E=384, top_k=6, H=5120, I=2304, shared=1`` with ``T in {1, 16, 128, 512, 1024,
4096}`` FP4 and ``T=16`` FP8, ``num_ranks=1`` only. Pipeline ``inputs`` dict:
``num_experts/top_k/num_tokens/hidden/intermediate``, ``routed_weight_dtype``
("fp4" / "fp8"), ``activation_clamp``, packed E4M3 ``x_fp8_packed`` + UE8M0
word ``x_sf_packed``, int64 ``topk_idx``, f32 ``topk_weights``, logical gate/up
halves ``w1_fp4/w2_fp4`` (or ``_fp8``) and FP32 power-of-two gran-32 scales
``w1_sf/w2_sf``; output BF16 ``[T, H]``. Source MegaMoE takes the packed
inputs plus a ``weights`` dict (``B1/B2`` uint8 interleaved, ``SFB1/SFB2``
uint32 group-folded, ``SB1/SB2/SSFB1/SSFB2`` one FP8 shared expert), a
zero-initialized uint8 workspace of the catalog ``layout.nbytes`` (128-byte
aligned; allocated when omitted) and an optional ``out``.

CUDA graphs: binding a plan during stream capture raises; bind and warm up
first, then capture ``run()``. Pipeline routes are self-cleaning (one kernel
per ``run()``, counters zeroed once at bind, ``reset()`` rejected); grouped
fused keeps a host ``l1_arrival.zero_()`` inside ``run()`` (captured graph
holds memset + kernel; never time ``launch_without_reset`` alone); grouped L2
repacks scales at prepare and on ``update_scales()``. Source MegaMoE has
kernel-owned reusable counters (no host reset); ``update_inputs`` is 4 copies +
1 scatter; a new plan is needed when addresses / layout change; no concurrent
streams.

Not supported here: other geometries / SM counts, multi-rank (``num_ranks > 1``),
SM120 / SM121.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from sglang.kernels.cake_kernels._support import SM100, SM103
from sglang.kernels.cake_kernels.moe_common import (
    cuda_device_in,
    current_cuda_index,
    device_sm_count,
    modules_available,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.mega_moe_v3"
FI_JIT_MODULE = "flashinfer.experimental.mega_moe_v3.runtime"
FI_SOURCE_MODULE = "flashinfer.source_mega_moe"
FI_SOURCE_JIT_MODULE = "flashinfer.experimental.source_mega_moe.runtime"
ARCHS = (SM100, SM103)
ARCH_NAMES = {SM100: "sm_100a", SM103: "sm_103a"}


def _device_catalogued(device, runtime_module: str) -> bool:
    """Arch + physical SM count must have catalogued routes (resolved from FI's catalog)."""
    import importlib

    from sglang.kernels.cake_kernels._support import device_capability

    index = current_cuda_index(device)
    if not cuda_device_in(index, ARCHS):
        return False
    runtime = importlib.import_module(runtime_module)
    arch = ARCH_NAMES[device_capability(index)]
    return device_sm_count(index) in set(runtime.supported_num_sms(arch))


def supports_mega_moe_v3(device: Optional[torch.device] = None) -> bool:
    """``True`` when ``device`` (default current) has catalogued ``mega_moe_v3`` routes; never raises."""
    try:
        return modules_available(FI_MODULE, FI_JIT_MODULE) and _device_catalogued(
            device, FI_JIT_MODULE
        )
    except Exception:
        return False


def supports_source_mega_moe(device: Optional[torch.device] = None) -> bool:
    """``True`` when ``device`` has catalogued ``source_mega_moe`` routes; never raises."""
    try:
        return modules_available(
            FI_SOURCE_MODULE, FI_SOURCE_JIT_MODULE
        ) and _device_catalogued(device, FI_SOURCE_JIT_MODULE)
    except Exception:
        return False


def get_mega_moe_v3_plan_class():
    from flashinfer.mega_moe_v3 import V3Plan

    return V3Plan


def get_source_mega_moe_plan_class():
    from flashinfer.source_mega_moe import MegaMoEPlan

    return MegaMoEPlan


def prepare_mega_moe_pipeline(inputs: Mapping[str, Any]):
    """Forward to ``flashinfer.mega_moe_v3.prepare_pipeline``; returns ``V3Plan`` (``run()`` -> BF16 ``[T, H]``)."""
    from flashinfer.mega_moe_v3 import prepare_pipeline

    return prepare_pipeline(inputs)


def prepare_mega_moe_grouped_l2(
    A: torch.Tensor,
    B: torch.Tensor,
    SFA: torch.Tensor,
    SFB: torch.Tensor,
    per_expert_M: Sequence[int],
    *,
    out: Optional[torch.Tensor] = None,
    packed_a: Optional[torch.Tensor] = None,
    packed_b: Optional[torch.Tensor] = None,
):
    """Forward to ``flashinfer.mega_moe_v3.prepare_grouped_l2``."""
    from flashinfer.mega_moe_v3 import prepare_grouped_l2

    return prepare_grouped_l2(
        A, B, SFA, SFB, per_expert_M, out=out, packed_a=packed_a, packed_b=packed_b
    )


def prepare_mega_moe_grouped_fused(
    bindings: Mapping[str, Any], per_expert_M: Sequence[int]
):
    """Forward to ``flashinfer.mega_moe_v3.prepare_grouped_fused``."""
    from flashinfer.mega_moe_v3 import prepare_grouped_fused

    return prepare_grouped_fused(bindings, per_expert_M)


def prepare_mega_moe_grouped_l1(
    bindings: Mapping[str, Any], per_expert_M: Sequence[int]
):
    """Forward to ``flashinfer.mega_moe_v3.prepare_grouped_l1``."""
    from flashinfer.mega_moe_v3 import prepare_grouped_l1

    return prepare_grouped_l1(bindings, per_expert_M)


def bind_mega_moe_prepared(
    surface: str,
    args: Mapping[str, Any],
    stage_bindings: Mapping[str, Mapping[str, Any]],
    outputs: Any,
    *,
    reset_storage: Any = None,
    reset_buffers: Sequence[Any] = (),
    preparation: Sequence[str] = (),
    owners: Sequence[Any] = (),
):
    """Forward to ``flashinfer.mega_moe_v3.bind_prepared`` (low-level catalog binding)."""
    from flashinfer.mega_moe_v3 import bind_prepared

    return bind_prepared(
        surface,
        args,
        stage_bindings,
        outputs,
        reset_storage=reset_storage,
        reset_buffers=reset_buffers,
        preparation=preparation,
        owners=owners,
    )


def prepare_source_mega_moe(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    weights: Mapping[str, torch.Tensor],
    num_experts: int,
    intermediate: int,
    routed_weight_dtype: str = "fp4",
    num_shared_experts: int = 1,
    activation_clamp: float = 10.0,
    fast_math: bool = True,
    workspace: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    descriptor_workspace: Optional[torch.Tensor] = None,
):
    """Forward to ``flashinfer.source_mega_moe.prepare_mega_moe``; returns ``MegaMoEPlan``."""
    from flashinfer.source_mega_moe import prepare_mega_moe

    return prepare_mega_moe(
        x,
        x_sf,
        topk_idx,
        topk_weights,
        weights=weights,
        num_experts=num_experts,
        intermediate=intermediate,
        routed_weight_dtype=routed_weight_dtype,
        num_shared_experts=num_shared_experts,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
        workspace=workspace,
        out=out,
        descriptor_workspace=descriptor_workspace,
    )
