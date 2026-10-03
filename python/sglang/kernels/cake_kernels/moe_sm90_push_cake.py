"""Cake SM90 native BF16 push mega-MoE backend (``sm90_bf16_bf16_bf16_push_cake``) via FlashInfer.

FlashInfer entries: ``flashinfer.moe_ep.Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig``
(dataclass megakernel config; pass as ``MegaConfig(megakernel=...)`` to
``flashinfer.moe_ep.MoEEpLayer``), ``Sm90CakeBf16MegaKernelBackend`` (the
``MegaKernelBackend`` the registry resolves for that config) and
``flashinfer.moe_ep.preprocess_sm90_push_cake_bf16_mega_weights(weights, *,
intermediate_size, hidden_size, num_local_experts)`` (implementation
``flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cake``; the
FC1 (fused SwiGLU) and FC2 grouped GEMMs are Cake-generated WGMMA kernels on
the SM90 push protocol, JIT generators in
``flashinfer.moe_ep.kernel_src.sm90.cake_bf16_megamoe``). Contract at
FlashInfer ``46340689a5ab``: SM90 (Hopper) only; BF16 activations / weights
into both expert GEMMs, fp32 accumulation, bf16 intermediate, dispatch
payload, combine wire and output (nothing quantized); ``top_k in {1, 2, 4, 6,
8}``; ``token_hidden_size % 256 == 0``; ``intermediate_size % 128 == 0``;
``num_experts % world_size == 0``; single-node EP group ``world_size <= 32``;
``capacity_factor in (0, 1]``; optional fp32 ``clamp_limit`` on the FC1
gate/up pre-activations; canonical BF16 ``MoEWeightPack`` only (no scale
planes): ``w13 [E_local, 2I, H]`` (gate rows then up rows), ``w2 [E_local, H, I]``.

Process-group requirement: the EP group is a ``torch.distributed`` group
(``TORCH_DIST`` runtime requirement); P2P / IPC peer handles are exchanged at
workspace setup under ``init_timeout_s``. ``BootstrapConfig.stream`` must be 0
(launches on the current torch stream). This adapter forwards the config /
weights preprocessor / backend class only; process groups, ``MoEEpLayer``
bootstrap and workspaces stay in the ``sglang.srt`` runtime integration.

Not supported here: quantized weights, ``world_size > 32``, multi-node EP,
non-Hopper devices.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

from sglang.kernels.cake_kernels._support import SM90
from sglang.kernels.cake_kernels.moe_common import (
    cuda_device_in,
    current_cuda_index,
    modules_available,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cake"
FI_JIT_MODULE = "flashinfer.moe_ep.kernel_src.sm90.cake_bf16_megamoe"
ARCHS = (SM90,)
KERNEL_NAME = "sm90_bf16_bf16_bf16_push_cake"
TOP_K_VALUES = (1, 2, 4, 6, 8)
HIDDEN_MULTIPLE = 256
INTERMEDIATE_MULTIPLE = 128
MAX_WORLD_SIZE = 32


def supports_sm90_push_cake_megamoe(
    *,
    intermediate_size: int,
    top_k: int,
    token_hidden_size: int,
    num_experts: int,
    world_size: int,
    capacity_factor: float = 1.0,
    clamp_limit: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> bool:
    """Admission check mirroring ``Sm90CakeBf16MegaKernelBackend.validate_init``; never raises."""
    try:
        if not (
            modules_available(FI_MODULE, FI_JIT_MODULE)
            and cuda_device_in(current_cuda_index(device), ARCHS)
        ):
            return False
        if int(top_k) not in TOP_K_VALUES:
            return False
        if (
            int(token_hidden_size) % HIDDEN_MULTIPLE
            or int(intermediate_size) % INTERMEDIATE_MULTIPLE
        ):
            return False
        if int(world_size) <= 0 or int(world_size) > MAX_WORLD_SIZE:
            return False
        if int(num_experts) % int(world_size):
            return False
        cf = float(capacity_factor)
        if not math.isfinite(cf) or not 0.0 < cf <= 1.0:
            return False
        if clamp_limit is not None:
            cl = float(clamp_limit)
            if not math.isfinite(cl) or cl <= 0.0:
                return False
        return True
    except Exception:
        return False


def sm90_push_cake_megamoe_config(
    intermediate_size: int,
    top_k: int,
    *,
    capacity_factor: float = 1.0,
    dedup_dispatch: bool = True,
    clamp_limit: Optional[float] = None,
    allow_unverified_p2p: bool = False,
    init_timeout_s: float = 600.0,
):
    """``Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig`` for ``MegaConfig(megakernel=...)``."""
    from flashinfer.moe_ep import Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig

    return Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig(
        intermediate_size=intermediate_size,
        top_k=top_k,
        kernel_name=KERNEL_NAME,
        capacity_factor=capacity_factor,
        dedup_dispatch=dedup_dispatch,
        clamp_limit=clamp_limit,
        allow_unverified_p2p=allow_unverified_p2p,
        init_timeout_s=init_timeout_s,
    )


def preprocess_sm90_push_cake_bf16_mega_weights(
    weights: Any,
    *,
    intermediate_size: int,
    hidden_size: int,
    num_local_experts: int,
):
    """Interleave canonical BF16 ``w13`` gate/up rows for the fused FC1 kernel (``TransformedMegaWeights``)."""
    from flashinfer.moe_ep import preprocess_sm90_push_cake_bf16_mega_weights as fi_pre

    return fi_pre(
        weights,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        num_local_experts=num_local_experts,
    )


def get_sm90_push_cake_backend_class():
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cake import (
        Sm90CakeBf16MegaKernelBackend,
    )

    return Sm90CakeBf16MegaKernelBackend


def get_sm90_push_cake_config_class():
    from flashinfer.moe_ep import Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig

    return Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig
