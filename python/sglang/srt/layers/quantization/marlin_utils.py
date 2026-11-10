# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

import numpy
import torch

from sglang.srt.layers.quantization.utils import (
    get_scalar_types,
    pack_cols,
    unpack_cols,
)
from sglang.srt.utils import get_device_capability, is_cuda
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

from sglang.kernels.sm70_paths import sm70_prebuilt_dir
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)

_is_cuda = is_cuda()

if _is_cuda:
    from sglang.kernels.ops.gemm.gptq_marlin import gptq_marlin_gemm

logger = logging.getLogger(__name__)

ScalarType, scalar_types = get_scalar_types()

GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_MAX_PARALLEL = 16

MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]
# NVFP SUPPORT 16, while MXFP4 supports 32 and 16.
FP4_MARLIN_SUPPORTED_GROUP_SIZES = [16, 32]

# In case there is a performance issue with Marlin, the variable below can be
# changed to False, which allows Marlin to perform global reductions in fp16
# precision (instead of fp32), and therefore, save on some memory movements.
USE_FP32_REDUCE_DEFAULT = True


@dataclass
class MarlinLinearLayerConfig:
    full_weight_shape: tuple[int, int]  # [in, out]
    partition_weight_shape: tuple[int, int]
    weight_type: ScalarType
    act_type: torch.dtype
    group_size: int
    zero_points: bool
    has_g_idx: bool


@lru_cache(maxsize=1)
def _sm70_marlin_v100_available() -> bool:
    """True iff the marlin_v100 SM70 MoE kernel is built and loadable.

    The stock JIT Marlin MoE kernel is an empty stub on SM70 (writes nothing,
    silently zeroing routed-expert output). marlin_v100 supplies real WMMA
    kernels; when it is installed (scripts/setup_v100_marlin.sh) we re-enable
    Marlin quant-type support and the gptq->gptq_marlin auto-promotion so that
    AWQ/GPTQ MoE models work on V100.
    """
    try:
        major, _ = get_device_capability()
        if major != 7:
            return False
        from sglang.kernels.ops.moe.moe_wna16_marlin import _load_marlin_v100_op

        return _load_marlin_v100_op() is not None
    except Exception:
        return False


@lru_cache(maxsize=1)
def _sm70_marlin_v100_repack_ops():
    """Load marlin_v100's dense _C extension on SM70 and return its repack ops.

    sglang's JIT ``gptq_marlin_repack``/``awq_marlin_repack`` are
    ``__CUDA_ARCH__ < 800`` stubs on SM70 (the kernel body is empty, so the
    output stays zeroed) -- meaning MoE expert weights are repacked to zeros
    during loading and the marlin GEMM then reads all-zero weights. marlin_v100
    ships real SM70 repack kernels in its dense extension; load it here.

    Returns ``(gptq_repack, awq_repack)`` callables, or ``(None, None)`` if the
    extension is unavailable (non-SM70 or not built).
    """
    if not _sm70_marlin_v100_available():
        return None, None
    import glob
    import os

    import torch  # noqa: F811

    here = os.path.dirname(os.path.abspath(__file__))
    # setup_v100_marlin.sh installs the portable artifacts beside SGLang's
    # JIT kernels.  The old lookup only checked this quantization source
    # directory and ~/marlin_v100, so container images (which intentionally do
    # not retain the full source checkout) silently fell back to SGLang's SM80+
    # zero-output repack stub on V100.
    jit_kernel_dir = str(sm70_prebuilt_dir())
    home = os.path.expanduser("~")
    candidates = sorted(
        glob.glob(os.path.join(jit_kernel_dir, "_sm70_marlin_v100_dense*.so"))
    )
    candidates += sorted(glob.glob(os.path.join(here, "_sm70_marlin_v100_dense*.so")))
    candidates += sorted(glob.glob(os.path.join(home, "marlin_v100", "vllm", "_C*.so")))
    for path in candidates:
        try:
            torch.ops.load_library(path)
        except Exception:
            continue
        gptq = getattr(torch.ops._C, "gptq_marlin_repack", None)
        awq = getattr(torch.ops._C, "awq_marlin_repack", None)
        if gptq is not None:
            # marlin_v100's op takes an extra trailing `is_a_8bit` bool; wrap it
            # so callers can use the sglang 5-arg signature unchanged.
            def _gptq(b_q_weight, perm, size_k, size_n, num_bits, _op=gptq):
                return _op(b_q_weight, perm, size_k, size_n, num_bits, False)

            def _awq(b_q_weight, size_k, size_n, num_bits, _op=awq):
                return _op(b_q_weight, size_k, size_n, num_bits, False)

            logger.info(
                "SM70 (V100): using marlin_v100 gptq/awq repack from %s "
                "(stock JIT repack is a zero-output stub below sm_80).",
                path,
            )
            return _gptq, _awq
    return None, None


@lru_cache(maxsize=1)
def _sm70_marlin_v100_gemm_op():
    """Return marlin_v100's dense SM70 GEMM op when it is installed.

    SGLang's in-tree Marlin CUDA source intentionally compiles empty kernels
    below SM80.  The V100 integration therefore has to use the dense
    ``marlin_v100`` extension for both repacking and execution; using only its
    repacker still produces zero output when the in-tree GEMM stub is called.
    Loading the repack ops also loads the shared library that registers
    ``torch.ops._C.marlin_gemm``.
    """
    if not _sm70_marlin_v100_available():
        return None

    gptq_repack, _ = _sm70_marlin_v100_repack_ops()
    if gptq_repack is None:
        return None

    op = getattr(torch.ops._C, "marlin_gemm", None)
    if op is None:
        logger.warning(
            "SM70 (V100): marlin_v100 repack is available, but its dense "
            "marlin_gemm op was not registered."
        )
        return None
    return op


# For binary size and compile time, we don't support the same types for with and
#  without runtime zero-point. We support common cases, i.e. AWQ and GPTQ.
#  TODO: we may want to move this into the C++ so its closer to the actual impl
def query_marlin_supported_quant_types(
    has_zp: Optional[bool] = None,
    include_fp_type: bool = True,
    device_capability: Optional[int] = None,
):
    if device_capability is None:
        major, minor = get_device_capability()
        capability = major * 10 + minor if major is not None else None
        device_capability = -1 if capability is None else capability

    if device_capability < 80:
        # The stock Marlin kernel is an SM80+ stub below sm_80. On SM70 (V100)
        # the marlin_v100 integration (scripts/setup_v100_marlin.sh) supplies
        # real WMMA kernels, so when it is loadable we report the same
        # quant-type support matrix as SM80+.
        if device_capability == 70 and _sm70_marlin_v100_available():
            pass
        else:
            return []

    # - has_zp is True: return quant_types that has zero points
    # - has_zp is False: return quant_types that has not zero points
    # - has_zp is None: both
    if has_zp is None:
        types0 = query_marlin_supported_quant_types(
            False, include_fp_type, device_capability
        )
        types1 = query_marlin_supported_quant_types(
            True, include_fp_type, device_capability
        )
        return types0 + types1

    if has_zp:
        # AWQ style, unsigned + runtime zero-point
        return [scalar_types.uint4]
    else:
        # GPTQ style, unsigned + symmetric bias
        res = [scalar_types.uint4b8, scalar_types.uint8b128]
        if include_fp_type:
            res += [scalar_types.float8_e4m3fn, scalar_types.float4_e2m1f]
        return res


def _check_marlin_supported(
    quant_type: ScalarType,
    group_size: Optional[int],
    has_zp: bool,
    device_capability: Optional[int] = None,
) -> tuple[bool, Optional[str]]:
    if device_capability is None:
        major, minor = get_device_capability()
        capability = major * 10 + minor if major is not None else None
        device_capability = -1 if capability is None else capability

    supported_types = query_marlin_supported_quant_types(
        has_zp, True, device_capability
    )

    if quant_type not in supported_types:
        return (
            False,
            f"Marlin does not support weight_bits = {quant_type}. "
            f"Only types = {supported_types} "
            f"are supported (for group_size = {group_size}, "
            f"device_capability = {device_capability}, zp = {has_zp}).",
        )
    if quant_type == scalar_types.float4_e2m1f:
        allowed_group_sizes = FP4_MARLIN_SUPPORTED_GROUP_SIZES
    else:
        allowed_group_sizes = MARLIN_SUPPORTED_GROUP_SIZES
    if group_size is None or group_size not in allowed_group_sizes:
        return (
            False,
            f"Marlin does not support group_size = {group_size} for "
            f"quant_type = {quant_type}. Only group_sizes = {allowed_group_sizes} "
            "are supported.",
        )

    return True, None


def check_marlin_supported(
    quant_type: ScalarType,
    group_size: int,
    has_zp: bool = False,
    device_capability: Optional[int] = None,
) -> bool:
    cond, _ = _check_marlin_supported(quant_type, group_size, has_zp, device_capability)
    return cond


def verify_marlin_supported(
    quant_type: ScalarType, group_size: int, has_zp: bool = False
) -> None:
    cond, err_msg = _check_marlin_supported(quant_type, group_size, has_zp)
    if not cond:
        assert err_msg is not None
        raise ValueError(err_msg)


def verify_marlin_supports_shape(
    output_size_per_partition: int,
    input_size_per_partition: int,
    input_size: int,
    group_size: int,
) -> None:
    # Validate output_size_per_partition
    if output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N != 0:
        raise ValueError(
            f"Weight output_size_per_partition = "
            f"{output_size_per_partition} is not divisible by "
            f" min_thread_n = {GPTQ_MARLIN_MIN_THREAD_N}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq."
        )

    # Validate input_size_per_partition
    if input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K != 0:
        raise ValueError(
            f"Weight input_size_per_partition = "
            f"{input_size_per_partition} is not divisible "
            f"by min_thread_k = {GPTQ_MARLIN_MIN_THREAD_K}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq."
        )

    if group_size < input_size and input_size_per_partition % group_size != 0:
        raise ValueError(
            f"Weight input_size_per_partition = {input_size_per_partition}"
            f" is not divisible by group_size = {group_size}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq."
        )


def check_marlin_supports_shape(
    output_size_per_partition: int,
    input_size_per_partition: int,
    input_size: int,
    group_size: int,
) -> tuple[bool, Optional[str]]:
    try:
        verify_marlin_supports_shape(
            output_size_per_partition, input_size_per_partition, input_size, group_size
        )
    except ValueError as e:
        return False, e.__str__()
    return True, None


def check_marlin_supports_layer(layer: LinearBase, group_size: int) -> bool:
    output_size_per_partition = (
        getattr(layer, "output_size_per_partition", None) or layer.output_size
    )
    input_size_per_partition = (
        getattr(layer, "input_size_per_partition", None) or layer.input_size
    )

    return check_marlin_supports_shape(
        output_size_per_partition=output_size_per_partition,
        input_size_per_partition=input_size_per_partition,
        input_size=layer.input_size,
        group_size=group_size,
    )[0]


def check_moe_marlin_supports_layer(
    layer: FusedMoE, group_size: int, allow_tile_padding: bool = False
) -> bool:
    hidden_size = layer.hidden_size
    intermediate_size_per_partition = layer.intermediate_size_per_partition
    # apply_router_weight_on_input is not supported for moe marlin
    supports_router_weight = not layer.moe_runner_config.apply_router_weight_on_input
    if layer.moe_runner_config.is_gated:
        supports_activation = layer.moe_runner_config.activation in {"silu", "situ"}
    else:
        supports_activation = layer.moe_runner_config.activation in {
            "silu",
            "relu2",
        }

    if allow_tile_padding:
        # Thread-tile misalignment can be fixed by zero-padding the expert
        # intermediate dimension before Marlin repack. The original K still
        # needs to fit a Marlin tile family, and quant groups must stay whole.
        supports_shape = (
            hidden_size % 64 == 0 and intermediate_size_per_partition % group_size == 0
        )
    else:
        # gate-up: (n, k) = (intermediate_size_per_partition * 2, hidden_size)
        # down: (n, k) = (hidden_size, intermediate_size_per_partition)
        # moe marlin requires n % 128 == 0 and k % 64 == 0
        supports_shape = (
            hidden_size % 128 == 0
            and intermediate_size_per_partition % max(64, group_size) == 0
        )
    supports_group_size = group_size in [-1, 32, 64, 128]
    return (
        supports_shape
        and supports_group_size
        and supports_router_weight
        and supports_activation
    )


def marlin_make_workspace(
    device: torch.device, max_blocks_per_sm: int = 1
) -> torch.Tensor:
    # In the new marlin kernel, we use the num of threadblocks as workspace
    # size. The num of threadblocks is sms_count * max_blocks_per_sm.
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(
        sms * max_blocks_per_sm, dtype=torch.int, device=device, requires_grad=False
    )


def marlin_is_k_full(act_order: bool, is_row_parallel: bool) -> bool:
    return (not act_order) or (act_order and not is_row_parallel)


def marlin_repeat_scales_on_all_ranks(
    act_order: bool, group_size: int, is_row_parallel: bool
) -> bool:
    # Need to repeat scales on every rank if act_ordering or
    # channelwise and RowParallelLinear
    is_channelwise = group_size == -1
    return act_order or (is_channelwise and is_row_parallel)


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(
        torch.empty(0, dtype=torch.int, device=device), requires_grad=False
    )


def marlin_sort_g_idx(g_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    g_idx_sort_indices = torch.argsort(g_idx).to(torch.int)
    return g_idx[g_idx_sort_indices], g_idx_sort_indices


def get_scale_perms():
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int
) -> torch.Tensor:
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s


def marlin_permute_bias(s: torch.Tensor) -> torch.Tensor:
    origin_shape = s.shape
    _, scale_perm_single = get_scale_perms()
    s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape(*origin_shape).contiguous()


def marlin_moe_permute_scales(
    s: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
):
    num_experts = s.shape[0]
    output = torch.empty(
        (num_experts, s.shape[1], s.shape[2]),
        device=s.device,
        dtype=s.dtype,
    )

    for e in range(num_experts):
        output[e] = marlin_permute_scales(s[e], size_k, size_n, group_size)
    return output


def sm70_marlin_moe_logical_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int
) -> torch.Tensor:
    """Keep SM70 MoE metadata in logical, N-contiguous order."""
    del size_k, group_size
    return s.reshape((s.shape[0], -1, size_n)).contiguous()


# SM70 marlin_v100 MXFP4 maps each UE8M0 byte `b` to FP16 as
#   e = clamp(b - 112, 0, 31); value = 2^(e - 15)
# which equals the OCP value 2^(b - 127) iff b ∈ [113, 142].
# Byte 0 (zero-padding) flushes to +0. Hopper BF16 bit-shifts the full
# E8M0 range (~2^-126 .. 2^127); we refuse out-of-range bytes rather than
# silently clamp.
SM70_MXFP4_UE8M0_FP16_EXACT_MIN = 113
SM70_MXFP4_UE8M0_FP16_EXACT_MAX = 142

# DeepSeek-V4.1-Flash routed-expert geometry. Not V4-Flash.
DSV41_FLASH_HIDDEN_SIZE = 5120
DSV41_FLASH_MOE_INTERMEDIATE_SIZE = 2304
DSV41_FLASH_NUM_ROUTED_EXPERTS = 384
DSV41_FLASH_NUM_EXPERTS_PER_TOK = 6
DSV41_FLASH_MXFP4_GROUP_SIZE = 32


def sm70_mxfp4_ue8m0_to_uint8(scales: torch.Tensor) -> torch.Tensor:
    """Return raw UE8M0 bytes without a numerical round-trip.

    Checkpoint-native ``float8_e8m0fnu`` / ``uint8`` are a view. Float
    containers must already hold exact powers of two (``2**(e-127)``).
    """
    if scales.dtype == torch.float8_e8m0fnu:
        return scales.view(torch.uint8)
    if scales.dtype == torch.uint8:
        return scales
    if scales.dtype == torch.int8:
        return scales.view(torch.uint8)
    if scales.dtype in (torch.float32, torch.float16, torch.bfloat16):
        finite = torch.isfinite(scales.float())
        positive = scales.float() > 0
        if bool((~finite | ~positive).any()):
            raise RuntimeError(
                "MXFP4 UE8M0 float scales must be finite and positive "
                f"(got min={float(scales.float().min())}, "
                f"max={float(scales.float().max())})."
            )
        log2_val = torch.log2(scales.float())
        rounded = torch.round(log2_val)
        if not torch.allclose(log2_val, rounded, atol=1e-4, rtol=0.0):
            raise RuntimeError(
                "MXFP4 UE8M0 float scales must be exact powers of two; "
                "refusing a close-enough reinterpret as E8M0."
            )
        byte = (rounded + 127).to(torch.int32)
        if bool((byte < 0).any() or (byte > 255).any()):
            raise RuntimeError(
                "MXFP4 UE8M0 exponent does not fit in a byte "
                f"(got {int(byte.min())} .. {int(byte.max())})."
            )
        return byte.to(torch.uint8)
    raise TypeError(f"Unsupported MXFP4 UE8M0 scale dtype: {scales.dtype}")


def sm70_mxfp4_validate_ue8m0_scales(scales: torch.Tensor) -> None:
    """Fail loud if any non-padding UE8M0 byte is outside the FP16-exact range."""
    raw = sm70_mxfp4_ue8m0_to_uint8(scales)
    in_range = (raw == 0) | (
        (raw >= SM70_MXFP4_UE8M0_FP16_EXACT_MIN)
        & (raw <= SM70_MXFP4_UE8M0_FP16_EXACT_MAX)
    )
    if bool(in_range.all()):
        return
    bad = raw[~in_range]
    raise RuntimeError(
        "SM70 marlin_v100 MXFP4 dequantizes UE8M0 in FP16 as "
        "clamp(byte-112, 0, 31) << 10, which is exact only for bytes in "
        f"{{0}} ∪ [{SM70_MXFP4_UE8M0_FP16_EXACT_MIN}, "
        f"{SM70_MXFP4_UE8M0_FP16_EXACT_MAX}] (2^-14 .. 2^15). "
        f"Found out-of-range bytes {int(bad.min())}..{int(bad.max())}. "
        "Refusing to clamp (Hopper BF16 Marlin would keep these exponents)."
    )


def sm70_mxfp4_logical_ue8m0_scales(
    scales: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    group_size: int = DSV41_FLASH_MXFP4_GROUP_SIZE,
) -> torch.Tensor:
    """Checkpoint ``[E, N, K/32]`` UE8M0 → logical ``[E, K/32, N]`` E8M0.

    marlin_v100 indexes ``scales_ + group * size_n + cache_n`` as raw E8M0
    bytes. Do **not** apply Hopper ``marlin_permute_scales`` or the 16-bit
    pair-swap in ``mxfp4_marlin_process_scales`` — both corrupt SM70 metadata.
    Do **not** encode NVFP4 S0E5M3; MXFP4 has no global_scale.
    """
    if group_size != 32:
        raise ValueError(
            f"Official MXFP4 is UE8M0 g32, got group_size={group_size}."
        )
    if size_k % group_size != 0:
        raise ValueError(f"size_k={size_k} is not divisible by group_size=32.")
    if scales.ndim != 3:
        raise ValueError(
            f"MXFP4 UE8M0 scales must be rank-3 [E, N, K/32] or "
            f"[E, K/32, N], got shape={tuple(scales.shape)}"
        )
    if scales.dtype == torch.float8_e4m3fn:
        raise RuntimeError(
            "MXFP4 experts are UE8M0 g32, not NVFP4 E4M3 g16. "
            "Refusing to approximately pack."
        )

    num_groups = size_k // group_size
    raw = sm70_mxfp4_ue8m0_to_uint8(scales)
    sm70_mxfp4_validate_ue8m0_scales(raw)

    if raw.shape[1] == size_n and raw.shape[2] == num_groups:
        logical = raw.transpose(1, 2).contiguous()
    elif raw.shape[1] == num_groups and raw.shape[2] == size_n:
        logical = raw.contiguous()
    else:
        raise ValueError(
            "MXFP4 UE8M0 scales must be [E, N, K/32] (checkpoint) or "
            f"[E, K/32, N] (logical); got shape={tuple(raw.shape)} "
            f"with N={size_n}, K/32={num_groups}."
        )
    return sm70_marlin_moe_logical_scales(
        logical.view(torch.float8_e8m0fnu), size_k, size_n, group_size
    )


def sm70_mxfp4_refuse_nvfp4_metadata(layer: torch.nn.Module) -> None:
    """MXFP4 has no NVFP4 global scale / g16 E4M3 block scales."""
    for name in (
        "w13_weight_scale_2",
        "w2_weight_scale_2",
        "w13_global_scale",
        "w2_global_scale",
        "weight_global_scale",
    ):
        value = getattr(layer, name, None)
        if value is not None:
            raise RuntimeError(
                "MXFP4 (e2m1 + UE8M0 g32) has no NVFP4 global_scale. "
                f"Found {name}={type(value)}; refusing to pack into "
                "sm70_nvfp4_moe_decode or NVFP4 Marlin metadata."
            )


def sm70_mxfp4_fused_decode_expert_limit() -> int | None:
    """Expert-count cap of the SM70 fused MXFP4 decode, or None if uncapped.

    ``marlin_v100 moe_wna16_marlin_gemm`` takes ``num_experts = b_q_weight.size(0)``
    with no cap, so DeepSeek-V4.1-Flash's 384 routed experts fit one fused
    decode (48 local with EP8). The specialized ``sm70_nvfp4_moe_decode``
    kernel is a different format (NVFP4 E4M3 g16 + FP32 global, hidden=2560,
    512 experts, top-10) and must not be used for V4.1-Flash.
    """
    return None


def sm70_nvfp4_marlin_process_scales(
    scales: torch.Tensor, activation_dtype: torch.dtype
) -> tuple[torch.Tensor, float]:
    """Encode logical E4M3 NVFP4 scales for the SM70 Marlin fast path.

    The V100 iterator converts four metadata bytes to FP16/BF16 with integer
    bit operations.  It therefore consumes the special non-negative S0E5M3
    representation used by marlin_v100, not checkpoint-native S1E4M3 bytes.
    ``scales`` must already be in logical ``[E, K/16, N]`` order.

    Returns the encoded metadata and a power-of-two scale factor.  The caller
    must divide the processed global scale by that factor.
    """
    if activation_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "SM70 NVFP4 Marlin supports FP16/BF16 activations, got "
            f"{activation_dtype}"
        )
    if scales.dtype != torch.float8_e4m3fn or scales.ndim != 3:
        raise ValueError(
            "SM70 NVFP4 Marlin expects [E,K/16,N] E4M3 scales, got "
            f"shape={tuple(scales.shape)}, dtype={scales.dtype}"
        )

    logical = scales.to(torch.float16)
    if bool((logical < 0).any()):
        raise ValueError("NVFP4 block scales must be non-negative")

    # FP16 has enough exponent range for ModelOpt's normalized E4M3 scales.
    # BF16 may need a shared rescale so every non-zero S0E5M3 byte keeps its
    # high bit set, which the integer dequantizer relies on.
    scale_factor = 1.0
    if activation_dtype == torch.bfloat16:
        nonzero = logical[logical > 0]
        if nonzero.numel() > 0:
            min_scaled = nonzero.float().min() * (2**7)
            if min_scaled < 2:
                scale_factor = float(torch.ceil(torch.log2(2 / min_scaled)).exp2())
                logical = (logical.float() * scale_factor).to(torch.float16)

    num_experts, num_groups, size_n = logical.shape
    if size_n % 4 != 0:
        raise ValueError(
            f"SM70 NVFP4 Marlin requires N divisible by 4, got N={size_n}"
        )
    flat = logical.reshape(-1, size_n)
    # The iterator deliberately reverses each pair when expanding metadata.
    flat = flat.view(-1, 4)[:, [0, 2, 1, 3]].reshape(-1, size_n)
    encoded = (flat * (2**7)).view(torch.int16) << 1
    encoded = encoded.view(torch.float8_e4m3fn).reshape(-1, size_n * 2)
    encoded = encoded[:, 1::2].contiguous()
    return encoded.reshape(num_experts, num_groups, size_n), scale_factor


def sm70_nvfp4_marlin_process_global_scale(
    global_scale: torch.Tensor, activation_dtype: torch.dtype
) -> torch.Tensor:
    """Compensate an NVFP4 global scale for SM70's integer dequantizer."""
    if activation_dtype == torch.float16:
        target_exponent = 5
    elif activation_dtype == torch.bfloat16:
        target_exponent = 8
    else:
        raise ValueError(
            "SM70 NVFP4 Marlin supports FP16/BF16 activations, got "
            f"{activation_dtype}"
        )
    fp4_exponent = 2
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    return (global_scale.to(torch.float32) * (2.0 ** (exponent_bias - 7))).contiguous()


def marlin_zero_points(
    zp: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    # Permute zero-points in a similar way to scales, but do not use the
    # "single" permutation, since zero-points are applied on every MMA
    scale_perm, _ = get_scale_perms()
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]

    # Interleave column dim (for the dequantize code) and pack it to int32
    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n)).contiguous()
    zp = pack_cols(zp, num_bits, size_k, size_n)

    return zp


def awq_to_marlin_zero_points(
    q_zp_packed: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    # AWQ zero-points are quantized and packed on the column dim.
    # In addition, the values are permuted based on dequantizer.
    # Here we undo both of these, and then apply marlin permutation
    # and pack it back.
    q_zp = unpack_cols(q_zp_packed, num_bits, size_k, size_n)

    # Undo interleaving (use argsort(..) to get inverse perm)
    if num_bits == 4:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 4, 6, 1, 3, 5, 7]))
    elif num_bits == 8:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 1, 3]))
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    q_zp = q_zp.reshape((-1, len(undo_interleave)))[:, undo_interleave].ravel()
    q_zp = q_zp.reshape((-1, size_n)).contiguous()

    marlin_zp = marlin_zero_points(q_zp, size_k, size_n, num_bits)
    return marlin_zp


def moe_awq_to_marlin_zero_points(
    q_zp_packed: torch.Tensor, size_k: int, size_n: int, num_bits: int
):
    num_experts = q_zp_packed.shape[0]
    output = torch.empty(
        (num_experts, q_zp_packed.shape[1], q_zp_packed.shape[2]),
        device=q_zp_packed.device,
        dtype=q_zp_packed.dtype,
    )
    for e in range(num_experts):
        output[e] = awq_to_marlin_zero_points(q_zp_packed[e], size_k, size_n, num_bits)
    return output


def moe_awq_to_sm70_marlin_zero_points_float(
    q_zp_packed: torch.Tensor,
    scales: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    """Expand AWQ zero-points to the FP16 metadata required by SM70 MoE.

    marlin_v100 consumes logical ``zero_point * scale`` values rather than the
    packed/permuted integer zero-points used by the SM80+ Marlin kernels.
    """
    q_zp = unpack_cols(
        q_zp_packed.reshape(-1, q_zp_packed.shape[-1]),
        num_bits,
        size_k * q_zp_packed.shape[0],
        size_n,
    ).reshape(q_zp_packed.shape[0], size_k, size_n)

    if num_bits == 4:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 4, 6, 1, 3, 5, 7]))
    elif num_bits == 8:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 1, 3]))
    else:
        raise ValueError(f"num_bits must be 4 or 8, got {num_bits}")

    q_zp = q_zp.reshape(-1, len(undo_interleave))[:, undo_interleave]
    q_zp = q_zp.reshape(q_zp_packed.shape[0], size_k, size_n)
    return (q_zp.float() * scales.contiguous().float()).to(scales.dtype).contiguous()


def maybe_warn_marlin_atomic_add(device, dtype):
    if torch.compiler.is_dynamo_compiling():
        return
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] < 9 and dtype == torch.bfloat16:
        logger.info_once(
            "You are running Marlin kernel with bf16 on GPUs before SM90. "
            "You can consider change to fp16 to achieve better performance "
            "if possible."
        )


def maybe_warn_marlin_atomic_add_env():
    if torch.compiler.is_dynamo_compiling():
        return
    # TODO(yiyun): Need to add sglang's MARLIN_USE_ATOMIC_ADD: bool = False
    if True:
        return
    # if envs.VLLM_MARLIN_USE_ATOMIC_ADD:
    #     return
    logger.info_once(
        "Marlin kernel can achieve better performance for small size_n "
        "with experimental use_atomic_add feature. "
        "You can consider set environment variable "
        "VLLM_MARLIN_USE_ATOMIC_ADD to 1 if possible."
    )


def should_use_atomic_add_reduce(
    m: int, n: int, k: int, device: torch.device, dtype: torch.dtype
) -> bool:
    # the performance of atomicAdd is better than global reduce
    # only when m*n is small and k is large
    if n >= 2048 or k < 2048 or device.type != "cuda":
        return False

    # disable atomicAdd reduce by default,
    # one can enable it with VLLM_MARLIN_USE_ATOMIC_ADD=1
    # TODO: Need to add sglang's MARLIN_USE_ATOMIC_ADD: bool = False
    if not True:
        maybe_warn_marlin_atomic_add_env()
        return False

    # sm8x doesn't support atomicAdd + bfloat16 natively
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] < 9 and dtype == torch.bfloat16:
        maybe_warn_marlin_atomic_add(device, dtype)
        return False

    return True


def apply_gptq_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    wtype: ScalarType,
    output_size_per_partition: int,
    input_size_per_partition: int,
    is_k_full: bool,
    bias: Optional[torch.Tensor] = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition,)

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=output_size_per_partition,
        k=reshaped_x.size(1),
        device=input.device,
        dtype=input.dtype,
    )

    forward_context = get_tc_piecewise_forward_context()
    if forward_context is None:
        output = gptq_marlin_gemm(
            reshaped_x,
            None,
            weight,
            weight_scale,
            None,
            weight_zp,
            g_idx,
            g_idx_sort_indices,
            workspace,
            wtype,
            size_m=reshaped_x.shape[0],
            size_n=output_size_per_partition,
            size_k=input_size_per_partition,
            is_k_full=is_k_full,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=use_fp32_reduce,
            is_zp_float=False,
        )
    else:
        output = unified_apply_gptq_marlin_gemm_with_wtype(
            input=reshaped_x,
            weight=weight,
            weight_scale=weight_scale,
            weight_zp=weight_zp,
            g_idx=g_idx,
            g_idx_sort_indices=g_idx_sort_indices,
            workspace=workspace,
            wtype_id=wtype.id,
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            is_k_full=is_k_full,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=use_fp32_reduce,
            is_zp_float=False,
        )

    if bias is not None:
        output.add_(bias)  # In-place add

    return output.reshape(out_shape)


def apply_awq_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    quant_type: ScalarType,
    output_size_per_partition: int,
    input_size_per_partition: int,
    bias: Optional[torch.Tensor] = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition,)

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=output_size_per_partition,
        k=reshaped_x.size(1),
        device=input.device,
        dtype=input.dtype,
    )

    forward_context = get_tc_piecewise_forward_context()
    if forward_context is None:
        output = gptq_marlin_gemm(
            reshaped_x,
            None,
            weight,
            weight_scale,
            None,
            weight_zp,
            g_idx,
            g_idx_sort_indices,
            workspace,
            quant_type,
            size_m=reshaped_x.shape[0],
            size_n=output_size_per_partition,
            size_k=input_size_per_partition,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=use_fp32_reduce,
            is_zp_float=False,
        )
    else:
        output = unified_apply_gptq_marlin_gemm(
            input=reshaped_x,
            weight=weight,
            weight_scale=weight_scale,
            weight_zp=weight_zp,
            g_idx=g_idx,
            g_idx_sort_indices=g_idx_sort_indices,
            workspace=workspace,
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=use_fp32_reduce,
            is_zp_float=False,
        )

    if bias is not None:
        output.add_(bias)  # In-place add

    return output.reshape(out_shape)


def fake_unified_apply_gptq_marlin_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    output_size_per_partition: int,
    input_size_per_partition: int,
    use_atomic_add: bool,
    use_fp32_reduce: bool,
    is_zp_float: bool,
) -> torch.Tensor:
    return input.new_empty(
        (input.shape[0], output_size_per_partition), dtype=input.dtype
    )


@register_custom_op(fake_impl=fake_unified_apply_gptq_marlin_gemm)
def unified_apply_gptq_marlin_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    output_size_per_partition: int,
    input_size_per_partition: int,
    use_atomic_add: bool,
    use_fp32_reduce: bool,
    is_zp_float: bool,
) -> torch.Tensor:
    quant_config = get_tc_piecewise_forward_context().quant_config
    quant_type = quant_config.quant_type
    return gptq_marlin_gemm(
        input,
        None,
        weight,
        weight_scale,
        None,
        weight_zp,
        g_idx,
        g_idx_sort_indices,
        workspace,
        quant_type,
        size_m=input.shape[0],
        size_n=output_size_per_partition,
        size_k=input_size_per_partition,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=is_zp_float,
    )


def fake_unified_apply_gptq_marlin_gemm_with_wtype(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    wtype_id: int,
    output_size_per_partition: int,
    input_size_per_partition: int,
    is_k_full: bool,
    use_atomic_add: bool,
    use_fp32_reduce: bool,
    is_zp_float: bool,
) -> torch.Tensor:
    return input.new_empty(
        (input.shape[0], output_size_per_partition), dtype=input.dtype
    )


@register_custom_op(fake_impl=fake_unified_apply_gptq_marlin_gemm_with_wtype)
def unified_apply_gptq_marlin_gemm_with_wtype(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    wtype_id: int,
    output_size_per_partition: int,
    input_size_per_partition: int,
    is_k_full: bool,
    use_atomic_add: bool,
    use_fp32_reduce: bool,
    is_zp_float: bool,
) -> torch.Tensor:
    # Reconstruct ScalarType from id
    wtype = None
    for attr_name in dir(scalar_types):
        if not attr_name.startswith("_"):
            st = getattr(scalar_types, attr_name)
            if hasattr(st, "id") and st.id == wtype_id:
                wtype = st
                break
    return gptq_marlin_gemm(
        input,
        None,
        weight,
        weight_scale,
        None,
        weight_zp,
        g_idx,
        g_idx_sort_indices,
        workspace,
        wtype,
        size_m=input.shape[0],
        size_n=output_size_per_partition,
        size_k=input_size_per_partition,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=is_zp_float,
    )
