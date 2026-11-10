from __future__ import annotations

import glob
import logging
import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.environ import envs
from sglang.kernels.sm70_paths import sm70_prebuilt_dir

if TYPE_CHECKING:
    from sgl_kernel.scalar_type import ScalarType
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

# Constants matching device::marlin_moe:: in marlin.cuh
_MAX_THREAD_N = 256


# --- SM70 (V100) marlin_v100 integration ---------------------------------------
# The stock JIT Marlin MoE kernel (csrc/gemm/marlin_moe/marlin_template.h) is an
# empty-body stub for __CUDA_ARCH__ < 800: it launches but writes nothing, so on
# V100 the routed experts silently contribute zero (the unquantized shared
# expert + residual mask the corruption). marlin_v100 (a vLLM-derived SM70 fork
# with real WMMA kernels) replaces it. Built+installed by
# scripts/setup_v100_marlin.sh; auto-detected here on SM70 with no env var.

_IS_SM70: bool = False
try:
    if torch.cuda.is_available():
        _IS_SM70 = torch.cuda.get_device_capability()[0] == 7
except Exception:
    pass

# Cache for the loaded op. States: None = not attempted yet; callable = loaded;
# False = attempted but unavailable (so we only warn once).
_marlin_v100_op = None

_SM70_MARLIN_MOE_ENV_NAMES = (
    "SM70_MARLIN_MOE_CTA_GEOMETRY",
    "SM70_MARLIN_MOE_SPLIT_K",
    "SM70_MARLIN_MOE_METADATA_CACHE",
)
_sm70_marlin_user_tuning = any(
    os.getenv(name) for name in _SM70_MARLIN_MOE_ENV_NAMES
)
_sm70_marlin_tuning_stage = None

# Decode sweep (scripts/dsv41_sm70_marlin_decode_sweep.py, 2026-09-16):
# auto (packed_macro_n=256 default 32x256x32x4x32x64x32 sk=1) was 588/274 us
# for w13/w2. Exact-match winner for both GEMMs is 32x128x32x4x32x32x32 sk=1
# (~390/184 us eager, 1.29x as a captured pair). split_k=4 is ~9 us faster
# on w13 but drifts by 0.0625 vs auto; keep sk=1.
_DSV41_DECODE_MARLIN_GEOM = "32x128x32x4x32x32x32"
_DSV41_DECODE_MARLIN_VALUES = (_DSV41_DECODE_MARLIN_GEOM, "1", "vector_words")


def sm70_qwen38_prefill_2k_enabled() -> bool:
    """Opt-in 2K-prefill stages; user-supplied geometry always wins."""
    return (
        _IS_SM70
        and not _sm70_marlin_user_tuning
        and envs.SGLANG_SM70_MOE_PREFILL_2K.get()
    )


def _sm70_marlin_moe_stage_values(
    scale_dtype: torch.dtype,
    moe_block_size: int,
    top_k: int,
    size_m: int,
    size_n: int,
    size_k: int,
) -> Optional[tuple[str, tuple[str, str, str]]]:
    """Return (stage, env-triple) for a measured SM70 Marlin MoE shape, or None."""
    # 2K prefill (opt-in): 64-row CTAs measured at the exact 2048-token w13 and
    # 20480-row w2 chunk shapes.  Off by default so the generic prefill
    # selector keeps the shapes it was tuned for.
    if (
        sm70_qwen38_prefill_2k_enabled()
        and scale_dtype == torch.float8_e4m3fn
        and moe_block_size == 64
        and top_k == 10
        and (size_m, size_n, size_k) == (2048, 320, 2560)
    ):
        return "qwen38_w13_prefill_2k", ("64x64x32x4x32x32x32", "1", "vector_words")
    if (
        sm70_qwen38_prefill_2k_enabled()
        and scale_dtype == torch.float8_e4m3fn
        and moe_block_size == 64
        and top_k == 1
        and (size_m, size_n, size_k) == (20480, 2560, 160)
    ):
        return "qwen38_w2_prefill_2k", ("64x256x32x4x64x64x32", "1", "vector_words")
    if (
        scale_dtype == torch.float8_e4m3fn
        and moe_block_size == 8
        and top_k == 10
        and size_m == 1
        and size_n == 320
        and size_k == 2560
    ):
        return "qwen38_w13_decode", ("32x64x64x4x32x64x16", "1", "vector_words")
    if (
        scale_dtype == torch.float8_e4m3fn
        and moe_block_size == 32
        and top_k == 10
        and 512 <= size_m <= 2048
        and size_n == 320
        and size_k == 2560
    ):
        return "qwen38_w13_prefill", ("32x64x64x4x32x32x32", "1", "vector_words")
    if (
        scale_dtype == torch.float8_e4m3fn
        and moe_block_size == 64
        and top_k == 1
        and size_m >= 40960
        and size_n == 2560
        and size_k == 160
    ):
        return "qwen38_w2_prefill", ("64x256x32x4x64x64x32", "1", "vector_words")
    if (
        scale_dtype == torch.float8_e4m3fn
        and moe_block_size == 8
        and top_k == 1
        and size_m == 10
        and size_n == 2560
        and size_k == 160
    ):
        return "qwen38_w2_decode", ("64x256x32x4x64x64x32", "1", "lane_vectors")
    # DeepSeek-V4.1-Flash MXFP4 decode (and 1-token extend). Prefill M>>1
    # keeps marlin_v100's generic selector.
    if (
        scale_dtype == torch.float8_e8m0fnu
        and moe_block_size == 8
        and top_k == 6
        and size_m == 1
        and size_n == 4608
        and size_k == 5120
    ):
        return "dsv41_w13_decode", _DSV41_DECODE_MARLIN_VALUES
    if (
        scale_dtype == torch.float8_e8m0fnu
        and moe_block_size == 8
        and top_k == 1
        and size_m == 6
        and size_n == 5120
        and size_k == 2304
    ):
        return "dsv41_w2_decode", _DSV41_DECODE_MARLIN_VALUES
    return None


def _configure_sm70_qwen38_nvfp4_stage(
    b_scales: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    size_m: int,
    size_n: int,
    size_k: int,
) -> None:
    """Select a measured CTA geometry before CUDA-graph capture.

    marlin_v100 reads these variables synchronously when its host launcher is
    called.  SGLang captures the resulting kernels in the decode CUDA graph,
    so there is no environment handling on graph replay. Unmatched shapes
    clear the override and retain marlin_v100's generic/model selectors.
    """

    global _sm70_marlin_tuning_stage
    if not _IS_SM70 or _sm70_marlin_user_tuning:
        return

    hit = _sm70_marlin_moe_stage_values(
        b_scales.dtype, moe_block_size, top_k, size_m, size_n, size_k
    )
    if hit is None:
        stage, values = None, None
    else:
        stage, values = hit

    if stage == _sm70_marlin_tuning_stage:
        return
    if values is None:
        for name in _SM70_MARLIN_MOE_ENV_NAMES:
            os.environ.pop(name, None)
    else:
        for name, value in zip(_SM70_MARLIN_MOE_ENV_NAMES, values):
            os.environ[name] = value
    _sm70_marlin_tuning_stage = stage


def _load_marlin_v100_op():
    """Lazily auto-detect and load the marlin_v100 MoE op on SM70.

    Returns the registered ``torch.ops._moe_C.moe_wna16_marlin_gemm`` callable,
    or ``None`` if it could not be found. Subsequent calls return the cached
    result. No-op (returns None) on non-SM70 devices.
    """
    global _marlin_v100_op
    if _marlin_v100_op is not False and _marlin_v100_op is not None:
        return _marlin_v100_op
    if _marlin_v100_op is False:
        return None
    _marlin_v100_op = False  # mark attempted

    if not _IS_SM70:
        return None

    here = os.path.dirname(os.path.abspath(__file__))
    home = os.path.expanduser("~")
    candidates = []
    # 1. installed next to sglang's jit_kernel package (setup_v100_marlin.sh)
    candidates += sorted(glob.glob(os.path.join(here, "_sm70_marlin_v100_moe*.so")))
    candidates += sorted(
        glob.glob(os.path.join(str(sm70_prebuilt_dir()), "_sm70_marlin_v100_moe*.so"))
    )
    # 2. dev build in a ~/marlin_v100 checkout
    candidates += sorted(glob.glob(os.path.join(home, "marlin_v100", "vllm", "_moe_C*.so")))

    for path in candidates:
        try:
            torch.ops.load_library(path)
        except Exception as e:  # noqa: BLE001
            logger.warning("SM70 (V100): failed to load marlin_v100 .so at %s: %s", path, e)
            continue
        op = getattr(torch.ops._moe_C, "moe_wna16_marlin_gemm", None)
        if op is not None:
            _marlin_v100_op = op
            logger.info("SM70 (V100): using marlin_v100 MoE kernel from %s", path)
            return op

    logger.warning(
        "SM70 (V100) detected but the marlin_v100 MoE kernel was not found. "
        "The stock JIT Marlin kernel is an empty stub on SM70, so routed-expert "
        "output will be ZERO (incorrect). Build it with: "
        "`bash scripts/setup_v100_marlin.sh`. Searched: %s",
        candidates,
    )
    return None


@cache_once
def _jit_moe_wna16_marlin_module(
    dtype: torch.dtype,
    is_ep: bool,
    has_bias: bool,
    expert_ids_are_sorted: bool = False,
    zero_inactive_output: bool = False,
    has_inactive_prefix_count: bool = False,
) -> Module:
    args = make_cpp_args(
        dtype,
        is_ep,
        has_bias,
        expert_ids_are_sorted,
        zero_inactive_output,
        has_inactive_prefix_count,
    )
    return load_jit(
        "moe_wna16_marlin",
        *args,
        cuda_files=["gemm/marlin_moe/moe_wna16_marlin.cuh"],
        cuda_wrappers=[
            (
                "moe_wna16_marlin_gemm",
                f"moe_wna16_marlin_gemm<{args}>",
            )
        ],
    )


def _or_empty(
    t: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return t if t is not None else torch.empty(0, device=device, dtype=dtype)


@debug_kernel_api
def moe_wna16_marlin_gemm(
    a: torch.Tensor,
    c_or_none: Optional[torch.Tensor],
    b_q_weight: torch.Tensor,
    b_bias_or_none: Optional[torch.Tensor],
    b_scales: torch.Tensor,
    global_scale_or_none: Optional[torch.Tensor],
    b_zeros_or_none: Optional[torch.Tensor],
    g_idx_or_none: Optional[torch.Tensor],
    perm_or_none: Optional[torch.Tensor],
    workspace: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    is_ep: bool,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
    expert_ids_are_sorted: bool = False,
    zero_inactive_output: bool = False,
    has_inactive_prefix_count: bool = False,
) -> torch.Tensor:
    """Project routes, preserving inactive rows in a caller-provided output.

    Inactive rows are zero when this function allocates the output. With
    ``zero_inactive_output=True``, the GEMM also zeros caller-owned rows present
    in inactive routing blocks; omitted routes remain untouched.
    ``expert_ids_are_sorted`` certifies ascending block expert IDs within the
    padded token count, with any -1 blocks first. Unused tail entries are ignored.
    Leave it false after remapping that can produce scattered inactive blocks.
    The sorted specialization is only used without expert parallelism.
    ``has_inactive_prefix_count`` additionally certifies that a supporting
    producer wrote an aligned int32 pair ``[padded_tokens, inactive_blocks]``
    for these tables and this block size. It is ignored outside sorted non-EP.
    """
    device = a.device

    # Allocate output if not provided
    if c_or_none is not None:
        c = c_or_none
    else:
        c = torch.zeros((size_m * top_k, size_n), dtype=a.dtype, device=device)

    # Early return for zero-size M
    if size_m == 0:
        return c

    # SM70 (V100): dispatch to the marlin_v100 kernel when available. Its op
    # signature differs from the JIT module: it takes `a_scales` (None for
    # W4A16) and tuning ints (thread_k/n, blocks_per_sm; -1 = auto-select), and
    # it computes has_act_order/has_bias/has_zp/num_groups/group_size/is_ep
    # internally from the tensor shapes, so those are not forwarded. The raw
    # Optional tensors are passed through unchanged (None => std::nullopt) so
    # the kernel's has_value()-based presence checks fire correctly; do NOT
    # convert None to an empty tensor (the kernel treats a present-but-empty
    # global_scale as an nvfp4-only input and rejects it for GPTQ/AWQ).
    if _IS_SM70:
        op = _load_marlin_v100_op()
        if op is not None:
            _configure_sm70_qwen38_nvfp4_stage(
                b_scales,
                moe_block_size,
                top_k,
                size_m,
                size_n,
                size_k,
            )
            # marlin_v100's epilogue reads topk_weights as float32
            # (`data_ptr<float>()`). Hopper fused_marlin_moe passes FP16/BF16.
            if topk_weights.dtype != torch.float32:
                topk_weights = topk_weights.float()
            op(
                a,
                c,
                b_q_weight,
                b_bias_or_none,
                b_scales,
                None,  # a_scales (W4A16 has no activation quantization)
                global_scale_or_none,
                b_zeros_or_none,
                g_idx_or_none,
                perm_or_none,
                workspace,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                topk_weights,
                moe_block_size,
                top_k,
                mul_topk_weights,
                b_q_type.id,
                size_m,
                size_n,
                size_k,
                is_k_full,
                use_atomic_add,
                use_fp32_reduce,
                is_zp_float,
                -1,  # thread_k  (-1 => C++ model-specific auto-select)
                -1,  # thread_n
                -1,  # blocks_per_sm
            )
            return c
        # fall through to the stock JIT path (empty stub on SM70) with the
        # warning already emitted by _load_marlin_v100_op.

    # Determine activation ordering
    has_act_order = (
        g_idx_or_none is not None
        and perm_or_none is not None
        and g_idx_or_none.numel() > 0
        and perm_or_none.numel() > 0
        and g_idx_or_none.size(-1) > 0
        and perm_or_none.size(-1) > 0
    )

    # Determine has_zp
    has_zp = b_zeros_or_none is not None and b_zeros_or_none.numel() > 0

    # Determine has_bias
    has_bias = b_bias_or_none is not None

    # Derive num_groups and group_size from b_scales
    num_groups = b_scales.size(1)
    if has_act_order:
        if is_k_full:
            group_size = size_k // num_groups
        else:
            group_size = 0
    else:
        if num_groups > 1:
            group_size = size_k // num_groups
        else:
            group_size = -1

    # Allocate a_tmp for act_order column permutation
    if has_act_order:
        a_tmp = torch.empty((size_m * top_k, size_k), dtype=a.dtype, device=device)
    else:
        a_tmp = torch.empty(0, dtype=a.dtype, device=device)

    # Allocate c_tmp for fp32 reduce
    if use_fp32_reduce and not use_atomic_add:
        sms = current_arch().num_sms
        # max num of threadblocks is sms * 4
        max_c_tmp_size = min(
            size_n * sorted_token_ids.size(0),
            sms * 4 * moe_block_size * _MAX_THREAD_N,
        )
        if moe_block_size == 8:
            max_c_tmp_size *= 2
        c_tmp = torch.empty(max_c_tmp_size, dtype=torch.float32, device=device)
    else:
        c_tmp = torch.empty(0, dtype=torch.float32, device=device)

    # Convert Optional tensors to empty tensors
    g_idx_t = _or_empty(g_idx_or_none, device, torch.int32)
    perm_t = _or_empty(perm_or_none, device, torch.int32)
    b_zeros_t = _or_empty(b_zeros_or_none, device, a.dtype)
    b_bias_t = _or_empty(b_bias_or_none, device, a.dtype)
    global_scale_t = _or_empty(global_scale_or_none, device, a.dtype)

    module = _jit_moe_wna16_marlin_module(
        a.dtype,
        is_ep,
        has_bias,
        expert_ids_are_sorted and not is_ep,
        zero_inactive_output,
        has_inactive_prefix_count and expert_ids_are_sorted and not is_ep,
    )
    module.moe_wna16_marlin_gemm(
        a,
        c,
        b_q_weight,
        b_bias_t,
        b_scales,
        global_scale_t,
        b_zeros_t,
        g_idx_t,
        perm_t,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        a_tmp,
        c_tmp,
        moe_block_size,
        top_k,
        mul_topk_weights,
        is_ep,
        b_q_type.id,
        size_m,
        size_n,
        size_k,
        has_act_order,
        has_bias,
        is_k_full,
        has_zp,
        num_groups,
        group_size,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )

    return c
