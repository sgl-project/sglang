from __future__ import annotations

from sglang.srt.runtime_context import get_parallel

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import (
    assert_buffer_fits,
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolFP4
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import (
    draft_kv_indices_buffer_width,
    draft_kv_indices_used_len,
    generate_draft_decode_kv_indices,
)
from sglang.srt.utils import (
    get_int_env_var,
    is_flashinfer_available,
    is_sm100_supported,
    is_sm120_supported,
    next_power_of_2,
    require_gathered_buffer,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _cuda_graph_capture_max_bs(server_args, max_bs: int) -> int:
    """Pad max_bs to the alignment cuda-graph capture uses (see get_batch_sizes_to_capture)."""
    mul_base = 1
    if server_args.enable_two_batch_overlap:
        mul_base *= 2
    if require_gathered_buffer(server_args):
        mul_base *= get_parallel().attn_tp_size
    if mul_base % get_parallel().attn_cp_size != 0:
        mul_base *= get_parallel().attn_cp_size
    return (max_bs + mul_base - 1) // mul_base * mul_base
def _fp4_kv_radix_trace_enabled() -> bool:
    return os.environ.get("SGLANG_FP4_KV_TRACE_RADIX") == "1"


def _fp4_kv_page_pair_trace_enabled() -> bool:
    return os.environ.get("SGLANG_FP4_KV_TRACE_PAGE_PAIR") == "1"


def _fp4_kv_merge_state_trace_enabled() -> bool:
    return os.environ.get("SGLANG_FP4_KV_TRACE_MERGE_STATE") == "1"


def _fp4_kv_prefix_ref_trace_enabled() -> bool:
    return os.environ.get("SGLANG_FP4_KV_TRACE_PREFIX_REF") == "1"


def _fp4_kv_dense_cache_trace_enabled() -> bool:
    return os.environ.get("SGLANG_FP4_KV_TRACE_DENSE_CACHE") == "1"


def _fp4_kv_dense_quant_attention_trace_enabled() -> bool:
    return os.environ.get("SGLANG_FP4_KV_TRACE_DENSE_QUANT_ATTENTION") == "1"


def _fp4_kv_module_trace_enabled() -> bool:
    return os.environ.get("SGLANG_FP4_KV_TRACE_MODULE") == "1"


def _gemma4_geometry_trace_enabled() -> bool:
    return os.environ.get("SGLANG_GEMMA4_TRACE_GEOMETRY") == "1"


def _flashinfer_vo_split_enabled() -> bool:
    return os.environ.get("SGLANG_FLASHINFER_VOSPLIT") == "1"


def _flashinfer_vo_split_head_dim_vo(head_dim_qk: int) -> int:
    # Gemma 4 global layers are the current target: Q/K stay 512-wide while
    # V/O are handled as two exact 256-wide passes.
    if _flashinfer_vo_split_enabled() and head_dim_qk == 512:
        return 256
    return head_dim_qk


def _trace_k_scale_multipliers() -> List[float]:
    raw = os.environ.get("SGLANG_FP4_KV_TRACE_K_SCALE_MULTIPLIERS")
    if raw in (None, ""):
        return []
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return values[:16]


def _trace_k_head_scale_policy_enabled() -> bool:
    return os.environ.get("SGLANG_FP4_KV_TRACE_K_HEAD_SCALE_POLICY") == "1"


def _trace_layer_enabled(layer_id: int) -> bool:
    raw = os.environ.get("SGLANG_FP4_KV_TRACE_LAYERS")
    if raw in (None, ""):
        return True
    enabled = {item.strip() for item in raw.split(",") if item.strip()}
    return str(layer_id) in enabled


def _trace_value_limit(default: int = 8) -> int:
    raw = os.environ.get("SGLANG_FP4_KV_TRACE_VALUES")
    if raw in (None, ""):
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _trace_token_limit(default: int = 128) -> int:
    raw = os.environ.get("SGLANG_FP4_KV_PREFIX_REF_MAX_TOKENS")
    if raw in (None, ""):
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _trace_q_row_limit(default: Optional[int] = None) -> Optional[int]:
    raw = os.environ.get("SGLANG_FP4_KV_PREFIX_REF_MAX_Q_ROWS")
    if raw in (None, ""):
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _trace_cpu_values(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    try:
        return list(value)
    except TypeError:
        return value


def _trace_tensor_values(value, limit: int = 8):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        flat = value.detach().flatten()
        length = flat.numel()
        return {
            "len": length,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "head": flat[:limit].to("cpu").tolist(),
            "tail": flat[max(0, length - limit) :].to("cpu").tolist(),
        }
    values = _trace_cpu_values(value)
    if isinstance(values, list):
        return {
            "len": len(values),
            "head": values[:limit],
            "tail": values[-limit:] if values else [],
        }
    return values


def _trace_simple_value(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _tensor_trace_summary(value)
    if isinstance(value, (list, tuple)):
        if len(value) > 8:
            return f"{type(value).__name__}(len={len(value)})"
        return [_trace_simple_value(item) for item in value]
    if isinstance(value, dict):
        if len(value) > 16:
            return f"dict(len={len(value)})"
        return {str(k): _trace_simple_value(v) for k, v in value.items()}
    if callable(value):
        return f"callable:{getattr(value, '__name__', type(value).__name__)}"
    text = repr(value)
    if len(text) > 240:
        text = text[:240] + "..."
    return text


def _flashinfer_wrapper_trace_summary(wrapper) -> dict:
    interesting = {}
    for name in sorted(dir(wrapper)):
        if name.startswith("__"):
            continue
        lower = name.lower()
        if not any(
            marker in lower
            for marker in (
                "backend",
                "cache",
                "dtype",
                "jit",
                "kv",
                "module",
                "paged",
                "plan",
                "uri",
            )
        ):
            continue
        try:
            value = getattr(wrapper, name)
        except Exception as exc:
            interesting[name] = f"error:{exc!r}"
            continue
        if callable(value) and not (
            "cache" in lower or "module" in lower or "plan" in lower
        ):
            continue
        interesting[name] = _trace_simple_value(value)
    return {"class": type(wrapper).__name__, "state": interesting}


def _trace_rids(forward_batch: ForwardBatch):
    rids = getattr(forward_batch, "rids", None)
    if rids is None:
        return None
    return [str(rid) for rid in rids]


if envs.SGLANG_ENABLE_TORCH_COMPILE.get():
    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True


if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
        fast_decode_plan,
    )
    from flashinfer.cascade import merge_state

    from sglang.srt.layers.attention.triton_ops.merge_state import merge_state_triton

    # FlashInfer's MergeState CUDA kernel uses blockDim = (head_dim/vec_size, num_heads).
    # When num_heads is large (e.g. with DP attention where attention_tp_size=1), the
    # total threads per block can exceed CUDA's limit of 1024 and the kernel launch fails
    # with `invalid configuration argument`. Fall back to the in-tree Triton implementation,
    # which uses (token, head) as the launch grid and is therefore unaffected.
    _MERGE_STATE_CUDA_MAX_THREADS_PER_BLOCK = 1024

    def _merge_state_max_safe_num_heads(head_dim: int, element_size: int) -> int:
        # Mirrors flashinfer's vec_size selection in include/flashinfer/attention/cascade.cuh.
        vec_size = max(16 // element_size, head_dim // 32)
        bdx = head_dim // vec_size
        if bdx <= 0:
            return _MERGE_STATE_CUDA_MAX_THREADS_PER_BLOCK
        return _MERGE_STATE_CUDA_MAX_THREADS_PER_BLOCK // bdx

    def _safe_merge_state(
        v_a: torch.Tensor,
        s_a: torch.Tensor,
        v_b: torch.Tensor,
        s_b: torch.Tensor,
    ):
        num_heads = v_a.shape[1]
        head_dim = v_a.shape[2]
        max_heads = _merge_state_max_safe_num_heads(head_dim, v_a.element_size())
        if num_heads <= max_heads:
            return merge_state(v_a, s_a, v_b, s_b)
        return merge_state_triton(v_a, s_a, v_b, s_b)


class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


@dataclass(frozen=True)
class FlashInferWrapperGeometry:
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    head_dim_vo: int


def _tp_sharded_kv_heads(total_num_kv_heads: int, tp_size: int) -> int:
    if total_num_kv_heads >= tp_size:
        assert total_num_kv_heads % tp_size == 0
        return total_num_kv_heads // tp_size
    assert tp_size % total_num_kv_heads == 0
    return 1


def _flashinfer_wrapper_geometries(
    model_config, dispatch_reason: Optional[WrapperDispatch], num_wrappers: int
) -> List[FlashInferWrapperGeometry]:
    tp_size = get_attention_tp_size()
    hf_config = model_config.hf_config
    hf_text_config = getattr(model_config, "hf_text_config", None)
    if hf_text_config is None:
        if hasattr(hf_config, "get_text_config"):
            hf_text_config = hf_config.get_text_config()
        else:
            hf_text_config = getattr(hf_config, "text_config", hf_config)

    num_attention_heads = getattr(
        hf_text_config, "num_attention_heads", model_config.num_attention_heads
    )
    total_num_kv_heads = getattr(hf_text_config, "num_key_value_heads", None)
    num_kv_heads = (
        _tp_sharded_kv_heads(total_num_kv_heads, tp_size)
        if total_num_kv_heads is not None
        else model_config.get_num_kv_heads(tp_size)
    )
    head_dim = getattr(hf_text_config, "head_dim", model_config.head_dim)
    base = FlashInferWrapperGeometry(
        num_qo_heads=num_attention_heads // tp_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        head_dim_vo=_flashinfer_vo_split_head_dim_vo(head_dim),
    )
    if dispatch_reason == WrapperDispatch.SLIDING_WINDOW and num_wrappers == 2:
        swa_head_dim = getattr(hf_text_config, "swa_head_dim", base.head_dim)
        swa_total_num_kv_heads = getattr(
            hf_text_config, "swa_num_key_value_heads", None
        )
        swa_num_kv_heads = (
            _tp_sharded_kv_heads(swa_total_num_kv_heads, tp_size)
            if swa_total_num_kv_heads is not None
            else base.num_kv_heads
        )
        return [
            FlashInferWrapperGeometry(
                num_qo_heads=base.num_qo_heads,
                num_kv_heads=swa_num_kv_heads,
                head_dim=swa_head_dim,
                head_dim_vo=_flashinfer_vo_split_head_dim_vo(swa_head_dim),
            ),
            base,
        ]
    return [base for _ in range(num_wrappers)]


@dataclass
class MultiItemScoringParams:
    """Parameters for multi-item scoring in attention computation.

    Used when processing sequences with multiple items separated by delimiters,
    where each item needs specific attention patterns that respect item boundaries.

    Attributes:
        prefix_len_ptr: A uint32 1D tensor indicating the prefix length of each prompt.
                       The tensor size is equal to the batch size.
        token_pos_in_items_ptr: A uint16 1D tensor indicating the token position of each item
                               starting from 0 (delimiter) for each item. For batch size > 1,
                               sequences are concatenated with zero padding to ensure same length.
        token_pos_in_items_len: Zero padding length for token_pos_in_items_ptr to handle
                               batch_size > 1 case. Defines the padded length for each sequence.
        max_item_len_ptr: A uint16 tensor containing the max token length of all items
                         for each prompt in the batch.

    """

    prefix_len_ptr: Optional[torch.Tensor] = None
    token_pos_in_items_ptr: Optional[torch.Tensor] = None
    token_pos_in_items_len: int = 0
    max_item_len_ptr: Optional[torch.Tensor] = None

    def is_enabled(self) -> bool:
        """Check if multi-item scoring is enabled."""
        return self.prefix_len_ptr is not None


@dataclass
class DecodeMetadata:
    decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper]
    # full->SWA translated out_cache_loc (SWA KV-store write target)
    swa_out_cache_loc: Optional[torch.Tensor] = None


@dataclass
class PrefillMetadata:
    prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper]
    use_ragged: bool
    extend_no_prefix: bool
    multi_item_params: Optional[MultiItemScoringParams] = None
    swa_out_cache_loc: Optional[torch.Tensor] = None


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None

# Use as a fast path to override the indptr in flashinfer's plan function
# This is used to remove some host-to-device copy overhead.
global_override_indptr_cpu = None


def fast_prefill_plan(
    self,
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    page_size: int,
    head_dim_vo: Optional[int] = None,
    custom_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    window_left: int = -1,
    q_data_type: Union[str, torch.dtype] = "float16",
    kv_data_type: Optional[Union[str, torch.dtype]] = None,
    o_data_type: Optional[Union[str, torch.dtype]] = None,
    non_blocking: bool = True,
    fixed_split_size: Optional[int] = None,
    prefix_len_ptr: Optional[torch.Tensor] = None,
    token_pos_in_items_ptr: Optional[torch.Tensor] = None,
    token_pos_in_items_len: int = 0,
    max_item_len_ptr: Optional[torch.Tensor] = None,
    # Required host-known metadata: lets us skip the per-replay device-to-host
    # copies upstream plan() always issues. Keyword-only with no default so a
    # caller that forgets them fails at the call boundary, not with a cryptic
    # None crash deeper in.
    *,
    qo_indptr_host: torch.Tensor,
    kv_indptr_host: torch.Tensor,
    kv_lens_host: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
) -> None:
    """Sync-free ``BatchPrefillWithPagedKVCacheWrapper.plan`` for the EAGLE
    draft-extend CUDA graph (FlashInfer fa2, cuda-graph mode only).

    Upstream plan() always does qo/paged_kv/last_page_len ``.to("cpu")`` to build
    its host scheduling metadata, a blocking D2H that drains the GPU queue every
    replay. The caller passes host-known qo/kv layout in, so we call the underlying
    ``_cached_module.plan`` directly with no readback; the ``_plan_info`` produced
    is identical to plan()'s.
    """
    assert self.is_cuda_graph_enabled, "fast_prefill_plan is cuda-graph only"
    assert (
        getattr(self, "_backend", None) == "fa2"
    ), "fast_prefill_plan supports the fa2 backend only"
    assert (
        getattr(self, "_cached_module", None) is not None
    ), "fast_prefill_plan requires _cached_module from a prior real plan() (capture)"

    if head_dim_vo is None:
        head_dim_vo = head_dim_qk
    batch_size = len(paged_kv_last_page_len)

    total_num_rows = int(qo_indptr_host[-1])
    self._qo_indptr_last = total_num_rows
    self._max_q_len = max_q_len
    self._max_kv_len = max_kv_len

    if self._max_total_num_rows is None:
        self._max_total_num_rows = total_num_rows

    self._batch_size = batch_size
    self._num_qo_heads = num_qo_heads
    self._num_kv_heads = num_kv_heads
    self._prefix_len_ptr = prefix_len_ptr
    self._token_pos_in_items_ptr = token_pos_in_items_ptr
    self._token_pos_in_items_len = token_pos_in_items_len
    self._max_item_len_ptr = max_item_len_ptr

    # Refresh the cuda-graph input buffers (device-to-device, non-blocking).
    self._qo_indptr_buf.copy_(qo_indptr, non_blocking=non_blocking)
    self._paged_kv_indptr_buf.copy_(paged_kv_indptr, non_blocking=non_blocking)
    self._paged_kv_last_page_len_buf.copy_(
        paged_kv_last_page_len, non_blocking=non_blocking
    )
    self._paged_kv_indices_buf[: len(paged_kv_indices)].copy_(
        paged_kv_indices,
        non_blocking=(paged_kv_indices.device == self.device) and non_blocking,
    )

    self._cached_q_data_type = q_data_type
    self._cached_kv_data_type = (
        kv_data_type if kv_data_type is not None else q_data_type
    )
    self._cached_o_data_type = o_data_type
    self._block_tables = None

    args = [
        self._float_workspace_buffer,
        self._int_workspace_buffer,
        self._pin_memory_int_workspace_buffer,
        qo_indptr_host,
        kv_indptr_host,
        kv_lens_host,
        self._max_total_num_rows or total_num_rows,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        page_size,
        self.is_cuda_graph_enabled,
        head_dim_qk,
        head_dim_vo,
        causal,
        window_left,
        fixed_split_size if fixed_split_size is not None else -1,
        False,  # disable_split_kv
        0,  # num_colocated_ctas
    ]
    self._plan_info = self._cached_module.plan(*args)
def _is_nvfp4_native_kv_pool(token_to_kv_pool) -> bool:
    if isinstance(token_to_kv_pool, MHATokenToKVPoolFP4):
        return True
    return (
        isinstance(token_to_kv_pool, SWAKVPool)
        and isinstance(token_to_kv_pool.full_kv_pool, MHATokenToKVPoolFP4)
        and isinstance(token_to_kv_pool.swa_kv_pool, MHATokenToKVPoolFP4)
    )


def _is_fp8_k_nvfp4_v_pool(token_to_kv_pool) -> bool:
    if isinstance(token_to_kv_pool, MHATokenToKVPoolFP4):
        return bool(getattr(token_to_kv_pool, "mixed_fp8_k_nvfp4_v", False))
    return (
        isinstance(token_to_kv_pool, SWAKVPool)
        and bool(
            getattr(token_to_kv_pool.full_kv_pool, "mixed_fp8_k_nvfp4_v", False)
        )
        and bool(getattr(token_to_kv_pool.swa_kv_pool, "mixed_fp8_k_nvfp4_v", False))
    )


def _nvfp4_inner_pool_and_layer_id(token_to_kv_pool, layer_id: int):
    if not isinstance(token_to_kv_pool, SWAKVPool):
        return token_to_kv_pool, layer_id

    token_to_kv_pool._wait_for_layer(layer_id)
    local_layer_id, is_swa_layer = token_to_kv_pool.layers_mapping[layer_id]
    inner_pool = (
        token_to_kv_pool.swa_kv_pool
        if is_swa_layer
        else token_to_kv_pool.full_kv_pool
    )
    return inner_pool, local_layer_id


def _shape_nvfp4_kv_scale_for_flashinfer(scale: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if scale is None:
        return None
    if scale.dim() == 3:
        return scale.unsqueeze(1)
    return scale


def _tensor_trace_summary(x):
    if isinstance(x, torch.Tensor):
        return {
            "shape": tuple(x.shape),
            "dtype": str(x.dtype),
            "stride": tuple(x.stride()),
            "device": str(x.device),
        }
    if isinstance(x, (tuple, list)):
        return tuple(_tensor_trace_summary(item) for item in x)
    return repr(x)


def _scale_trace_value(x):
    try:
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return float(x.detach().float().cpu().item())
            return _tensor_trace_summary(x)
        return float(x)
    except Exception:
        return repr(x)


def _float_tensor_trace_values(x: torch.Tensor, limit: int = 32):
    try:
        flat = x.detach().float().reshape(-1).cpu()
        values = [float(v) for v in flat[:limit]]
        return {
            "count": int(flat.numel()),
            "values": values,
            "truncated": int(flat.numel()) > limit,
            "min": float(flat.min().item()) if flat.numel() else None,
            "max": float(flat.max().item()) if flat.numel() else None,
        }
    except Exception:
        return repr(x)


def _page_view_trace_summary(x):
    if not isinstance(x, torch.Tensor):
        return _tensor_trace_summary(x)
    return {
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "stride": tuple(x.stride()),
        "device": str(x.device),
        "storage_offset": x.storage_offset(),
        "data_ptr": x.data_ptr(),
    }


def _trace_numeric_tensor_stats(x, limit: Optional[int] = None):
    if not isinstance(x, torch.Tensor):
        return _tensor_trace_summary(x)
    if limit is None:
        limit = _trace_value_limit()

    summary = _tensor_trace_summary(x)
    try:
        flat = x.detach().flatten()
        sample = flat[:limit].to("cpu")
        summary["sample"] = sample.tolist()
    except Exception as exc:
        summary["sample_error"] = repr(exc)
        flat = None

    try:
        work = x.detach().float()
        finite = torch.isfinite(work)
        summary["finite"] = bool(finite.all().detach().cpu().item())
        if work.numel() > 0 and bool(finite.any().detach().cpu().item()):
            finite_values = work[finite]
            summary["min"] = float(finite_values.min().detach().cpu().item())
            summary["max"] = float(finite_values.max().detach().cpu().item())
            summary["mean"] = float(finite_values.mean().detach().cpu().item())
    except Exception as exc:
        summary["stats_error"] = repr(exc)

    return summary


def _trace_tensor_raw_bytes(x, limit: Optional[int] = None):
    if not isinstance(x, torch.Tensor):
        return _tensor_trace_summary(x)
    if limit is None:
        limit = _trace_value_limit(16)

    try:
        byte_view = x.detach().contiguous().view(torch.uint8).flatten()
        return {
            "shape": tuple(x.shape),
            "dtype": str(x.dtype),
            "stride": tuple(x.stride()),
            "bytes": byte_view[:limit].to("cpu").tolist(),
        }
    except Exception as exc:
        summary = _tensor_trace_summary(x)
        summary["bytes_error"] = repr(exc)
        return summary


def _trace_sample_page_bytes(x, page_ids, limit: Optional[int] = None):
    if not isinstance(x, torch.Tensor):
        return _tensor_trace_summary(x)
    if limit is None:
        limit = _trace_value_limit(16)

    samples = []
    for page_id in page_ids:
        try:
            page = int(page_id)
        except (TypeError, ValueError):
            samples.append({"page": repr(page_id), "error": "invalid page id"})
            continue

        if page < 0 or page >= x.shape[0]:
            samples.append(
                {
                    "page": page,
                    "error": "page out of range",
                    "first_dim": int(x.shape[0]),
                }
            )
            continue

        samples.append(
            {
                "page": page,
                "value": _trace_tensor_raw_bytes(x[page], limit=limit),
            }
        )
    return samples


def _trace_plan_sample_pages(plan, limit: int = 4):
    if not isinstance(plan, dict):
        return []
    kv_indices = plan.get("kv_indices_used")
    if not isinstance(kv_indices, dict):
        return []
    values = list(kv_indices.get("head") or []) + list(kv_indices.get("tail") or [])
    pages = []
    for value in values:
        try:
            page = int(value)
        except (TypeError, ValueError):
            continue
        if page not in pages:
            pages.append(page)
        if len(pages) >= limit:
            break
    return pages


def _trace_tensor_key(x):
    if not isinstance(x, torch.Tensor):
        return repr(x)
    return (
        tuple(x.shape),
        str(x.dtype),
        str(x.device),
        tuple(x.stride()),
        x.storage_offset(),
    )


def _trace_sample_rows(forward_batch: ForwardBatch, fallback_tokens: int):
    try:
        extend_seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_seq_lens_cpu:
            rows = []
            cursor = 0
            for seq_len in extend_seq_lens_cpu:
                cursor += int(seq_len)
                rows.append(cursor - 1)
            return rows
    except Exception:
        pass
    if fallback_tokens <= 0:
        return []
    return [fallback_tokens - 1]


def _trace_tensor_rows(x, rows, limit: Optional[int] = None):
    if not isinstance(x, torch.Tensor):
        return _tensor_trace_summary(x)
    if limit is None:
        limit = _trace_value_limit()
    samples = []
    for row in rows:
        try:
            row_id = int(row)
        except (TypeError, ValueError):
            samples.append({"row": repr(row), "error": "invalid row"})
            continue
        if row_id < 0 or row_id >= x.shape[0]:
            samples.append(
                {
                    "row": row_id,
                    "error": "row out of range",
                    "first_dim": int(x.shape[0]),
                }
            )
            continue
        samples.append({"row": row_id, "value": _trace_numeric_tensor_stats(x[row_id], limit)})
    return {
        "tensor": _tensor_trace_summary(x),
        "rows": samples,
    }


def _trace_nvfp4_write_samples(token_to_kv_pool, layer_id: int, page_ids):
    try:
        kv_pool, local_layer_id = _nvfp4_inner_pool_and_layer_id(
            token_to_kv_pool, layer_id
        )
        getter = getattr(kv_pool, "get_fp4_kv_write_trace_samples", None)
        if getter is None:
            return None
        return getter(local_layer_id, page_ids)
    except Exception as exc:
        return {"error": repr(exc)}


def _trace_compare_tensors(a: torch.Tensor, b: torch.Tensor):
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return {
            "a": _tensor_trace_summary(a),
            "b": _tensor_trace_summary(b),
            "error": "non-tensor input",
        }
    summary = {
        "a": _tensor_trace_summary(a),
        "b": _tensor_trace_summary(b),
        "shape_match": tuple(a.shape) == tuple(b.shape),
    }
    if tuple(a.shape) != tuple(b.shape):
        return summary
    try:
        af = a.detach().float()
        bf = b.detach().float()
        diff = af - bf
        finite = torch.isfinite(af) & torch.isfinite(bf)
        summary["finite_pair"] = bool(finite.all().detach().cpu().item())
        if diff.numel() > 0:
            summary["max_abs"] = float(diff.abs().max().detach().cpu().item())
            summary["rms"] = float(
                torch.sqrt(torch.mean(diff * diff)).detach().cpu().item()
            )
            summary["cosine"] = float(
                torch.nn.functional.cosine_similarity(
                    af.flatten(), bf.flatten(), dim=0, eps=1e-12
                )
                .detach()
                .cpu()
                .item()
            )
    except Exception as exc:
        summary["compare_error"] = repr(exc)
    return summary


def _trace_select_lse_slice(s: torch.Tensor, qo_start: int, qo_end: int, q_len: int):
    if not isinstance(s, torch.Tensor):
        return None
    if s.dim() < 2:
        return None
    try:
        if s.shape[0] >= qo_end:
            return s[qo_start:qo_end]
        if s.shape[-1] >= qo_end:
            moved = s.transpose(0, -1)
            return moved[qo_start:qo_end]
        if s.shape[0] == q_len:
            return s
        if s.shape[-1] == q_len:
            return s.transpose(0, -1)
    except Exception:
        return None
    return None


class FlashInferAttnBackend(AttentionBackend):
    """Flashinfer attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
        init_new_workspace: bool = False,
    ):
        super().__init__()
        self.prefill_backend = "fa2"
        self.decode_backend = "fa2"

        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self._swa_kv_pool: Optional[BaseSWAKVPool] = self._resolve_swa_kv_pool(
            model_runner
        )
        self.use_sliding_window_kv_pool = self._swa_kv_pool is not None
        self.enable_mis = model_runner.server_args.enable_mis
        self.is_nvfp4_native = _is_nvfp4_native_kv_pool(
            self.token_to_kv_pool
        ) and is_sm120_supported()
        self.is_fp8_k_nvfp4_v = self.is_nvfp4_native and _is_fp8_k_nvfp4_v_pool(
            self.token_to_kv_pool
        )
        self.enable_vo_split = _flashinfer_vo_split_enabled()

        # FIXME: remove dllm workarounds from flashinfer
        self.dllm_config = DllmConfig.from_server_args(model_runner.server_args)
        self.is_dllm_model = self.dllm_config is not None

        # Parse constants
        self.decode_use_tensor_cores = should_use_tensor_core(
            kv_cache_dtype=model_runner.kv_cache_dtype,
            num_attention_heads=model_runner.model_config.num_attention_heads
            // get_parallel().attn_tp_size,
            num_kv_heads=model_runner.model_config.get_num_kv_heads(
                get_parallel().attn_tp_size
            ),
        )
        if self.is_nvfp4_native:
            self.decode_use_tensor_cores = True
        if self.is_fp8_k_nvfp4_v:
            logger.warning(
                "SGLang FP4 KV mixed mode enabled: K cache uses FP8 e4m3, "
                "V cache uses packed NVFP4. Capacity claims must use the "
                "mixed-KV denominator, not full NVFP4 K+V."
            )
        if self.enable_vo_split:
            logger.warning(
                "SGLang FlashInfer VO split enabled: D=512 paged prefill "
                "and decode-as-prefill use two D_VO=256 passes."
            )
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill
        self.is_multimodal = model_runner.model_config.is_multimodal
        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        if model_runner.sliding_window_size is not None:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.SLIDING_WINDOW
        elif model_runner.model_config.is_encoder_decoder:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.CROSS_ATTENTION
        else:
            self.num_wrappers = 1
            self.dispatch_reason = None

        # Qwen2/Qwen3 models require higher flashinfer workspace size
        if (
            "Qwen2ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "Qwen3ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "MiMoForCausalLM" in model_runner.model_config.hf_config.architectures
            or "Qwen3VLForConditionalGeneration"
            in model_runner.model_config.hf_config.architectures
            or "Qwen3VLMoeForConditionalGeneration"
            in model_runner.model_config.hf_config.architectures
        ):
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(512 * 1024 * 1024)

        # When deterministic inference is enabled, tensor cores should be used for decode
        # Also set split tile sizes for prefill and decode from environment variables, and disable kv split for cuda graph
        # More information can be found here: https://github.com/flashinfer-ai/flashinfer/pull/1675
        self.enable_deterministic = (
            model_runner.server_args.enable_deterministic_inference
        )
        self.prefill_split_tile_size = None
        self.decode_split_tile_size = None
        self.disable_cuda_graph_kv_split = False
        if self.enable_deterministic:
            self.decode_use_tensor_cores = True
            self.prefill_split_tile_size = get_int_env_var(
                "SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096
            )
            self.decode_split_tile_size = get_int_env_var(
                "SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE", 2048
            )
            self.disable_cuda_graph_kv_split = True
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(2048 * 1024 * 1024)

        self.use_paged = envs.SGLANG_FLASHINFER_USE_PAGED.get()

        # Allocate buffers
        global global_workspace_buffer
        if global_workspace_buffer is None:
            # different from flashinfer zero_init_global_workspace_buffer
            global_workspace_size = envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get()
            global_workspace_buffer = torch.empty(
                global_workspace_size,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        if init_new_workspace:
            self.workspace_buffer = torch.empty(
                envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                dtype=torch.uint8,
                device=model_runner.device,
            )
        else:
            self.workspace_buffer = global_workspace_buffer
        max_bs = _cuda_graph_capture_max_bs(
            model_runner.server_args, model_runner.req_to_token_pool.size
        )
        if kv_indptr_buf is None:
            self.kv_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
                for _ in range(self.num_wrappers)
            ]
        else:
            assert self.num_wrappers == 1
            self.kv_indptr = [kv_indptr_buf]

        if kv_last_page_len_buf is None:
            self.kv_last_page_len = torch.ones(
                (max_bs,), dtype=torch.int32, device=model_runner.device
            )
        else:
            assert self.num_wrappers == 1
            self.kv_last_page_len = kv_last_page_len_buf

        if not self.skip_prefill:
            self.qo_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
                for _ in range(self.num_wrappers)
            ]

        fmha_backend = "auto"
        if is_sm100_supported():
            # Disable CUTLASS backend when piecewise cuda graph is enabled
            # due to TMA descriptor initialization issues on SM100 GPUs.
            if not check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE):
                fmha_backend = "cutlass"
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend=fmha_backend
        )

        # Two wrappers: one for sliding window attention and one for full attention.
        # Using two wrappers is unnecessary in the current PR, but are prepared for future PRs
        self.prefill_wrappers_paged = []
        self.prefill_wrappers_verify = []
        self.decode_wrappers = []
        for _ in range(self.num_wrappers):
            if not skip_prefill:
                self.prefill_wrappers_paged.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend=self.prefill_backend,
                    )
                )
                self.prefill_wrappers_verify.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend=self.prefill_backend,
                    )
                )
            self.decode_wrappers.append(
                BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    backend=self.decode_backend,
                    use_tensor_cores=self.decode_use_tensor_cores,
                )
            )

        self._geometry_trace_seen = set()
        self.wrapper_geometries = _flashinfer_wrapper_geometries(
            model_runner.model_config, self.dispatch_reason, self.num_wrappers
        )
        if _gemma4_geometry_trace_enabled():
            logger.warning(
                "SGLang FlashInfer wrapper geometries dispatch=%s geometries=%s",
                self.dispatch_reason,
                self.wrapper_geometries,
            )

        # Create indices updater
        if not skip_prefill:
            self.indices_updater_prefill = FlashInferIndicesUpdaterPrefill(
                model_runner, self
            )  # for verify
        self.indices_updater_decode = FlashInferIndicesUpdaterDecode(model_runner, self)

        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None

        self.decode_cuda_graph_metadata = {}
        self.prefill_cuda_graph_metadata = {}  # For verify
        self.draft_extend_cuda_graph_metadata = {}  # For draft extend
        self._nvfp4_trace_seen = set()
        self._nvfp4_batch_trace_seen = set()
        self._nvfp4_page_pair_trace_seen = set()
        self._nvfp4_merge_state_trace_seen = set()
        self._nvfp4_prefix_ref_trace_seen = set()
        self._nvfp4_dense_cache_trace_seen = set()
        self._nvfp4_module_trace_seen = set()
        self._nvfp4_last_paged_plan = {}
        self._nvfp4_last_paged_plan_tensors = {}
        self._nvfp4_dense_reference_by_layer = {}

    def _trace_gemma4_geometry_dispatch(
        self,
        *,
        label: Optional[str],
        layer: Optional[RadixAttention],
        paged_kv_cache=None,
        paged_kv_kwargs=None,
        wrapper=None,
        vo_split: bool = False,
    ) -> None:
        if (
            not _gemma4_geometry_trace_enabled()
            or layer is None
            or label is None
        ):
            return
        layer_id = getattr(layer, "layer_id", None)
        key = (label, int(layer_id) if layer_id is not None else None)
        if key in self._geometry_trace_seen:
            return
        self._geometry_trace_seen.add(key)
        k_sf = v_sf = None
        if isinstance(paged_kv_kwargs, dict):
            k_sf, v_sf = paged_kv_kwargs.get("kv_cache_sf", (None, None))
        wrapper_id = self._get_wrapper_idx(layer)
        planned = (
            self.wrapper_geometries[wrapper_id]
            if 0 <= wrapper_id < len(self.wrapper_geometries)
            else None
        )
        logger.warning(
            "SGLang Gemma4 FlashInfer geometry label=%s layer=%s "
            "wrapper_id=%s planned=%s layer_q_heads=%s layer_k_heads=%s "
            "layer_v_heads=%s layer_head_dim=%s sliding_window=%s "
            "vo_split=%s wrapper=%s kv_cache=%s k_sf=%s v_sf=%s",
            label,
            layer_id,
            wrapper_id,
            planned,
            getattr(layer, "tp_q_head_num", None),
            getattr(layer, "tp_k_head_num", None),
            getattr(layer, "tp_v_head_num", None),
            getattr(layer, "head_dim", None),
            getattr(layer, "sliding_window_size", None),
            vo_split,
            _flashinfer_wrapper_trace_summary(wrapper),
            _tensor_trace_summary(paged_kv_cache),
            _tensor_trace_summary(k_sf),
            _tensor_trace_summary(v_sf),
        )

    def _trace_nvfp4_native_call(
        self,
        *,
        label: str,
        layer: RadixAttention,
        q: torch.Tensor,
        paged_kv_cache,
        paged_kv_kwargs,
    ):
        if not self.is_nvfp4_native or os.environ.get("SGLANG_FP4_KV_TRACE_BACKEND") != "1":
            return
        key = (label, int(layer.layer_id))
        if key in self._nvfp4_trace_seen:
            return
        self._nvfp4_trace_seen.add(key)

        metadata = self.forward_metadata
        metadata_summary = {
            "type": type(metadata).__name__ if metadata is not None else None,
            "use_ragged": getattr(metadata, "use_ragged", None),
            "extend_no_prefix": getattr(metadata, "extend_no_prefix", None),
        }
        k_sf, v_sf = paged_kv_kwargs.get("kv_cache_sf", (None, None))
        logger.warning(
            "NVFP4 KV backend trace label=%s layer=%s wrapper_idx=%s "
            "metadata=%s q=%s kv_cache=%s k_sf=%s v_sf=%s "
            "k_scale=%s v_scale=%s",
            label,
            layer.layer_id,
            self._get_wrapper_idx(layer),
            metadata_summary,
            _tensor_trace_summary(q),
            _tensor_trace_summary(paged_kv_cache),
            _tensor_trace_summary(k_sf),
            _tensor_trace_summary(v_sf),
            _scale_trace_value(paged_kv_kwargs.get("k_scale")),
            _scale_trace_value(paged_kv_kwargs.get("v_scale")),
        )

    def _trace_nvfp4_forward_batch(
        self,
        *,
        label: str,
        forward_batch: ForwardBatch,
        use_ragged: Optional[bool],
        extend_no_prefix: Optional[bool],
    ):
        if not self.is_nvfp4_native or not _fp4_kv_radix_trace_enabled():
            return

        rids = _trace_rids(forward_batch)
        rid_filter = os.environ.get("SGLANG_FP4_KV_TRACE_RID")
        if rid_filter not in (None, "") and (rids is None or rid_filter not in rids):
            return

        extend_prefix_lens_cpu = _trace_cpu_values(
            getattr(forward_batch, "extend_prefix_lens_cpu", None)
        )
        seq_lens_cpu = _trace_cpu_values(getattr(forward_batch, "seq_lens_cpu", None))
        key = (
            label,
            tuple(rids or ()),
            tuple(extend_prefix_lens_cpu or ()),
            tuple(seq_lens_cpu or ()),
            bool(use_ragged),
            bool(extend_no_prefix),
        )
        if key in self._nvfp4_batch_trace_seen:
            return
        self._nvfp4_batch_trace_seen.add(key)

        logger.warning(
            "FP4 KV FlashInfer batch trace label=%s rids=%s mode=%s "
            "seq_lens_cpu=%s extend_prefix_lens_cpu=%s use_ragged=%s "
            "extend_no_prefix=%s req_pool_indices=%s out_cache_loc=%s",
            label,
            rids,
            getattr(forward_batch, "forward_mode", None),
            seq_lens_cpu,
            extend_prefix_lens_cpu,
            use_ragged,
            extend_no_prefix,
            _tensor_trace_summary(getattr(forward_batch, "req_pool_indices", None)),
            _tensor_trace_summary(getattr(forward_batch, "out_cache_loc", None)),
        )

    def _capture_nvfp4_paged_plan(
        self,
        *,
        label: str,
        wrapper_id: int,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_start_idx: Optional[torch.Tensor],
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        use_ragged: bool,
    ):
        if not self.is_nvfp4_native:
            return
        if (
            not _fp4_kv_page_pair_trace_enabled()
            and not _fp4_kv_prefix_ref_trace_enabled()
        ):
            return

        try:
            used = int(kv_indptr[-1].detach().to("cpu").item())
        except Exception:
            used = 0
        kv_indices_used = kv_indices[:used]
        tensor_plan = None
        if _fp4_kv_prefix_ref_trace_enabled():
            try:
                tensor_plan = {
                    "label": label,
                    "wrapper_id": int(wrapper_id),
                    "qo_indptr": qo_indptr.detach().clone(),
                    "kv_indptr": kv_indptr.detach().clone(),
                    "kv_indices": kv_indices_used.detach().clone(),
                    "prefix_lens": prefix_lens.detach().clone(),
                    "seq_lens": seq_lens.detach().clone(),
                }
            except Exception as exc:
                tensor_plan = {"error": repr(exc)}
            self._nvfp4_last_paged_plan_tensors[int(wrapper_id)] = tensor_plan

        plan = {
            "label": label,
            "wrapper_id": int(wrapper_id),
            "use_ragged": bool(use_ragged),
            "req_pool_indices": _trace_tensor_values(req_pool_indices),
            "paged_kernel_lens": _trace_tensor_values(paged_kernel_lens),
            "prefix_lens": _trace_tensor_values(prefix_lens),
            "seq_lens": _trace_tensor_values(seq_lens),
            "kv_start_idx": _trace_tensor_values(kv_start_idx),
            "kv_indptr": _trace_tensor_values(kv_indptr),
            "kv_indices_used": _trace_tensor_values(kv_indices_used),
        }
        self._nvfp4_last_paged_plan[int(wrapper_id)] = plan

        if not _fp4_kv_page_pair_trace_enabled():
            return

        key = (
            label,
            int(wrapper_id),
            tuple(plan["kv_indptr"]["head"] if plan["kv_indptr"] else ()),
            tuple(plan["kv_indices_used"]["head"] if plan["kv_indices_used"] else ()),
            tuple(plan["kv_indices_used"]["tail"] if plan["kv_indices_used"] else ()),
        )
        if key in self._nvfp4_page_pair_trace_seen:
            return
        self._nvfp4_page_pair_trace_seen.add(key)
        logger.warning("FP4 KV paged plan trace %s", plan)

    def _trace_nvfp4_page_pair(
        self,
        *,
        label: str,
        layer: RadixAttention,
        paged_kv_cache,
        paged_kv_kwargs,
    ):
        if not self.is_nvfp4_native or not _fp4_kv_page_pair_trace_enabled():
            return

        wrapper_id = int(self._get_wrapper_idx(layer))
        plan = self._nvfp4_last_paged_plan.get(wrapper_id)
        k_cache, v_cache = paged_kv_cache
        k_sf, v_sf = paged_kv_kwargs.get("kv_cache_sf", (None, None))
        first_dims = [
            tensor.shape[0] if isinstance(tensor, torch.Tensor) else None
            for tensor in (k_cache, v_cache, k_sf, v_sf)
        ]
        summary = {
            "label": label,
            "layer": int(layer.layer_id),
            "wrapper_id": wrapper_id,
            "plan": plan,
            "k_cache": _page_view_trace_summary(k_cache),
            "v_cache": _page_view_trace_summary(v_cache),
            "k_sf": _page_view_trace_summary(k_sf),
            "v_sf": _page_view_trace_summary(v_sf),
            "first_dims": first_dims,
            "first_dim_match": len(set(first_dims)) == 1,
            "k_scale": _scale_trace_value(paged_kv_kwargs.get("k_scale")),
            "v_scale": _scale_trace_value(paged_kv_kwargs.get("v_scale")),
        }
        key = (label, int(layer.layer_id), wrapper_id, repr(plan))
        if key in self._nvfp4_page_pair_trace_seen:
            return
        self._nvfp4_page_pair_trace_seen.add(key)
        logger.warning("FP4 KV page-pair trace %s", summary)

    def _trace_nvfp4_merge_state(
        self,
        *,
        label: str,
        layer: RadixAttention,
        paged_kv_cache,
        paged_kv_kwargs,
        o1: torch.Tensor,
        s1: torch.Tensor,
        o2: torch.Tensor,
        s2: torch.Tensor,
        merged: Optional[torch.Tensor],
        swa_window_left: Optional[int],
    ):
        if (
            not self.is_nvfp4_native
            or not _fp4_kv_merge_state_trace_enabled()
            or not _trace_layer_enabled(int(layer.layer_id))
        ):
            return

        wrapper_id = int(self._get_wrapper_idx(layer))
        plan = self._nvfp4_last_paged_plan.get(wrapper_id)
        page_ids = _trace_plan_sample_pages(plan)
        key = (
            label,
            int(layer.layer_id),
            wrapper_id,
            tuple(page_ids),
            _trace_tensor_key(o1),
            _trace_tensor_key(o2),
        )
        if key in self._nvfp4_merge_state_trace_seen:
            return
        self._nvfp4_merge_state_trace_seen.add(key)

        k_cache, v_cache = paged_kv_cache
        k_sf, v_sf = paged_kv_kwargs.get("kv_cache_sf", (None, None))
        summary = {
            "label": label,
            "layer": int(layer.layer_id),
            "wrapper_id": wrapper_id,
            "swa_window_left": swa_window_left,
            "plan": plan,
            "sample_page_ids": page_ids,
            "page_bytes": {
                "k_cache": _trace_sample_page_bytes(k_cache, page_ids),
                "v_cache": _trace_sample_page_bytes(v_cache, page_ids),
                "k_sf": _trace_sample_page_bytes(k_sf, page_ids),
                "v_sf": _trace_sample_page_bytes(v_sf, page_ids),
            },
            "write_trace": _trace_nvfp4_write_samples(
                self.token_to_kv_pool, int(layer.layer_id), page_ids
            ),
            "k_scale": _scale_trace_value(paged_kv_kwargs.get("k_scale")),
            "v_scale": _scale_trace_value(paged_kv_kwargs.get("v_scale")),
            "merge_inputs": {
                "o1_ragged": _trace_numeric_tensor_stats(o1),
                "s1_ragged": _trace_numeric_tensor_stats(s1),
                "o2_paged": _trace_numeric_tensor_stats(o2),
                "s2_paged": _trace_numeric_tensor_stats(s2),
            },
            "merged": _trace_numeric_tensor_stats(merged),
        }
        logger.warning("FP4 KV merge-state trace %s", summary)

    def _trace_nvfp4_dense_cache_state(
        self,
        *,
        label: str,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        q: Optional[torch.Tensor] = None,
        o: Optional[torch.Tensor] = None,
        o1: Optional[torch.Tensor] = None,
        s1: Optional[torch.Tensor] = None,
        o2: Optional[torch.Tensor] = None,
        s2: Optional[torch.Tensor] = None,
        merged: Optional[torch.Tensor] = None,
        paged_kv_kwargs: Optional[dict] = None,
    ):
        if (
            not self.is_nvfp4_native
            or not _fp4_kv_dense_cache_trace_enabled()
            or not _trace_layer_enabled(int(layer.layer_id))
        ):
            return

        rids = _trace_rids(forward_batch)
        rid_filter = os.environ.get("SGLANG_FP4_KV_TRACE_RID")
        if rid_filter not in (None, "") and (rids is None or rid_filter not in rids):
            return

        extend_prefix_lens_cpu = _trace_cpu_values(
            getattr(forward_batch, "extend_prefix_lens_cpu", None)
        )
        extend_seq_lens_cpu = _trace_cpu_values(
            getattr(forward_batch, "extend_seq_lens_cpu", None)
        )
        seq_lens_cpu = _trace_cpu_values(getattr(forward_batch, "seq_lens_cpu", None))
        row_source = o if isinstance(o, torch.Tensor) else merged
        if not isinstance(row_source, torch.Tensor):
            row_source = o1 if isinstance(o1, torch.Tensor) else q
        sample_rows = _trace_sample_rows(
            forward_batch,
            int(row_source.shape[0]) if isinstance(row_source, torch.Tensor) else 0,
        )
        key = (
            label,
            int(layer.layer_id),
            tuple(rids or ()),
            tuple(extend_prefix_lens_cpu or ()),
            tuple(extend_seq_lens_cpu or ()),
            tuple(sample_rows),
            _trace_tensor_key(row_source),
        )
        if key in self._nvfp4_dense_cache_trace_seen:
            return
        self._nvfp4_dense_cache_trace_seen.add(key)

        wrapper_id = int(self._get_wrapper_idx(layer))
        summary = {
            "kind": "attention",
            "label": label,
            "layer": int(layer.layer_id),
            "forward_pass_id": getattr(forward_batch, "forward_pass_id", None),
            "wrapper_id": wrapper_id,
            "rids": rids,
            "mode": repr(getattr(forward_batch, "forward_mode", None)),
            "seq_lens_cpu": seq_lens_cpu,
            "extend_prefix_lens_cpu": extend_prefix_lens_cpu,
            "extend_seq_lens_cpu": extend_seq_lens_cpu,
            "sample_rows": sample_rows,
            "req_pool_indices": _trace_tensor_values(
                getattr(forward_batch, "req_pool_indices", None)
            ),
            "out_cache_loc": _trace_tensor_values(
                getattr(forward_batch, "out_cache_loc", None)
            ),
            "q_rows": _trace_tensor_rows(q, sample_rows) if q is not None else None,
            "o_rows": _trace_tensor_rows(o, sample_rows) if o is not None else None,
            "o1_rows": _trace_tensor_rows(o1, sample_rows) if o1 is not None else None,
            "s1_rows": _trace_tensor_rows(s1, sample_rows) if s1 is not None else None,
            "o2_rows": _trace_tensor_rows(o2, sample_rows) if o2 is not None else None,
            "s2_rows": _trace_tensor_rows(s2, sample_rows) if s2 is not None else None,
            "merged_rows": (
                _trace_tensor_rows(merged, sample_rows) if merged is not None else None
            ),
        }
        if paged_kv_kwargs is not None:
            summary["k_scale"] = _scale_trace_value(paged_kv_kwargs.get("k_scale"))
            summary["v_scale"] = _scale_trace_value(paged_kv_kwargs.get("v_scale"))
        logger.warning("FP4 KV dense-cache attention trace %s", summary)

    def _trace_nvfp4_dense_quant_attention_loss(
        self,
        *,
        label: str,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        sm_scale: float,
        logits_soft_cap: Optional[float],
    ):
        if (
            not self.is_nvfp4_native
            or not _fp4_kv_dense_quant_attention_trace_enabled()
            or not _trace_layer_enabled(int(layer.layer_id))
        ):
            return
        try:
            from sglang.srt.layers.quantization.kvfp4_tensor import (
                E2M1_MAX,
                MAX_BLOCK_SCALE_FP8,
                NVFP4KVQuantizeUtil,
            )

            k_global, v_global = self.token_to_kv_pool._get_kv_global_scale_tensor(
                layer.layer_id
            )
            q3 = q.view(-1, layer.tp_q_head_num, layer.head_dim).contiguous()
            k3 = k.view(-1, layer.tp_k_head_num, layer.head_dim).contiguous()
            v3 = v.view(-1, layer.tp_v_head_num, layer.head_dim).contiguous()
            o3 = o.view(-1, layer.tp_q_head_num, layer.head_dim).contiguous()
            if label == "forward_extend_ragged_no_prefix":
                self._nvfp4_dense_reference_by_layer[int(layer.layer_id)] = {
                    "q": q3.detach(),
                    "k": k3.detach(),
                    "v": v3.detach(),
                    "o": o3.detach(),
                    "rids": _trace_rids(forward_batch),
                    "seq_lens_cpu": _trace_cpu_values(
                        getattr(forward_batch, "seq_lens_cpu", None)
                    ),
                    "extend_seq_lens_cpu": _trace_cpu_values(
                        getattr(forward_batch, "extend_seq_lens_cpu", None)
                    ),
                    "forward_pass_id": getattr(forward_batch, "forward_pass_id", None),
                }

            k_fp4, k_sf, _ = NVFP4KVQuantizeUtil.quantize(k3, k_global)
            v_fp4, v_sf, _ = NVFP4KVQuantizeUtil.quantize(v3, v_global)
            k_deq = NVFP4KVQuantizeUtil.dequantize(
                k_fp4.view(torch.uint8),
                k_sf.reshape(k3.shape[0], -1),
                k_global,
                dtype=k3.dtype,
            ).view_as(k3)
            v_deq = NVFP4KVQuantizeUtil.dequantize(
                v_fp4.view(torch.uint8),
                v_sf.reshape(v3.shape[0], -1),
                v_global,
                dtype=v3.dtype,
            ).view_as(v3)
            k_scale_multiplier_refs = []
            for multiplier in _trace_k_scale_multipliers():
                alt_k_global = (k_global.float() * multiplier).contiguous()
                alt_k_fp4, alt_k_sf, _ = NVFP4KVQuantizeUtil.quantize(
                    k3, alt_k_global
                )
                alt_k_deq = NVFP4KVQuantizeUtil.dequantize(
                    alt_k_fp4.view(torch.uint8),
                    alt_k_sf.reshape(k3.shape[0], -1),
                    alt_k_global,
                    dtype=k3.dtype,
                ).view_as(k3)
                k_scale_multiplier_refs.append(
                    {
                        "multiplier": multiplier,
                        "k_global": _scale_trace_value(alt_k_global),
                        "k_deq": alt_k_deq,
                    }
                )
            k_head_scale_policy_refs = []
            if _trace_k_head_scale_policy_enabled():
                denom = E2M1_MAX * MAX_BLOCK_SCALE_FP8
                head_amax = k3.detach().abs().amax(dim=(0, 2)).float()
                base_k_global = k_global.float().reshape(1).contiguous()
                head_multipliers = _trace_k_scale_multipliers() or [1.0]
                for multiplier in head_multipliers[:16]:
                    head_globals = (head_amax / denom * multiplier).clamp(min=1e-8)
                    head_chunks = []
                    ratio_values = []
                    sf_before_values = []
                    sf_after_values = []
                    for head_id in range(k3.shape[1]):
                        head_global = head_globals[head_id : head_id + 1].contiguous()
                        head_fp4, head_sf, _ = NVFP4KVQuantizeUtil.quantize(
                            k3[:, head_id : head_id + 1, :].contiguous(),
                            head_global,
                        )
                        # FlashInfer's paged attention receives a single global scale.
                        # This simulates carrying per-head effective globals by folding
                        # the ratio into the stored FP8 block-scale buffer.
                        ratio = (head_global / base_k_global).contiguous()
                        head_sf_for_base = (head_sf.float() * ratio).to(
                            torch.float8_e4m3fn
                        )
                        head_deq = NVFP4KVQuantizeUtil.dequantize(
                            head_fp4.view(torch.uint8),
                            head_sf_for_base.reshape(k3.shape[0], -1),
                            base_k_global,
                            dtype=k3.dtype,
                        ).view(k3.shape[0], 1, k3.shape[2])
                        head_chunks.append(head_deq)
                        ratio_values.append(ratio.reshape(()))
                        sf_before_values.append(head_sf.float().reshape(-1))
                        sf_after_values.append(head_sf_for_base.float().reshape(-1))
                    k_head_scale_policy_refs.append(
                        {
                            "policy": "per_kv_head_amax_folded_sf",
                            "multiplier": multiplier,
                            "k_globals": _float_tensor_trace_values(head_globals),
                            "global_to_base_ratios": _float_tensor_trace_values(
                                torch.stack(ratio_values)
                            ),
                            "sf_before": _float_tensor_trace_values(
                                torch.cat(sf_before_values)
                            ),
                            "sf_after": _float_tensor_trace_values(
                                torch.cat(sf_after_values)
                            ),
                            "k_deq": torch.cat(head_chunks, dim=1),
                        }
                    )

            sample_rows = _trace_sample_rows(forward_batch, int(o3.shape[0]))
            head_repeat = layer.tp_q_head_num // layer.tp_k_head_num
            if head_repeat <= 0 or layer.tp_q_head_num % layer.tp_k_head_num != 0:
                raise RuntimeError(
                    f"unsupported GQA mapping: q_heads={layer.tp_q_head_num}, "
                    f"kv_heads={layer.tp_k_head_num}"
                )

            def attention_ref(kv_k: torch.Tensor, kv_v: torch.Tensor, row_id: int):
                q_row = q3[row_id].float()
                k_prefix = kv_k[: row_id + 1].float()
                v_prefix = kv_v[: row_id + 1].float()
                k_rep = k_prefix.repeat_interleave(head_repeat, dim=1)
                v_rep = v_prefix.repeat_interleave(head_repeat, dim=1)
                scores = torch.einsum("hd,thd->ht", q_row, k_rep) * sm_scale
                if logits_soft_cap is not None and float(logits_soft_cap) > 0:
                    scores = float(logits_soft_cap) * torch.tanh(
                        scores / float(logits_soft_cap)
                    )
                probs = torch.softmax(scores, dim=-1)
                return torch.einsum("ht,thd->hd", probs, v_rep).to(o3.dtype)

            rows = []
            for row in sample_rows:
                row_id = int(row)
                if row_id < 0 or row_id >= o3.shape[0]:
                    continue
                bf16_ref = attention_ref(k3, v3, row_id)
                fp4_ref = attention_ref(k_deq, v_deq, row_id)
                fp4_k_ref = attention_ref(k_deq, v3, row_id)
                fp4_v_ref = attention_ref(k3, v_deq, row_id)
                multiplier_rows = []
                for alt in k_scale_multiplier_refs:
                    alt_k_deq = alt["k_deq"]
                    alt_fp4_ref = attention_ref(alt_k_deq, v_deq, row_id)
                    alt_fp4_k_ref = attention_ref(alt_k_deq, v3, row_id)
                    multiplier_rows.append(
                        {
                            "multiplier": alt["multiplier"],
                            "k_global": alt["k_global"],
                            "k_dequant_prefix_vs_bf16_prefix": (
                                _trace_compare_tensors(
                                    alt_k_deq[: row_id + 1],
                                    k3[: row_id + 1],
                                )
                            ),
                            "fp4_ref_vs_bf16_ref": _trace_compare_tensors(
                                alt_fp4_ref, bf16_ref
                            ),
                            "fp4_k_only_ref_vs_bf16_ref": (
                                _trace_compare_tensors(alt_fp4_k_ref, bf16_ref)
                            ),
                        }
                    )
                head_policy_rows = []
                for policy in k_head_scale_policy_refs:
                    policy_k_deq = policy["k_deq"]
                    policy_fp4_ref = attention_ref(policy_k_deq, v_deq, row_id)
                    policy_fp4_k_ref = attention_ref(policy_k_deq, v3, row_id)
                    head_policy_rows.append(
                        {
                            "policy": policy["policy"],
                            "multiplier": policy["multiplier"],
                            "k_globals": policy["k_globals"],
                            "global_to_base_ratios": policy["global_to_base_ratios"],
                            "sf_before": policy["sf_before"],
                            "sf_after": policy["sf_after"],
                            "k_dequant_prefix_vs_bf16_prefix": (
                                _trace_compare_tensors(
                                    policy_k_deq[: row_id + 1],
                                    k3[: row_id + 1],
                                )
                            ),
                            "fp4_ref_vs_bf16_ref": _trace_compare_tensors(
                                policy_fp4_ref, bf16_ref
                            ),
                            "fp4_k_only_ref_vs_bf16_ref": (
                                _trace_compare_tensors(policy_fp4_k_ref, bf16_ref)
                            ),
                        }
                    )
                actual = o3[row_id]
                row_record = {
                    "row": row_id,
                    "actual_vs_bf16_ref": _trace_compare_tensors(
                        actual, bf16_ref
                    ),
                    "actual_vs_fp4_ref": _trace_compare_tensors(actual, fp4_ref),
                    "fp4_ref_vs_bf16_ref": _trace_compare_tensors(
                        fp4_ref, bf16_ref
                    ),
                    "fp4_k_only_ref_vs_bf16_ref": _trace_compare_tensors(
                        fp4_k_ref, bf16_ref
                    ),
                    "fp4_v_only_ref_vs_bf16_ref": _trace_compare_tensors(
                        fp4_v_ref, bf16_ref
                    ),
                    "actual": _trace_numeric_tensor_stats(actual),
                    "bf16_ref": _trace_numeric_tensor_stats(bf16_ref),
                    "fp4_ref": _trace_numeric_tensor_stats(fp4_ref),
                    "fp4_k_only_ref": _trace_numeric_tensor_stats(fp4_k_ref),
                    "fp4_v_only_ref": _trace_numeric_tensor_stats(fp4_v_ref),
                }
                if multiplier_rows:
                    row_record["k_scale_multiplier_refs"] = multiplier_rows
                if head_policy_rows:
                    row_record["k_head_scale_policy_refs"] = head_policy_rows
                rows.append(row_record)

            if rows:
                logger.warning(
                    "FP4 KV dense-quant attention trace %s",
                    {
                        "kind": "dense_quant_attention",
                        "label": label,
                        "layer": int(layer.layer_id),
                        "forward_pass_id": getattr(
                            forward_batch, "forward_pass_id", None
                        ),
                        "rids": _trace_rids(forward_batch),
                        "mode": repr(getattr(forward_batch, "forward_mode", None)),
                        "sample_rows": sample_rows,
                        "k_global": _scale_trace_value(k_global),
                        "v_global": _scale_trace_value(v_global),
                        "logits_soft_cap": (
                            None if logits_soft_cap is None else float(logits_soft_cap)
                        ),
                        "rows": rows,
                    },
                )
        except Exception as exc:
            logger.warning(
                "FP4 KV dense-quant attention trace failed layer=%s error=%r",
                int(layer.layer_id),
                exc,
            )

    def _trace_nvfp4_prefix_reference(
        self,
        *,
        label: str,
        layer: RadixAttention,
        q: torch.Tensor,
        suffix_k: torch.Tensor,
        suffix_v: torch.Tensor,
        paged_kv_cache,
        paged_kv_kwargs,
        o1: torch.Tensor,
        s1: torch.Tensor,
        o2: torch.Tensor,
        s2: torch.Tensor,
        merged: torch.Tensor,
        swa_window_left: Optional[int],
        sm_scale: float,
        logits_soft_cap: Optional[float],
    ):
        if (
            not self.is_nvfp4_native
            or not _fp4_kv_prefix_ref_trace_enabled()
            or not _trace_layer_enabled(int(layer.layer_id))
        ):
            return

        wrapper_id = int(self._get_wrapper_idx(layer))
        plan = self._nvfp4_last_paged_plan_tensors.get(wrapper_id)
        key = (label, int(layer.layer_id), wrapper_id, _trace_tensor_key(q))
        if key in self._nvfp4_prefix_ref_trace_seen:
            return
        self._nvfp4_prefix_ref_trace_seen.add(key)

        summary = {
            "label": label,
            "layer": int(layer.layer_id),
            "wrapper_id": wrapper_id,
            "swa_window_left": swa_window_left,
            "plan_error": None,
        }
        try:
            if not isinstance(plan, dict) or "error" in plan:
                summary["plan_error"] = plan
                logger.warning("FP4 KV prefix-reference trace %s", summary)
                return

            qo_indptr = plan["qo_indptr"].to(device=q.device, dtype=torch.long)
            kv_indptr = plan["kv_indptr"].to(device=q.device, dtype=torch.long)
            kv_indices = plan["kv_indices"].to(device=q.device, dtype=torch.long)
            prefix_lens = plan["prefix_lens"].to(device=q.device, dtype=torch.long)

            req_idx = None
            for idx, prefix_len in enumerate(prefix_lens.detach().cpu().tolist()):
                if int(prefix_len) > 0:
                    req_idx = idx
                    break
            if req_idx is None:
                summary["skip"] = "no cached prefix"
                logger.warning("FP4 KV prefix-reference trace %s", summary)
                return

            qo_start = int(qo_indptr[req_idx].detach().cpu().item())
            qo_end = int(qo_indptr[req_idx + 1].detach().cpu().item())
            kv_start = int(kv_indptr[req_idx].detach().cpu().item())
            kv_end = int(kv_indptr[req_idx + 1].detach().cpu().item())
            token_limit = _trace_token_limit()
            kv_end = min(kv_end, kv_start + token_limit)
            q_row_limit = _trace_q_row_limit()
            if q_row_limit is not None:
                qo_end = min(qo_end, qo_start + q_row_limit)
            q_req = q[qo_start:qo_end].float()
            prefix_slots = kv_indices[kv_start:kv_end]
            summary.update(
                {
                    "request_index": req_idx,
                    "qo_range": [qo_start, qo_end],
                    "kv_range": [kv_start, kv_end],
                    "prefix_slots": _trace_tensor_values(prefix_slots),
                    "token_limit": token_limit,
                    "q_row_limit": q_row_limit,
                }
            )
            if q_req.numel() == 0 or prefix_slots.numel() == 0:
                summary["skip"] = "empty q or prefix"
                logger.warning("FP4 KV prefix-reference trace %s", summary)
                return

            k_cache, v_cache = paged_kv_cache
            k_sf, v_sf = paged_kv_kwargs.get("kv_cache_sf", (None, None))
            if not all(isinstance(x, torch.Tensor) for x in (k_cache, v_cache, v_sf)):
                summary["error"] = "missing tensor cache or scale views"
                summary["cache"] = _tensor_trace_summary(paged_kv_cache)
                summary["scale"] = _tensor_trace_summary((k_sf, v_sf))
                logger.warning("FP4 KV prefix-reference trace %s", summary)
                return

            k_packed = k_cache[prefix_slots]
            v_packed = v_cache[prefix_slots]
            k_scale = k_sf[prefix_slots] if isinstance(k_sf, torch.Tensor) else None
            v_scale = v_sf[prefix_slots]
            if k_scale is not None and k_scale.dim() == 4 and k_scale.shape[1] == 1:
                k_scale = k_scale[:, 0]
            if v_scale.dim() == 4 and v_scale.shape[1] == 1:
                v_scale = v_scale[:, 0]
            if k_scale is not None:
                k_scale = k_scale.view(torch.float8_e4m3fn)
            v_scale = v_scale.view(torch.float8_e4m3fn)
            k_scale_for_dequant = (
                k_scale.reshape(k_scale.shape[0], -1)
                if k_scale is not None
                else None
            )
            v_scale_for_dequant = v_scale.reshape(v_scale.shape[0], -1)

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                NVFP4KVQuantizeUtil,
            )

            if k_scale_for_dequant is None:
                k_ref = k_packed.float()
            else:
                k_ref = NVFP4KVQuantizeUtil.dequantize(
                    k_packed.view(torch.uint8),
                    k_scale_for_dequant,
                    paged_kv_kwargs["k_scale"],
                    dtype=torch.float32,
                ).float()
            v_ref = NVFP4KVQuantizeUtil.dequantize(
                v_packed.view(torch.uint8),
                v_scale_for_dequant,
                paged_kv_kwargs["v_scale"],
                dtype=torch.float32,
            ).float()

            q_heads = q_req.shape[1]
            kv_heads = k_ref.shape[1]
            if q_heads % kv_heads != 0:
                summary["error"] = "q heads are not divisible by kv heads"
                summary["q_heads"] = int(q_heads)
                summary["kv_heads"] = int(kv_heads)
                logger.warning("FP4 KV prefix-reference trace %s", summary)
                return

            kv_head_for_q = torch.arange(q_heads, device=q.device) // (
                q_heads // kv_heads
            )
            k_for_q = k_ref[:, kv_head_for_q, :]
            v_for_q = v_ref[:, kv_head_for_q, :]
            logits = torch.einsum("qhd,thd->qht", q_req, k_for_q) * float(sm_scale)
            if logits_soft_cap is not None and float(logits_soft_cap) > 0:
                logits = float(logits_soft_cap) * torch.tanh(
                    logits / float(logits_soft_cap)
                )
            lse_ref = torch.logsumexp(logits, dim=-1)
            lse_ref_base2 = lse_ref * 1.4426950408889634
            probs = torch.softmax(logits, dim=-1)
            o2_ref = torch.einsum("qht,thd->qhd", probs, v_for_q).to(o2.dtype)

            o2_slice = o2[qo_start:qo_end]
            s2_slice = _trace_select_lse_slice(s2, qo_start, qo_end, qo_end - qo_start)
            summary["reference"] = {
                "q": _trace_numeric_tensor_stats(q_req),
                "k_dequant": _trace_numeric_tensor_stats(k_ref),
                "v_dequant": _trace_numeric_tensor_stats(v_ref),
                "lse_ref": _trace_numeric_tensor_stats(lse_ref),
                "lse_ref_base2": _trace_numeric_tensor_stats(lse_ref_base2),
                "o2_ref": _trace_numeric_tensor_stats(o2_ref),
                "o2_flashinfer": _trace_numeric_tensor_stats(o2_slice),
                "o2_compare": _trace_compare_tensors(o2_ref, o2_slice),
            }
            if isinstance(s2_slice, torch.Tensor):
                summary["reference"]["s2_flashinfer"] = _trace_numeric_tensor_stats(
                    s2_slice
                )
                summary["reference"]["s2_compare"] = _trace_compare_tensors(
                    lse_ref_base2, s2_slice
                )
                summary["reference"]["s2_compare_natural_log"] = (
                    _trace_compare_tensors(lse_ref, s2_slice)
                )
            else:
                summary["reference"]["s2_compare"] = {
                    "error": "could not select comparable s2 slice",
                    "s2": _tensor_trace_summary(s2),
                }

            s1_slice = _trace_select_lse_slice(s1, qo_start, qo_end, qo_end - qo_start)
            merged_slice = merged[qo_start:qo_end]
            o1_slice = o1[qo_start:qo_end].float()
            if isinstance(s1_slice, torch.Tensor) and isinstance(s2_slice, torch.Tensor):
                o2_slice_f = o2_slice.float()
                s1_work = s1_slice.float().unsqueeze(-1)
                s2_work = s2_slice.float().unsqueeze(-1)
                m = torch.maximum(s1_work, s2_work)
                w1 = torch.exp2(s1_work - m)
                w2 = torch.exp2(s2_work - m)
                manual_merged = ((o1_slice * w1) + (o2_slice_f * w2)) / (w1 + w2)
                summary["merge_compare"] = _trace_compare_tensors(
                    manual_merged.to(merged.dtype), merged_slice
                )
            else:
                summary["merge_compare"] = {
                    "error": "could not select comparable s1/s2 slices",
                    "s1": _tensor_trace_summary(s1),
                    "s2": _tensor_trace_summary(s2),
                }

            suffix_k_req_raw = suffix_k.view(
                -1, layer.tp_k_head_num, layer.head_dim
            )[qo_start:qo_end].contiguous()
            suffix_v_req_raw = suffix_v.view(
                -1, layer.tp_v_head_num, layer.head_dim
            )[qo_start:qo_end].contiguous()
            suffix_k_req = suffix_k_req_raw.float()
            suffix_v_req = suffix_v_req_raw.float()

            def attention_ref_for_q(
                kv_k: torch.Tensor,
                kv_v: torch.Tensor,
                q_rows: torch.Tensor,
            ):
                if kv_k.shape[0] == 0:
                    out = torch.zeros_like(q_rows)
                    lse = torch.full(
                        q_rows.shape[:2],
                        float("-inf"),
                        dtype=torch.float32,
                        device=q_rows.device,
                    )
                    return out, lse
                q_heads_ref = q_rows.shape[1]
                kv_heads_ref = kv_k.shape[1]
                kv_head_for_q_ref = torch.arange(
                    q_heads_ref, device=q_rows.device
                ) // (q_heads_ref // kv_heads_ref)
                k_for_q_ref = kv_k[:, kv_head_for_q_ref, :]
                v_for_q_ref = kv_v[:, kv_head_for_q_ref, :]
                logits_ref = (
                    torch.einsum("qhd,thd->qht", q_rows, k_for_q_ref)
                    * float(sm_scale)
                )
                if logits_soft_cap is not None and float(logits_soft_cap) > 0:
                    logits_ref = float(logits_soft_cap) * torch.tanh(
                        logits_ref / float(logits_soft_cap)
                    )
                lse_ref_local = torch.logsumexp(logits_ref, dim=-1)
                probs_ref = torch.softmax(logits_ref, dim=-1)
                out_ref = torch.einsum("qht,thd->qhd", probs_ref, v_for_q_ref)
                return out_ref, lse_ref_local

            def causal_suffix_ref(
                kv_k: torch.Tensor,
                kv_v: torch.Tensor,
                q_rows: torch.Tensor,
            ):
                outs = []
                lses = []
                for row_idx in range(q_rows.shape[0]):
                    out_i, lse_i = attention_ref_for_q(
                        kv_k[: row_idx + 1],
                        kv_v[: row_idx + 1],
                        q_rows[row_idx : row_idx + 1],
                    )
                    outs.append(out_i)
                    lses.append(lse_i)
                return torch.cat(outs, dim=0), torch.cat(lses, dim=0)

            def full_with_prefix_ref(
                prefix_k: torch.Tensor,
                prefix_v: torch.Tensor,
                suffix_k_local: torch.Tensor,
                suffix_v_local: torch.Tensor,
                q_rows: torch.Tensor,
            ):
                outs = []
                lses = []
                for row_idx in range(q_rows.shape[0]):
                    full_k = torch.cat(
                        [prefix_k, suffix_k_local[: row_idx + 1]], dim=0
                    )
                    full_v = torch.cat(
                        [prefix_v, suffix_v_local[: row_idx + 1]], dim=0
                    )
                    out_i, lse_i = attention_ref_for_q(
                        full_k, full_v, q_rows[row_idx : row_idx + 1]
                    )
                    outs.append(out_i)
                    lses.append(lse_i)
                return torch.cat(outs, dim=0), torch.cat(lses, dim=0)

            runtime_suffix_o, runtime_suffix_lse = causal_suffix_ref(
                suffix_k_req, suffix_v_req, q_req
            )
            runtime_full_o, runtime_full_lse = full_with_prefix_ref(
                k_ref.float(), v_ref.float(), suffix_k_req, suffix_v_req, q_req
            )

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                NVFP4KVQuantizeUtil as _NVFP4KVQuantizeUtilForSuffix,
            )

            suffix_k_fp4, suffix_k_sf, _ = _NVFP4KVQuantizeUtilForSuffix.quantize(
                suffix_k_req_raw, paged_kv_kwargs["k_scale"]
            )
            suffix_v_fp4, suffix_v_sf, _ = _NVFP4KVQuantizeUtilForSuffix.quantize(
                suffix_v_req_raw, paged_kv_kwargs["v_scale"]
            )
            suffix_k_fp4_ref = _NVFP4KVQuantizeUtilForSuffix.dequantize(
                suffix_k_fp4.view(torch.uint8),
                suffix_k_sf.reshape(suffix_k_req.shape[0], -1),
                paged_kv_kwargs["k_scale"],
                dtype=torch.float32,
            ).view_as(suffix_k_req)
            suffix_v_fp4_ref = _NVFP4KVQuantizeUtilForSuffix.dequantize(
                suffix_v_fp4.view(torch.uint8),
                suffix_v_sf.reshape(suffix_v_req.shape[0], -1),
                paged_kv_kwargs["v_scale"],
                dtype=torch.float32,
            ).view_as(suffix_v_req)
            all_fp4_suffix_o, all_fp4_suffix_lse = causal_suffix_ref(
                suffix_k_fp4_ref, suffix_v_fp4_ref, q_req
            )
            all_fp4_full_o, all_fp4_full_lse = full_with_prefix_ref(
                k_ref.float(),
                v_ref.float(),
                suffix_k_fp4_ref,
                suffix_v_fp4_ref,
                q_req,
            )
            runtime_suffix_lse_base2 = runtime_suffix_lse * 1.4426950408889634
            runtime_full_lse_base2 = runtime_full_lse * 1.4426950408889634
            all_fp4_suffix_lse_base2 = all_fp4_suffix_lse * 1.4426950408889634
            all_fp4_full_lse_base2 = all_fp4_full_lse * 1.4426950408889634
            summary["runtime_full_reference"] = {
                "suffix_o_bf16_vs_o1": (
                    _trace_compare_tensors(
                        runtime_suffix_o.to(o1_slice.dtype), o1_slice
                    )
                    if isinstance(s1_slice, torch.Tensor)
                    else {"error": "missing s1 slice"}
                ),
                "suffix_lse_bf16_base2_vs_s1": (
                    _trace_compare_tensors(runtime_suffix_lse_base2, s1_slice)
                    if isinstance(s1_slice, torch.Tensor)
                    else {"error": "missing s1 slice"}
                ),
                "suffix_lse_bf16_natural_vs_s1": (
                    _trace_compare_tensors(runtime_suffix_lse, s1_slice)
                    if isinstance(s1_slice, torch.Tensor)
                    else {"error": "missing s1 slice"}
                ),
                "full_prefix_fp4_suffix_bf16_vs_merged": _trace_compare_tensors(
                    runtime_full_o.to(merged_slice.dtype), merged_slice
                ),
                "full_prefix_fp4_suffix_bf16_lse_base2": _trace_numeric_tensor_stats(
                    runtime_full_lse_base2
                ),
                "suffix_o_fp4_vs_o1": (
                    _trace_compare_tensors(
                        all_fp4_suffix_o.to(o1_slice.dtype), o1_slice
                    )
                    if isinstance(s1_slice, torch.Tensor)
                    else {"error": "missing s1 slice"}
                ),
                "suffix_lse_fp4_base2_vs_s1": (
                    _trace_compare_tensors(all_fp4_suffix_lse_base2, s1_slice)
                    if isinstance(s1_slice, torch.Tensor)
                    else {"error": "missing s1 slice"}
                ),
                "full_prefix_fp4_suffix_fp4_vs_merged": _trace_compare_tensors(
                    all_fp4_full_o.to(merged_slice.dtype), merged_slice
                ),
                "full_prefix_fp4_suffix_fp4_lse_base2": _trace_numeric_tensor_stats(
                    all_fp4_full_lse_base2
                ),
            }

            dense_ref = self._nvfp4_dense_reference_by_layer.get(int(layer.layer_id))
            if isinstance(dense_ref, dict):
                q_len = qo_end - qo_start
                dense_row_start = int(prefix_lens[req_idx].detach().cpu().item())
                dense_row_end = dense_row_start + q_len
                dense_q = dense_ref["q"]
                dense_k = dense_ref["k"]
                dense_v = dense_ref["v"]
                dense_o = dense_ref["o"]
                summary["dense_reference"] = {
                    "dense_rids": dense_ref.get("rids"),
                    "dense_forward_pass_id": dense_ref.get("forward_pass_id"),
                    "dense_seq_lens_cpu": dense_ref.get("seq_lens_cpu"),
                    "dense_extend_seq_lens_cpu": dense_ref.get("extend_seq_lens_cpu"),
                    "dense_row_range": [dense_row_start, dense_row_end],
                }
                if dense_row_end <= dense_q.shape[0]:
                    dense_q_req = dense_q[dense_row_start:dense_row_end].float()
                    dense_prefix_k = dense_k[:dense_row_start].float()
                    dense_prefix_v = dense_v[:dense_row_start].float()
                    dense_suffix_k = dense_k[dense_row_start:dense_row_end].float()
                    dense_suffix_v = dense_v[dense_row_start:dense_row_end].float()
                    dense_full_k = dense_k[:dense_row_end].float()
                    dense_full_v = dense_v[:dense_row_end].float()
                    dense_o_slice = dense_o[dense_row_start:dense_row_end]
                    q_req_f = q_req.float()

                    def dense_attention_ref(kv_k: torch.Tensor, kv_v: torch.Tensor):
                        if kv_k.shape[0] == 0:
                            out = torch.zeros_like(q_req_f)
                            lse = torch.full(
                                q_req_f.shape[:2],
                                float("-inf"),
                                dtype=torch.float32,
                                device=q_req_f.device,
                            )
                            return out, lse
                        q_heads = q_req_f.shape[1]
                        kv_heads = kv_k.shape[1]
                        kv_head_for_q = torch.arange(q_heads, device=q_req_f.device) // (
                            q_heads // kv_heads
                        )
                        k_for_q = kv_k[:, kv_head_for_q, :]
                        v_for_q = kv_v[:, kv_head_for_q, :]
                        logits = (
                            torch.einsum("qhd,thd->qht", q_req_f, k_for_q)
                            * float(sm_scale)
                        )
                        if logits_soft_cap is not None and float(logits_soft_cap) > 0:
                            logits = float(logits_soft_cap) * torch.tanh(
                                logits / float(logits_soft_cap)
                            )
                        lse = torch.logsumexp(logits, dim=-1)
                        probs = torch.softmax(logits, dim=-1)
                        out = torch.einsum("qht,thd->qhd", probs, v_for_q)
                        return out, lse

                    def dense_causal_suffix_ref(kv_k: torch.Tensor, kv_v: torch.Tensor):
                        outs = []
                        lses = []
                        for row_idx in range(q_req_f.shape[0]):
                            out_i, lse_i = dense_attention_ref(
                                kv_k[: row_idx + 1],
                                kv_v[: row_idx + 1],
                            )
                            outs.append(out_i[row_idx : row_idx + 1])
                            lses.append(lse_i[row_idx : row_idx + 1])
                        return torch.cat(outs, dim=0), torch.cat(lses, dim=0)

                    def dense_causal_full_ref(kv_k: torch.Tensor, kv_v: torch.Tensor):
                        outs = []
                        lses = []
                        for row_idx in range(q_req_f.shape[0]):
                            end = dense_row_start + row_idx + 1
                            out_i, lse_i = dense_attention_ref(kv_k[:end], kv_v[:end])
                            outs.append(out_i[row_idx : row_idx + 1])
                            lses.append(lse_i[row_idx : row_idx + 1])
                        return torch.cat(outs, dim=0), torch.cat(lses, dim=0)

                    dense_prefix_o, dense_prefix_lse = dense_attention_ref(
                        dense_prefix_k, dense_prefix_v
                    )
                    dense_suffix_o, dense_suffix_lse = dense_causal_suffix_ref(
                        dense_suffix_k, dense_suffix_v
                    )
                    dense_full_o, dense_full_lse = dense_causal_full_ref(
                        dense_full_k, dense_full_v
                    )
                    dense_prefix_lse_base2 = dense_prefix_lse * 1.4426950408889634
                    dense_suffix_lse_base2 = dense_suffix_lse * 1.4426950408889634
                    dense_full_lse_base2 = dense_full_lse * 1.4426950408889634
                    dense_section = summary["dense_reference"]
                    dense_section.update(
                        {
                            "q_compare": _trace_compare_tensors(dense_q_req, q_req_f),
                            "dense_o_flashinfer": _trace_numeric_tensor_stats(
                                dense_o_slice
                            ),
                            "dense_full_ref": _trace_numeric_tensor_stats(dense_full_o),
                            "dense_o_compare": _trace_compare_tensors(
                                dense_full_o.to(dense_o_slice.dtype), dense_o_slice
                            ),
                            "prefix_o_vs_o2": _trace_compare_tensors(
                                dense_prefix_o.to(o2_slice.dtype), o2_slice
                            ),
                            "prefix_lse_base2_vs_s2": (
                                _trace_compare_tensors(dense_prefix_lse_base2, s2_slice)
                                if isinstance(s2_slice, torch.Tensor)
                                else {"error": "missing s2 slice"}
                            ),
                            "suffix_o_vs_o1": (
                                _trace_compare_tensors(
                                    dense_suffix_o.to(o1_slice.dtype), o1_slice
                                )
                                if isinstance(s1_slice, torch.Tensor)
                                else {"error": "missing s1 slice"}
                            ),
                            "suffix_lse_base2_vs_s1": (
                                _trace_compare_tensors(dense_suffix_lse_base2, s1_slice)
                                if isinstance(s1_slice, torch.Tensor)
                                else {"error": "missing s1 slice"}
                            ),
                            "full_o_vs_merged": _trace_compare_tensors(
                                dense_full_o.to(merged_slice.dtype), merged_slice
                            ),
                            "full_lse_base2": _trace_numeric_tensor_stats(
                                dense_full_lse_base2
                            ),
                        }
                    )
                else:
                    summary["dense_reference"]["error"] = "dense reference too short"
                    summary["dense_reference"]["dense_shape"] = _tensor_trace_summary(
                        dense_q
                    )
            else:
                summary["dense_reference"] = {
                    "error": "missing dense no-prefix reference"
                }
        except Exception as exc:
            summary["error"] = repr(exc)

        logger.warning("FP4 KV prefix-reference trace %s", summary)

    def _get_paged_kv_cache_and_kwargs(self, layer: RadixAttention):
        kv_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        if not self.is_nvfp4_native:
            return kv_cache, {
                "k_scale": layer.k_scale_float,
                "v_scale": layer.v_scale_float,
            }

        kv_pool, local_layer_id = _nvfp4_inner_pool_and_layer_id(
            self.token_to_kv_pool, layer.layer_id
        )
        k_sf, v_sf = kv_pool.get_kv_scale_buffer(local_layer_id)
        k_sf = _shape_nvfp4_kv_scale_for_flashinfer(k_sf)
        v_sf = _shape_nvfp4_kv_scale_for_flashinfer(v_sf)
        k_global, v_global = kv_pool.get_kv_global_scale(local_layer_id)
        return kv_cache, {
            "kv_cache_sf": (k_sf, v_sf),
            "k_scale": k_global,
            "v_scale": v_global,
        }

    def _suffix_attention_inputs(
        self,
        layer: RadixAttention,
        k: torch.Tensor,
        v: torch.Tensor,
        paged_kv_kwargs,
    ):
        if (
            not self.is_nvfp4_native
            or self.is_fp8_k_nvfp4_v
            or k is None
            or v is None
        ):
            if k is None or v is None:
                return k, v, {}
            return (
                k.view(-1, layer.tp_k_head_num, layer.head_dim),
                v.view(-1, layer.tp_v_head_num, layer.head_dim),
                {},
            )

        from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

        k3 = k.view(-1, layer.tp_k_head_num, layer.head_dim).contiguous()
        v3 = v.view(-1, layer.tp_v_head_num, layer.head_dim).contiguous()
        k_fp4, k_sf, _ = NVFP4KVQuantizeUtil.quantize(k3, paged_kv_kwargs["k_scale"])
        v_fp4, v_sf, _ = NVFP4KVQuantizeUtil.quantize(v3, paged_kv_kwargs["v_scale"])
        return (
            k_fp4.view(torch.uint8),
            v_fp4.view(torch.uint8),
            {
                "kv_cache_sf": (k_sf, v_sf),
                "k_scale": paged_kv_kwargs["k_scale"],
                "v_scale": paged_kv_kwargs["v_scale"],
            },
        )

    def _should_vo_split(self, layer: RadixAttention) -> bool:
        return self.enable_vo_split and layer.head_dim == 512

    @staticmethod
    def _slice_last_dim_for_vo_split(x: Optional[torch.Tensor], pass_id: int):
        if x is None:
            return None
        half = x.shape[-1] // 2
        return x[..., pass_id * half : (pass_id + 1) * half]

    def _vo_split_paged_inputs(self, paged_kv_cache, paged_kv_kwargs, pass_id: int):
        k_cache, v_cache = paged_kv_cache
        split_kwargs = dict(paged_kv_kwargs)
        if split_kwargs.get("kv_cache_sf") is not None:
            k_sf, v_sf = split_kwargs["kv_cache_sf"]
            split_kwargs["kv_cache_sf"] = (
                k_sf,
                self._slice_last_dim_for_vo_split(v_sf, pass_id),
            )
        return (
            k_cache,
            self._slice_last_dim_for_vo_split(v_cache, pass_id),
        ), split_kwargs

    def _run_paged_native(
        self,
        wrapper,
        q,
        paged_kv_cache,
        *,
        causal,
        sm_scale,
        window_left,
        logits_soft_cap,
        return_lse,
        paged_kv_kwargs,
        trace_label=None,
        trace_layer=None,
        vo_split: bool = False,
    ):
        if vo_split:
            outs = []
            lse = None
            for pass_id in range(2):
                split_cache, split_kwargs = self._vo_split_paged_inputs(
                    paged_kv_cache, paged_kv_kwargs, pass_id
                )
                result = self._run_paged_native(
                    wrapper,
                    q,
                    split_cache,
                    causal=causal,
                    sm_scale=sm_scale,
                    window_left=window_left,
                    logits_soft_cap=logits_soft_cap,
                    return_lse=return_lse,
                    paged_kv_kwargs=split_kwargs,
                    trace_label=(
                        f"{trace_label}_vosplit{pass_id}"
                        if trace_label is not None
                        else None
                    ),
                    trace_layer=trace_layer,
                    vo_split=False,
                )
                if return_lse:
                    out_i, lse_i = result
                    if lse is None:
                        lse = lse_i
                    outs.append(out_i)
                else:
                    outs.append(result)
            out = torch.cat(outs, dim=-1)
            return (out, lse) if return_lse else out

        wrapper._causal = causal
        wrapper._pos_encoding_mode = "NONE"
        wrapper._use_fp16_qk_reduction = False
        wrapper._window_left = -1 if window_left is None else window_left
        wrapper._logits_soft_cap = 0.0 if logits_soft_cap is None else logits_soft_cap
        wrapper._sm_scale = sm_scale
        wrapper._rope_scale = None
        wrapper._rope_theta = None
        self._trace_gemma4_geometry_dispatch(
            label=trace_label,
            layer=trace_layer,
            paged_kv_cache=paged_kv_cache,
            paged_kv_kwargs=paged_kv_kwargs,
            wrapper=wrapper,
            vo_split=vo_split,
        )
        out = wrapper.run(
            q, paged_kv_cache, return_lse=return_lse, **paged_kv_kwargs
        )
        if self.is_nvfp4_native and _fp4_kv_module_trace_enabled():
            layer_id = getattr(trace_layer, "layer_id", None)
            key = (trace_label, int(layer_id) if layer_id is not None else None)
            if key not in self._nvfp4_module_trace_seen:
                self._nvfp4_module_trace_seen.add(key)
                extra_flags = os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", "")
                k_sf, v_sf = paged_kv_kwargs.get("kv_cache_sf", (None, None))
                logger.warning(
                    "FP4 KV FlashInfer module trace label=%s layer=%s "
                    "extra_cuda_flags=%r deswizzle_macro_active=%s "
                    "wrapper=%s kv_cache=%s k_sf=%s v_sf=%s k_scale=%s v_scale=%s",
                    trace_label,
                    layer_id,
                    extra_flags,
                    "FLASHINFER_PAGED_V_SF_DESWIZZLE" in extra_flags,
                    _flashinfer_wrapper_trace_summary(wrapper),
                    _tensor_trace_summary(paged_kv_cache),
                    _tensor_trace_summary(k_sf),
                    _tensor_trace_summary(v_sf),
                    _scale_trace_value(paged_kv_kwargs.get("k_scale")),
                    _scale_trace_value(paged_kv_kwargs.get("v_scale")),
                )
        return out

    def _plan_decode_as_prefill_vo_split(
        self,
        *,
        decode_wrapper,
        prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper,
        q: torch.Tensor,
        layer: RadixAttention,
    ) -> None:
        if self.skip_prefill:
            raise RuntimeError("VO-split decode-as-prefill requires prefill wrappers")

        wrapper_id = self._get_wrapper_idx(layer)
        geom = self.wrapper_geometries[wrapper_id]
        bs = q.shape[0]
        qo_indptr = self.qo_indptr[wrapper_id]
        qo_indptr[: bs + 1].copy_(
            torch.arange(bs + 1, dtype=torch.int32, device=qo_indptr.device)
        )
        qo_indptr = qo_indptr[: bs + 1]

        if not all(
            hasattr(decode_wrapper, name)
            for name in (
                "_paged_kv_indptr_buf",
                "_paged_kv_indices_buf",
                "_paged_kv_last_page_len_buf",
            )
        ):
            raise RuntimeError(
                "VO-split decode-as-prefill requires a planned decode wrapper"
            )

        updater = self.indices_updater_prefill
        plan_kwargs = {
            "head_dim_vo": geom.head_dim_vo,
            "q_data_type": updater.q_data_type,
            "kv_data_type": updater.kv_data_type,
            "custom_mask": None,
            "non_blocking": True,
            "fixed_split_size": self.prefill_split_tile_size,
        }
        if updater.k_data_type != updater.v_data_type:
            plan_kwargs.update(
                k_data_type=updater.k_data_type,
                v_data_type=updater.v_data_type,
            )

        prefill_wrapper.begin_forward(
            qo_indptr,
            decode_wrapper._paged_kv_indptr_buf[: bs + 1],
            decode_wrapper._paged_kv_indices_buf,
            decode_wrapper._paged_kv_last_page_len_buf[:bs],
            geom.num_qo_heads,
            geom.num_kv_heads,
            geom.head_dim,
            1,
            **plan_kwargs,
        )

    @staticmethod
    def _resolve_swa_kv_pool(model_runner: ModelRunner) -> Optional[BaseSWAKVPool]:
        """Return the SWA KV pool to translate against, or None for non-SWA models.

        EAGLE-like draft workers share the target allocator for token bookkeeping,
        but own a separate draft KV pool. Do not use the target allocator's SWA
        mapping for that draft pool. FROZEN_KV MTP is the exception: its draft
        path reads target KV directly, so it still needs the allocator pool when
        the active pool is not SWA.
        """
        active_pool = model_runner.token_to_kv_pool
        if isinstance(active_pool, BaseSWAKVPool):
            return active_pool

        if model_runner.is_draft_worker:
            if not model_runner.spec_algorithm.is_frozen_kv_mtp():
                return None

        kvcache = model_runner.token_to_kv_pool_allocator.get_kvcache()
        return kvcache if isinstance(kvcache, BaseSWAKVPool) else None

    def _process_multi_item_scoring(
        self, forward_batch: ForwardBatch
    ) -> MultiItemScoringParams:
        """Process multi-item scoring tensors for FlashInfer attention.

        This method handles sequences containing multiple "items" separated by delimiter tokens,
        where each item needs specific attention patterns that respect item boundaries.

        The method produces four key tensors for FlashInfer:
        - prefix_len_ptr: uint32 tensor with prefix length for each prompt in batch
        - token_pos_in_items_ptr: uint16 tensor with token positions starting from 0 at delimiters
        - token_pos_in_items_len: padding length for batch processing
        - max_item_len_ptr: uint16 tensor with max item length for each prompt

        Args:
            forward_batch: The forward batch containing input sequences and delimiter info

        Returns:
            MultiItemScoringParams: The processed multi-item scoring parameters

        Examples:
            Following FlashInfer definition: for 3 items of length 3, 2, 4 respectively:
            token_pos_in_items_ptr = [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0]

            Case 1: Single sequence
            Text: "What is the capital of France? <delim> London <delim> Paris <delim> Berlin <delim>"
            Tokens: [What, is, the, capital, of, France, ?, <delim>, London, <delim>, Paris, <delim>, Berlin, <delim>]
            Indices: [ 0,   1,  2,   3,      4,  5,     6,   7,     8,      9,     10,    11,    12,     13]
            - prefix_len_ptr: [7] (query length before first delimiter)
            - token_pos_in_items_ptr: [0, 1, 0, 1, 0, 1, 0] (delim=0, London=1, delim=0, Paris=1, delim=0, Berlin=1, delim=0)
            - token_pos_in_items_len: 7 (actual length)
            - max_item_len_ptr: [1] (max item length is 1 token - all options are single tokens)

            Case 2: Batch processing (batch_size=2)
            Sequence 1: 2 items of length 2, 1 → [0, 1, 2, 0, 1, 0] (6 elements)
            Sequence 2: 3 items of length 1, 3, 2 → [0, 1, 0, 1, 2, 3, 0, 1, 2, 0] (10 elements)
            After padding both to length 10:
            - token_pos_in_items_ptr: [0, 1, 2, 0, 1, 0, 0, 0, 0, 0,    0, 1, 0, 1, 2, 3, 0, 1, 2, 0]
            - token_pos_in_items_len: 10 (padded length for batch processing)
            - max_item_len_ptr: [2, 3] (max lengths per sequence)
        """

        if not self.enable_mis or forward_batch.forward_mode == ForwardMode.DECODE:
            return MultiItemScoringParams()

        precomputed_indices = forward_batch.multi_item_delimiter_indices
        if precomputed_indices is None:
            return MultiItemScoringParams()

        prefix_cache_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        prefix_len_ptr, token_pos_in_items_ptr = [], []
        token_pos_in_items_len = 0
        device = forward_batch.input_ids.device

        # If no extend_seq_lens, treat whole batch as one sequence
        if extend_seq_lens is None or len(extend_seq_lens) <= 1:
            extend_seq_lens = [forward_batch.input_ids.size(0)]

        seq_start = 0
        for i, seq_len in enumerate(extend_seq_lens):
            seq_end = seq_start + seq_len
            delimiter_indices_cpu = precomputed_indices[i]
            if len(delimiter_indices_cpu) == 0:
                seq_start = seq_end
                continue

            first_delim = delimiter_indices_cpu[0].item()  # CPU .item(), no GPU sync
            delimiter_indices = delimiter_indices_cpu.to(device, non_blocking=True)
            prefix_len = first_delim + (
                prefix_cache_lens[i] if prefix_cache_lens is not None else 0
            )
            prefix_len_ptr.append(prefix_len)

            # Compute relative positions within items using searchsorted (no GPU sync).
            #   suffix_range      = [0, 1, 2, 3, 4, ...]
            #   searchsorted      = bucket index for each position
            #   last_delim        = delimiter offset at start of current bucket
            #   pos_within_item   = suffix_range - last_delim
            suffix_len = seq_len - first_delim
            relative_positions = delimiter_indices - first_delim

            suffix_range = torch.arange(suffix_len, dtype=torch.int64, device=device)
            bucket_idx = torch.searchsorted(
                relative_positions, suffix_range, right=True
            )
            last_delim = relative_positions[torch.clamp(bucket_idx - 1, min=0)]
            pos_within_item = suffix_range - last_delim

            token_pos_in_items_ptr.append(pos_within_item.to(torch.uint16))

            forward_batch.positions[seq_start + first_delim : seq_end] = (
                prefix_len + pos_within_item - 1
            )

            seq_start = seq_end

        # Pad token_pos_in_items_ptr for batch processing
        if token_pos_in_items_ptr:
            token_pos_in_items_len = max(t.numel() for t in token_pos_in_items_ptr)
            token_pos_in_items_ptr = [
                torch.cat(
                    [
                        t,
                        torch.zeros(
                            token_pos_in_items_len - t.numel(),
                            dtype=torch.uint16,
                            device=device,
                        ),
                    ]
                )
                for t in token_pos_in_items_ptr
            ]

        if not prefix_len_ptr or not token_pos_in_items_ptr:
            return MultiItemScoringParams()

        return MultiItemScoringParams(
            prefix_len_ptr=torch.tensor(
                prefix_len_ptr, dtype=torch.uint32, device=device
            ),
            token_pos_in_items_ptr=torch.cat(token_pos_in_items_ptr, dim=0),
            token_pos_in_items_len=token_pos_in_items_len & 0xFFFFFFFF,
            max_item_len_ptr=torch.stack(
                [
                    t.to(torch.int32).max().to(torch.uint16)
                    for t in token_pos_in_items_ptr
                ],
                dim=0,
            ),
        )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        bs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        seq_lens_cpu = forward_batch.seq_lens_cpu
        seq_lens_sum = forward_batch.seq_lens_sum
        encoder_lens = forward_batch.encoder_lens
        forward_mode = forward_batch.forward_mode
        spec_info = forward_batch.spec_info

        if in_capture:
            num_tokens = forward_batch.positions.numel()
            self._prepare_cuda_graph_metadata(bs, num_tokens, forward_mode, spec_info)

        if forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                decode_wrappers=self.decode_cuda_graph_metadata[bs],
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
                fixed_split_size=None,
                disable_split_kv=self.disable_cuda_graph_kv_split,
            )
        elif forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
            )
        elif forward_mode.is_dllm_extend():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=seq_lens - self.dllm_config.block_size,
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                use_ragged=not self.use_paged,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=None,
            )
        elif forward_mode.is_draft_extend_v2():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.draft_extend_cuda_graph_metadata[bs],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
            )
        else:
            raise ValueError("Invalid forward mode")

        if in_capture and forward_mode.is_decode_or_idle():
            # fast_decode_plan needs _cached_module from the initial begin_forward
            # above, so install it only after that first plan has run.
            for w in self.decode_cuda_graph_metadata[bs]:
                w.begin_forward = partial(fast_decode_plan, w)

        if (
            in_capture
            and forward_mode.is_draft_extend_v2()
            and self.prefill_backend == "fa2"
            # Host-rebuilt layout only matches full attention (single wrapper);
            # SWA/cross-attn keep the plain plan().
            and self.dispatch_reason is None
        ):
            # Like decode: swap in fast_prefill_plan for replay, after the real
            # plan() above set up _cached_module (host metadata supplied per-replay
            # in call_begin_forward).
            for w in self.draft_extend_cuda_graph_metadata[bs]:
                w.begin_forward = partial(fast_prefill_plan, w)

        # Refill the SWA write-target buffer from the live out_cache_loc before
        # replay (bound onto the metadata at capture below).
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            assert self._swa_kv_pool is not None
            n = forward_batch.out_cache_loc.shape[0]
            self.cuda_graph_swa_out_cache_loc[n:].zero_()
            self.cuda_graph_swa_out_cache_loc[:n].copy_(
                self._swa_kv_pool.translate_loc_from_full_to_swa(
                    forward_batch.out_cache_loc
                )
            )
            if in_capture:
                self.forward_metadata.swa_out_cache_loc = (
                    self.cuda_graph_swa_out_cache_loc[:n]
                )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        swa_out_cache_loc = None
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            assert self._swa_kv_pool is not None
            swa_out_cache_loc = self._swa_kv_pool.translate_loc_from_full_to_swa(
                forward_batch.out_cache_loc
            )

        if forward_batch.forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                decode_wrappers=self.decode_wrappers,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
                fixed_split_size=self.decode_split_tile_size,
                disable_split_kv=False,
            )
            self.forward_metadata = DecodeMetadata(
                self.decode_wrappers, swa_out_cache_loc=swa_out_cache_loc
            )
        elif forward_batch.forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_wrappers_verify,
                use_ragged=False,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_verify,
                False,
                False,
                swa_out_cache_loc=swa_out_cache_loc,
            )
        else:
            prefix_lens = forward_batch.extend_prefix_lens

            # Disable ragged wrapper and ensure prefix handling for multimodal and multi-item scoring
            if self.is_multimodal or self.enable_mis:
                # use_ragged = False: Multi-item scoring requires the paged wrapper because:
                # 1. Ragged wrapper doesn't support the specialized multi-item parameters
                #    (prefix_len_ptr, token_pos_in_items_ptr, etc.)
                # 2. Paged wrapper provides better control over attention masking needed
                #    for respecting item boundaries in multi-item sequences
                # 3. Custom masking logic conflicts with ragged wrapper's assumptions
                use_ragged = False
                extend_no_prefix = False
            else:
                use_ragged = (
                    not self.enable_deterministic
                    and not is_in_tc_piecewise_cuda_graph()
                    and not self.use_paged
                )
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)

            # Process multi-item scoring in attention backend instead of ForwardBatch
            multi_item_params = MultiItemScoringParams()
            if self.enable_mis:
                # Use new backend-specific implementation
                multi_item_params = self._process_multi_item_scoring(forward_batch)

            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens,
                prefill_wrappers=self.prefill_wrappers_paged,
                use_ragged=use_ragged,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=None,
                fixed_split_size=self.prefill_split_tile_size,
                multi_item_params=multi_item_params,
                cross_attention_custom_mask=forward_batch.cross_attention_custom_mask,
                extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_paged,
                use_ragged,
                extend_no_prefix,
                multi_item_params,
                swa_out_cache_loc=swa_out_cache_loc,
            )
            self._trace_nvfp4_forward_batch(
                label="init_forward_metadata_extend",
                forward_batch=forward_batch,
                use_ragged=use_ragged,
                extend_no_prefix=extend_no_prefix,
            )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        if kv_indices_buf is None:
            cuda_graph_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len,),
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = kv_indices_buf

        self.cuda_graph_kv_indices = [cuda_graph_kv_indices] + [
            cuda_graph_kv_indices.clone() for _ in range(self.num_wrappers - 1)
        ]

        # SWA write-target buffer; refilled and bound onto forward_metadata in
        # init_forward_metadata_out_graph before each replay.
        self.cuda_graph_swa_out_cache_loc = (
            torch.zeros(max_num_tokens, dtype=torch.int64, device="cuda")
            if self.use_sliding_window_kv_pool
            else None
        )

        # Ensure tensors are properly allocated
        for i in range(self.num_wrappers):
            # Force allocation by performing a small operation
            if len(self.cuda_graph_kv_indices[i]) > 0:
                self.cuda_graph_kv_indices[i][0] = 0

        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device="cuda",
            )
            self.cuda_graph_qk_indptr = [x.clone() for x in self.kv_indptr]
            self.cuda_graph_qo_indptr = [x.clone() for x in self.kv_indptr]

    def _create_decode_wrappers(self, bs: int, num_tokens: int) -> list:
        return [
            BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                backend=self.decode_backend,
                use_cuda_graph=True,
                use_tensor_cores=self.decode_use_tensor_cores,
                paged_kv_indptr_buffer=self.kv_indptr[i][: num_tokens + 1],
                paged_kv_indices_buffer=self.cuda_graph_kv_indices[i],
                paged_kv_last_page_len_buffer=self.kv_last_page_len[:num_tokens],
            )
            for i in range(self.num_wrappers)
        ]

    def _create_prefill_wrappers(self, bs: int, use_custom_mask: bool = False) -> list:
        # FlashInfer's prefill wrapper decides mask mode based on whether
        # `custom_mask_buf` is initialized (not whether a custom mask is provided).
        # For cases like DFLASH draft (ENCODER_ONLY / non-causal) we do NOT use a
        # custom mask, so we must avoid initializing `custom_mask_buf`, otherwise
        # FlashInfer will treat the (zero) buffer as a real mask and block attention.
        wrappers = []
        for i in range(self.num_wrappers):
            extra = (
                {
                    "custom_mask_buf": self.cuda_graph_custom_mask,
                    "mask_indptr_buf": self.cuda_graph_qk_indptr[i][: bs + 1],
                }
                if use_custom_mask
                else {}
            )
            wrappers.append(
                BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_cuda_graph=True,
                    backend=self.prefill_backend,
                    qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                    paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                    paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                    paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                    **extra,
                )
            )
        return wrappers

    def _prepare_cuda_graph_metadata(
        self,
        bs: int,
        num_tokens: int,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ) -> None:
        if forward_mode.is_decode_or_idle():
            decode_wrappers = self._create_decode_wrappers(bs, num_tokens)
            self.decode_cuda_graph_metadata[bs] = decode_wrappers
            self.forward_metadata = DecodeMetadata(decode_wrappers)
        elif forward_mode.is_target_verify() or forward_mode.is_dllm_extend():
            use_custom_mask = (
                forward_mode.is_target_verify()
                and spec_info is not None
                and getattr(spec_info, "custom_mask", None) is not None
            )
            prefill_wrappers = self._create_prefill_wrappers(bs, use_custom_mask)
            self.prefill_cuda_graph_metadata[bs] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(
                prefill_wrappers, forward_mode.is_dllm_extend(), False
            )
        elif forward_mode.is_draft_extend_v2():
            # Draft-extend: causal paged prefill over the full sequence (no mask).
            prefill_wrappers = self._create_prefill_wrappers(bs, use_custom_mask=False)
            self.draft_extend_cuda_graph_metadata[bs] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(prefill_wrappers, False, False)
        else:
            raise ValueError(f"Invalid mode: {forward_mode=}")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    @debug_kernel_api
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        prefill_wrapper_paged = self.forward_metadata.prefill_wrappers[
            self._get_wrapper_idx(layer)
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        logits_soft_cap = layer.logit_cap

        q = q.contiguous()
        if not self.forward_metadata.use_ragged:
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                        k,
                        v,
                        layer.k_scale,
                        layer.v_scale,
                    )

            causal = (
                not layer.is_cross_attention
                and layer.attn_type != AttentionType.ENCODER_ONLY
            )
            paged_kv_cache, paged_kv_kwargs = self._get_paged_kv_cache_and_kwargs(
                layer
            )
            paged_window_left = (
                layer.sliding_window_size
                if not (
                    self.forward_metadata.multi_item_params
                    and self.forward_metadata.multi_item_params.is_enabled()
                )
                else -1
            )
            if self.is_nvfp4_native:
                self._trace_nvfp4_forward_batch(
                    label="forward_extend_paged",
                    forward_batch=forward_batch,
                    use_ragged=self.forward_metadata.use_ragged,
                    extend_no_prefix=self.forward_metadata.extend_no_prefix,
                )
                q_native = q.view(-1, layer.tp_q_head_num, layer.head_dim)
                self._trace_nvfp4_native_call(
                    label="extend_paged",
                    layer=layer,
                    q=q_native,
                    paged_kv_cache=paged_kv_cache,
                    paged_kv_kwargs=paged_kv_kwargs,
                )
                self._trace_nvfp4_page_pair(
                    label="extend_paged",
                    layer=layer,
                    paged_kv_cache=paged_kv_cache,
                    paged_kv_kwargs=paged_kv_kwargs,
                )
                o = self._run_paged_native(
                    prefill_wrapper_paged,
                    q_native,
                    paged_kv_cache,
                    causal=causal,
                    sm_scale=layer.scaling,
                    window_left=paged_window_left,
                    logits_soft_cap=logits_soft_cap,
                    return_lse=False,
                    paged_kv_kwargs=paged_kv_kwargs,
                    trace_label="extend_paged",
                    trace_layer=layer,
                    vo_split=self._should_vo_split(layer),
                )
                self._trace_nvfp4_dense_cache_state(
                    label="forward_extend_paged",
                    layer=layer,
                    forward_batch=forward_batch,
                    q=q_native,
                    o=o,
                    paged_kv_kwargs=paged_kv_kwargs,
                )
            else:
                o = self._run_paged_native(
                    prefill_wrapper_paged,
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    paged_kv_cache,
                    causal=causal,
                    sm_scale=layer.scaling,
                    window_left=paged_window_left,
                    logits_soft_cap=logits_soft_cap,
                    return_lse=False,
                    paged_kv_kwargs=paged_kv_kwargs,
                    trace_label="extend_paged",
                    trace_layer=layer,
                    vo_split=self._should_vo_split(layer),
                )
        else:
            # If `k`/`v` are not explicitly provided, fall back to the KV cache stored in
            # `self.token_to_kv_pool` for this layer. This enables attention over
            # previously cached context without re-materializing KV tensors (e.g., the
            # IQuestLoopCoder path uses token_to_kv_pool as the KV source).
            if k is None and v is None:
                if self.is_nvfp4_native:
                    raise RuntimeError(
                        "FlashInfer ragged fallback cannot read packed NVFP4 KV "
                        "as dense K/V. Use paged attention for native FP4 KV."
                    )
                k = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)[0]
                v = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)[1]
            causal = True
            if (
                layer.is_cross_attention
                or layer.attn_type == AttentionType.ENCODER_ONLY
            ):
                causal = False
            if not self.is_dllm_model and layer.attn_type == AttentionType.ENCODER_ONLY:
                save_kv_cache = False

            if self.forward_metadata.extend_no_prefix:
                self._trace_nvfp4_forward_batch(
                    label="forward_extend_ragged_no_prefix",
                    forward_batch=forward_batch,
                    use_ragged=self.forward_metadata.use_ragged,
                    extend_no_prefix=self.forward_metadata.extend_no_prefix,
                )
                # NOTE: FlashInfer currently has limitations with head_dim = 32 or other dimensions
                # The FlashInfer head_dim limitation itself is tracked here:
                # https://github.com/flashinfer-ai/flashinfer/issues/1048
                o = self.prefill_wrapper_ragged.forward(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    causal=causal,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )
                if self.is_nvfp4_native:
                    self._trace_nvfp4_dense_cache_state(
                        label="forward_extend_ragged_no_prefix",
                        layer=layer,
                        forward_batch=forward_batch,
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        o=o,
                    )

            else:
                swa_window_left = (
                    layer.sliding_window_size
                    if not (
                        self.forward_metadata.multi_item_params
                        and self.forward_metadata.multi_item_params.is_enabled()
                    )
                    else -1
                )
                paged_kv_cache, paged_kv_kwargs = self._get_paged_kv_cache_and_kwargs(
                    layer
                )
                (
                    suffix_k_for_attention,
                    suffix_v_for_attention,
                    suffix_attention_kwargs,
                ) = self._suffix_attention_inputs(
                    layer, k, v, paged_kv_kwargs
                )
                q_native = q.view(-1, layer.tp_q_head_num, layer.head_dim)
                if suffix_attention_kwargs:
                    self.prefill_wrapper_ragged._causal = causal
                    self.prefill_wrapper_ragged._pos_encoding_mode = "NONE"
                    self.prefill_wrapper_ragged._use_fp16_qk_reduction = False
                    self.prefill_wrapper_ragged._window_left = swa_window_left
                    self.prefill_wrapper_ragged._logits_soft_cap = logits_soft_cap
                    self.prefill_wrapper_ragged._sm_scale = layer.scaling
                    self.prefill_wrapper_ragged._rope_scale = None
                    self.prefill_wrapper_ragged._rope_theta = None
                    o1, s1 = self.prefill_wrapper_ragged.run_return_lse(
                        q_native,
                        suffix_k_for_attention,
                        suffix_v_for_attention,
                        **suffix_attention_kwargs,
                    )
                else:
                    o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                        q_native,
                        suffix_k_for_attention,
                        suffix_v_for_attention,
                        causal=causal,
                        sm_scale=layer.scaling,
                        window_left=swa_window_left,
                        logits_soft_cap=logits_soft_cap,
                    )
                if self.is_nvfp4_native:
                    self._trace_nvfp4_forward_batch(
                        label="forward_extend_merge_paged",
                        forward_batch=forward_batch,
                        use_ragged=self.forward_metadata.use_ragged,
                        extend_no_prefix=self.forward_metadata.extend_no_prefix,
                    )
                    self._trace_nvfp4_native_call(
                        label="extend_merge_paged",
                        layer=layer,
                        q=q_native,
                        paged_kv_cache=paged_kv_cache,
                        paged_kv_kwargs=paged_kv_kwargs,
                    )
                    self._trace_nvfp4_page_pair(
                        label="extend_merge_paged",
                        layer=layer,
                        paged_kv_cache=paged_kv_cache,
                        paged_kv_kwargs=paged_kv_kwargs,
                    )
                    o2, s2 = self._run_paged_native(
                        prefill_wrapper_paged,
                        q_native,
                        paged_kv_cache,
                        causal=False,
                        sm_scale=layer.scaling,
                        window_left=swa_window_left,
                        logits_soft_cap=logits_soft_cap,
                        return_lse=True,
                        paged_kv_kwargs=paged_kv_kwargs,
                        trace_label="extend_merge_paged",
                        trace_layer=layer,
                        vo_split=self._should_vo_split(layer),
                    )
                else:
                    o2, s2 = self._run_paged_native(
                        prefill_wrapper_paged,
                        q_native,
                        paged_kv_cache,
                        causal=False,
                        sm_scale=layer.scaling,
                        window_left=swa_window_left,
                        logits_soft_cap=logits_soft_cap,
                        return_lse=True,
                        paged_kv_kwargs=paged_kv_kwargs,
                        trace_label="extend_merge_paged",
                        trace_layer=layer,
                        vo_split=self._should_vo_split(layer),
                    )

                o, _ = _safe_merge_state(o1, s1, o2, s2)
                if self.is_nvfp4_native:
                    self._trace_nvfp4_dense_cache_state(
                        label="forward_extend_merge_paged",
                        layer=layer,
                        forward_batch=forward_batch,
                        q=q_native,
                        o1=o1,
                        s1=s1,
                        o2=o2,
                        s2=s2,
                        merged=o,
                        paged_kv_kwargs=paged_kv_kwargs,
                    )
                    self._trace_nvfp4_merge_state(
                        label="extend_merge_paged",
                        layer=layer,
                        paged_kv_cache=paged_kv_cache,
                        paged_kv_kwargs=paged_kv_kwargs,
                        o1=o1,
                        s1=s1,
                        o2=o2,
                        s2=s2,
                        merged=o,
                        swa_window_left=swa_window_left,
                    )
                    self._trace_nvfp4_prefix_reference(
                        label="extend_merge_paged",
                        layer=layer,
                        q=q_native,
                        suffix_k=k,
                        suffix_v=v,
                        paged_kv_cache=paged_kv_cache,
                        paged_kv_kwargs=paged_kv_kwargs,
                        o1=o1,
                        s1=s1,
                        o2=o2,
                        s2=s2,
                        merged=o,
                        swa_window_left=swa_window_left,
                        sm_scale=layer.scaling,
                        logits_soft_cap=logits_soft_cap,
                    )

            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )
            if (
                self.is_nvfp4_native
                and self.forward_metadata.use_ragged
                and self.forward_metadata.extend_no_prefix
                and k is not None
                and v is not None
            ):
                self._trace_nvfp4_dense_quant_attention_loss(
                    label="forward_extend_ragged_no_prefix",
                    layer=layer,
                    forward_batch=forward_batch,
                    q=q,
                    k=k,
                    v=v,
                    o=o,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    @debug_kernel_api
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        decode_wrapper = self.forward_metadata.decode_wrappers[
            self._get_wrapper_idx(layer)
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

        paged_kv_cache, paged_kv_kwargs = self._get_paged_kv_cache_and_kwargs(layer)
        if self._should_vo_split(layer):
            q_native = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            prefill_wrapper = self.prefill_wrappers_paged[self._get_wrapper_idx(layer)]
            self._plan_decode_as_prefill_vo_split(
                decode_wrapper=decode_wrapper,
                prefill_wrapper=prefill_wrapper,
                q=q_native,
                layer=layer,
            )
            o = self._run_paged_native(
                prefill_wrapper,
                q_native,
                paged_kv_cache,
                causal=False,
                sm_scale=layer.scaling,
                window_left=-1,
                logits_soft_cap=layer.logit_cap,
                return_lse=False,
                paged_kv_kwargs=paged_kv_kwargs,
                trace_label="decode_as_prefill",
                trace_layer=layer,
                vo_split=True,
            )
            return o.view(-1, layer.tp_q_head_num * layer.head_dim)

        if self.is_nvfp4_native:
            q_native = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            self._trace_nvfp4_native_call(
                label="decode",
                layer=layer,
                q=q_native,
                paged_kv_cache=paged_kv_cache,
                paged_kv_kwargs=paged_kv_kwargs,
            )
            o = self._run_paged_native(
                decode_wrapper,
                q_native,
                paged_kv_cache,
                causal=False,
                sm_scale=layer.scaling,
                window_left=-1,
                logits_soft_cap=layer.logit_cap,
                return_lse=False,
                paged_kv_kwargs=paged_kv_kwargs,
                trace_label="decode",
                trace_layer=layer,
            )
        else:
            self._trace_gemma4_geometry_dispatch(
                label="decode",
                layer=layer,
                paged_kv_cache=paged_kv_cache,
                paged_kv_kwargs=paged_kv_kwargs,
                wrapper=decode_wrapper,
                vo_split=False,
            )
            o = decode_wrapper.forward(
                q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                paged_kv_cache,
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                **paged_kv_kwargs,
            )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _get_wrapper_idx(self, layer: RadixAttention):
        if self.num_wrappers == 1:
            return 0

        if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            return layer.sliding_window_size == -1
        if self.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            return layer.is_cross_attention

        raise ValueError(f"Unknown dispatch reason: {self.dispatch_reason}")


class FlashInferIndicesUpdaterDecode:
    def __init__(self, model_runner: ModelRunner, attn_backend: FlashInferAttnBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.head_dim_vo = _flashinfer_vo_split_head_dim_vo(self.head_dim)
        self.data_type = model_runner.kv_cache_dtype
        self.kv_data_type = (
            torch.uint8 if attn_backend.is_nvfp4_native else self.data_type
        )
        self.k_data_type = (
            torch.float8_e4m3fn
            if attn_backend.is_fp8_k_nvfp4_v
            else self.kv_data_type
        )
        self.v_data_type = self.kv_data_type
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend
        self.wrapper_geometries = attn_backend.wrapper_geometries

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self._swa_kv_pool = attn_backend._swa_kv_pool

        # Dispatch the update function
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        decode_wrappers = decode_wrappers or self.decode_wrappers
        self.call_begin_forward(
            decode_wrappers[0],
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            self.kv_indptr[0],
            None,
            spec_info,
            seq_lens_cpu,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        assert self.sliding_window_size is not None
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # Sliding window attention
                paged_kernel_lens_tmp = torch.clamp(
                    seq_lens, max=self.sliding_window_size + 1
                )
                if seq_lens_cpu is not None:
                    seq_lens_cpu_tmp = torch.clamp(
                        seq_lens_cpu, max=self.sliding_window_size + 1
                    )
                    paged_kernel_lens_sum_tmp = seq_lens_cpu_tmp.sum().item()
                else:
                    paged_kernel_lens_sum_tmp = paged_kernel_lens_tmp.sum().item()
                kv_start_idx_tmp = seq_lens - paged_kernel_lens_tmp
            else:
                # Full attention
                paged_kernel_lens_tmp = seq_lens
                paged_kernel_lens_sum_tmp = seq_lens_sum
                seq_lens_cpu_tmp = seq_lens_cpu
                kv_start_idx_tmp = None

            use_sliding_window_kv_pool = (
                wrapper_id == 0 and self._swa_kv_pool is not None
            )

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens_tmp,
                paged_kernel_lens_sum_tmp,
                self.kv_indptr[wrapper_id],
                kv_start_idx_tmp,
                spec_info,
                seq_lens_cpu=seq_lens_cpu_tmp,
                use_sliding_window_kv_pool=use_sliding_window_kv_pool,
                fixed_split_size=fixed_split_size,
                disable_split_kv=disable_split_kv,
                wrapper_id=wrapper_id,
            )

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        # Cache encoder_lens on CPU to avoid GPU→CPU transfer per call
        encoder_lens_cpu = encoder_lens.cpu() if encoder_lens is not None else None
        for wrapper_id in range(2):
            if wrapper_id == 0:
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
                kv_lens_cpu = seq_lens_cpu
            else:
                # Cross-attention: attend to encoder tokens only
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                seq_lens_sum = encoder_lens.sum().item()
                kv_lens_cpu = encoder_lens_cpu

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                seq_lens_sum,
                self.kv_indptr[wrapper_id],
                kv_start_idx,
                spec_info,
                seq_lens_cpu=kv_lens_cpu,
                fixed_split_size=fixed_split_size,
                disable_split_kv=disable_split_kv,
                wrapper_id=wrapper_id,
            )

    def call_begin_forward(
        self,
        wrapper: BatchDecodeWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        kv_indptr: torch.Tensor,
        kv_start_idx: torch.Tensor,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        use_sliding_window_kv_pool: bool = False,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
        wrapper_id: int = 0,
    ):
        if spec_info is None or getattr(spec_info, "kv_indptr", None) is None:
            bs = len(req_pool_indices)
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]

            if wrapper.is_cuda_graph_enabled:
                # Directly write to the cuda graph input buffer
                kv_indices = wrapper._paged_kv_indices_buf
            else:
                kv_indices = torch.empty(
                    paged_kernel_lens_sum, dtype=torch.int32, device="cuda"
                )

            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )
        else:
            kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
            bs = kv_indptr.shape[0] - 1

        if use_sliding_window_kv_pool:
            assert self._swa_kv_pool is not None
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = (
                self._swa_kv_pool.translate_loc_from_full_to_swa(
                    kv_indices[:kv_last_index]
                )
            )

        global global_override_indptr_cpu
        locally_override = False
        if seq_lens_cpu is not None and global_override_indptr_cpu is None:
            locally_override = True
            global_override_indptr_cpu = torch.empty_like(kv_indptr, device="cpu")
            global_override_indptr_cpu[0] = 0
            global_override_indptr_cpu[1 : bs + 1] = torch.cumsum(seq_lens_cpu, dim=0)

        # Check if this specific wrapper's begin_forward has been replaced with fast_decode_plan
        # by checking if it's a partial function with fast_decode_plan as the func
        wrapper_uses_fast_decode_plan = (
            hasattr(wrapper.begin_forward, "func")
            and wrapper.begin_forward.func == fast_decode_plan
        )

        if wrapper_uses_fast_decode_plan:
            # When begin_forward is replaced with fast_decode_plan, pass global_override_indptr_cpu
            plan_kwargs = {
                "data_type": self.kv_data_type,
                "q_data_type": self.q_data_type,
                "non_blocking": True,
                "fixed_split_size": fixed_split_size,
                "disable_split_kv": (
                    disable_split_kv if disable_split_kv is not None else False
                ),
                "global_override_indptr_cpu": global_override_indptr_cpu,
            }
            if self.k_data_type != self.v_data_type:
                plan_kwargs.update(
                    k_data_type=self.k_data_type,
                    v_data_type=self.v_data_type,
                )
            geom = self.wrapper_geometries[wrapper_id]
            wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                self.kv_last_page_len[:bs],
                geom.num_qo_heads,
                geom.num_kv_heads,
                geom.head_dim,
                1,
                **plan_kwargs,
            )
        else:
            # When using original begin_forward, don't pass global_override_indptr_cpu
            plan_kwargs = {
                "data_type": self.kv_data_type,
                "q_data_type": self.q_data_type,
                "non_blocking": True,
                "fixed_split_size": fixed_split_size,
                "disable_split_kv": (
                    disable_split_kv if disable_split_kv is not None else False
                ),
            }
            if self.k_data_type != self.v_data_type:
                plan_kwargs.update(
                    k_data_type=self.k_data_type,
                    v_data_type=self.v_data_type,
                )
            geom = self.wrapper_geometries[wrapper_id]
            wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                self.kv_last_page_len[:bs],
                geom.num_qo_heads,
                geom.num_kv_heads,
                geom.head_dim,
                1,
                **plan_kwargs,
            )

        if locally_override:
            global_override_indptr_cpu = None


class FlashInferIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: FlashInferAttnBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.head_dim_vo = _flashinfer_vo_split_head_dim_vo(self.head_dim)
        self.data_type = model_runner.kv_cache_dtype
        self.kv_data_type = (
            torch.uint8 if attn_backend.is_nvfp4_native else self.data_type
        )
        self.k_data_type = (
            torch.float8_e4m3fn
            if attn_backend.is_fp8_k_nvfp4_v
            else self.kv_data_type
        )
        self.v_data_type = self.kv_data_type
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend
        self.wrapper_geometries = attn_backend.wrapper_geometries
        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self._swa_kv_pool = attn_backend._swa_kv_pool
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged

        # Dispatch the update function
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        extend_prefix_lens_cpu: Optional[List[int]] = None,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        extend_prefix_lens_cpu: Optional[List[int]] = None,
    ):
        if use_ragged:
            assert prefix_lens is not None
            paged_kernel_lens = prefix_lens
            if extend_prefix_lens_cpu is not None:
                # Host-known prefix lens; avoids a per-step D2H sync.
                paged_kernel_lens_sum = sum(extend_prefix_lens_cpu)
            else:
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()
        else:
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum

        self.call_begin_forward(
            self.prefill_wrapper_ragged,
            prefill_wrappers[0],
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            None,
            self.kv_indptr[0],
            self.qo_indptr[0],
            use_ragged,
            spec_info,
            fixed_split_size=fixed_split_size,
            multi_item_params=multi_item_params,
            seq_lens_cpu=seq_lens_cpu,
            wrapper_id=0,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        extend_prefix_lens_cpu: Optional[List[int]] = None,
    ):
        if prefix_lens is None:
            num_accept_tokens = getattr(spec_info, "num_accept_tokens", None)
            prefix_lens = (
                seq_lens
                if num_accept_tokens is None
                else seq_lens
                - num_accept_tokens[: seq_lens.shape[0]].to(
                    device=seq_lens.device, dtype=seq_lens.dtype
                )
            )
        sliding_window_size = self.sliding_window_size
        assert sliding_window_size is not None
        for wrapper_id in range(2):
            swa_paged_custom_mask = None
            if wrapper_id == 0:
                if use_ragged:
                    # K for extend tokens is written after the paged wrapper runs, so
                    # the paged wrapper sees prefix-only. Trim to the last `window` tokens
                    # (required for SWATokenToKVPoolAllocator; also keeps mask O(window)).
                    effective_start = torch.clamp(
                        prefix_lens - sliding_window_size, min=0
                    )
                    paged_kernel_lens = prefix_lens - effective_start
                    paged_kernel_lens_sum = paged_kernel_lens.sum().item()
                    kv_start_idx = effective_start
                    swa_paged_custom_mask = self._build_swa_prefix_custom_mask(
                        prefix_lens, seq_lens, effective_start
                    )
                else:
                    # window attention use paged only
                    paged_kernel_lens = torch.minimum(
                        seq_lens,
                        sliding_window_size + seq_lens - prefix_lens,
                    )
                    paged_kernel_lens_sum = paged_kernel_lens.sum().item()
                    kv_start_idx = seq_lens - paged_kernel_lens
            else:
                # full attention
                paged_kernel_lens = seq_lens
                paged_kernel_lens_sum = seq_lens_sum
                kv_start_idx = seq_lens - paged_kernel_lens
            use_sliding_window_kv_pool = (
                wrapper_id == 0 and self._swa_kv_pool is not None
            )

            self.call_begin_forward(
                self.prefill_wrapper_ragged,
                prefill_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
                spec_info,
                use_sliding_window_kv_pool=use_sliding_window_kv_pool,
                fixed_split_size=fixed_split_size,
                multi_item_params=multi_item_params,
                cross_attention_custom_mask=swa_paged_custom_mask,
                wrapper_id=wrapper_id,
            )

    def _build_swa_prefix_custom_mask(
        self,
        prefix_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_start_idx: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Custom SWA mask for the paged wrapper in the ragged merge_state EXTEND path.

        Paged KV covers absolute positions [kv_start_idx[i], prefix_lens[i]).
        Returns None when every key is in-window for every extend query.
        """
        window = self.sliding_window_size
        if window is None or window < 0:
            return None

        prefix_lens_cpu = prefix_lens.detach().cpu().tolist()
        extend_lens_cpu = (seq_lens - prefix_lens).detach().cpu().tolist()
        kv_start_cpu = kv_start_idx.detach().cpu().tolist()
        if all(p == 0 for p in prefix_lens_cpu):
            return None

        device = prefix_lens.device
        mask_parts: List[torch.Tensor] = []
        need_mask = False
        for prefix_len, extend_len, kv_start in zip(
            prefix_lens_cpu, extend_lens_cpu, kv_start_cpu
        ):
            paged_len = int(prefix_len - kv_start)  # = min(prefix_len, window)
            if paged_len == 0 or extend_len == 0:
                continue
            q_abs = torch.arange(extend_len, device=device).view(-1, 1) + prefix_len
            k_abs = torch.arange(paged_len, device=device).view(1, -1) + kv_start
            block = (k_abs >= (q_abs - window)).to(torch.uint8)
            if not bool(block.all()):
                need_mask = True
            mask_parts.append(block.view(-1))

        if not need_mask or not mask_parts:
            return None
        return torch.cat(mask_parts)

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        extend_prefix_lens_cpu: Optional[List[int]] = None,
    ):
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # normal attention
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
                paged_kernel_lens_sum = seq_lens_sum
            else:
                # cross attention
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()

            self.call_begin_forward(
                self.prefill_wrapper_ragged,
                prefill_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
                spec_info,
                fixed_split_size=fixed_split_size,
                multi_item_params=multi_item_params,
                cross_attention_custom_mask=(
                    cross_attention_custom_mask if wrapper_id == 1 else None
                ),
                wrapper_id=wrapper_id,
            )

    def call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: BatchPrefillWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: Optional[torch.Tensor],
        kv_start_idx: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
        spec_info: Optional[SpecInput],
        use_sliding_window_kv_pool: bool = False,
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        wrapper_id: int = 0,
    ):
        bs = len(seq_lens)
        geom = self.wrapper_geometries[wrapper_id]
        if spec_info is None:
            assert prefix_lens is not None
            assert len(seq_lens) == len(req_pool_indices)
            # Normal extend
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                paged_kernel_lens_sum + 256,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )
            qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]

            custom_mask = cross_attention_custom_mask
        else:
            assert isinstance(spec_info, SpecInput)
            if spec_info.spec_input_type == SpecInputType.DFLASH_VERIFY:
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        req_pool_indices,
                        paged_kernel_lens,
                        paged_kernel_lens_sum,
                        self.req_to_token,
                        kv_start_idx=kv_start_idx,
                    )
                )
            else:
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        req_pool_indices,
                        paged_kernel_lens,
                        paged_kernel_lens_sum,
                        self.req_to_token,
                    )
                )

        # extend part
        if use_ragged:
            has_cached_prefix = bool(torch.any(prefix_lens[:bs] > 0).item())
            ragged_kv_data_type = (
                self.kv_data_type
                if (
                    has_cached_prefix
                    and
                    self.attn_backend.is_nvfp4_native
                    and not self.attn_backend.is_fp8_k_nvfp4_v
                )
                else self.q_data_type
            )
            wrapper_ragged.begin_forward(
                qo_indptr,
                qo_indptr,
                geom.num_qo_heads,
                geom.num_kv_heads,
                geom.head_dim,
                q_data_type=self.q_data_type,
                kv_data_type=ragged_kv_data_type,
            )

        if use_sliding_window_kv_pool:
            assert self._swa_kv_pool is not None
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = (
                self._swa_kv_pool.translate_loc_from_full_to_swa(
                    kv_indices[:kv_last_index]
                )
            )

        self.attn_backend._capture_nvfp4_paged_plan(
            label="prefill",
            wrapper_id=wrapper_id,
            req_pool_indices=req_pool_indices,
            paged_kernel_lens=paged_kernel_lens,
            prefix_lens=prefix_lens,
            seq_lens=seq_lens,
            kv_start_idx=kv_start_idx,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            use_ragged=use_ragged,
        )

        # cached part
        # Conditionally set multi-item parameters
        if multi_item_params is not None and multi_item_params.is_enabled():
            # Multi-item scoring is active - use specialized parameters and disable generic custom_mask
            use_custom_mask = None
            prefix_len_ptr = multi_item_params.prefix_len_ptr
            token_pos_in_items_ptr = multi_item_params.token_pos_in_items_ptr
            token_pos_in_items_len = multi_item_params.token_pos_in_items_len
            max_item_len_ptr = multi_item_params.max_item_len_ptr
        else:
            # No multi-item scoring - use standard parameters
            use_custom_mask = custom_mask
            prefix_len_ptr = None
            token_pos_in_items_ptr = None
            token_pos_in_items_len = 0
            max_item_len_ptr = None

        paged_plan_kwargs = {
            "head_dim_vo": geom.head_dim_vo,
            "q_data_type": self.q_data_type,
            "kv_data_type": self.kv_data_type,
            "custom_mask": use_custom_mask,
            "non_blocking": True,
            "fixed_split_size": fixed_split_size,
            "prefix_len_ptr": prefix_len_ptr,
            "token_pos_in_items_ptr": token_pos_in_items_ptr,
            "token_pos_in_items_len": token_pos_in_items_len,
            "max_item_len_ptr": max_item_len_ptr,
        }
        if self.k_data_type != self.v_data_type:
            paged_plan_kwargs.update(
                k_data_type=self.k_data_type,
                v_data_type=self.v_data_type,
            )
        # fast_prefill_plan (installed at capture) is sync-free: it needs the
        # host-known qo/kv layout from the caller. Assert rather than silently
        # fall back to plan()'s blocking D2H on the replay hot-path.
        num_tokens_per_req = getattr(spec_info, "num_tokens_per_req", None)
        uses_fast_prefill = (
            hasattr(wrapper_paged.begin_forward, "func")
            and wrapper_paged.begin_forward.func is fast_prefill_plan
        )
        if uses_fast_prefill:
            assert (
                seq_lens_cpu is not None
            ), "fast_prefill_plan replay requires host-known seq_lens_cpu (got None)"
            assert (
                num_tokens_per_req is not None and num_tokens_per_req > 0
            ), f"fast_prefill_plan replay requires num_tokens_per_req > 0 (got {num_tokens_per_req})"
            seq_lens_cpu_i32 = seq_lens_cpu.to(torch.int32)
            qo_indptr_host = torch.arange(
                0,
                (bs + 1) * num_tokens_per_req,
                step=num_tokens_per_req,
                dtype=torch.int32,
                device="cpu",
            )
            kv_indptr_host = torch.zeros(bs + 1, dtype=torch.int32, device="cpu")
            kv_indptr_host[1:] = torch.cumsum(seq_lens_cpu_i32, dim=0)
            paged_plan_kwargs.update(
                qo_indptr_host=qo_indptr_host,
                kv_indptr_host=kv_indptr_host,
                kv_lens_host=seq_lens_cpu_i32,
                max_q_len=num_tokens_per_req,
                max_kv_len=int(seq_lens_cpu_i32.max()),
            )

        wrapper_paged.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            geom.num_qo_heads,
            geom.num_kv_heads,
            geom.head_dim,
            1,
            **paged_plan_kwargs,
        )


class FlashInferMultiStepDraftBackend:
    """
    Wrap multiple flashinfer attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices
        self.page_size = model_runner.page_size

        max_bs = _cuda_graph_capture_max_bs(
            model_runner.server_args, model_runner.req_to_token_pool.size * self.topk
        )
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        self.attn_backends: List[FlashInferAttnBackend] = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                FlashInferAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    kv_last_page_len_buf=self.kv_last_page_len,
                )
            )

        self.max_context_len = self.attn_backends[0].max_context_len

        # Cached variables for generate_draft_decode_kv_indices
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.req_to_token_pool = model_runner.req_to_token_pool

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: torch.Tensor,
        call_fn: Callable,
    ):
        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        required_kv_indices_len = draft_kv_indices_used_len(
            seq_lens_sum, self.topk, bs, self.speculative_num_steps
        )
        assert_buffer_fits(
            required_kv_indices_len,
            kv_indices_buffer.shape[1],
            "EAGLE draft kv_indices row (size max_bs * topk * max_context_len)",
            bs=bs,
            seq_lens_sum=seq_lens_sum,
        )

        self.generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps),
            next_power_of_2(bs),
            self.page_size,
        )

        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        # Copy the kv_indptr once to avoid multiple device-to-host copies in flashinfer's plan.
        indptr_cpu_whole = self.kv_indptr[:, : bs + 1].cpu()
        global global_override_indptr_cpu

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : draft_kv_indices_used_len(seq_lens_sum, self.topk, bs, i + 1)
            ]
            global_override_indptr_cpu = indptr_cpu_whole[i]
            call_fn(i, forward_batch)

        global_override_indptr_cpu = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices_width = draft_kv_indices_buffer_width(
            forward_batch.batch_size, self.topk, self.max_context_len
        )
        kv_indices = torch.empty(
            (self.speculative_num_steps, kv_indices_width),
            dtype=torch.int32,
            device="cuda",
        )

        def call_fn(i, forward_batch):
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # generate_draft_decode_kv_indices packs topk per-branch sequences per row,
        # so the row needs the topk factor -- same as the eager init_forward_metadata
        # (batch_size * topk * max_context_len). Dropping it overflows the buffer.
        kv_indices_width = draft_kv_indices_buffer_width(
            max_bs, self.topk, self.max_context_len
        )
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, kv_indices_width),
            dtype=torch.int32,
            device="cuda",
        )

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, kv_indices_buf=self.cuda_graph_kv_indices[i]
            )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        bs = forward_batch.batch_size

        def call_fn(i, fb):
            inner_fb = build_inner_fb_view(fb, bs=bs, forward_mode=ForwardMode.DECODE)
            self.attn_backends[i].init_forward_metadata_out_graph(
                inner_fb, in_capture=in_capture
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        for attn_backend in self.attn_backends:
            attn_backend.init_forward_metadata_in_graph(forward_batch)


def should_use_tensor_core(
    kv_cache_dtype: torch.dtype,
    num_attention_heads: int,
    num_kv_heads: int,
) -> bool:
    """
    Determine whether to use tensor cores for attention computation.

    Args:
        kv_cache_dtype: Data type of the KV cache
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of key/value heads

    Returns:
        bool: Whether to use tensor cores
    """
    # Try to use environment variable first
    env_override = os.environ.get("SGLANG_FLASHINFER_USE_TENSOR_CORE")
    if env_override is not None:
        return env_override.lower() == "true"

    # Try to use _grouped_size_compiled_for_decode_kernels if available
    # This is for flashinfer <=0.1.6. Otherwise, there is an accuracy bug
    try:
        from flashinfer.decode import _grouped_size_compiled_for_decode_kernels

        if not _grouped_size_compiled_for_decode_kernels(
            num_attention_heads,
            num_kv_heads,
        ):
            return True
        else:
            return False
    except (ImportError, AttributeError):
        pass

    # Calculate GQA group size
    gqa_group_size = num_attention_heads // num_kv_heads

    # For Flashinfer, a GQA group size of at least 4 is needed to efficiently
    # use Tensor Cores, as it fuses the head group with the token dimension in MMA.
    if kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return True
    elif kv_cache_dtype in (torch.float16, torch.half, torch.bfloat16):
        return gqa_group_size >= 4
    else:
        return False
