from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from einops import rearrange

from sglang.kernels.fused_op import BaseFusedOp
from sglang.kernels.ops.attention.fused_store_index_cache import (
    can_use_dsa_fused_store,
    fused_quantize_index_k_packed,
    fused_store_index_k_cache,
    store_packed_index_k_cache,
    store_rank_major_packed_index_k_cache,
)
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, is_fp8_fnuz
from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa import litetopk
from sglang.srt.layers.attention.dsa.dsa_indexer_metadata import BaseIndexerMetadata
from sglang.srt.layers.attention.dsa.dsa_npu_indexer import DSANPUIndexerMixin
from sglang.srt.layers.attention.dsa.dsa_prefill_cuda_graph import (
    GRAPH_WEIGHTS_PROJ_LORA_ERROR,
    _is_in_piecewise_or_breakable_cuda_graph,
    bcg_dsa_indexer_prefill_split,
    pcg_dsa_indexer_prefill_split,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
    DSAPagedMQALogitsBackend,
)
from sglang.srt.layers.attention.dsa.utils import (
    aiter_can_use_preshuffle_paged_mqa,
    is_dsa_enable_prefill_cp,
    is_dsa_prefill_cp_in_seq_split,
    is_graph_dsa_split_op_surface,
)
from sglang.srt.layers.layernorm import LayerNorm, RMSNorm
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import (
    get_device,
    get_exec,
    get_parallel,
    get_schedule,
    get_stream,
)
from sglang.srt.state_capturer.indexer_topk import (
    maybe_capture_indexer_topk,
)
from sglang.srt.utils import (
    add_prefix,
    ceil_align,
    get_bool_env_var,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_npu,
    is_xpu,
)
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)
_litetopk_exec_logged = False
# One process-wide index-K communication stream serializes every producer
# layer, so its packed transport scratch can be shared safely across layers.
_CP_PACKED_INDEX_K_LOCAL_BUFS: Dict[str, torch.Tensor] = {}
_CP_PACKED_INDEX_K_AG_EXEC_LOGGED = False

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_xpu = is_xpu()


def _litetopk_single_request_cpu_range(
    metadata: BaseIndexerMetadata,
    forward_batch: ForwardBatch,
    *,
    dsa_cp_enabled: bool,
) -> Optional[Tuple[int, int]]:
    """Return ``(window_start, common_end)`` without synchronizing CUDA.

    LiteTopK is currently restricted to one request.  For that layout the
    flattened K window starts at zero, and the earliest causal endpoint is
    known from scheduler CPU metadata.  CP-v2 interleave assigns query rows
    ``rank, rank + cp_size, ...``, so rank contributes the only offset.

    Unknown/uneven CP layouts return ``None`` and retain the conservative
    device-tensor fallback at the call site.
    """
    seq_lens_cpu = metadata.get_indexer_seq_len_cpu()
    if len(seq_lens_cpu) != 1:
        return None
    sequence_length = int(seq_lens_cpu[0].item())

    if dsa_cp_enabled:
        cp_metadata = forward_batch.attn_cp_metadata
        strategy = get_cp_strategy()
        if (
            cp_metadata is None
            or strategy is None
            or strategy.name != "interleave"
            or int(cp_metadata.bs) != 1
        ):
            return None
        global_query_length = int(cp_metadata.total_seq_lens)
        # The experimental Q=1024/1016 shapes are evenly divided.  Avoid
        # making assumptions about padded uneven tails until they are tested.
        if global_query_length <= 0 or global_query_length % int(strategy.cp_size) != 0:
            return None
        common_end = sequence_length - global_query_length + 1 + int(strategy.cp_rank)
    else:
        extend_lens_cpu = metadata.get_dsa_extend_len_cpu()
        if len(extend_lens_cpu) != 1:
            return None
        common_end = sequence_length - int(extend_lens_cpu[0]) + 1

    if common_end < 0 or common_end > sequence_length:
        return None
    return 0, common_end


def _litetopk_carry_recent_rows_hint(
    forward_batch: ForwardBatch,
    *,
    dsa_cp_enabled: bool,
) -> Optional[int]:
    """Keep LiteTopK's recent-query carry window in global row units."""
    if not dsa_cp_enabled or forward_batch.attn_cp_metadata is None:
        return None
    strategy = get_cp_strategy()
    if strategy is None or strategy.name != "interleave":
        return None
    return litetopk.carry_recent_rows_for_cp(int(strategy.cp_size))


def _transform_litetopk_single_request_indices(
    metadata: BaseIndexerMetadata,
    logical_indices: torch.Tensor,
) -> None:
    """Match the stock indexer's post-top-k ABI for one request.

    LiteTopK emits request-logical token IDs.  RAGGED single-request prefill
    consumes those IDs directly, while PAGED prefill consumes physical token
    locations from the page-size-1 table.  The stock path performs this mapping
    inside ``topk_transform``; the fused path must do the equivalent explicitly.
    """
    method = getattr(metadata, "topk_transform_method", None)
    if method == TopkTransformMethod.RAGGED:
        return
    if method != TopkTransformMethod.PAGED:
        raise RuntimeError(f"Unsupported LiteTopK top-k transform method: {method}")
    # Match DSATopKBackend.topk_transform: when fused transform is disabled,
    # the attention backend consumes logical IDs and performs PAGED mapping
    # later.  Mapping here in that mode would transform twice.
    if not envs.SGLANG_DSA_FUSE_TOPK.get() or bool(
        getattr(metadata, "force_unfused_topk", False)
    ):
        return

    page_table_1 = metadata.get_page_table_1()
    if page_table_1 is None or page_table_1.dim() != 2 or page_table_1.shape[0] != 1:
        raise RuntimeError(
            "LiteTopK PAGED prefill requires a one-request page-size-1 table"
        )
    if (
        logical_indices.device.type == "cuda"
        and logical_indices.dtype == torch.int32
        and logical_indices.dim() == 2
        and logical_indices.shape[1] == 2048
        and logical_indices.is_contiguous()
        and page_table_1.dtype == torch.int32
        and page_table_1.is_contiguous()
        and page_table_1.stride(1) == 1
        and page_table_1.device == logical_indices.device
    ):
        # One Triton launch replaces long/ge/clamp/take/where/copy and its
        # roughly 50 MiB of temporary traffic for Q=1024,K=2048.  A stride-0
        # row expansion presents the one-request table to the decode kernel;
        # each CTA reads its complete input row before writing that same row,
        # so result=input is a safe in-place transform.  The kernel preserves
        # every negative sentinel as -1.
        from sglang.kernels.ops.attention.dsa.transform_index import (
            transform_index_page_table_decode_fast,
        )

        transform_index_page_table_decode_fast(
            page_table_1.expand(logical_indices.shape[0], -1),
            logical_indices,
            result=logical_indices,
            page_size=1,
        )
        return
    logical_indices_i64 = logical_indices.long()
    valid = logical_indices_i64 >= 0
    # torch.take treats -1 as the final element.  Stock top-k preserves -1
    # padding/sentinels, so gather through a non-negative scratch index and
    # restore every invalid lane instead of turning underfill into a real slot.
    mapped = torch.take(page_table_1[0], logical_indices_i64.clamp_min(0))
    logical_indices.copy_(torch.where(valid, mapped, logical_indices))


if not _is_npu:
    from sglang.kernels.ops.attention.dsa import (
        aiter_paged_mqa_logits,
        cutedsl_paged_mqa_logits,
        deepgemm_paged_mqa_logits_native,
        deepgemm_paged_mqa_logits_split,
    )
else:
    aiter_paged_mqa_logits = None
    cutedsl_paged_mqa_logits = None
    deepgemm_paged_mqa_logits_native = None
    deepgemm_paged_mqa_logits_split = None

if _is_cuda:
    from sglang.kernels.ops.attention.dsa import pick_dsl_expand
else:
    pick_dsl_expand = None

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_fp8_fnuz = is_fp8_fnuz()
_is_gfx95_supported = is_gfx95_supported()
# Whether the aiter preshuffle paged-MQA path (page_size=64 + Preshuffle=True +
# KVBlockSize=64) can be used. Falls back to the legacy page_size=1 / KVBlockSize=1
# path when the gluon kernel is unavailable (Triton<3.5 and no AOT bundle).
_use_aiter_preshuffle = aiter_can_use_preshuffle_paged_mqa()
if _use_aiter and not _use_aiter_preshuffle:
    logger.warning(
        "ROCm DSA indexer: aiter preshuffle paged-MQA path is unavailable "
        "(needs Triton>=3.5.0 or AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1); "
        "falling back to legacy page_size=1 / KVBlockSize=1 path."
    )
if _is_cuda:
    try:
        import deep_gemm
    except ImportError as e:
        deep_gemm = e

if _use_aiter:
    from aiter.ops.cache import indexer_k_quant_and_cache

from sglang.srt.distributed import (
    get_attn_tp_group,
)
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.cp.base import get_cp_strategy
from sglang.srt.layers.cp.utils import is_cp_v2_active
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.model_executor.runner import get_is_capture_mode

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool


DUAL_STREAM_TOKEN_THRESHOLD = 1024 if _is_cuda else 0


if _is_cuda or _is_hip:
    # Plain-torch graph helpers: usable wherever the split-op surface is.
    from sglang.srt.layers.attention.dsa.dsa_prefill_cuda_graph import (
        logits_head_gate_graph,
        scale_head_gate_graph,
    )

if _is_cuda:
    from sglang.kernels.ops.attention.dsv4 import fused_q_indexer_rope_first_quant
    from sglang.kernels.ops.quantization.dsv32 import (
        fused_k_indexer_norm_rope,
        fused_k_indexer_norm_rope_store,
    )

    @register_custom_op(mutates_args=["topk_indices"])
    @register_split_op()
    def broadcast_indexer_topk_from_rank0_(topk_indices: torch.Tensor) -> None:
        _broadcast_indexer_topk_from_rank0_impl(topk_indices)


def _broadcast_indexer_topk_from_rank0_impl(topk_indices: torch.Tensor) -> None:
    group = get_attn_tp_group()
    if group.world_size == 1:
        return

    # PyNCCL is the faster path under capture, but it is not a precondition:
    # a split attn-TP group is built without one (parallel_state.py), and the
    # process-group broadcast captures and replays correctly.
    if (
        topk_indices.device.type == "cuda"
        and torch.cuda.is_current_stream_capturing()
        and group.pynccl_comm is not None
    ):
        with group.pynccl_comm.change_state(enable=True):
            group.pynccl_comm.broadcast(topk_indices, src=0)
    else:
        group.broadcast(topk_indices, src=0)


def _broadcast_indexer_topk_from_rank0(
    topk_indices: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    # Sync only the finalized indexer output. Internal topk_transform calls can
    # be chunked differently across ranks, which would make collectives diverge.
    if topk_indices is None or not envs.SGLANG_DSA_TOPK_BROADCAST.get():
        return topk_indices

    if is_in_tc_piecewise_cuda_graph():
        broadcast_indexer_topk_from_rank0_(topk_indices)
    else:
        _broadcast_indexer_topk_from_rank0_impl(topk_indices)
    return topk_indices


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    # from sgl_kernel import hadamard_transform
    if _is_hip:
        from fast_hadamard_transform import hadamard_transform
    elif _is_xpu:
        from sgl_kernel import hadamard_transform
    else:
        from sglang.kernels.ops.quantization.hadamard import hadamard_transform

    hidden_size = x.size(-1)
    assert (hidden_size & (hidden_size - 1)) == 0, (
        "Hidden size must be a power of 2 for Hadamard transform."
    )
    return hadamard_transform(x, scale=hidden_size**-0.5)


class Indexer(DSANPUIndexerMixin, BaseFusedOp):
    _MQA_LOGITS_BYTES_PER_ELEM = 4
    _MQA_LOGITS_STATIC_SKIP_ELEMS = 8_000_000
    _MQA_LOGITS_TOTAL_MEM_FRACTION = 0.3
    # aiter's fp8_mqa_logits only compiles below 2 GiB of logits (buffer_store).
    _MQA_LOGITS_MAX_BYTES_ROCM = 2**31 - 1
    _mqa_logits_budget_bytes: Dict[int, int] = {}

    @staticmethod
    def _mqa_logits_free_mem_fraction() -> float:
        return envs.SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION.get()

    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        layer_id: int,
        scale_fmt: Optional[str],
        block_size: int = 128,
        rope_scaling: Optional[Dict[str, Any]] = None,
        is_neox_style: bool = True,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        config=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.layer_id = layer_id
        self.use_dsa_indexer_fusion = (
            _is_cuda
            and not envs.SGLANG_DISABLE_DSA_INDEXER_FUSION.get()
            and not is_neox_style
        )
        self.alt_stream = alt_stream
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        if self.dsa_enable_prefill_cp:
            self.cp_size = get_parallel().attn_cp_size
        else:
            self.cp_size = None
        if litetopk.CP_GLOBAL_CARRY:
            if not self.dsa_enable_prefill_cp:
                raise RuntimeError(
                    "SGLANG_LITETOPK_CP_GLOBAL_CARRY requires DSA prefill CP"
                )
            cp_strategy = get_cp_strategy()
            if cp_strategy is None or cp_strategy.name != "interleave":
                raise RuntimeError(
                    "SGLANG_LITETOPK_CP_GLOBAL_CARRY currently requires "
                    "CP-v2 interleave"
                )
            litetopk.configure_cp_global_carry(get_parallel().attn_cp_group)
        # One process-wide stream preserves the order of every index-K CP
        # collective.  Leasing it during model construction keeps the CUDA
        # driver call outside graph capture.  The stream is used only by the
        # opt-in eager LiteTopK pipeline below.
        self.cp_indexer_comm_stream = (
            get_stream("dsa_cp_index_k_comm")
            if _is_cuda
            and self.dsa_enable_prefill_cp
            and litetopk.ENABLED
            and litetopk.CP_ASYNC_PREP
            else None
        )
        if _is_cuda:
            self.sm_count = deep_gemm.get_num_sms()
            self.half_device_sm_count = ceil_align(self.sm_count // 2, 8)
            pp_size = get_parallel().pp_size
            self.logits_with_pp_recv = pp_size > 1 and not get_pp_group().is_last_rank
        else:
            self.logits_with_pp_recv = False

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )

        if self.use_dsa_indexer_fusion:
            self.wk_weights_proj = ReplicatedLinear(
                self.hidden_size,
                self.head_dim + self.n_heads,
                bias=False,
                params_dtype=torch.bfloat16,
                prefix=add_prefix("wk_weights_proj", prefix),
            )
        else:
            self.wk = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wk", prefix),
            )
            self.weights_proj = ReplicatedLinear(
                self.hidden_size,
                self.n_heads,
                bias=False,
                params_dtype=torch.bfloat16,
                prefix=add_prefix("weights_proj", prefix),
            )
        if (
            config is not None
            and getattr(config, "index_k_norm_type", "layer") == "rms"
        ):
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.k_norm = LayerNorm(
                self.head_dim, dtype=torch.bfloat16 if _use_aiter else torch.float32
            )
        self.rotary_emb = get_rope_wrapper(
            rope_head_dim,
            rotary_dim=rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,  # type: ignore
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            device=get_device().device,
        )
        self.block_size = block_size
        self.scale_fmt = scale_fmt
        self.softmax_scale = self.head_dim**-0.5
        self.num_init_tokens = self.num_local_tokens = 0
        if config is not None:
            self.num_init_tokens = getattr(config, "index_init_tokens", 0)
            self.num_local_tokens = getattr(config, "index_local_tokens", 0)

        self.paged_mqa_logits_backend = DSAPagedMQALogitsBackend.resolve(
            get_exec().kernel.dsa_paged_mqa_logits_backend
        )

        # LiteTopk fused indexer top-k (SM100, prefill only, opt-in): scoring +
        # top-k in one pass, never materializing the [num_q, seq_len] logits.
        # GLM DSA shape only (H=32, D=128), and incompatible with forced
        # init/local token inclusion (no logits buffer to mask).
        if _is_cuda and envs.SGLANG_ENABLE_DSA_LITETOPK.get():
            from sglang.kernels.ops.attention.dsa import dsa_litetopk_is_supported

            self.use_dsa_litetopk = (
                self.n_heads == 32
                and self.head_dim == 128
                and self.num_init_tokens == 0
                and self.num_local_tokens == 0
                and dsa_litetopk_is_supported()
            )
        else:
            self.use_dsa_litetopk = False

    @contextlib.contextmanager
    def _with_real_sm_count(self):
        # When pipeline parallelism is enabled, each PP rank initiates a recv operation after the _pp_launch_batch
        # request to receive the PP proxy tensor or output from the previous stage, occupying one SM resource.
        # Model execution runs in parallel with the recv operation, so the SMs available to the indexer must be reduced
        # by 1. Currently, the last rank starts the send result + recv request only after waiting for execution results.
        if self.logits_with_pp_recv:
            pp_recv_sm_count = 1
            with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                self.sm_count - pp_recv_sm_count
            ):
                yield
        else:
            yield

    @property
    def _indexer_cos_sin_cache(self) -> torch.Tensor:
        return self.rotary_emb.cos_sin_cache

    def _weights_proj_bf16_in_fp32_out(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> torch.Tensor:
        # aiter (ROCm gfx95): extract the passthrough bf16 tensor from the
        # 3-tuple (fp8, scale, bf16) produced by fused_rms_fp8_group_quant,
        # avoiding an expensive FP8-to-bf16 dequantization.
        if _use_aiter and _is_gfx95_supported and isinstance(x, tuple) and len(x) == 3:
            x = x[2]
        if _is_cuda:
            return torch.mm(x, self.weights_proj.weight.t(), out_dtype=torch.float32)

        weights, _ = self.weights_proj(x)
        if _is_hip:
            # Return bf16; multiplying with q_scale promotes back to fp32.
            return weights
        return weights.float()

    @torch.compile(dynamic=True)
    def _project_and_scale_head_gates(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ):
        weights = self._weights_proj_bf16_in_fp32_out(x)
        weights = weights * self.n_heads**-0.5
        return weights

    @torch.compile(dynamic=True)
    def _get_logits_head_gate(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]], q_scale: torch.Tensor
    ):
        weights = self._weights_proj_bf16_in_fp32_out(x)
        weights = weights * self.n_heads**-0.5
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        return weights

    @torch.compile(dynamic=True)
    def _apply_q_scale_and_softmax_scale(
        self, weights: torch.Tensor, q_scale: torch.Tensor
    ):
        return weights.unsqueeze(-1) * q_scale * self.softmax_scale

    @torch.compile(dynamic=True)
    def _scale_head_gates(self, weights_raw: torch.Tensor, q_scale: torch.Tensor):
        weights = weights_raw * self.n_heads**-0.5
        return weights.unsqueeze(-1) * q_scale * self.softmax_scale

    def _fused_k_weights(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        kw, _ = self.wk_weights_proj(x)
        return kw.split([self.head_dim, self.n_heads], dim=-1)

    def _maybe_rotate(self, x: torch.Tensor) -> torch.Tensor:
        # Fusion drops the (logit-preserving) Hadamard rotation; without it the
        # index-K cache here matches the fused path that decode reads back.
        return x if self.use_dsa_indexer_fusion else rotate_activation(x)

    def _should_skip_logits_computation(self, forward_batch: ForwardBatch) -> bool:
        # When kv_len <= index_topk the top-k selects ALL valid positions, so the
        # indexer's logits GEMM + paged_mqa_logits + top-k are wasted work: a plain
        # topk_transform(dummy_logits) already yields the correct "select-all"
        # (physical page-slot) indices. Skipping the logits path is safe here.
        #
        # Prefill/extend: original fast path, all platforms.
        # Decode: new here, and ROCm-only for now (see the _is_hip gate below).
        # Under a captured decode cuda graph the chosen branch is frozen at
        # capture time and would replay incorrectly for kv_len > index_topk, so
        # the decode skip is not decided per-step during capture; it is driven by
        # which graph variant is being captured instead.
        fb = forward_batch

        # Prefill/extend: original per-step gate (host sync on seq_lens_cpu is fine).
        if fb.forward_mode.is_extend_without_speculative():
            if fb.seq_lens_cpu is None or fb.seq_lens_cpu.numel() == 0:
                return False
            return int(fb.seq_lens_cpu.max().item()) <= self.index_topk

        # Decode/idle.
        if fb.forward_mode.is_decode_or_idle():
            # Decode k-only skip (both the captured dual-graph "dense" variant
            # and the eager per-step skip below) is currently HIP-only. On CUDA
            # this common code keeps the original behavior (decode never skips
            # the indexer, i.e. always runs the full logits path) because the
            # decode k-only path has not been validated on CUDA yet. Mirrors the
            # is_hip() gate on dsa_dual_graph in decode_cuda_graph_runner, which
            # already prevents the CUDA capture path from setting a "dense"
            # variant.
            if not _is_hip:
                return False
            if get_is_capture_mode():
                # Under a captured decode cuda graph the taken branch is frozen at
                # capture time, so we must NOT branch on a runtime seq_len (also a
                # host sync would break capture). The chosen branch is instead
                # driven by which graph variant is being captured.
                #
                # The decode runner captures a "dense" (k-only) and a "sparse"
                # (full indexer) graph per bs bucket and dispatches on max_kv_len
                # at replay. The capture-variant signal tells us which one to
                # bake in.
                from sglang.srt.model_executor.runner_utils.capture_mode import (
                    get_capture_dsa_variant,
                )

                variant = get_capture_dsa_variant()
                if variant == "dense":
                    return True
                if variant == "sparse":
                    return False

                # No dual-variant capture signal: default to the correct-for-all
                # full-indexer (sparse) path.
                return False
            # Eager decode: safe to check per-step (host sync OK); correct for both
            # kv_len<=index_topk (k-only) and kv_len>index_topk (falls through).
            if fb.seq_lens_cpu is not None and fb.seq_lens_cpu.numel() > 0:
                max_kv_len = int(fb.seq_lens_cpu.max().item())
            elif fb.seq_lens is not None and fb.seq_lens.numel() > 0:
                max_kv_len = int(fb.seq_lens.max().item())
            else:
                return False
            return max_kv_len <= self.index_topk

        return False

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        positions: torch.Tensor,
        enable_dual_stream: bool,
        forward_batch: ForwardBatch,
    ):
        weights_raw = None
        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                self.half_device_sm_count
            ):
                query, _ = self.wq_b(q_lora)
                query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
                q_rope, _ = torch.split(
                    query,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )
            with torch.cuda.stream(self.alt_stream):
                # TODO we should also put DeepGEMM half SM here?
                if self.use_dsa_indexer_fusion:
                    key, weights_raw = self._fused_k_weights(x)
                else:
                    key, _ = self.wk(x)
                key = self.k_norm(key)

                k_rope, _ = torch.split(
                    key,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )

            current_stream.wait_stream(self.alt_stream)
        else:
            query, _ = self.wq_b(q_lora)
            query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
            q_rope, _ = torch.split(
                query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )
            if self.use_dsa_indexer_fusion:
                key, weights_raw = self._fused_k_weights(x)
            else:
                key, _ = self.wk(x)
            key = self.k_norm(key)
            k_rope, _ = torch.split(
                key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )

        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

        self._update_rope_guarded(query[..., : self.rope_head_dim], q_rope)
        self._update_rope_guarded(key[..., : self.rope_head_dim], k_rope)

        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            query = self._maybe_rotate(query)

            with torch.cuda.stream(self.alt_stream):
                key = self._maybe_rotate(key)
            current_stream.wait_stream(self.alt_stream)
        elif (
            self.alt_stream is not None
            and forward_batch.attn_cp_metadata is not None
            and self.dsa_enable_prefill_cp
        ):
            key = self._maybe_rotate(key)
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            query = self._maybe_rotate(query)

            # Gather the full key on alt_stream so the CP all-gather overlaps
            # with the query rotate above on the current stream.
            with torch.cuda.stream(self.alt_stream):
                if is_cp_v2_active(forward_batch):
                    key = get_cp_strategy().materialize_full_indexer_k_cache(
                        key, forward_batch
                    )
                else:
                    key = cp_all_gather_rerange_output(
                        key.contiguous(),
                        self.cp_size,
                        forward_batch,
                        torch.cuda.current_stream(),
                    )
            current_stream.wait_stream(self.alt_stream)
            return query, key, weights_raw
        else:
            query = self._maybe_rotate(query)
            key = self._maybe_rotate(key)

        # allgather+rerrange
        if is_cp_v2_active(forward_batch):
            key = get_cp_strategy().materialize_full_indexer_k_cache(key, forward_batch)
        elif forward_batch.attn_cp_metadata is not None and self.dsa_enable_prefill_cp:
            key = cp_all_gather_rerange_output(
                key.contiguous(),
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )
        return query, key, weights_raw

    def _get_k_bf16(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ):
        # Non-fusion path only; self.wk does not exist when fusion is on.
        key, _ = self.wk(x)
        key = self.k_norm(key)
        k_rope, _ = torch.split(
            key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

        # Rotary may update both inputs in place, so the K-only path must not
        # alias its dummy query with the key.
        if _is_cuda or _is_hip or _is_xpu:
            dummy_q_rope = torch.empty_like(k_rope)
        else:
            dummy_q_rope = k_rope
        _, k_rope = self.rotary_emb(positions, dummy_q_rope, k_rope)
        self._update_rope_guarded(key[..., : self.rope_head_dim], k_rope)
        key = rotate_activation(key)

        return key

    def _fused_k_prepare_and_store(
        self,
        key_raw: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        act_quant,
        out_cache_loc: Optional[torch.Tensor] = None,
    ) -> None:
        if out_cache_loc is None:
            out_cache_loc = forward_batch.out_cache_loc
        pool = get_token_to_kv_pool()
        page_size = pool.page_size
        if hasattr(pool, "invalidate_index_buffer_for_layer"):
            pool.invalidate_index_buffer_for_layer(layer_id)
        if hasattr(pool, "_is_layer_owned") and not pool._is_layer_owned(layer_id):
            return
        if (
            not _is_fp8_fnuz
            and out_cache_loc is not None
            and can_use_dsa_fused_store(torch.bfloat16, out_cache_loc.dtype, page_size)
        ):
            fused_k_indexer_norm_rope_store(
                key_raw,
                pool.get_index_k_with_scale_buffer(layer_id=layer_id),
                out_cache_loc,
                self.k_norm.weight,
                self.k_norm.bias,
                self.k_norm.variance_epsilon,
                self._indexer_cos_sin_cache,
                positions,
                page_size,
            )
            return

        # Fallback: separate K kernel + store kernel.
        key = fused_k_indexer_norm_rope(
            key_raw,
            self.k_norm.weight,
            self.k_norm.bias,
            self.k_norm.variance_epsilon,
            self._indexer_cos_sin_cache,
            positions,
        )
        self._store_index_k_cache(
            forward_batch=forward_batch,
            layer_id=layer_id,
            key=key,
            act_quant=act_quant,
            out_cache_loc=out_cache_loc,
        )

    def _fused_q_prepare_and_store(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        act_quant,
        *,
        num_tokens: Optional[int] = None,
        enable_dual_stream: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # num_tokens (graph split-op contract) slices q/k/positions/out_cache_loc
        # to the unpadded count; the returned q_fp8/weights are sliced to match.
        q_scale_gate = self.softmax_scale * self.n_heads**-0.5
        out_cache_loc = forward_batch.out_cache_loc
        if num_tokens is not None:
            positions = positions[:num_tokens]
            out_cache_loc = out_cache_loc[:num_tokens]

        if self.alt_stream is None or not enable_dual_stream:
            kw, _ = self.wk_weights_proj(x)
            key, weights_raw = kw.split([self.head_dim, self.n_heads], dim=-1)
            if num_tokens is not None:
                key = key[:num_tokens]
                weights_raw = weights_raw[:num_tokens]
            self._fused_k_prepare_and_store(
                key,
                positions,
                forward_batch,
                layer_id,
                act_quant,
                out_cache_loc=out_cache_loc,
            )
            q = self.wq_b(q_lora)[0].view(-1, self.n_heads, self.head_dim)
            if num_tokens is not None:
                q = q[:num_tokens]
            return fused_q_indexer_rope_first_quant(
                q.contiguous(),
                weights_raw,
                q_scale_gate,
                self._indexer_cos_sin_cache,
                positions,
            )

        # Two overlap stages: wq_b GEMM (alt) || wk_weights_proj GEMM (current),
        # then fused Q kernel (current) || fused K kernel + cache store (alt).
        # wait_stream calls are ordered by issue position so each side waits only
        # on the GEMMs it consumes, not on the other side's fused kernel.
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.alt_stream):
            q = self.wq_b(q_lora)[0].view(-1, self.n_heads, self.head_dim)
            if num_tokens is not None:
                q = q[:num_tokens]

        kw, _ = self.wk_weights_proj(x)
        key, weights_raw = kw.split([self.head_dim, self.n_heads], dim=-1)
        if num_tokens is not None:
            key = key[:num_tokens]
            weights_raw = weights_raw[:num_tokens]

        current_stream.wait_stream(self.alt_stream)
        self.alt_stream.wait_stream(current_stream)
        q_fp8, weights = fused_q_indexer_rope_first_quant(
            q.contiguous(),
            weights_raw,
            q_scale_gate,
            self._indexer_cos_sin_cache,
            positions,
        )
        with torch.cuda.stream(self.alt_stream):
            self._fused_k_prepare_and_store(
                key,
                positions,
                forward_batch,
                layer_id,
                act_quant,
                out_cache_loc=out_cache_loc,
            )

        current_stream.wait_stream(self.alt_stream)
        return q_fp8, weights

    def _fused_q_prepare_and_cp_store(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        act_quant,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare a fused CP indexer on three ordered CUDA streams.

        ``wq_b`` runs on the model's ordinary alternate stream while the
        fused K/head-gate projection runs on the caller's main stream.  Once
        the latter is ready, a dedicated process-wide communication stream
        performs local K norm/RoPE, the CP-v2 all-gather/reorder, and the
        global index-cache store.  The main stream prepares Q concurrently and
        joins the communication stream exactly once before LiteTopK reads the
        newly written tail.

        Every DSA layer leases the same communication stream, which keeps the
        collective order identical across ranks and avoids cross-layer NCCL
        reordering.
        """
        global _CP_PACKED_INDEX_K_AG_EXEC_LOGGED

        if self.alt_stream is None or self.cp_indexer_comm_stream is None:
            raise RuntimeError(
                "CP LiteTopK async preparation requires two side streams"
            )
        if not is_cp_v2_active(forward_batch):
            raise RuntimeError("CP LiteTopK async preparation requires CP-v2")

        current_stream = torch.cuda.current_stream()
        q_stream = self.alt_stream
        comm_stream = self.cp_indexer_comm_stream
        q_scale_gate = self.softmax_scale * self.n_heads**-0.5

        # Snapshot the caller's dependencies before either side stream starts.
        q_stream.wait_stream(current_stream)
        with torch.cuda.stream(q_stream):
            q = self.wq_b(q_lora)[0].view(-1, self.n_heads, self.head_dim)

        # Keep the larger fused K/head-gate projection on the main stream.
        kw, _ = self.wk_weights_proj(x)
        key_raw, weights_raw = kw.split([self.head_dim, self.n_heads], dim=-1)

        # The communication stream consumes the main-stream projection, then
        # owns all K work through the cache write.  record_stream protects the
        # projection allocation if Python releases ``kw`` before that work is
        # complete.
        comm_stream.wait_stream(current_stream)
        key_raw.record_stream(comm_stream)
        positions.record_stream(comm_stream)
        if forward_batch.out_cache_loc is not None:
            forward_batch.out_cache_loc.record_stream(comm_stream)
        with torch.cuda.stream(comm_stream):
            key = fused_k_indexer_norm_rope(
                key_raw,
                self.k_norm.weight,
                self.k_norm.bias,
                self.k_norm.variance_epsilon,
                self._indexer_cos_sin_cache,
                positions,
            )
            if litetopk.CP_PACKED_INDEX_K_AG:
                local_rows = int(key.shape[0])
                packed_key = str(key.device)
                packed_entry = _CP_PACKED_INDEX_K_LOCAL_BUFS.get(packed_key)
                if packed_entry is None or packed_entry.shape[0] < local_rows:
                    capacity = 1 << max(0, local_rows - 1).bit_length()
                    packed_entry = torch.empty(
                        (capacity, 132),
                        dtype=torch.uint8,
                        device=key.device,
                    )
                    _CP_PACKED_INDEX_K_LOCAL_BUFS[packed_key] = packed_entry
                packed_local = packed_entry[:local_rows]
                fused_quantize_index_k_packed(
                    key,
                    packed_local,
                    page_size=get_token_to_kv_pool().page_size,
                )
                cp_strategy = get_cp_strategy()
                rank_major = cp_strategy.try_materialize_rank_major_indexer_k_cache(
                    packed_local, forward_batch
                )
                if rank_major is not None:
                    packed_rank_major, packed_cp_size = rank_major
                    self._store_rank_major_packed_index_k_cache(
                        forward_batch=forward_batch,
                        layer_id=layer_id,
                        packed=packed_rank_major,
                        cp_size=packed_cp_size,
                    )
                    packed_route = "rank_major_direct_store"
                else:
                    packed_full = cp_strategy.materialize_full_indexer_k_cache(
                        packed_local, forward_batch
                    )
                    self._store_packed_index_k_cache(
                        forward_batch=forward_batch,
                        layer_id=layer_id,
                        packed=packed_full,
                    )
                    packed_route = "reordered_fallback"
                if not _CP_PACKED_INDEX_K_AG_EXEC_LOGGED:
                    logger.info(
                        "LITETOPK_CP_PACKED_INDEX_K_AG_EXECUTED "
                        "local_fp8_pack_before_CP_allgather %s",
                        packed_route,
                    )
                    _CP_PACKED_INDEX_K_AG_EXEC_LOGGED = True
            else:
                key = get_cp_strategy().materialize_full_indexer_k_cache(
                    key, forward_batch
                )
                self._store_index_k_cache(
                    forward_batch=forward_batch,
                    layer_id=layer_id,
                    key=key,
                    act_quant=act_quant,
                )

        # Q preparation only depends on q_stream and weights_raw.  It overlaps
        # the local-K transform, collective, reorder, quantization and store.
        current_stream.wait_stream(q_stream)
        q.record_stream(current_stream)
        q_fp8, weights = fused_q_indexer_rope_first_quant(
            q.contiguous(),
            weights_raw,
            q_scale_gate,
            self._indexer_cos_sin_cache,
            positions,
        )

        # LiteTopK immediately gathers the current index-cache tail, so this
        # is the semantic join point.  No earlier main-stream wait is needed.
        current_stream.wait_stream(comm_stream)
        return q_fp8, weights

    @staticmethod
    def _update_rope_guarded(dst: torch.Tensor, src: torch.Tensor) -> None:
        # On AMD with in-place RoPE kernels, self-aliasing can occur;
        # skip write-back when src/dst tensors point to a single memory.
        # data_ptr() is not comparable inside torch.compile, so skip the guard there.
        if not torch.compiler.is_compiling() and src.data_ptr() == dst.data_ptr():
            return
        dst.copy_(src)

    @staticmethod
    def _get_index_k_read_buffer(pool, layer_id: int) -> torch.Tensor:
        # Read path: prefer the owner-broadcast scratch buffer under DSA cache
        # layer split; fall back to the owned buffer for plain pools. Stores go
        # through get_index_k_with_scale_buffer() (owned buffer) instead.
        if hasattr(pool, "get_broadcastable_index_k_with_scale_buffer"):
            return pool.get_broadcastable_index_k_with_scale_buffer(layer_id)
        return pool.get_index_k_with_scale_buffer(layer_id=layer_id)

    @staticmethod
    def _pad_heads_for_deep_gemm(q_fp8, weights):
        """Pad q and weights to 32 heads when num_heads < 32,
        so that block_q = 128/num_heads doesn't exceed seq_len_alignment(4)."""
        num_heads = q_fp8.shape[1]
        if num_heads >= 32:
            return q_fp8, weights, num_heads
        target_heads = 32
        q_fp8 = torch.nn.functional.pad(q_fp8, (0, 0, 0, target_heads - num_heads))
        weights = torch.nn.functional.pad(weights, (0, target_heads - num_heads))
        return q_fp8, weights, num_heads

    def _mask_init_and_local_tokens(
        self,
        logits: torch.Tensor,
        lengths: torch.Tensor,
        row_starts: Optional[torch.Tensor] = None,
    ):
        if self.num_init_tokens == 0 and self.num_local_tokens == 0:
            return logits
        if row_starts is None:
            row_starts = lengths.new_zeros(lengths.shape[0])
        num_init_tokens = self.num_init_tokens
        num_local_tokens = self.num_local_tokens
        if num_init_tokens > 0:
            init_idxs = (
                torch.arange(
                    num_init_tokens, dtype=lengths.dtype, device=lengths.device
                )[None, :]
                + row_starts[:, None]
            )
            init_idxs.clamp_max_(logits.shape[-1] - 1)
            logits.scatter_(dim=1, index=init_idxs, value=float("inf"))
        if num_local_tokens > 0:
            local_idxs = (
                lengths[:, None]
                - 1
                + row_starts[:, None]
                - torch.arange(
                    num_local_tokens, dtype=lengths.dtype, device=lengths.device
                )[None, :]
            )
            local_idxs.clamp_min_(0)
            logits.scatter_(dim=1, index=local_idxs, value=float("inf"))
        return logits

    def _get_topk_paged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(get_token_to_kv_pool(), DSATokenToKVPool)

        page_size = get_token_to_kv_pool().page_size
        # NOTE(dark): blocksize = 64 is hardcoded in deep_gemm
        if _is_hip:
            if _use_aiter_preshuffle:
                assert page_size % 16 == 0, (
                    f"HIP preshuffle requires page_size to be a multiple of 16, got {page_size}"
                )
            else:
                assert page_size == 1, (
                    f"HIP legacy DSA path requires page_size == 1, got {page_size}"
                )
        else:
            assert page_size == 64, "only support page size 64"
        # NOTE(dark): this support extend/decode/decode+graph
        if _is_hip and not _use_aiter_preshuffle:
            block_tables = metadata.get_page_table_1()
        else:
            block_tables = metadata.get_page_table_64()

        max_seq_len = block_tables.shape[1] * page_size
        kv_cache_fp8 = self._get_index_k_read_buffer(get_token_to_kv_pool(), layer_id)

        blocksize = page_size
        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            seqlens_32 = metadata.get_seqlens_expanded()
        else:
            seqlens_32 = metadata.get_seqlens_int32()
        # Reuse pre-computed schedule metadata if available (from init_forward_metadata),
        # otherwise fall back to computing it here.
        schedule_metadata = getattr(metadata, "paged_mqa_schedule_metadata", None)
        assert len(q_fp8.shape) == 3
        # attn_tp_size > 1 or MAX_LEN padding mode can leave padding in the
        # hidden states; q_offset is the real (unpadded) q length.
        q_offset = sum(metadata.get_dsa_extend_len_cpu())

        B = metadata.get_seqlens_int32().shape[0]
        next_n = q_offset // B if B > 0 else 0
        use_cute_dsl = (
            self.paged_mqa_logits_backend.is_cutedsl()
            and not forward_batch.forward_mode.is_draft_extend_v2()
        )
        dsl_expand_factor, dsl_atom = 1, 1
        if (
            use_cute_dsl
            and forward_batch.forward_mode.is_target_verify()
            and next_n >= 2
        ):
            assert pick_dsl_expand is not None, "CuTe DSL paged MQA is CUDA-only."
            dsl_expand_factor, dsl_atom = pick_dsl_expand(
                next_n,
                batch_size=B,
                max_ctx=max_seq_len,
                num_sms=self.sm_count,
                kernel_atoms=(1, 2, 3, 4),
                num_heads=self.n_heads,
            )
        ctx_2d = getattr(metadata, "paged_mqa_ctx_lens_2d", None)
        use_dg_native = (
            not use_cute_dsl
            and _is_cuda
            and forward_batch.forward_mode.is_target_verify()
            and next_n >= 2
            and ctx_2d is not None
            and ctx_2d.shape == (B, next_n)
        )

        if use_dg_native:
            seqlens_32_2d = ctx_2d
        elif seqlens_32.dim() == 2:
            seqlens_32_2d = seqlens_32
        else:
            seqlens_32_2d = seqlens_32.unsqueeze(-1)
        if _is_cuda:
            if schedule_metadata is None:
                schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32_2d, blocksize, self.sm_count
                )

        assert len(kv_cache_fp8.shape) == 2
        block_kv = page_size
        num_heads_kv = 1
        head_dim_with_sf = 132
        kv_cache_fp8 = kv_cache_fp8.view(
            kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)

        if self.paged_mqa_logits_backend.is_aiter():
            logits = aiter_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                seqlens_32,
                block_tables,
                max_seq_len,
                preshuffle=_use_aiter_preshuffle,
                kv_block_size=block_kv,
            )
        elif use_cute_dsl:
            logits = cutedsl_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                metadata.get_seqlens_int32(),
                block_tables,
                schedule_metadata,
                max_seq_len,
                q_offset=q_offset,
                B=B,
                next_n=next_n,
                is_target_verify=forward_batch.forward_mode.is_target_verify(),
                dsl_expand_factor=dsl_expand_factor,
                dsl_atom=dsl_atom,
                blocksize=blocksize,
                sm_count=self.sm_count,
                get_paged_mqa_logits_metadata_fn=deep_gemm.get_paged_mqa_logits_metadata,
            )
        elif use_dg_native:
            logits = deepgemm_paged_mqa_logits_native(
                deep_gemm.fp8_paged_mqa_logits,
                q_fp8,
                kv_cache_fp8,
                weights,
                seqlens_32_2d,
                block_tables,
                schedule_metadata,
                max_seq_len,
                q_offset=q_offset,
                B=B,
                next_n=next_n,
            )
        else:
            logits = deepgemm_paged_mqa_logits_split(
                deep_gemm.fp8_paged_mqa_logits,
                q_fp8,
                kv_cache_fp8,
                weights,
                seqlens_32_2d,
                block_tables,
                schedule_metadata,
                max_seq_len,
                q_offset=q_offset,
            )

        # NOTE(dark): logits should be cleaned in topk_transform
        self._mask_init_and_local_tokens(logits, seqlens_32)
        topk_result = metadata.topk_transform(logits, self.index_topk)
        # Restore possible padding exist in the hidden states.
        if not _is_hip and q_offset < q_fp8.shape[0]:
            pad_len = q_fp8.shape[0] - q_offset
            padding = torch.full(
                (pad_len, topk_result.shape[1]),
                -1,
                dtype=topk_result.dtype,
                device=topk_result.device,
            )
            topk_result = torch.cat([topk_result, padding], dim=0)
        return topk_result

    def _get_mqa_logits_budget_bytes(self, device_index: int) -> int:
        free_mem_fraction = self._mqa_logits_free_mem_fraction()
        cached_budget = self._mqa_logits_budget_bytes.get(device_index)
        if cached_budget is not None:
            return cached_budget

        total_mem = torch.cuda.get_device_properties(device_index).total_memory

        total_mem_budget = int(total_mem * self._MQA_LOGITS_TOTAL_MEM_FRACTION)
        mem_fraction_static = get_schedule().mem_fraction_static
        if mem_fraction_static is None:
            static_budget = total_mem_budget
        else:
            static_free_mem = int(total_mem * max(0.0, 1.0 - mem_fraction_static))
            static_budget = min(
                int(static_free_mem * free_mem_fraction),
                total_mem_budget,
            )
        static_budget = max(1, static_budget)

        # Keep the static serving-memory guard during CUDA graph capture without
        # caching it. The first non-capture prefill path will cache the real
        # free-memory budget below.
        if get_is_capture_mode():
            return static_budget

        # Match the original free-memory guard: logits_bytes * 2 > free_mem.
        # torch.cuda.mem_get_info synchronizes the host, so cache the result,
        # capped by the workload-independent serving-memory headroom.
        free_mem, _ = torch.cuda.mem_get_info(device_index)
        budget_bytes = min(int(free_mem * free_mem_fraction), static_budget)

        budget_bytes = max(1, budget_bytes)
        self._mqa_logits_budget_bytes[device_index] = budget_bytes
        return budget_bytes

    def _should_chunk_mqa_logits(
        self, num_q: int, num_k: int, device_index: int
    ) -> Tuple[bool, int]:
        """
        Detect whether we need to chunk the MQA logits computation to avoid OOM,
        and on ROCm to stay under aiter's 2 GiB logits limit
        Return: (need_chunk, logits_budget_bytes)
        """
        # Quick static check for normal batches
        if num_q * num_k < self._MQA_LOGITS_STATIC_SKIP_ELEMS:
            return False, 0

        logits_bytes = num_q * num_k * self._MQA_LOGITS_BYTES_PER_ELEM
        logits_budget_bytes = self._get_mqa_logits_budget_bytes(device_index)
        if _is_hip:
            logits_budget_bytes = min(
                logits_budget_bytes, self._MQA_LOGITS_MAX_BYTES_ROCM
            )

        need_chunk = logits_bytes > logits_budget_bytes
        return need_chunk, logits_budget_bytes

    def _get_topk_ragged_litetopk(
        self,
        *,
        forward_batch: ForwardBatch,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        ks: torch.Tensor,
        ke: torch.Tensor,
        indexer_seq_lens_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """LiteTopk fused prefill top-k: scoring + gate + exact select in one
        KV pass; no [num_q, seq_len] logits buffer and no chunk loop. Output
        indices are gathered-KV absolute positions, -1 padded -- identical to
        the dense RAGGED transform contract."""
        from sglang.kernels.ops.attention.dsa import dsa_litetopk_indexer

        assert forward_batch.extend_seq_lens_cpu is not None
        kv_lens = indexer_seq_lens_cpu.tolist()
        req_bounds = []
        row0 = kv0 = 0
        for rows, kv_len in zip(forward_batch.extend_seq_lens_cpu, kv_lens):
            req_bounds.append((row0, row0 + rows, kv0, kv0 + int(kv_len)))
            row0 += rows
            kv0 += int(kv_len)
        return dsa_litetopk_indexer(
            q_fp8.contiguous(),
            k_fp8,
            k_scale,
            weights.contiguous(),
            ks,
            ke,
            self.index_topk,
            req_bounds=req_bounds,
        )

    def _get_topk_ragged(
        self,
        enable_dual_stream: bool,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
        topk_result: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(get_token_to_kv_pool(), DSATokenToKVPool)

        assert forward_batch.forward_mode.is_extend_without_speculative()

        page_size = get_token_to_kv_pool().page_size
        if _is_hip:
            if _use_aiter_preshuffle:
                assert page_size % 16 == 0, (
                    f"HIP preshuffle requires page_size to be a multiple of 16, got {page_size}"
                )
            else:
                assert page_size == 1, (
                    f"HIP legacy DSA path requires page_size == 1, got {page_size}"
                )
        else:
            assert page_size == 64, "only support page size 64"

        assert len(weights.shape) == 3
        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        weights = weights.squeeze(-1)

        if _is_hip and not _use_aiter_preshuffle:
            block_tables = metadata.get_page_table_1()
        else:
            block_tables = metadata.get_page_table_64()

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )

        batch_size = len(block_tables)
        global _litetopk_exec_logged

        token_nums, _, _ = q_fp8.shape
        device = q_fp8.device
        device_index = device.index
        assert device_index is not None, "q_fp8 must be on an indexed CUDA device"

        if topk_result is None:
            topk_result = torch.full(
                (token_nums, self.index_topk), -1, device=device, dtype=torch.int32
            )
        if batch_size == 0:
            return topk_result

        ks, ke = metadata.get_indexer_kvcache_range()

        indexer_seq_lens_cpu = metadata.get_indexer_seq_len_cpu()
        seq_len_sum = int(torch.sum(indexer_seq_lens_cpu).item())
        max_seq_len = int(torch.max(indexer_seq_lens_cpu).item())

        # Detect a new single-request epoch before the native dense prefix can
        # allocate [Q,S] logits.  The first indexer layer that observes the
        # sequence rollback retires all previous-request LiteTopK workspaces;
        # later layers see no old carry and make this a no-op.  The boundary
        # stock chunk still seeds a fresh carry normally before S reaches MIN.
        if (
            envs.SGLANG_LITETOPK.get()
            and forward_batch.attn_cp_metadata is not None
            and int(forward_batch.attn_cp_metadata.bs) == 1
            and not get_is_capture_mode()
        ):
            litetopk.retire_if_carry_extent_rollback(device, seq_len_sum)

        # LiteTopK performs its own pair-swapped paged gather.  Dispatch it
        # before the stock contiguous gather so the full index K is not copied
        # twice.  For supported single-request prefill shapes it scans the
        # suffix without materializing [Q, S] logits and maps winners back to
        # the exact ABI expected by the selected attention backend.
        # REQUIRED is checked before eligibility so an unsupported scheduler
        # shape cannot silently turn a qualification run into the stock path.
        litetopk_enabled = envs.SGLANG_LITETOPK.get()
        litetopk_required = envs.SGLANG_LITETOPK_REQUIRED.get()
        litetopk_capturing = (
            get_is_capture_mode() if litetopk_enabled or litetopk_required else False
        )
        cp_metadata = forward_batch.attn_cp_metadata
        litetopk_num_reqs = (
            int(cp_metadata.bs) if cp_metadata is not None else batch_size
        )
        cp_global_carry_layout_ok = True
        cp_global_carry_layout_reasons: Tuple[str, ...] = ()
        if litetopk.CP_GLOBAL_CARRY:
            cp_strategy = get_cp_strategy()
            global_query_length = (
                -1 if cp_metadata is None else int(cp_metadata.total_seq_lens)
            )
            cp_size = -1 if cp_strategy is None else int(cp_strategy.cp_size)
            cp_global_carry_layout_ok = (
                cp_metadata is not None
                and cp_strategy is not None
                and cp_strategy.name == "interleave"
                and cp_size > 1
                and global_query_length > 0
                and global_query_length % cp_size == 0
            )
            if not cp_global_carry_layout_ok:
                cp_global_carry_layout_reasons = (
                    "global carry requires an evenly divided CP-v2 interleave "
                    f"chunk, got global_Q={global_query_length}, cp_size={cp_size}",
                )
        required_path_args = {
            "enabled": litetopk_enabled,
            "required": litetopk_required,
            "use_fp4": False,
            "query_length": token_nums,
            "sequence_length": seq_len_sum,
            "num_reqs": litetopk_num_reqs,
            "capturing": litetopk_capturing,
            "route": "SGLang FP8 fused indexer",
            "extra_reasons": cp_global_carry_layout_reasons,
        }
        litetopk_production_ready = litetopk.enforce_required_path(**required_path_args)
        litetopk_common_eligible = (
            litetopk_enabled
            and litetopk_num_reqs == 1
            and not litetopk_capturing
            and self.num_init_tokens == 0
            and self.num_local_tokens == 0
            and self.index_topk == 2048
            and cp_global_carry_layout_ok
            and litetopk.supports_fused_query_len(token_nums, use_fp4=False)
        )
        litetopk_eligible = litetopk_common_eligible and litetopk_production_ready
        litetopk_carry_recent_rows = _litetopk_carry_recent_rows_hint(
            forward_batch,
            dsa_cp_enabled=self.dsa_enable_prefill_cp,
        )
        if litetopk_eligible:
            cpu_range = _litetopk_single_request_cpu_range(
                metadata,
                forward_batch,
                dsa_cp_enabled=self.dsa_enable_prefill_cp,
            )
            if cpu_range is None:
                # Conservative fallback for an unqualified CP layout.  The
                # normal CP8 interleave path never synchronizes here.
                common_end = int(ke.min().item())
                window_start = int(ks.min().item())
            else:
                window_start, common_end = cpu_range

            k_fp8 = torch.empty(
                (seq_len_sum, self.head_dim),
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            k_scale_u8 = torch.empty((seq_len_sum, 4), dtype=torch.uint8, device=device)
            cache_u8 = self._get_index_k_read_buffer(get_token_to_kv_pool(), layer_id)
            cache_u8 = cache_u8.view(cache_u8.shape[0], page_size, 132)
            plan = litetopk.prepare_permuted_gather(
                cache_u8,
                k_fp8,
                k_scale_u8,
                block_tables,
                sequence_length=seq_len_sum,
                query_length=token_nums,
                num_reqs=1,
                common_end=common_end,
                window_start=window_start,
                hot_key=layer_id,
            )
            if plan is not None:
                q_padded, w_padded, _ = self._pad_heads_for_deep_gemm(
                    q_fp8[:token_nums], weights[:token_nums]
                )
                k_scale = k_scale_u8.view(torch.float32).squeeze(-1)
                if litetopk.try_large_exact_once_chunk(
                    q_padded,
                    k_fp8,
                    k_scale,
                    w_padded,
                    ks,
                    ke,
                    topk_result[:token_nums],
                    self.index_topk,
                    permuted_plan=plan,
                    num_reqs=1,
                    ke_min_hint=common_end,
                    hot_key=layer_id,
                    ks_common_hint=window_start,
                    carry_extent_hint=seq_len_sum,
                    carry_recent_rows_hint=litetopk_carry_recent_rows,
                ):
                    _transform_litetopk_single_request_indices(
                        metadata, topk_result[:token_nums]
                    )
                    if not _litetopk_exec_logged:
                        print(
                            "LITETOPK_KERNEL_EXECUTED SGLang FP8 fused indexer "
                            f"dispatched device={q_fp8.device}",
                            flush=True,
                        )
                        _litetopk_exec_logged = True
                    return topk_result
        if litetopk_required and litetopk_production_ready:
            litetopk.enforce_required_path(
                **required_path_args,
                dispatched=False,
            )

        k_fp8, k_scale = get_token_to_kv_pool().get_index_k_scale_buffer(
            layer_id,
            metadata.get_indexer_seq_len(),
            block_tables,
            seq_len_sum,
            max_seq_len,
        )
        if _is_fp8_fnuz:
            k_fp8 = k_fp8.view(torch.float8_e4m3fnuz)
        else:
            k_fp8 = k_fp8.view(torch.float8_e4m3fn)

        k_scale = k_scale.view(torch.float32).squeeze(-1)
        kv_fp8 = (k_fp8, k_scale)

        # Check if we need to chunk to avoid OOM
        seq_lens_expanded = metadata.get_seqlens_expanded()
        token_to_batch_idx = metadata.get_token_to_batch_idx()
        q_offset = ks.shape[0]
        k_offset = k_fp8.shape[0]
        if self.use_dsa_litetopk:
            raw_topk_result = self._get_topk_ragged_litetopk(
                forward_batch=forward_batch,
                q_fp8=q_fp8[:q_offset],
                weights=weights[:q_offset],
                k_fp8=k_fp8,
                k_scale=k_scale,
                ks=ks,
                ke=ke,
                indexer_seq_lens_cpu=indexer_seq_lens_cpu,
            )
            topk_result[:q_offset] = raw_topk_result
            return topk_result

        need_chunk, logits_budget_bytes = self._should_chunk_mqa_logits(
            q_offset, k_offset, device_index
        )

        min_s = litetopk.production_min_s(False)
        # q_offset is rank-local under CP interleave, while k_offset and the
        # LiteTopK crossover live in the global sequence coordinate.  Use the
        # current global scheduler chunk as the next-chunk hint so the final
        # stock step seeds carry before a normal CP step crosses MIN_S.
        next_query_span = q_offset
        if self.dsa_enable_prefill_cp and cp_metadata is not None:
            next_query_span = int(cp_metadata.total_seq_lens)
        seed_litetopk_carry = (
            litetopk_common_eligible and k_offset < min_s <= k_offset + next_query_span
        )

        if not need_chunk:
            assert q_fp8[:q_offset].shape[0] != 0
            with self._with_real_sm_count():
                if _is_hip:
                    from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

                    kv, scale = kv_fp8
                    # Match the CUDA deep_gemm path (clean_logits=False): the topk
                    # transform masks invalid positions via ks/ke/lengths, so the
                    # -inf pre-fill of the [tokens x seq_len_kv] logits buffer is
                    # redundant and grows quadratically with context length.
                    logits = fp8_mqa_logits(
                        q_fp8[:q_offset],
                        kv,
                        scale,
                        weights[:q_offset],
                        ks,
                        ke,
                        clean_logits=False,
                    )
                else:
                    q_padded, w_padded, _ = self._pad_heads_for_deep_gemm(
                        q_fp8[:q_offset], weights[:q_offset]
                    )
                    logits = deep_gemm.fp8_mqa_logits(
                        q_padded,
                        kv_fp8,
                        w_padded,
                        ks,
                        ke,
                        clean_logits=False,
                    )
            assert logits.shape[0] == len(seq_lens_expanded)
            assert logits.shape[1] == k_offset

            self._mask_init_and_local_tokens(logits, seq_lens_expanded, ks)
            if seed_litetopk_carry:
                # The backend transform may turn logical winners into physical
                # cache slots (PAGED).  Carry planning must stay in logical
                # sequence coordinates, so select logical IDs first, map only
                # the attention-facing copy, and seed from the unmapped IDs.
                logical_topk_result = metadata.logical_topk(
                    logits, self.index_topk, ks=ks
                )
                topk_result[:q_offset].copy_(logical_topk_result)
                _transform_litetopk_single_request_indices(
                    metadata, topk_result[:q_offset]
                )
                litetopk.stash_carry(
                    layer_id,
                    logical_topk_result,
                    k_offset,
                    recent_rows_hint=litetopk_carry_recent_rows,
                )
            else:
                transformed_topk_result = metadata.topk_transform(
                    logits, self.index_topk, ks=ks
                )
                topk_result[:q_offset] = transformed_topk_result
            return topk_result

        bytes_per_row = k_offset * self._MQA_LOGITS_BYTES_PER_ELEM
        max_rows = max(1, int(logits_budget_bytes // max(bytes_per_row, 1)))
        max_rows = min(max_rows, q_offset)

        global_topk_offset = metadata.attn_metadata.topk_indices_offset
        cu_seqlens_q_full = None
        if global_topk_offset is None:
            cu_seqlens_q_full = torch.ones(q_offset, dtype=torch.int32, device=device)

        # The stock dense path may itself split Q to stay inside the logits
        # budget.  Preserve the same official->LiteTopK boundary contract by
        # retaining only the most recent logical winner rows across those
        # internal chunks, then publishing one carry after every output row is
        # finalized.  Attention-facing rows are transformed independently.
        logical_carry_tail = None
        carry_recent_budget = (
            litetopk.CARRY_RECENT_ROWS
            if litetopk_carry_recent_rows is None
            else int(litetopk_carry_recent_rows)
        )

        assert seq_lens_expanded.shape[0] == q_offset, (
            f"seq_lens_expanded length mismatch: {seq_lens_expanded.shape[0]} != {q_offset}"
        )
        if global_topk_offset is not None:
            assert global_topk_offset.shape[0] >= q_offset, (
                f"topk_indices_offset too short: {global_topk_offset.shape[0]} < {q_offset}"
            )

        start = 0
        while start < q_offset:
            end = min(start + max_rows, q_offset)

            with self._with_real_sm_count():
                if _is_hip:
                    from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

                    kv, scale = kv_fp8
                    # clean_logits=False: topk transform handles masking (see above)
                    logits_chunk = fp8_mqa_logits(
                        q_fp8[start:end],
                        kv,
                        scale,
                        weights[start:end],
                        ks[start:end],
                        ke[start:end],
                        clean_logits=False,
                    )
                else:
                    q_padded, w_padded, _ = self._pad_heads_for_deep_gemm(
                        q_fp8[start:end], weights[start:end]
                    )
                    logits_chunk = deep_gemm.fp8_mqa_logits(
                        q_padded,
                        kv_fp8,
                        w_padded,
                        ks[start:end],
                        ke[start:end],
                        clean_logits=False,
                    )

            lengths_chunk = seq_lens_expanded[start:end]
            self._mask_init_and_local_tokens(logits_chunk, lengths_chunk, ks[start:end])

            # RAGGED: use global offset; PAGED: construct local cu_seqlens_q per chunk
            if global_topk_offset is not None:
                # RAGGED path
                topk_offset_chunk = global_topk_offset[start:end]
                cu_seqlens_q_chunk = None
                batch_idx_chunk = None
            else:
                # PAGED path: treat each token as a length-1 sequence
                topk_offset_chunk = None
                cu_seqlens_q_chunk = cu_seqlens_q_full[start:end]
                batch_idx_chunk = token_to_batch_idx[start:end]

            if seed_litetopk_carry:
                logical_topk_chunk = metadata.logical_topk(
                    logits_chunk,
                    self.index_topk,
                    ks=ks[start:end],
                    lengths=lengths_chunk,
                )
                topk_result[start:end].copy_(logical_topk_chunk)
                _transform_litetopk_single_request_indices(
                    metadata, topk_result[start:end]
                )
                chunk_tail = logical_topk_chunk[-carry_recent_budget:]
                if (
                    logical_carry_tail is None
                    or chunk_tail.shape[0] >= carry_recent_budget
                ):
                    logical_carry_tail = chunk_tail
                else:
                    logical_carry_tail = torch.cat(
                        (logical_carry_tail, chunk_tail), dim=0
                    )[-carry_recent_budget:]
            else:
                raw_topk_chunk = metadata.topk_transform(
                    logits_chunk,
                    self.index_topk,
                    ks=ks[start:end],
                    cu_seqlens_q=cu_seqlens_q_chunk,
                    ke_offset=lengths_chunk,
                    batch_idx_list=batch_idx_chunk,
                    topk_indices_offset_override=topk_offset_chunk,
                )
                topk_result[start:end] = raw_topk_chunk
            start = end

        if seed_litetopk_carry:
            assert logical_carry_tail is not None
            litetopk.stash_carry(
                layer_id,
                logical_carry_tail,
                k_offset,
                recent_rows_hint=litetopk_carry_recent_rows,
            )

        return topk_result

    def _forward_cuda_k_only(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        act_quant,
        metadata: BaseIndexerMetadata,
        return_indices: bool = True,
        *,
        num_tokens: Optional[int] = None,
        topk_result: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        # Shared by the eager path and the graph DSA split-op dispatch. The two
        # keyword args carry the graph contract and default to the eager behavior:
        #   - num_tokens: slice key/out_cache_loc to the unpadded count (the graph
        #     runs at a static padded shape). None => full (eager) shape.
        #   - topk_result: pre-allocated padded buffer to fill in place (a downstream
        #     captured graph reads it at a fixed address). None => return a fresh,
        #     naturally-sized tensor.
        assert (
            forward_batch.forward_mode.is_extend_without_speculative()
            or forward_batch.forward_mode.is_decode_or_idle()
        )
        x_meta = x[0] if isinstance(x, tuple) else x

        # Fast path: only compute and store k cache, skip all q and weights ops.
        # num_tokens (graph contract) slices to the unpadded count.
        out_cache_loc = None
        if num_tokens is not None:
            assert num_tokens <= forward_batch.out_cache_loc.shape[0]
            out_cache_loc = forward_batch.out_cache_loc[:num_tokens]
        elif not forward_batch.out_cache_loc.is_contiguous():
            forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()

        # Write the same K representation the decode path reads back: fused
        # (no-Hadamard) when fusion is on, else the legacy Hadamard path.
        if self.use_dsa_indexer_fusion:
            key_raw, _ = self._fused_k_weights(x)
            if num_tokens is not None:
                assert num_tokens <= key_raw.shape[0]
                key_raw = key_raw[:num_tokens]
                positions = positions[:num_tokens]
            self._fused_k_prepare_and_store(
                key_raw,
                positions,
                forward_batch,
                layer_id,
                act_quant,
                out_cache_loc=out_cache_loc,
            )
        else:
            key = self._get_k_bf16(x, positions)
            if num_tokens is not None:
                assert num_tokens <= key.shape[0]
                key = key[:num_tokens]
            self._store_index_k_cache(
                forward_batch=forward_batch,
                layer_id=layer_id,
                key=key,
                act_quant=act_quant,
                out_cache_loc=out_cache_loc,
            )

        # MHA doesn't need topk_indices
        if not return_indices:
            return None

        # MLA: use dummy logits with topk kernel's fast path to generate indices
        # When length <= 2048, naive_topk_cuda directly generates [0,1,...,length-1,-1,...]
        seq_lens_expanded = metadata.get_seqlens_expanded()
        dummy_logits = torch.zeros(
            seq_lens_expanded.shape[0],
            self.index_topk,
            dtype=torch.float32,
            device=x_meta.device,
        )
        raw_topk_result = metadata.topk_transform(dummy_logits, self.index_topk)
        if topk_result is not None:
            # PCG/BCG: fill the valid prefix of the padded static buffer and
            # leave padded rows at the -1 sentinel.
            topk_result[: raw_topk_result.shape[0]] = raw_topk_result
            return None
        return raw_topk_result

    def _get_topk_ragged_with_cp(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
        kv_len: int,
        actual_seq_q: int,
        cp_index: List[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        assert not _is_in_piecewise_or_breakable_cuda_graph(), (
            "DSA context parallel (_get_topk_ragged_with_cp) not supported under "
            "piecewise/breakable CUDA graph"
        )
        if TYPE_CHECKING:
            assert isinstance(get_token_to_kv_pool(), DSATokenToKVPool)

        page_size = get_token_to_kv_pool().page_size
        assert page_size == 64, "only support page size 64"
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)
        k_fp8_list = []
        k_scale_list = []
        ks_list = []
        ke_offset_list = []
        offset = 0
        actual_seq_q_list = []
        batch_idx_list = []

        block_tables = metadata.get_page_table_64()

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        if cp_index is not None:
            # TODO Multi-batch support has accuracy issues
            for batch_idx, start_seq_position, end_seq_position in cp_index:
                pre_chunk_offset = (
                    forward_batch.seq_lens_cpu[batch_idx].item()
                    - forward_batch.extend_seq_lens_cpu[batch_idx]
                )
                start_seq_position += pre_chunk_offset
                end_seq_position += pre_chunk_offset
                if offset == 0 and batch_idx != 0:
                    offset += forward_batch.extend_seq_lens_cpu[batch_idx - 1]
                k_fp8 = get_token_to_kv_pool().get_index_k_continuous(
                    layer_id,
                    end_seq_position,
                    block_tables[batch_idx],
                )
                k_scale = get_token_to_kv_pool().get_index_k_scale_continuous(
                    layer_id,
                    end_seq_position,
                    block_tables[batch_idx],
                )

                extend_seq_len = end_seq_position - start_seq_position
                ks = torch.full(
                    (extend_seq_len,), offset, dtype=torch.int32, device="cuda"
                )
                k_fp8_list.append(k_fp8)
                k_scale_list.append(k_scale)
                ks_list.append(ks)
                ke_offset = torch.arange(
                    start_seq_position + 1,
                    end_seq_position + 1,
                    dtype=torch.int32,
                    device="cuda",
                )
                ke_offset_list.append(ke_offset)
                actual_seq_q = torch.tensor(
                    [extend_seq_len], dtype=torch.int32, device="cuda"
                )
                actual_seq_q_list.append(actual_seq_q)
                batch_idx_list.append(batch_idx)

            k_fp8 = torch.cat(k_fp8_list, dim=0).view(torch.float8_e4m3fn)
            k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
            kv_fp8 = (k_fp8, k_scale)
            ks = torch.cat(ks_list, dim=0)
            ke_offset = torch.cat(ke_offset_list, dim=0)
            ke = ks + ke_offset
            actual_seq_q = torch.cat(actual_seq_q_list, dim=0)
            with self._with_real_sm_count():
                q_padded, w_padded, _ = self._pad_heads_for_deep_gemm(q_fp8, weights)
                logits = deep_gemm.fp8_mqa_logits(
                    q_padded,
                    kv_fp8,
                    w_padded,
                    ks,
                    ke,
                    clean_logits=False,
                )
            topk_result = metadata.topk_transform(
                logits,
                self.index_topk,
                ks=ks,
                cu_seqlens_q=actual_seq_q,
                ke_offset=ke_offset,
                batch_idx_list=batch_idx_list,
            )
        else:
            kv_len = (
                forward_batch.seq_lens_cpu[0].item()
                - forward_batch.extend_seq_lens_cpu[0]
                + kv_len
            )
            k_fp8 = get_token_to_kv_pool().get_index_k_continuous(
                layer_id,
                kv_len,
                block_tables[0],
            )
            k_scale = get_token_to_kv_pool().get_index_k_scale_continuous(
                layer_id,
                kv_len,
                block_tables[0],
            )

            k_fp8 = k_fp8.view(torch.float8_e4m3fn)
            k_scale = k_scale.view(torch.float32).squeeze(-1)
            kv_fp8 = (k_fp8, k_scale)
            ks = torch.full((actual_seq_q,), offset, dtype=torch.int32, device="cuda")
            ke_offset = torch.arange(
                (kv_len - actual_seq_q) + 1,
                kv_len + 1,
                dtype=torch.int32,
                device="cuda",
            )
            ke = ks + ke_offset

            with self._with_real_sm_count():
                q_padded, w_padded, _ = self._pad_heads_for_deep_gemm(q_fp8, weights)
                logits = deep_gemm.fp8_mqa_logits(
                    q_padded,
                    kv_fp8,
                    w_padded,
                    ks,
                    ke,
                    clean_logits=False,
                )
            actual_seq_q = torch.tensor([actual_seq_q], dtype=torch.int32).to(
                device="cuda", non_blocking=True
            )
            topk_result = metadata.topk_transform(
                logits,
                self.index_topk,
                ks=ks,
                cu_seqlens_q=actual_seq_q,
                ke_offset=ke_offset,
            )

        return topk_result

    def _store_index_k_cache(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        key: torch.Tensor,
        *,
        act_quant=None,  # fallback only
        out_cache_loc: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Store DSA indexer K cache for current step.

        Preferred: fused_store_index_k_cache(key, cache, out_cache_loc, page_size)
        Fallback : act_quant(key) + token_to_kv_pool.set_index_k_scale_buffer(...)

        out_cache_loc will default to forward_batch.out_cache_loc if not provided.
        """

        if out_cache_loc is None:
            out_cache_loc = forward_batch.out_cache_loc

        pool = get_token_to_kv_pool()
        if hasattr(pool, "invalidate_index_buffer_for_layer"):
            pool.invalidate_index_buffer_for_layer(layer_id)
        if hasattr(pool, "_is_layer_owned") and not pool._is_layer_owned(layer_id):
            return

        if (
            _is_cuda
            and (not _is_fp8_fnuz)
            and can_use_dsa_fused_store(
                key.dtype,
                out_cache_loc.dtype,
                pool.page_size,
            )
        ):
            # NOTE: wrapper already normalizes shape/contiguity and asserts dtypes.
            buf = pool.get_index_k_with_scale_buffer(layer_id=layer_id)
            fused_store_index_k_cache(
                key,
                buf,
                out_cache_loc,
                pool.page_size,
            )
            return

        # Fast path: AITER fused quant + cache store
        # When _use_aiter_preshuffle is True we use the new MFMA 16x16 preshuffle
        # layout (page_size>=16). Otherwise we fall back to the legacy row-major
        # layout with page_size=1; the same kv_cache.view works for both cases
        # because page_size is 1 there.
        if _use_aiter:
            page_size = pool.page_size
            buf = pool.get_index_k_with_scale_buffer(layer_id=layer_id)
            kv_cache = buf.view(-1, page_size, 132).view(fp8_dtype)
            out_loc = forward_batch.out_cache_loc
            if not out_loc.is_contiguous():
                out_loc = out_loc.contiguous()
            indexer_k_quant_and_cache(
                key,
                kv_cache,
                out_loc,
                self.block_size,
                self.scale_fmt,
                preshuffle=_use_aiter_preshuffle,
            )
            return

        # Fallback: original path
        assert act_quant is not None
        k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)

        if not out_cache_loc.is_contiguous():
            out_cache_loc = out_cache_loc.contiguous()

        pool.set_index_k_scale_buffer(
            layer_id=layer_id,
            loc=out_cache_loc,
            index_k=k_fp8,
            index_k_scale=k_scale,
        )

    def _store_packed_index_k_cache(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        packed: torch.Tensor,
    ) -> None:
        """Store one gathered ``[global_Q, 132]`` fp8+scale transport slab."""
        pool = get_token_to_kv_pool()
        if hasattr(pool, "invalidate_index_buffer_for_layer"):
            pool.invalidate_index_buffer_for_layer(layer_id)
        if hasattr(pool, "_is_layer_owned") and not pool._is_layer_owned(layer_id):
            return
        out_cache_loc = forward_batch.out_cache_loc
        if out_cache_loc is None:
            raise RuntimeError("packed index-K CP store requires out_cache_loc")
        if packed.shape[0] != out_cache_loc.shape[0]:
            raise RuntimeError(
                "packed index-K CP gather/store row mismatch: "
                f"packed={packed.shape[0]}, locations={out_cache_loc.shape[0]}"
            )
        store_packed_index_k_cache(
            packed,
            pool.get_index_k_with_scale_buffer(layer_id=layer_id),
            out_cache_loc,
            pool.page_size,
        )

    def _store_rank_major_packed_index_k_cache(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        packed: torch.Tensor,
        cp_size: int,
    ) -> None:
        """Store equal-shard interleave AG rows without a global reorder."""
        pool = get_token_to_kv_pool()
        if hasattr(pool, "invalidate_index_buffer_for_layer"):
            pool.invalidate_index_buffer_for_layer(layer_id)
        if hasattr(pool, "_is_layer_owned") and not pool._is_layer_owned(layer_id):
            return
        out_cache_loc = forward_batch.out_cache_loc
        if out_cache_loc is None:
            raise RuntimeError("rank-major packed index-K store requires out_cache_loc")
        if packed.shape[0] != out_cache_loc.shape[0]:
            raise RuntimeError(
                "rank-major packed index-K gather/store row mismatch: "
                f"packed={packed.shape[0]}, locations={out_cache_loc.shape[0]}"
            )
        store_rank_major_packed_index_k_cache(
            packed,
            pool.get_index_k_with_scale_buffer(layer_id=layer_id),
            out_cache_loc,
            int(cp_size),
            pool.page_size,
        )

    def forward_native(self, *args, **kwargs):
        # The indexer has no pure-torch reference path; it only runs on
        # platforms with a dedicated forward below.
        raise NotImplementedError("Indexer has no native (pure-torch) path")

    def forward_xpu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        return self.forward_cuda(
            x, q_lora, positions, forward_batch, layer_id, return_indices
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        if _is_hip:
            from sglang.kernels.ops.attention.dsa.tilelang_kernel import act_quant
        elif not _is_npu:
            from sglang.kernels.ops.attention.dsa.triton_kernel import act_quant

        if TYPE_CHECKING:
            assert isinstance(get_token_to_kv_pool(), DSATokenToKVPool)

        # When upstream uses fused FP8 RMSNorm+quant, activations may be passed as
        # a tuple like (x_fp8, x_scale[, y]). Use `x_meta` for shape/device queries.
        x_meta = x[0] if isinstance(x, tuple) else x

        in_piecewise_or_breakable_cuda_graph = (
            _is_in_piecewise_or_breakable_cuda_graph()
        )

        # In piecewise/breakable CUDA graph mode, metadata is fetched inside
        # custom ops via get_tc_piecewise_forward_context() to prevent Dynamo
        # from guarding on forward_metadata identity, which changes each replay
        # when init_forward_metadata creates a new ForwardMetadata object.
        if not in_piecewise_or_breakable_cuda_graph:
            metadata = get_attn_backend().get_indexer_metadata(layer_id, forward_batch)
            if metadata is None:
                return None
        else:
            metadata = None

        enable_dual_stream = (
            self.alt_stream is not None
            and get_is_capture_mode()
            and q_lora.shape[0] > 0
            and q_lora.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
        )

        # Determine if should skip topk based on sequence length
        # We can only skip the logits computation if cuda graph is not involved
        skip_logits_computation = False
        if not in_piecewise_or_breakable_cuda_graph:
            skip_logits_computation = self._should_skip_logits_computation(
                forward_batch
            )

        # Optimization: fast path when skipping topk computation
        if skip_logits_computation and (not self.dsa_enable_prefill_cp):
            topk_result = self._forward_cuda_k_only(
                x,
                positions,
                forward_batch,
                layer_id,
                act_quant,
                metadata,
                return_indices,
            )
            topk_result = _broadcast_indexer_topk_from_rank0(topk_result)
            return maybe_capture_indexer_topk(layer_id, topk_result)

        # When weights_proj is LoRA-wrapped, use an eager module call so the
        # wrapper owns base+delta and no LoRA kernel runs under torch.compile.
        # Fusion folds weights_proj into wk_weights_proj, so weights_proj is
        # absent then; short-circuit before touching it.
        weights_proj_lora = not self.use_dsa_indexer_fusion and getattr(
            self.weights_proj, "set_lora", False
        )

        litetopk_production_ready_cpu = (
            metadata is not None
            and forward_batch.attn_cp_metadata is not None
            and int(forward_batch.attn_cp_metadata.bs) == 1
            and litetopk.single_request_production_ready_cpu(
                forward_batch.seq_lens_cpu, use_fp4=False
            )
        )

        cp_litetopk_async_prep = (
            litetopk.ENABLED
            and litetopk.CP_ASYNC_PREP
            and litetopk_production_ready_cpu
            and self.use_dsa_indexer_fusion
            and not in_piecewise_or_breakable_cuda_graph
            and self.dsa_enable_prefill_cp
            and forward_batch.attn_cp_metadata is not None
            and is_cp_v2_active(forward_batch)
            and forward_batch.forward_mode.is_extend_without_speculative()
            and self.alt_stream is not None
            and self.cp_indexer_comm_stream is not None
            and litetopk.supports_fused_query_len(int(q_lora.shape[0]), use_fp4=False)
        )

        if cp_litetopk_async_prep:
            q_fp8, weights = self._fused_q_prepare_and_cp_store(
                x,
                q_lora,
                positions,
                forward_batch,
                layer_id,
                act_quant,
            )
        elif (
            self.use_dsa_indexer_fusion
            and not in_piecewise_or_breakable_cuda_graph
            and forward_batch.attn_cp_metadata is None
        ):
            q_fp8, weights = self._fused_q_prepare_and_store(
                x, q_lora, positions, forward_batch, layer_id, act_quant
            )
        elif (
            is_graph_dsa_split_op_surface(forward_batch)
            and not self.dsa_enable_prefill_cp
        ):
            # Default path for non-CP prefill under PCG/BCG: run the whole indexer
            # (q/k proj, head gate, k-cache store, topk) as a single eager split op
            # instead of capturing it piecemeal in the graph. The split op is
            # fusion-aware, so this also covers the fused path here.
            if weights_proj_lora:
                raise RuntimeError(GRAPH_WEIGHTS_PROJ_LORA_ERROR)
            if return_indices:
                topk_result = torch.full(
                    (x.shape[0], self.index_topk),
                    -1,
                    device=x.device,
                    dtype=torch.int32,
                )
            else:
                topk_result = torch.empty(
                    (0, self.index_topk), device=x.device, dtype=torch.int32
                )
            graph_dispatch_fn = (
                bcg_dsa_indexer_prefill_split
                if is_in_breakable_cuda_graph()
                else pcg_dsa_indexer_prefill_split
            )
            graph_dispatch_fn(
                layer_id=layer_id,
                x=x,
                q_lora=q_lora,
                positions=positions,
                topk_result=topk_result,
            )
            result = _broadcast_indexer_topk_from_rank0(
                topk_result if return_indices else None
            )
            return maybe_capture_indexer_topk(layer_id, result)

        elif enable_dual_stream and forward_batch.forward_mode.is_decode_or_idle():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            if not self.use_dsa_indexer_fusion:
                if weights_proj_lora:
                    weights = self.weights_proj(x)[0].float() * self.n_heads**-0.5
                else:
                    weights = self._project_and_scale_head_gates(x)
            query, key, weights_raw = self._get_q_k_bf16(
                q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch
            )
            q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
            with torch.cuda.stream(self.alt_stream):
                self._store_index_k_cache(
                    forward_batch=forward_batch,
                    layer_id=layer_id,
                    key=key,
                    act_quant=act_quant,
                )
            current_stream.wait_stream(self.alt_stream)
            if self.use_dsa_indexer_fusion:
                weights = self._scale_head_gates(weights_raw, q_scale)
            else:
                weights = self._apply_q_scale_and_softmax_scale(weights, q_scale)
        else:
            query, key, weights_raw = self._get_q_k_bf16(
                q_lora,
                x,
                positions,
                enable_dual_stream,
                forward_batch=forward_batch,
            )

            if enable_dual_stream:
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)

                q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
                with torch.cuda.stream(self.alt_stream):
                    self._store_index_k_cache(
                        forward_batch=forward_batch,
                        layer_id=layer_id,
                        key=key,
                        act_quant=act_quant,
                    )
                current_stream.wait_stream(self.alt_stream)
            elif not in_piecewise_or_breakable_cuda_graph:
                q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
                self._store_index_k_cache(
                    forward_batch=forward_batch,
                    layer_id=layer_id,
                    key=key,
                    act_quant=act_quant,
                )
            else:
                # Graph paths not handled by the full DSA indexer split op
                # still need q_fp8 for paged topk and q_scale for
                # logits_head_gate_graph. K-cache storage is handled by the
                # full graph split path when prefill requires it.
                q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)

            # aiter (ROCm gfx95): the 3-tuple (fp8, scale, bf16) from
            # fused_rms_fp8_group_quant is passed directly to _get_logits_head_gate,
            # which extracts the bf16 tensor via _weights_proj_bf16_in_fp32_out,
            # completely skipping the FP8 dequantization path below.
            if (
                _use_aiter
                and _is_gfx95_supported
                and isinstance(x, tuple)
                and len(x) == 3
            ):
                x_for_gate = x
            elif isinstance(x, tuple):
                assert len(x) in (
                    2,
                    3,
                ), "For tuple input, only (x, x_s) or (x, x_s, y) formats are accepted"
                x_q, x_s = x[0], x[1]
                if (
                    x_s is not None
                    and x_q.dim() == 2
                    and x_s.dim() == 2
                    and x_q.shape[0] == x_s.shape[0]
                ):
                    m, n = x_q.shape
                    ng = x_s.shape[1]
                    if ng > 0 and n % ng == 0:
                        group = n // ng
                        x_for_gate = (
                            x_q.to(torch.float32)
                            .view(m, ng, group)
                            .mul_(x_s.to(torch.float32).unsqueeze(-1))
                            .view(m, n)
                            .to(torch.bfloat16)
                        )
                    else:
                        x_for_gate = x_q.to(torch.bfloat16)
                else:
                    x_for_gate = x_q.to(torch.bfloat16)
            else:
                x_for_gate = x

            if in_piecewise_or_breakable_cuda_graph:
                if self.use_dsa_indexer_fusion:
                    weights = scale_head_gate_graph(
                        weights_raw,
                        self.n_heads**-0.5,
                        self.softmax_scale,
                        q_scale,
                    )
                else:
                    if weights_proj_lora:
                        raise RuntimeError(GRAPH_WEIGHTS_PROJ_LORA_ERROR)
                    weights = logits_head_gate_graph(
                        x_for_gate,
                        self.weights_proj.weight,
                        self.n_heads**-0.5,
                        self.softmax_scale,
                        q_scale,
                    )
            elif self.use_dsa_indexer_fusion:
                weights = self._scale_head_gates(weights_raw, q_scale)
            elif weights_proj_lora:
                weights = self.weights_proj(x_for_gate)[0].float() * self.n_heads**-0.5
                weights = self._apply_q_scale_and_softmax_scale(weights, q_scale)
            else:
                weights = self._get_logits_head_gate(x_for_gate, q_scale)

        if _is_cuda or _is_hip:
            # In piecewise/breakable CUDA graph, any access to seq_lens_cpu
            # creates a Dynamo shape guard. These graph modes never have empty
            # batches.
            if not in_piecewise_or_breakable_cuda_graph:
                if forward_batch.seq_lens.numel() == 0:
                    # this seems b/c max-pad, no worries?
                    # if x.shape[0] != 0:
                    #     print(
                    #         "HACK: seq_lens empty but x not empty, hackily return all-invalid topk_result"
                    #     )
                    topk_result = torch.full(
                        (x_meta.shape[0], self.index_topk),
                        -1,
                        dtype=torch.int,
                        device=x_meta.device,
                    )
                    topk_result = _broadcast_indexer_topk_from_rank0(topk_result)
                    return maybe_capture_indexer_topk(layer_id, topk_result)

            if (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend_v2()
            ):
                topk_result = self._get_topk_paged(
                    forward_batch, layer_id, q_fp8, weights, metadata
                )
            else:
                if (
                    forward_batch.attn_cp_metadata is not None
                    and is_dsa_prefill_cp_in_seq_split()
                ):
                    kv_len_prev = forward_batch.attn_cp_metadata.kv_len_prev_list[0]
                    kv_len_next = forward_batch.attn_cp_metadata.kv_len_next_list[0]
                    actual_seq_q_prev = (
                        forward_batch.attn_cp_metadata.actual_seq_q_prev_list[0]
                    )
                    actual_seq_q_next = (
                        forward_batch.attn_cp_metadata.actual_seq_q_next_list[0]
                    )

                    # TODO support mutil-batch
                    # cp_batch_seq_index_prev = forward_batch.attn_cp_metadata["cp_batch_seq_index_prev"]
                    # cp_batch_seq_index_next = forward_batch.attn_cp_metadata["cp_batch_seq_index_next"]
                    # TODO prev, next, combined into a single call
                    q_fp8_prev, q_fp8_next = torch.split(
                        q_fp8, (q_fp8.shape[0] + 1) // 2, dim=0
                    )
                    weights_prev, weights_next = torch.split(
                        weights, (weights.shape[0] + 1) // 2, dim=0
                    )
                    topk_result_prev = self._get_topk_ragged_with_cp(
                        forward_batch,
                        layer_id,
                        q_fp8_prev,
                        weights_prev,
                        metadata,
                        kv_len_prev,
                        actual_seq_q_prev,
                    )

                    topk_result_next = self._get_topk_ragged_with_cp(
                        forward_batch,
                        layer_id,
                        q_fp8_next,
                        weights_next,
                        metadata,
                        kv_len_next,
                        actual_seq_q_next,
                    )
                    topk_result = torch.cat([topk_result_prev, topk_result_next], dim=0)
                    topk_result = _broadcast_indexer_topk_from_rank0(topk_result)
                    return maybe_capture_indexer_topk(layer_id, topk_result)
                else:
                    # In-graph (PCG/BCG) non-CP prefill is handled earlier by the
                    # graph DSA split-op dispatch, so only the eager path reaches
                    # here.
                    assert not in_piecewise_or_breakable_cuda_graph, (
                        "Internal error: in-graph DSA prefill must go through the "
                        "graph DSA split-op dispatch"
                    )
                    topk_result = self._get_topk_ragged(
                        enable_dual_stream,
                        forward_batch,
                        layer_id,
                        q_fp8,
                        weights,
                        metadata,
                    )
        else:
            raise NotImplementedError("DSA indexer only supports CUDA, HIP, and NPU")
        topk_result = _broadcast_indexer_topk_from_rank0(topk_result)
        return maybe_capture_indexer_topk(layer_id, topk_result)
