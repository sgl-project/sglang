from __future__ import annotations

from sglang.srt.runtime_context import (
    get_buffer,
    get_disagg,
    get_exec,
    get_parallel,
    get_platform,
    get_schedule,
)

"""
Support attention backend for flashinfer MLA.
The flashinfer_mla_disable_ragged flag controls whether to use ragged prefill wrapper and defaults to be false.
When it's set to false, all wrappers are BatchMLAPaged wrapper.
When it's set to true, the backend uses BatchRagged and BatchMLAPaged wrapper for prefilling,
and uses BatchMLAPaged wrapper for decoding.
More details can be found in https://docs.flashinfer.ai/api/mla.html
"""

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch

from sglang.kernels.ops.attention.dsa.quant_k_cache import gather_dsa_kv_scales
from sglang.kernels.ops.attention.utils import assert_buffer_fits
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashinfer_backend import (
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.layers.dcp import (
    DecodeContextParallelMetadata,
    update_local_kv_lens_for_dcp,
)
from sglang.srt.layers.dcp.planner import plan_dcp_decode_metadata
from sglang.srt.mem_cache.kv_index_translator import KVIndexTable
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    is_in_breakable_cuda_graph,
)
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
    is_flashinfer_available,
    next_power_of_2,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.flashinfer_mla_backend import (
        FlashInferMlaAttnBackend,
    )
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

if envs.SGLANG_ENABLE_TORCH_COMPILE.get():
    import logging

    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True

if is_flashinfer_available():
    from flashinfer import (
        BatchMLAPagedAttentionWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )


_MLA_BF16_DEQUANT_SCRATCH = {}


def _referenced_kv_to_bf16(k_fp8, kv_indices):
    """Dequantize only referenced KV rows instead of the entire pool.
    Allocate the persistent CUDA graph buffer outside inference mode.
    """
    key = (k_fp8.device, tuple(k_fp8.shape))
    scratch = _MLA_BF16_DEQUANT_SCRATCH.get(key)
    if scratch is None:
        with torch.inference_mode(False):
            scratch = torch.empty(
                k_fp8.shape, dtype=torch.bfloat16, device=k_fp8.device
            )
        _MLA_BF16_DEQUANT_SCRATCH[key] = scratch
    idx = kv_indices.long()
    scratch[idx] = k_fp8[idx].to(torch.bfloat16)
    return scratch


def _verify_plan_indptr_cpu(bs, draft_token_num, seq_lens_cpu):
    """Build verify indptr on CPU to avoid FlashInfer plan's blocking GPU-to-host synchronization."""
    n = draft_token_num
    qo_indptr = torch.arange(0, (bs + 1) * n, step=n, dtype=torch.int32)
    kv_len_arr = seq_lens_cpu[:bs].to(torch.int32) + n
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(kv_len_arr, dim=0)
    return qo_indptr, kv_indptr, kv_len_arr


def _extend_plan_indptr_cpu(bs, seq_lens_cpu, extend_seq_lens_cpu):
    """Build extend indptr on CPU to avoid FlashInfer plan's blocking GPU-to-host synchronization."""
    kv_len_arr = seq_lens_cpu[:bs].to(torch.int32)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(kv_len_arr, dim=0)
    qo_indptr = torch.zeros(bs + 1, dtype=torch.int32)
    qo_indptr[1:] = torch.cumsum(
        torch.tensor(extend_seq_lens_cpu[:bs], dtype=torch.int32), dim=0
    )
    return qo_indptr, kv_indptr, kv_len_arr


@dataclass
class DecodeMetadata:
    decode_wrapper: BatchMLAPagedAttentionWrapper


@dataclass
class PrefillMetadata:
    prefill_wrapper: BatchMLAPagedAttentionWrapper
    use_ragged: bool


# Reuse this workspace buffer across all flashinfer wrappers


class FlashInferMhaChunkKVRunner:
    def __init__(
        self, model_runner: ModelRunner, attn_backend: FlashInferMlaAttnBackend
    ):
        # Parse Constants
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        self.data_type = model_runner.dtype
        self.q_data_type = model_runner.dtype

        # Buffers and wrappers
        self.qo_indptr = attn_backend.qo_indptr
        self.kv_indptr = attn_backend.kv_indptr
        self.workspace_buffer = attn_backend.workspace_buffer
        self.fmha_backend = attn_backend.fmha_backend

        self.chunk_ragged_wrappers = []
        self.ragged_wrapper = attn_backend.prefill_wrapper_ragged

    def update_prefix_chunks(self, num_prefix_chunks: int):
        while num_prefix_chunks > len(self.chunk_ragged_wrappers):
            ragged_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace_buffer, "NHD", backend=self.fmha_backend
            )
            self.chunk_ragged_wrappers.append(ragged_wrapper)

    def update_wrapper(
        self,
        forward_batch: ForwardBatch,
        disable_flashinfer_ragged: bool = False,
    ):
        assert forward_batch.num_prefix_chunks is not None
        num_prefix_chunks = forward_batch.num_prefix_chunks
        self.update_prefix_chunks(num_prefix_chunks)

        prefix_lens = forward_batch.extend_prefix_lens
        seq_lens = forward_batch.seq_lens

        bs = len(seq_lens)
        qo_indptr = self.qo_indptr
        qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
        qo_indptr = qo_indptr[: bs + 1]

        for chunk_idx in range(forward_batch.num_prefix_chunks):
            # MHA for chunked prefix kv cache when running model with MLA
            assert forward_batch.prefix_chunk_idx is not None
            assert forward_batch.prefix_chunk_cu_seq_lens is not None
            assert forward_batch.prefix_chunk_max_seq_lens is not None

            kv_indptr = forward_batch.prefix_chunk_cu_seq_lens[chunk_idx]
            wrapper = self.chunk_ragged_wrappers[chunk_idx]
            wrapper.begin_forward(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                num_qo_heads=self.num_local_heads,
                num_kv_heads=self.num_local_heads,
                head_dim_qk=self.qk_nope_head_dim + self.qk_rope_head_dim,
                head_dim_vo=self.v_head_dim,
                q_data_type=self.q_data_type,
                causal=False,
            )
        # ragged prefill
        if not disable_flashinfer_ragged:
            kv_indptr = (
                qo_indptr
                if not forward_batch.mha_one_shot
                else self.kv_indptr[: bs + 1]
            )
            self.ragged_wrapper.begin_forward(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                num_qo_heads=self.num_local_heads,
                num_kv_heads=self.num_local_heads,
                head_dim_qk=self.qk_nope_head_dim + self.qk_rope_head_dim,
                head_dim_vo=self.v_head_dim,
                q_data_type=self.q_data_type,
                causal=True,
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ):
        logits_soft_cap = layer.logit_cap
        if forward_batch.attn_attend_prefix_cache:
            chunk_idx = forward_batch.prefix_chunk_idx
            assert chunk_idx >= 0
            wrapper = self.chunk_ragged_wrappers[chunk_idx]
            o = wrapper.forward_return_lse(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                v.view(-1, layer.tp_v_head_num, layer.v_head_dim).to(q.dtype),
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )
        else:
            forward = (
                self.ragged_wrapper.forward_return_lse
                if forward_batch.mha_return_lse
                else self.ragged_wrapper.forward
            )
            o = forward(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                v.view(-1, layer.tp_v_head_num, layer.v_head_dim).to(q.dtype),
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )
        return o


class FlashInferMLAAttnBackend(AttentionBackend):
    """Flashinfer attention kernels."""

    # kv_indptr/qo_indptr are preallocated at (req pool + 1); an extend batch
    # can never carry more seqs than the pool.
    extend_dummy_seqs_capped_by_req_pool: bool = True

    # Verify metadata is ragged-layout aware via generate_attn_arg_prefill;
    # graphs key their wrappers by token tier (_verify_graph_key).
    supports_ragged_verify_graph: bool = True

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Parse constants
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.skip_prefill = skip_prefill
        # Pool refs — captured at construction so they survive deletion of the
        # corresponding ForwardBatch fields.
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.kv_index_translator = model_runner.kv_index_translator
        self.kv_read_tables = None
        self.enable_chunk_kv = (
            not skip_prefill
            and get_disagg().disaggregation_mode != "decode"
            and not get_schedule().disable_chunked_prefix_cache
            and not get_exec().kernel.flashinfer_mla_disable_ragged
        )
        self.page_size = model_runner.page_size
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.dsa_scaled_kv_layout = self.token_to_kv_pool.dsa_kv_cache_store_fp8
        self.use_dsa_scaled_kv = (
            self.dsa_scaled_kv_layout and self.qk_rope_head_dim == 0
        )
        self._dsa_num_tiles = self.kv_lora_rank // 128
        self._dsa_ckv_scale_buf = None
        if self.use_dsa_scaled_kv:
            kv_pool = self.token_to_kv_pool
            self._dsa_ckv_scale_buf = torch.empty(
                (kv_pool.size + kv_pool.page_size, 1, self._dsa_num_tiles),
                dtype=torch.float32,
                device=model_runner.device,
            )
        self.extend_scale_kv_indices = None
        self.extend_scale_kv_indptr = None
        self.extend_scale_kv_indptr_idx = 0

        # Allocate buffers
        # different from flashinfer zero_init_global_workspace_buffer
        self.workspace_buffer = get_buffer(
            "flashinfer_mla_workspace",
            lambda: torch.empty(
                envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                dtype=torch.uint8,
                device=model_runner.device,
            ),
        )

        max_bs = model_runner.req_to_token_pool.size
        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        if not self.skip_prefill:
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

        if q_indptr_decode_buf is None:
            self.q_indptr_decode = torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=model_runner.device
            )
        else:
            self.q_indptr_decode = q_indptr_decode_buf

        if get_platform().is_sm100:
            self.fmha_backend = "cutlass"
        else:
            self.fmha_backend = "auto"

        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend=self.fmha_backend
        )

        self.decode_bf16_kv_indices = None
        self.extend_bf16_kv_indices = None
        self.decode_fp8_native = get_platform().is_sm90

        if not self.skip_prefill:
            self.prefill_wrapper_paged = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                backend="auto",
            )

            # FlashinferMLA backend uses mla wrapper for target verify
            self.prefill_wrapper_verify = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                backend="auto",
            )

        self.decode_wrapper = BatchMLAPagedAttentionWrapper(
            self.workspace_buffer, backend="auto"
        )

        # Create indices updater
        if not skip_prefill:
            self.indices_updater_prefill = FlashInferMLAIndicesUpdaterPrefill(
                model_runner, self
            )
            if self.enable_chunk_kv:
                self.mha_chunk_kv_cache = FlashInferMhaChunkKVRunner(model_runner, self)

        self.indices_updater_decode = FlashInferMLAIndicesUpdaterDecode(
            model_runner, self
        )

        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None
        self.decode_cuda_graph_metadata = {}
        self.prefill_cuda_graph_metadata = {}  # For verify

        # Pinned host buffers for the fast prefill path plan
        if not skip_prefill:
            self.fast_plan_qo_indptr_cpu = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device="cpu", pin_memory=True
            )
            self.fast_plan_kv_indptr_cpu = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device="cpu", pin_memory=True
            )
            self.fast_plan_kv_len_arr_cpu = torch.zeros(
                (max_bs,), dtype=torch.int32, device="cpu", pin_memory=True
            )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        bs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        forward_mode = forward_batch.forward_mode
        spec_info = forward_batch.spec_info

        # All flashinfer gathers run OUT-of-graph (plan time), so the
        # capture-stable table is buffer reuse, not pointer stability.
        kv_view = self.kv_index_translator.build_index_table(
            req_pool_indices=req_pool_indices[:bs],
            seq_lens=seq_lens[:bs],
            into=self.kv_read_tables,
        )

        if in_capture:
            num_tokens = forward_batch.positions.numel()
            seq_lens_sum = seq_lens.sum().item()
            seq_lens_cpu = seq_lens.cpu()

            if forward_mode.is_decode_or_idle():
                decode_wrapper = BatchMLAPagedAttentionWrapper(
                    self.workspace_buffer,
                    use_cuda_graph=True,
                    qo_indptr=self.cuda_graph_qo_indptr[: num_tokens + 1],
                    kv_indptr=self.cuda_graph_kv_indptr[: num_tokens + 1],
                    kv_indices=self.cuda_graph_kv_indices,
                    kv_len_arr=self.cuda_graph_kv_lens[:num_tokens],
                    backend="auto",
                )
                self.indices_updater_decode.update(
                    seq_lens,
                    seq_lens_sum,
                    decode_wrapper=decode_wrapper,
                    init_metadata_replay=False,
                    spec_info=spec_info,
                    kv_view=kv_view,
                )
                self.decode_cuda_graph_metadata[bs] = decode_wrapper
                self.forward_metadata = DecodeMetadata(decode_wrapper)
                # fast_mla_decode_plan needs _cached_module from the initial
                # begin_forward above, so install it only after that call completes.
                decode_wrapper.plan = partial(fast_mla_decode_plan, decode_wrapper)
            elif forward_mode.is_target_verify():
                prefill_wrapper = BatchMLAPagedAttentionWrapper(
                    self.workspace_buffer,
                    use_cuda_graph=True,
                    qo_indptr=self.cuda_graph_qo_indptr[: bs + 1],
                    kv_indptr=self.cuda_graph_kv_indptr[: bs + 1],
                    kv_indices=self.cuda_graph_kv_indices,
                    kv_len_arr=self.cuda_graph_kv_lens[:bs],
                    backend="auto",
                )
                self.prefill_cuda_graph_metadata[
                    self._verify_graph_key(bs, spec_info)
                ] = prefill_wrapper
                self.forward_metadata = PrefillMetadata(prefill_wrapper, False)
            else:
                raise ValueError(f"Invalid mode: {forward_mode=}")

            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_sum=seq_lens_sum,
                forward_mode=forward_mode,
                spec_info=spec_info,
                seq_lens_cpu=seq_lens_cpu,
                in_capture=True,
                kv_view=kv_view,
            )
            if forward_mode.is_target_verify() and (
                spec_info is None
                or spec_info.spec_input_type != SpecInputType.DFLASH_VERIFY
            ):
                # use sync-free fast_mla_prefill_plan for replay
                prefill_wrapper.plan = partial(fast_mla_prefill_plan, prefill_wrapper)
        else:
            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_sum=forward_batch.seq_lens_sum,
                forward_mode=forward_mode,
                spec_info=spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                kv_view=kv_view,
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_view = self.kv_index_translator.index_table_for_batch(forward_batch)
        if forward_batch.forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                decode_wrapper=self.decode_wrapper,
                init_metadata_replay=False,
                kv_view=kv_view,
            )
            self.forward_metadata = DecodeMetadata(self.decode_wrapper)
        elif forward_batch.forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrapper_paged=self.prefill_wrapper_verify,
                use_ragged=False,
                spec_info=forward_batch.spec_info,
                is_verify=True,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                kv_view=kv_view,
            )
            self.forward_metadata = PrefillMetadata(self.prefill_wrapper_verify, False)
            self.extend_scale_kv_indices = self.extend_bf16_kv_indices
            self.extend_scale_kv_indptr = None
            self.extend_scale_kv_indptr_idx = 0
        else:
            prefix_lens = forward_batch.extend_prefix_lens
            extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            use_ragged = (
                not get_exec().kernel.flashinfer_mla_disable_ragged
                and self.qk_rope_head_dim > 0
                and extend_no_prefix
                # Captured prefill (tc_piecewise or breakable) must use paged
                # prefill: it stays compatible with prefix cache, and the ragged
                # wrapper rejects the absorbed-MLA head dims (qk=576, vo=512).
                and not is_in_tc_piecewise_cuda_graph()
                and not is_in_breakable_cuda_graph()
            )

            # build host indptr/len arrays for eager DRAFT_EXTEND_V2 fast plan path
            qo_indptr_cpu = kv_indptr_cpu = kv_len_arr_cpu = None
            spec_info = forward_batch.spec_info
            if (
                not use_ragged
                and forward_batch.forward_mode.is_draft_extend_v2()
                and forward_batch.seq_lens_cpu is not None
                and spec_info is not None
            ):
                ndt = spec_info.num_tokens_per_req
                bs = forward_batch.batch_size
                self.fast_plan_qo_indptr_cpu[: bs + 1] = torch.arange(
                    0, (bs + 1) * ndt, ndt, dtype=torch.int32
                )
                self.fast_plan_kv_len_arr_cpu[:bs] = forward_batch.seq_lens_cpu[:bs]
                self.fast_plan_kv_indptr_cpu[1 : bs + 1] = torch.cumsum(
                    self.fast_plan_kv_len_arr_cpu[:bs], dim=0
                )
                qo_indptr_cpu = self.fast_plan_qo_indptr_cpu[: bs + 1]
                kv_indptr_cpu = self.fast_plan_kv_indptr_cpu[: bs + 1]
                kv_len_arr_cpu = self.fast_plan_kv_len_arr_cpu[:bs]

            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                prefix_lens,
                prefill_wrapper_paged=self.prefill_wrapper_paged,
                use_ragged=use_ragged,
                attn_dcp_metadata=forward_batch.attn_dcp_metadata,
                qo_indptr_cpu=qo_indptr_cpu,
                kv_indptr_cpu=kv_indptr_cpu,
                kv_len_arr_cpu=kv_len_arr_cpu,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
                kv_view=kv_view,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrapper_paged, use_ragged
            )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        self.kv_read_tables = self.kv_index_translator.make_capture_tables(
            max_bs=max_bs, max_context_len=self.max_context_len
        )
        if kv_indices_buf is None:
            cuda_graph_kv_indices = torch.zeros(
                (max_bs * self.max_context_len,),
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = kv_indices_buf

        self.cuda_graph_kv_indices = cuda_graph_kv_indices
        self.cuda_graph_qo_indptr = self.q_indptr_decode.clone()
        self.cuda_graph_kv_indptr = self.kv_indptr.clone()
        self.cuda_graph_kv_lens = torch.ones(
            (max_bs,), dtype=torch.int32, device=self.device
        )

        # For fast decode plan in graph replaying
        self.cuda_graph_qo_indptr_cpu = self.cuda_graph_qo_indptr.to("cpu")
        self.cuda_graph_kv_indptr_cpu = self.cuda_graph_kv_indptr.to("cpu")
        self.fast_decode_kwargs = {
            "qo_indptr_cpu": self.cuda_graph_qo_indptr_cpu,
            "kv_indptr_cpu": self.cuda_graph_kv_indptr_cpu,
            "kv_indices": self.cuda_graph_kv_indices,
        }

    @staticmethod
    def _verify_graph_key(bs: int, spec_info: Optional[SpecInput]):
        """bs for uniform graphs; token tier for ragged (tiers share slot
        counts but each graph must replay its own recorded plan buffers)."""
        layout = spec_info.ragged_verify_layout if spec_info is not None else None
        if layout is None:
            return bs
        return ("ragged", layout.graph_num_tokens)

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        kv_view: KVIndexTable,
        in_capture: bool = False,
    ):
        """Shared capture+replay body for the cuda-graph init path.

        Public entry: :py:meth:`init_forward_metadata_out_graph`.
        """
        if forward_mode.is_decode_or_idle():
            assert seq_lens_cpu is not None
            kv_len_arr_cpu = seq_lens_cpu[:bs].to(torch.int32)
            update_local_kv_lens_for_dcp(kv_len_arr_cpu)
            self.cuda_graph_kv_indptr_cpu[1 : bs + 1] = torch.cumsum(
                kv_len_arr_cpu, dim=0
            )
            self.fast_decode_kwargs.update(
                {
                    "qo_indptr_cpu": self.cuda_graph_qo_indptr_cpu[: bs + 1],
                    "kv_indptr_cpu": self.cuda_graph_kv_indptr_cpu[: bs + 1],
                    "kv_len_arr_cpu": kv_len_arr_cpu,
                }
            )

            self.indices_updater_decode.update(
                seq_lens[:bs],
                seq_lens_sum,
                decode_wrapper=self.decode_cuda_graph_metadata[bs],
                init_metadata_replay=True,
                spec_info=spec_info,
                kv_view=kv_view,
                **self.fast_decode_kwargs,
            )
        elif forward_mode.is_target_verify():
            # build host indptr/len arrays for target-verify fast plan path
            assert (
                seq_lens_cpu is not None and spec_info is not None
            ), "target-verify cuda-graph replay requires host-resident seq_lens_cpu"
            ndt = spec_info.draft_token_num
            self.fast_plan_qo_indptr_cpu[: bs + 1] = torch.arange(
                0, (bs + 1) * ndt, ndt, dtype=torch.int32
            )
            self.fast_plan_kv_len_arr_cpu[:bs] = seq_lens_cpu[:bs] + ndt
            self.fast_plan_kv_indptr_cpu[1 : bs + 1] = torch.cumsum(
                self.fast_plan_kv_len_arr_cpu[:bs], dim=0
            )
            fast_verify_plan_kwargs = self._build_fast_verify_plan_kwargs(
                bs=bs,
                spec_info=spec_info,
                seq_lens_cpu=seq_lens_cpu,
                in_capture=in_capture,
            )
            use_generic_fast_plan = (
                spec_info.spec_input_type != SpecInputType.DFLASH_VERIFY
            )
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrapper_paged=self.prefill_cuda_graph_metadata[
                    self._verify_graph_key(bs, spec_info)
                ],
                use_ragged=False,
                spec_info=spec_info,
                fast_verify_plan_kwargs=fast_verify_plan_kwargs,
                qo_indptr_cpu=(
                    self.fast_plan_qo_indptr_cpu[: bs + 1]
                    if use_generic_fast_plan
                    else None
                ),
                kv_indptr_cpu=(
                    self.fast_plan_kv_indptr_cpu[: bs + 1]
                    if use_generic_fast_plan
                    else None
                ),
                kv_len_arr_cpu=(
                    self.fast_plan_kv_len_arr_cpu[:bs]
                    if use_generic_fast_plan
                    else None
                ),
                is_verify=True,
                seq_lens_cpu=seq_lens_cpu,
                kv_view=kv_view,
            )
            self.extend_scale_kv_indices = self.cuda_graph_kv_indices
            self.extend_scale_kv_indptr = self.cuda_graph_kv_indptr
            self.extend_scale_kv_indptr_idx = bs
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

    def _build_fast_verify_plan_kwargs(
        self,
        *,
        bs: int,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        in_capture: bool,
    ) -> Optional[dict]:
        """Host-known plan inputs for the sync-free TARGET_VERIFY fast plan.

        Upstream ``BatchMLAPagedAttentionWrapper.plan`` issues three blocking
        ``.to("cpu")`` copies per call (qo_indptr / kv_indptr / kv_len_arr); on
        the graph-replay hot path each of those drains the whole GPU queue and
        stalls the scheduler CPU behind the in-flight draft graph. All three
        arrays are host-derivable, so we feed ``fast_mla_decode_plan`` directly.

        Returns None when the slow (device-fed) plan must run instead: at
        capture (the real plan() populates ``_cached_module`` and the wrapper's
        cuda-graph buffers), for non-DFLASH spec inputs, for ragged/compact
        verify layouts or custom masks, under DCP, or when seq_lens_cpu is
        unavailable.

        DFLASH invariant this relies on: the verify ForwardBatch carries
        seq_lens_cpu = prefix + draft_token_num (dspark_verify.run_non_compact
        and dflash_worker_v2 both add the verify window host-side before
        prepare_for_verify), which equals the device kv length that
        generate_attn_arg_prefill produces (seq_lens + draft_token_num). The
        reserved_seq_lens_cpu fallback (an upper bound, not the exact value) is
        only reachable when seq_lens_cpu is resolved as None, and this fast
        path requires flashinfer's needs_cpu_seq_lens=True resolve, so the
        exact value is always the one seen here.
        """
        if in_capture or seq_lens_cpu is None or spec_info is None:
            return None
        if spec_info.spec_input_type != SpecInputType.DFLASH_VERIFY:
            return None
        if (
            spec_info.ragged_verify_layout is not None
            or spec_info.custom_mask is not None
        ):
            return None
        if get_parallel().dcp_enabled:
            return None
        draft_token_num = int(spec_info.draft_token_num)
        kv_len_arr_cpu = seq_lens_cpu[:bs].to(torch.int32)
        kv_indptr_cpu = torch.zeros(bs + 1, dtype=torch.int32)
        torch.cumsum(kv_len_arr_cpu, dim=0, out=kv_indptr_cpu[1:])
        qo_indptr_cpu = torch.arange(
            0, (bs + 1) * draft_token_num, draft_token_num, dtype=torch.int32
        )
        return {
            "qo_indptr_cpu": qo_indptr_cpu,
            "kv_indptr_cpu": kv_indptr_cpu,
            "kv_len_arr_cpu": kv_len_arr_cpu,
            "kv_indices_buf": self.cuda_graph_kv_indices,
        }

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def init_mha_chunk_metadata(
        self, forward_batch: ForwardBatch, disable_flashinfer_ragged: bool = False
    ):
        """Init the metadata for a forward pass."""
        self.mha_chunk_kv_cache.update_wrapper(forward_batch, disable_flashinfer_ragged)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if forward_batch.attn_attend_prefix_cache is not None and any(
            forward_batch.extend_prefix_lens_cpu
        ):  # MHA Chunk
            assert self.enable_chunk_kv
            assert q_rope is None
            assert k_rope is None
            return self.mha_chunk_kv_cache.forward(q, k, v, layer, forward_batch)

        cache_loc = forward_batch.out_cache_loc
        logits_soft_cap = layer.logit_cap
        prefill_wrapper_paged = self.forward_metadata.prefill_wrapper

        # Save kv cache
        if save_kv_cache and k is not None:
            assert v is not None
            if save_kv_cache:
                if k_rope is not None:
                    self.token_to_kv_pool.set_mla_kv_buffer(layer, cache_loc, k, k_rope)
                else:
                    self.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
        if self.qk_rope_head_dim == 0:
            q_rope = None
        if q_rope is not None:
            q = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )

        if self.forward_metadata.use_ragged:
            # ragged prefill
            if q_rope is not None:
                q = torch.cat([q, q_rope], dim=-1)
            qall = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            if k_rope is not None:
                k = torch.cat([k, k_rope], dim=-1)
            o = self.prefill_wrapper_ragged.forward(
                qall,
                k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )
        else:
            # mla paged prefill
            if (
                forward_batch.attn_dcp_metadata is not None
                and forward_batch.attn_dcp_metadata.dcp_kv_buffer is not None
            ):
                k_buf = forward_batch.attn_dcp_metadata.dcp_kv_buffer
            else:
                k_buf = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
            if q_rope is None:
                qall = q.view(-1, layer.tp_q_head_num, layer.head_dim)
                q, q_rope = (
                    qall[:, :, : layer.v_head_dim],
                    qall[:, :, layer.v_head_dim :],
                )
            o = q.new_empty(q.shape)
            fp8_run_kwargs = {}
            if k_buf.dtype == torch.float8_e4m3fn:
                if self.use_dsa_scaled_kv:
                    k_buf, fp8_run_kwargs = self._scaled_fp8_kv_kwargs(
                        k_buf,
                        (
                            self.extend_scale_kv_indices
                            if forward_batch.forward_mode.is_target_verify()
                            else self.extend_bf16_kv_indices
                        ),
                        (
                            self.extend_scale_kv_indptr
                            if forward_batch.forward_mode.is_target_verify()
                            else None
                        ),
                        self.extend_scale_kv_indptr_idx,
                    )
                elif (
                    forward_batch.forward_mode.is_target_verify()
                    and self.decode_fp8_native
                ):
                    kv_scale = (
                        layer.k_scale_float if layer.k_scale_float is not None else 1.0
                    )
                    fp8_run_kwargs = {"ckv_scale": kv_scale, "kpe_scale": kv_scale}
                else:
                    k_buf = self._dequant_referenced_kv_extend(k_buf)
            elif k_buf.dtype != q.dtype:
                k_buf = k_buf.to(q.dtype)
            o = prefill_wrapper_paged.run(
                q,
                q_rope,
                k_buf[:, :, : layer.v_head_dim],
                k_buf[:, :, layer.v_head_dim :],
                out=o,
                **fp8_run_kwargs,
            )

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _dequant_referenced_kv(self, k_fp8):
        idx = self.decode_bf16_kv_indices
        if idx is None:
            return k_fp8.to(torch.bfloat16)
        return _referenced_kv_to_bf16(k_fp8, idx)

    def _dequant_referenced_kv_extend(self, k_fp8):
        idx = self.extend_bf16_kv_indices
        if idx is None:
            return k_fp8.to(torch.bfloat16)
        return _referenced_kv_to_bf16(k_fp8, idx)

    def _scaled_fp8_kv_kwargs(self, k_fp8, kv_indices, kv_indptr=None, kv_indptr_idx=0):
        """Gather referenced per-group FP8 scales into an index-compatible persistent buffer."""
        assert self.use_dsa_scaled_kv
        assert self._dsa_ckv_scale_buf is not None
        scale_src = k_fp8[
            :, :, self.kv_lora_rank : self.kv_lora_rank + self._dsa_num_tiles * 4
        ].view(torch.float32)
        if kv_indptr is not None:
            gather_dsa_kv_scales(
                scale_src,
                self._dsa_ckv_scale_buf,
                kv_indices,
                kv_indptr,
                kv_indptr_idx,
            )
        elif kv_indices is None:
            self._dsa_ckv_scale_buf.copy_(scale_src)
        else:
            idx = kv_indices.long()
            self._dsa_ckv_scale_buf[idx] = scale_src[idx]
        return k_fp8[:, :, : self.kv_lora_rank], {
            "ckv_scale": 1.0,
            "kpe_scale": 1.0,
            "ckv_scale_arr": self._dsa_ckv_scale_buf,
        }

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        decode_wrapper = self.forward_metadata.decode_wrapper
        cache_loc = forward_batch.out_cache_loc

        if k is not None:
            assert v is not None
            if save_kv_cache:
                if k_rope is not None:
                    self.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )
                else:
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        v,
                    )

        # Reshape inputs
        if self.qk_rope_head_dim == 0:
            q_rope = None
        if q_rope is not None:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
        else:
            reshaped_q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = reshaped_q[:, :, : layer.v_head_dim]
            q_rope = reshaped_q[:, :, layer.v_head_dim :]

        k_buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        fp8_run_kwargs = {}
        if k_buffer.dtype == torch.float8_e4m3fn:
            if self.use_dsa_scaled_kv:
                k_buffer, fp8_run_kwargs = self._scaled_fp8_kv_kwargs(
                    k_buffer, self.decode_bf16_kv_indices
                )
            elif self.decode_fp8_native:
                kv_scale = (
                    layer.k_scale_float if layer.k_scale_float is not None else 1.0
                )
                fp8_run_kwargs = {"ckv_scale": kv_scale, "kpe_scale": kv_scale}
            else:
                k_buffer = self._dequant_referenced_kv(k_buffer)
        elif k_buffer.dtype != q.dtype:
            k_buffer = k_buffer.to(q.dtype)

        o = q_nope.new_empty(q_nope.shape)
        # for decode and dcp_world_size > 1, lse should be returned to compute final attn_out
        # Direct call to run without the wrapper
        o = decode_wrapper.run(
            q_nope,
            q_rope,
            k_buffer[:, :, : layer.v_head_dim],
            k_buffer[:, :, layer.v_head_dim :],
            out=o,
            # for decode forward_batch, each dcp rank computes total q and partial kv, thus, we need to return_lse for online softmax to get final attn_output
            return_lse=(
                forward_batch.forward_mode.is_decode() and get_parallel().dcp_enabled
            ),
            **fp8_run_kwargs,
        )
        if isinstance(o, tuple):
            out, lse = o
            out = out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            return (out, lse)
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


class FlashInferMLAIndicesUpdaterDecode:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads
            // get_parallel().attn_tp_size
            * get_parallel().attn_dcp_size
        )
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.scaling = model_runner.model_config.scaling
        self.data_type = model_runner.dtype
        self.kv_data_type = (
            model_runner.kv_cache_dtype
            if model_runner.kv_cache_dtype == torch.float8_e4m3fn
            else model_runner.dtype
        )
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.q_indptr = attn_backend.q_indptr_decode

    def update(
        self,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrapper: BatchMLAPagedAttentionWrapper,
        init_metadata_replay: bool = False,
        spec_info: Optional[SpecInput] = None,
        *,
        kv_view: KVIndexTable,
        **fast_decode_kwargs,
    ):
        decode_wrapper = decode_wrapper or self.decode_wrapper
        self.call_begin_forward(
            decode_wrapper,
            seq_lens,
            seq_lens_sum,
            self.q_indptr,
            self.kv_indptr,
            init_metadata_replay,
            spec_info,
            kv_view=kv_view,
            **fast_decode_kwargs,
        )

    def call_begin_forward(
        self,
        wrapper: BatchMLAPagedAttentionWrapper,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        q_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        init_metadata_replay: bool = False,
        spec_info: Optional[SpecInput] = None,
        *,
        kv_view: KVIndexTable,
        **fast_decode_kwargs,
    ):
        bs = len(paged_kernel_lens)
        q_indptr = q_indptr[: bs + 1]
        kv_lens = paged_kernel_lens.to(torch.int32)
        sm_scale = self.scaling
        if spec_info is None:
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            # On replay `kv_indices` IS the capture-stable buffer the captured
            # wrapper reads -- never rebind it. The builder fills only the
            # [:paged_kernel_lens_sum] prefix; the stale tail is unread.
            kv_indices = (
                torch.empty(paged_kernel_lens_sum, dtype=torch.int32, device="cuda")
                if not init_metadata_replay
                else fast_decode_kwargs["kv_indices"]
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                kv_view.ids,
                kv_view.row_ids,
                paged_kernel_lens,
                kv_indptr,
                None,
                kv_indices,
                kv_view.row_stride,
                ENTRY_PAGE_SIZE=kv_view.entry_page_size,
            )

            if get_parallel().dcp_enabled:
                plan_dcp_decode_metadata(
                    kv_lens,
                    kv_indptr,
                    kv_indices,
                    init_metadata_replay,
                    fast_decode_kwargs,
                    bs,
                )
        else:
            kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

        self.attn_backend.decode_bf16_kv_indices = kv_indices
        plan_kv_data_type = (
            self.kv_data_type if self.attn_backend.decode_fp8_native else self.data_type
        )

        if not init_metadata_replay:
            wrapper.plan(
                q_indptr,
                kv_indptr,
                kv_indices,
                kv_lens,
                self.num_local_heads,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                1,
                False,
                sm_scale,
                self.data_type,
                plan_kv_data_type,
            )
        else:
            wrapper.plan(
                fast_decode_kwargs["qo_indptr_cpu"],
                fast_decode_kwargs["kv_indptr_cpu"],
                kv_indices,
                fast_decode_kwargs["kv_len_arr_cpu"],
                self.num_local_heads,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                1,
                False,
                sm_scale,
                self.data_type,
                plan_kv_data_type,
            )


class FlashInferMLAIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        self.scaling = model_runner.model_config.scaling
        self.data_type = model_runner.dtype
        self.q_data_type = model_runner.dtype
        self.kv_data_type = (
            model_runner.kv_cache_dtype
            if model_runner.kv_cache_dtype == torch.float8_e4m3fn
            else model_runner.dtype
        )
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.qo_indptr = attn_backend.qo_indptr
        # Kept ONLY for the spec-info branch (generate_attn_arg_prefill), which
        # is static-pool-only: unified memory asserts spec off. The normal
        # builder sources from the per-batch KVIndexTable.
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrapper_paged: BatchMLAPagedAttentionWrapper,
        use_ragged: bool,
        spec_info: Optional[SpecInput] = None,
        attn_dcp_metadata: Optional[DecodeContextParallelMetadata] = None,
        fast_verify_plan_kwargs: Optional[dict] = None,
        *,
        kv_view: KVIndexTable,
        qo_indptr_cpu: Optional[torch.Tensor] = None,
        kv_indptr_cpu: Optional[torch.Tensor] = None,
        kv_len_arr_cpu: Optional[torch.Tensor] = None,
        is_verify: bool = False,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        extend_seq_lens_cpu: Optional[List[int]] = None,
    ):
        if use_ragged:
            paged_kernel_lens = prefix_lens
            paged_kernel_lens_sum = paged_kernel_lens.sum().item()
        else:
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum

        self.call_begin_forward(
            self.prefill_wrapper_ragged,
            prefill_wrapper_paged,
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            self.kv_indptr,
            self.qo_indptr,
            use_ragged,
            spec_info,
            attn_dcp_metadata=attn_dcp_metadata,
            fast_verify_plan_kwargs=fast_verify_plan_kwargs,
            qo_indptr_cpu=qo_indptr_cpu,
            kv_indptr_cpu=kv_indptr_cpu,
            kv_len_arr_cpu=kv_len_arr_cpu,
            is_verify=is_verify,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            kv_view=kv_view,
        )

    def call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: BatchMLAPagedAttentionWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
        spec_info: Optional[SpecInput] = None,
        attn_dcp_metadata: Optional[DecodeContextParallelMetadata] = None,
        fast_verify_plan_kwargs: Optional[dict] = None,
        *,
        kv_view: KVIndexTable,
        qo_indptr_cpu: Optional[torch.Tensor] = None,
        kv_indptr_cpu: Optional[torch.Tensor] = None,
        kv_len_arr_cpu: Optional[torch.Tensor] = None,
        is_verify: bool = False,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        extend_seq_lens_cpu: Optional[List[int]] = None,
    ):
        bs = len(seq_lens)
        sm_scale = self.scaling

        if spec_info is None:
            assert len(seq_lens) == len(req_pool_indices)
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                paged_kernel_lens_sum,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                kv_view.ids,
                kv_view.row_ids,
                paged_kernel_lens,
                kv_indptr,
                None,
                kv_indices,
                kv_view.row_stride,
                ENTRY_PAGE_SIZE=kv_view.entry_page_size,
            )
            qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
        elif fast_verify_plan_kwargs is not None:
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    paged_kernel_lens,
                    paged_kernel_lens_sum,
                    self.req_to_token,
                    kv_indices_buf=fast_verify_plan_kwargs["kv_indices_buf"],
                )
            )
        else:
            assert isinstance(spec_info, SpecInput)
            # TODO: Support topk > 1 with custom mask
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    paged_kernel_lens,
                    paged_kernel_lens_sum,
                    self.req_to_token,
                )
            )

        self.attn_backend.extend_bf16_kv_indices = kv_indices
        if use_ragged:
            # ragged prefill
            wrapper_ragged.begin_forward(
                qo_indptr=qo_indptr,
                kv_indptr=qo_indptr,
                num_qo_heads=self.num_local_heads,
                num_kv_heads=self.num_local_heads,
                head_dim_qk=self.qk_nope_head_dim + self.qk_rope_head_dim,
                head_dim_vo=self.v_head_dim,
                q_data_type=self.q_data_type,
                causal=True,
            )
        elif fast_verify_plan_kwargs is not None:
            fast_mla_decode_plan(
                wrapper_paged,
                fast_verify_plan_kwargs["qo_indptr_cpu"],
                fast_verify_plan_kwargs["kv_indptr_cpu"],
                kv_indices,
                fast_verify_plan_kwargs["kv_len_arr_cpu"],
                self.num_local_heads,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                1,
                True,
                sm_scale,
                self.q_data_type,
                self.data_type,
            )
        else:
            # mla paged prefill
            if attn_dcp_metadata is not None:
                if attn_dcp_metadata.dcp_kv_indptr is not None:
                    kv_indptr = attn_dcp_metadata.dcp_kv_indptr
                if attn_dcp_metadata.dcp_kv_indices is not None:
                    kv_indices = attn_dcp_metadata.dcp_kv_indices
                # DCP splits kv across CP ranks; the host-side fast-plan arrays are
                # built from full seq_lens and don't match, so fall back to GPU.
                qo_indptr_cpu = kv_indptr_cpu = kv_len_arr_cpu = None

            if qo_indptr_cpu is not None:
                plan_qo_indptr = qo_indptr_cpu
                plan_kv_indptr = kv_indptr_cpu
                plan_kv_len_arr = kv_len_arr_cpu
            elif (
                is_verify
                and seq_lens_cpu is not None
                and attn_dcp_metadata is None
                and spec_info is not None
            ):
                plan_qo_indptr, plan_kv_indptr, plan_kv_len_arr = (
                    _verify_plan_indptr_cpu(bs, spec_info.draft_token_num, seq_lens_cpu)
                )
            elif (
                spec_info is None
                and seq_lens_cpu is not None
                and extend_seq_lens_cpu is not None
                and attn_dcp_metadata is None
            ):
                plan_qo_indptr, plan_kv_indptr, plan_kv_len_arr = (
                    _extend_plan_indptr_cpu(bs, seq_lens_cpu, extend_seq_lens_cpu)
                )
            else:
                plan_qo_indptr = qo_indptr
                plan_kv_indptr = kv_indptr
                plan_kv_len_arr = kv_indptr[1:] - kv_indptr[:-1]
            plan_kv_data_type = (
                self.kv_data_type
                if (is_verify and self.attn_backend.decode_fp8_native)
                or self.attn_backend.use_dsa_scaled_kv
                else self.data_type
            )
            wrapper_paged.plan(
                plan_qo_indptr,
                plan_kv_indptr,
                kv_indices,
                plan_kv_len_arr,
                self.num_local_heads,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                1,
                True,
                sm_scale,
                self.q_data_type,
                plan_kv_data_type,
            )


class FlashInferMLAMultiStepDraftBackend:
    """
    Wrap multiple flashinfer mla attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        if topk > 1:
            raise ValueError(
                "Currently Flashinfer MLA only supports topk=1 for speculative decoding"
            )
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices

        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.q_indptr_decode = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=model_runner.device
        )

        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                FlashInferMLAAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    q_indptr_decode_buf=self.q_indptr_decode,
                )
            )

        self.max_context_len = self.attn_backends[0].max_context_len
        self.attn_backend_list = self.attn_backends
        self.forward_metadata = None

        # Cached variables for generate_draft_decode_kv_indices
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.page_size = get_schedule().page_size

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

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : draft_kv_indices_used_len(seq_lens_sum, self.topk, bs, i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices_width = draft_kv_indices_buffer_width(
            forward_batch.batch_size, self.topk, self.max_context_len
        )
        kv_indices = torch.zeros(
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
        # Row holds topk per-branch sequences (generate_draft_decode_kv_indices), so
        # it needs the topk factor, matching the eager init_forward_metadata.
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

        inner_fb = build_inner_fb_view(
            forward_batch,
            bs=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
        )

        def call_fn(i, _forward_batch):
            self.attn_backends[i].init_forward_metadata_out_graph(
                inner_fb, in_capture=in_capture
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        for attn_backend in self.attn_backends:
            attn_backend.init_forward_metadata_in_graph(forward_batch)


def fast_mla_decode_plan(
    self,
    qo_indptr_cpu: torch.Tensor,
    kv_indptr_cpu: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_len_arr_cpu: torch.Tensor,
    num_heads: int,
    head_dim_ckv: int,
    head_dim_kpe: int,
    page_size: int,
    causal: bool,
    sm_scale: float,
    q_data_type: torch.dtype,
    kv_data_type: torch.dtype,
) -> None:
    """A faster version of BatchMLAPagedAttentionWrapper::plan,
    for skipping the stream synchronization in original plan function during
    cuda graph replaying.
    """
    self._causal = causal
    self._page_size = page_size
    self._sm_scale = sm_scale

    try:
        # Standard version with just the required arguments (no use_profiler)
        self._cached_module.plan(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_cpu,
            kv_indptr_cpu,
            kv_len_arr_cpu,
            num_heads,
            head_dim_ckv,
            causal,
        )
    except Exception as e:
        raise RuntimeError(f"Error in alternate MLA plan: {e}")


def fast_mla_prefill_plan(
    self,
    qo_indptr_cpu: torch.Tensor,
    kv_indptr_cpu: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_len_arr_cpu: torch.Tensor,
    num_heads: int,
    head_dim_ckv: int,
    head_dim_kpe: int,
    page_size: int,
    causal: bool,
    sm_scale: float,
    q_data_type: torch.dtype,
    kv_data_type: torch.dtype,
) -> None:
    """Sync-free BatchMLAPagedAttentionWrapper.plan for the target-verify CUDA
    graph replay. Like fast_mla_decode_plan it hands host-known qo/kv indptr +
    lengths straight to _cached_module.plan (no per-replay device-to-host copy).
    Decode's indices updater writes the cuda-graph buffers in place so its fast
    plan can skip them; verify metadata is freshly built each step, so refresh
    the bound buffers here exactly as stock plan()'s use_cuda_graph branch does
    (host->device / device->device, non-blocking).
    """
    self._causal = causal
    self._page_size = page_size
    self._sm_scale = sm_scale
    self._qo_indptr_buf.copy_(qo_indptr_cpu, non_blocking=True)
    self._kv_indptr_buf.copy_(kv_indptr_cpu, non_blocking=True)
    self._kv_indices_buf[: len(kv_indices)].copy_(kv_indices, non_blocking=True)
    self._kv_len_arr_buf.copy_(kv_len_arr_cpu, non_blocking=True)

    try:
        self._cached_module.plan(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_cpu,
            kv_indptr_cpu,
            kv_len_arr_cpu,
            num_heads,
            head_dim_ckv,
            causal,
        )
    except Exception as e:
        raise RuntimeError(f"Error in alternate MLA prefill plan: {e}")
