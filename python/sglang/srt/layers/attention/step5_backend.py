"""Step5 attention backend, forward/graph metadata, indexer state and execution."""

from __future__ import annotations

import torch

from sglang.kernels.ops.attention.step5.indexer import build_compression_metadata
from sglang.kernels.ops.attention.step5.sparse_attention import (
    step5_sparse_attention,
)
from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    SharedReadEnds,
)
from sglang.srt.layers.attention.graph_utils import padded_extend_real_tokens
from sglang.srt.layers.attention.step5.compressor import (
    build_prefill_compression_inputs,
)
from sglang.srt.layers.attention.step5.metadata import (
    Step5IndexerMetadata,
    Step5LayerIndexerState,
    allocate_selection_workspace,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_utils.prefill_graph import (
    is_in_full_prefill_graph,
    is_in_prefill_graph_scope,
)
from sglang.srt.runtime_context import get_spec

STEP5_PAGE_SIZE = 64
STEP5_COMPRESSION_RATIO = 8
STEP5_INDEX_HEAD_DIM = 256
STEP5_INDEX_RING_SIZE = 8


def wrap_step5_attention_backend(runner, full_attn_backend: AttentionBackend):
    """Compose target sparse attention with the configured public SWA backend."""
    if get_spec().speculative_algorithm is not None:
        raise ValueError("Step5 sparse attention does not support speculative decoding")
    return Step5AttentionBackend(
        swa_backend=full_attn_backend,
        kv_pool=runner.token_to_kv_pool,
        req_to_token_pool=runner.req_to_token_pool,
        context_len=runner.model_config.context_len,
    )


def _validate_kv_pool(kv_pool) -> None:
    if not isinstance(kv_pool, SWAKVPool):
        raise ValueError("Step5 sparse attention requires the hybrid SWA KV pool")
    if kv_pool.page_size != STEP5_PAGE_SIZE:
        raise ValueError(f"Step5 indexer kernels require page size {STEP5_PAGE_SIZE}")
    if kv_pool.full_kv_pool.dtype != torch.bfloat16:
        raise ValueError("Step5 sparse attention requires BF16 KV cache")
    if kv_pool.post_capture_active:
        raise ValueError("Step5 indexer state requires an allocated full KV pool")


def _allocate_indexer_states(
    *, kv_pool: SWAKVPool, num_req_slots: int
) -> dict[int, Step5LayerIndexerState]:
    full_pool = kv_pool.full_kv_pool
    device = full_pool.device
    num_layers = kv_pool.full_layer_nums
    num_pages = full_pool.size // kv_pool.page_size + 1
    ring_shape = (
        num_layers,
        num_req_slots,
        STEP5_INDEX_RING_SIZE,
        1,
        STEP5_INDEX_HEAD_DIM,
    )
    compressed_index_keys = torch.empty(
        (
            num_layers,
            num_pages,
            kv_pool.page_size // STEP5_COMPRESSION_RATIO,
            1,
            STEP5_INDEX_HEAD_DIM,
        ),
        dtype=torch.uint8,
        device=device,
    )
    # Pending index keys and compression logits for non-online R8,
    # indexed by layer, request slot and absolute-position residue.
    index_key_ring = torch.empty(ring_shape, dtype=torch.bfloat16, device=device)
    # Z contains per-channel compression logits, not block_idx scores.
    compression_logits_ring = torch.empty(
        ring_shape, dtype=torch.bfloat16, device=device
    )
    index_positions = torch.full(
        (num_layers, num_req_slots, STEP5_INDEX_RING_SIZE),
        -1,
        dtype=torch.int64,
        device=device,
    )
    return {
        layer_id: Step5LayerIndexerState(
            compressed_index_keys=compressed_index_keys[index],
            index_key_ring=index_key_ring[index],
            compression_logits_ring=compression_logits_ring[index],
            index_positions=index_positions[index],
        )
        for layer_id, (index, is_swa) in kv_pool.layers_mapping.items()
        if not is_swa
    }


def _check_forward_mode(mode: ForwardMode) -> None:
    if not (
        mode.is_decode_or_idle() or mode in (ForwardMode.EXTEND, ForwardMode.MIXED)
    ):
        raise ValueError(
            f"Step5 sparse attention does not support forward mode {mode.name}; "
            "speculative decoding is not supported"
        )


def _build_prefill_request_spans(fb: ForwardBatch):
    if fb.forward_mode not in (
        ForwardMode.EXTEND,
        ForwardMode.MIXED,
    ):
        raise ValueError("Step5 packed spans require target extend")
    if fb.extend_prefix_lens_cpu is None or fb.extend_seq_lens_cpu is None:
        raise ValueError("Step5 eager extend requires native CPU packed spans")
    prefixes = list(fb.extend_prefix_lens_cpu)
    lengths = list(fb.extend_seq_lens_cpu)
    if not (len(prefixes) == len(lengths) == fb.batch_size):
        raise ValueError("Step5 native request/token batch dimensions differ")
    tokens = sum(lengths)
    padded_tokens = fb.input_ids.numel()
    if tokens != padded_tokens:
        # Native DP pads token arrays to attention-TP alignment without adding
        # an extend request. Represent only that tail as ownerless CSA rows;
        # never extend a live request's logical frontier or mutate ForwardBatch.
        if (
            not fb.forward_mode.is_extend()
            or fb.dp_padding_mode is None
            or fb._original_num_tokens != tokens
            or padded_tokens < tokens
        ):
            raise ValueError("Step5 native request/token batch dimensions differ")
        prefixes.append(0)
        lengths.append(padded_tokens - tokens)
    return prefixes, lengths


def _prefill_query_layout(compression_inputs: dict):
    starts = compression_inputs["input_start_positions"]
    sources = compression_inputs["input_start_rows"]
    lengths = compression_inputs["input_lengths"]
    num_tokens = compression_inputs["num_input_tokens"]
    device = lengths.device
    token_requests = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=device),
        lengths,
        output_size=num_tokens,
    )
    requests = compression_inputs["request_slots"][token_requests].to(torch.int32)
    positions = (
        starts[token_requests]
        + torch.arange(num_tokens, device=device)
        - sources[token_requests]
    ).to(torch.int32)
    return requests, torch.where(requests != 0, positions, 0)


class Step5RadixAttention(RadixAttention):
    """Keep aligned CSA tail rows when dispatching to the Step5 backend."""

    def forward(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        key_value_num_tokens: int | None = None,
        **kwargs,
    ):
        backend = get_attn_backend()
        # Graph wrappers still own real-row slicing and output padding.
        if (
            isinstance(backend, Step5AttentionBackend)
            and not (
                self.use_prefill_attention_wrapper
                and is_in_prefill_graph_scope()
                and forward_batch.forward_mode.is_extend()
                and (is_in_breakable_cuda_graph() or is_in_full_prefill_graph())
            )
            and padded_extend_real_tokens(q.shape[0], forward_batch) is not None
        ):
            if k is not None:
                assert v is not None
                k = k.view(
                    -1,
                    self.tp_k_head_num,
                    self.v_head_dim if "k_rope" in kwargs else self.qk_head_dim,
                )
                if "k_rope" not in kwargs:
                    v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
            return backend.forward(
                q, k, v, self, forward_batch, save_kv_cache, **kwargs
            )
        return super().forward(
            q,
            k,
            v,
            forward_batch,
            save_kv_cache,
            key_value_num_tokens=key_value_num_tokens,
            **kwargs,
        )


class Step5AttentionBackend(AttentionBackend):
    def __init__(
        self,
        *,
        swa_backend: AttentionBackend,
        kv_pool: SWAKVPool,
        req_to_token_pool,
        context_len: int,
    ):
        _validate_kv_pool(kv_pool)
        self.swa_backend = swa_backend
        # Expose the child to the shared metadata snapshot/restore lifecycle.
        self.attn_backend_list = [swa_backend]
        self.runner_ids = swa_backend.runner_ids
        self.extend_dummy_seqs_capped_by_req_pool = (
            swa_backend.extend_dummy_seqs_capped_by_req_pool
        )
        self.token_to_kv_pool = kv_pool
        self.req_to_token_pool = req_to_token_pool
        self.context_len = context_len
        self.full_num_pages = kv_pool.full_kv_pool.size // kv_pool.page_size + 1
        self.indexer_states = _allocate_indexer_states(
            kv_pool=kv_pool,
            num_req_slots=req_to_token_pool.req_to_token.shape[0],
        )
        self.dsa_layer_ids = frozenset(self.indexer_states)
        # Metadata storage only. CUDA graph objects belong to the public runner.
        self._cuda_graph_max_bs = 0
        self._cuda_graph_metadata: dict[int, Step5IndexerMetadata] = {}
        self._graph_valid_rows: torch.Tensor | None = None
        self.forward_metadata: Step5IndexerMetadata | None = None

    def window_capacity_children(self) -> list:
        return [self.swa_backend]

    @property
    def needs_cpu_seq_lens(self):
        return self.swa_backend.needs_cpu_seq_lens

    def shared_read_ends(self, fm):
        return SharedReadEnds.max_of(
            (SharedReadEnds.POST_REPLAY, self.swa_backend.shared_read_ends(fm))
        )

    def init_forward_metadata(self, forward_batch):
        self.forward_metadata = None
        mode = forward_batch.forward_mode
        _check_forward_mode(mode)
        if mode.is_idle():
            return
        if mode.is_decode():
            self.forward_metadata = self._build_eager_decode_metadata(forward_batch)
        else:
            self.forward_metadata = self._build_prefill_metadata(forward_batch)
        self.swa_backend.init_forward_metadata(forward_batch)

    def _build_indexer_metadata(
        self, *, compression_inputs, row_to_req, positions, score_width
    ) -> Step5IndexerMetadata:
        req_to_token = self.req_to_token_pool.req_to_token
        request_slots = compression_inputs["request_slots"]
        return Step5IndexerMetadata(
            layer_states=self.indexer_states,
            req_to_token=req_to_token,
            **compression_inputs,
            query_request_slots=row_to_req,
            query_positions=positions,
            score_width=score_width,
            selection_workspace=allocate_selection_workspace(
                rows=positions.numel(),
                score_width=score_width,
                device=request_slots.device,
                max_score_width=req_to_token.shape[1] // 8,
                row_to_req=row_to_req,
                positions=positions,
            ),
            compression_enabled=torch.empty_like(request_slots, dtype=torch.bool),
            compression_success=torch.empty_like(request_slots, dtype=torch.bool),
        )

    def _build_prefill_metadata(self, fb: ForwardBatch) -> Step5IndexerMetadata:
        req_to_token = self.req_to_token_pool.req_to_token
        prefixes, lengths = _build_prefill_request_spans(fb)
        slots = fb.req_pool_indices
        if len(lengths) > fb.batch_size:
            slots = torch.cat((slots, slots.new_zeros(1)))
        compression_inputs = build_prefill_compression_inputs(
            request_slots=slots,
            prefix_lengths=prefixes,
            append_lengths=lengths,
            req_to_token=req_to_token,
            compressed_index_key_pages=self.full_num_pages,
        )
        requests, positions = _prefill_query_layout(compression_inputs)
        torch._assert_async(
            ((requests == 0) | (fb.positions == positions)).all(),
            "Step5 positions differ from native packed spans",
        )
        torch._assert_async(
            (fb.out_cache_loc == req_to_token[requests.long(), positions.long()]).all(),
            "Step5 write locations differ from FULL mapping",
        )
        score_width = max(
            1,
            (max((p + n for p, n in zip(prefixes, lengths)), default=0) + 7) // 8,
        )
        return self._build_indexer_metadata(
            compression_inputs=compression_inputs,
            row_to_req=requests,
            positions=positions,
            score_width=score_width,
        )

    def _build_eager_decode_metadata(self, fb: ForwardBatch) -> Step5IndexerMetadata:
        # Fill on device: constructing from a host list synchronizes the
        # current stream even though this metadata contains only one scalar.
        valid_rows = torch.full(
            (1,),
            (
                fb._original_batch_size
                if fb._original_batch_size is not None
                else fb.batch_size
            ),
            dtype=torch.int32,
            device=self.req_to_token_pool.req_to_token.device,
        )
        return self._build_decode_metadata(fb, valid_rows=valid_rows)

    def _build_decode_metadata(
        self, fb: ForwardBatch, *, valid_rows: torch.Tensor
    ) -> Step5IndexerMetadata:
        """Compute decode indices and compression plans on the device.

        Shared by eager and capture; replay executes the captured GPU operations.
        Decode positions name the current token (seq_len - 1).
        """
        request_slots = fb.req_pool_indices
        rows = request_slots.numel()
        if (
            rows < 1
            or fb.positions.shape != (rows,)
            or fb.seq_lens.shape != (rows,)
            or fb.out_cache_loc.shape != (rows,)
        ):
            raise ValueError("Step5 token metadata shape differs from its batch")
        (
            compression_inputs,
            row_to_req,
            positions,
            _,
            native_valid,
        ) = build_compression_metadata(
            request_slots=request_slots,
            positions=fb.positions,
            valid_rows=valid_rows,
            req_to_token=self.req_to_token_pool.req_to_token,
            compressed_index_key_pages=self.full_num_pages,
            native_positions=fb.positions,
            seq_lens=fb.seq_lens,
            out_cache_loc=fb.out_cache_loc,
            context_len=self.context_len,
        )
        torch._assert_async(
            native_valid.all(),
            "Step5 graph position/FULL mapping differs from native batch",
        )
        return self._build_indexer_metadata(
            compression_inputs=compression_inputs,
            row_to_req=row_to_req,
            positions=positions,
            score_width=(self.context_len + 7) // 8,
        )

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        bs = forward_batch.batch_size
        if (
            not forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.spec_info is not None
        ):
            raise ValueError("Step5 CUDA graph supports only decode/idle batches")
        if not 1 <= bs <= self._cuda_graph_max_bs:
            raise ValueError("Step5 graph batch exceeds initialized capacity")
        if in_capture:
            self._graph_valid_rows.zero_()
            # The in-graph hook binds the final capture's tensors after warmup.
            self._cuda_graph_metadata.pop(bs, None)
            self.forward_metadata = None
        else:
            # Resolve the bucket and validate the live count before any device write.
            if bs not in self._cuda_graph_metadata:
                raise ValueError("Step5 graph metadata bucket has not been captured")
            raw_bs = bs - forward_batch.num_padding
            if not 0 <= raw_bs <= bs:
                raise ValueError("Step5 live rows exceed graph capacity")
            self._graph_valid_rows.fill_(raw_bs)
            self.forward_metadata = self._cuda_graph_metadata[bs]
        self.swa_backend.init_forward_metadata_out_graph(
            forward_batch, in_capture=in_capture
        )

    def init_forward_metadata_in_graph(self, forward_batch):
        self.forward_metadata = self._build_decode_metadata(
            forward_batch, valid_rows=self._graph_valid_rows
        )
        self._cuda_graph_metadata[forward_batch.batch_size] = self.forward_metadata
        self.swa_backend.init_forward_metadata_in_graph(forward_batch)

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        if max_bs < 1 or max_num_tokens != max_bs:
            raise ValueError("Step5 decode graph requires one token per request")
        req_to_token = self.req_to_token_pool.req_to_token
        if (
            not 1 <= self.context_len <= 1048576
            or (self.context_len + 7) // 8 * 8 > req_to_token.shape[1]
        ):
            raise ValueError("Step5 graph context exceeds native request capacity")
        self._cuda_graph_max_bs = max_bs
        self._cuda_graph_metadata.clear()
        self.forward_metadata = None
        self._graph_valid_rows = torch.zeros(
            1, dtype=torch.int32, device=req_to_token.device
        )
        self.swa_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def get_cuda_graph_seq_len_fill_value(self):
        return self.swa_backend.get_cuda_graph_seq_len_fill_value()

    def get_indexer_metadata(self, layer_id: int, forward_batch: ForwardBatch):
        return self.forward_metadata

    def forward_decode(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        if layer.layer_id in self.dsa_layer_ids:
            return self._forward_dsa(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )
        return self.swa_backend.forward(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )

    def forward_extend(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        if layer.layer_id in self.dsa_layer_ids:
            return self._forward_dsa(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )
        output = self.swa_backend.forward(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )
        if (
            forward_batch.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED)
            and forward_batch.dp_padding_mode is not None
        ):
            # DP aligns token rows without adding an attention sequence. SWA
            # only writes the real rows, leaving the aligned tail uninitialized.
            # Keep that tail finite before residuals and subsequent KV writes.
            real_tokens = forward_batch._original_num_tokens
            if real_tokens is not None and real_tokens < output.shape[0]:
                output[real_tokens:].zero_()
        return output

    def _forward_dsa(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        save_kv_cache=True,
        *,
        index_selection,
    ):
        if self.forward_metadata is None:
            raise RuntimeError("Step5 backend metadata has not been initialized")
        if save_kv_cache:
            if k is None or v is None:
                raise ValueError("Step5 DSA KV store requires both K and V")
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                KVWriteLoc(forward_batch.out_cache_loc),
                k,
                v,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
            )
        k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        packed, counts, lengths = index_selection
        return step5_sparse_attention(
            q,
            k_cache,
            v_cache,
            packed=packed,
            counts=counts,
            lengths=lengths,
            out=torch.empty_like(q),
            scale=layer.scaling,
        )
