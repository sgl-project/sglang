"""Step5 attention backend: SWA child plus DSA indexer metadata and sparse attention."""

from __future__ import annotations

import msgspec
import torch
import triton

from sglang.kernels.ops.attention.deep_select import is_deep_select_supported
from sglang.kernels.ops.attention.step5_dsa_indexer import (
    STEP5_PAGE_SIZE,
    deepselect_score_width,
)
from sglang.kernels.ops.attention.step5_dsa_sparse_attention import (
    step5_dsa_sparse_attention,
)
from sglang.srt.configs.step5 import (
    Step5AttentionProfile,
    parse_step5_attention_profile,
)
from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    SharedReadEnds,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_spec
from sglang.srt.utils import is_cuda
from sglang.srt.utils.device import async_h2d

# 512 MiB FP32 score budget, independent of packed prefill length.
STEP5_SCORE_WORKSPACE_ELEMENTS = 128 * 1024 * 1024


class Step5LayerIndexerState(msgspec.Struct, frozen=True, kw_only=True):
    compressed_index_keys: torch.Tensor
    index_key_ring: torch.Tensor
    compression_logits_ring: torch.Tensor
    index_positions: torch.Tensor


class Step5CompressionPlan(msgspec.Struct, frozen=True, kw_only=True):
    """Compression access plan shared by every DSA layer of one forward.

    R = request rows, G = maximum completed blocks per request. Request rows need
    not equal query rows: prefill packs a variable number of tokens per request.
    """

    request_slots: torch.Tensor  # [R], request-pool row, zero means padding
    input_start_positions: torch.Tensor  # [R], absolute logical token position
    input_start_rows: torch.Tensor  # [R], offset into this forward's packed K/Z
    input_lengths: torch.Tensor  # [R], number of input K/Z rows for the request
    # [G, R], flattened compressed-key address, NOT a physical KV page number;
    # ordinal g names block floor(input_start_positions / region size) + g.
    compressed_key_write_indices: torch.Tensor
    num_input_tokens: int  # capacity of this forward's packed K/Z, including padding
    # Input validity, shared by all layers; it does not cover layer-local ring tags.
    request_active: torch.Tensor  # bool [R], live request row, excluding slot 0
    request_mapping_valid: torch.Tensor  # bool [R], admission from input checks
    request_count_valid: torch.Tensor  # bool [1], live count fits request capacity

    @classmethod
    def for_prefill(
        cls,
        *,
        request_slots: torch.Tensor,
        prefix_lengths: list[int],
        append_lengths: list[int],
        req_to_token: torch.Tensor,
        compressed_index_key_pages: int,
        region_size: int,
    ) -> Step5CompressionPlan:
        """A live request with an invalid completed block fails the ring admission
        assert; padding rows consume input rows but never own ring cells."""
        rows, source = [], 0
        for base, length in zip(prefix_lengths, append_lengths):
            rows.append((base, source, length))
            source += length
        device = req_to_token.device
        raw_slots = torch.as_tensor(request_slots, dtype=torch.int64, device=device)
        slots = raw_slots.clamp(0, req_to_token.shape[0] - 1)
        packed = async_h2d(rows, dtype=torch.int64, device=device).reshape(-1, 3)
        bases, sources, lengths = packed.T.contiguous().unbind(0)
        active = (raw_slots != 0) & (lengths > 0)
        uses = torch.zeros(req_to_token.shape[0], dtype=torch.int32, device=device)
        uses.scatter_add_(0, slots, active.to(torch.int32))
        admitted = (raw_slots == slots) & (uses[slots] == 1)
        ratio = region_size
        max_blocks = max(
            ((row[0] % ratio + row[2]) // ratio for row in rows), default=0
        )
        block_idx = torch.arange(max_blocks, device=device)[:, None]
        offsets = torch.arange(ratio, device=device)
        logical = (bases // ratio + block_idx) * ratio
        complete = logical + ratio <= bases + lengths
        mapped = req_to_token[
            slots[None, :, None],
            (logical[:, :, None] + offsets).clamp_max(req_to_token.shape[1] - 1),
        ].long()
        first = mapped[:, :, 0]
        valid = (
            (first >= STEP5_PAGE_SIZE)
            & (first < compressed_index_key_pages * STEP5_PAGE_SIZE)
            & (first.remainder(ratio) == 0)
            & (mapped == first[:, :, None] + offsets).all(2)
        )
        return cls(
            request_slots=slots,
            input_start_positions=bases,
            input_start_rows=sources,
            input_lengths=lengths,
            request_active=active,
            request_mapping_valid=admitted & (~complete | valid).all(0),
            request_count_valid=torch.ones(1, dtype=torch.bool, device=device),
            compressed_key_write_indices=(first // ratio).contiguous(),
            num_input_tokens=source,
        )

    @classmethod
    def for_decode(
        cls,
        *,
        request_slots: torch.Tensor,
        positions: torch.Tensor,
        valid_rows: torch.Tensor,
        req_to_token: torch.Tensor,
        compressed_index_key_pages: int,
        context_len: int,
        region_size: int,
    ) -> tuple[Step5CompressionPlan, torch.Tensor, torch.Tensor]:
        """Decode plan with one input token per request, built on device only so it
        can be captured; also returns int32 per-query request slots and positions."""
        device = req_to_token.device
        num_slots, num_tokens = req_to_token.shape
        rows = request_slots.numel()
        raw_slots = request_slots.to(torch.int64)
        slots = raw_slots.clamp(0, num_slots - 1)
        live = valid_rows.to(torch.int64)
        active = (torch.arange(rows, device=device) < live) & (raw_slots != 0)
        count_valid = (live >= 0) & (live <= rows)
        uses = torch.zeros(num_slots, dtype=torch.int32, device=device)
        uses.index_add_(0, slots, active.to(torch.int32))
        bases = positions.to(torch.int64)
        safe_positions = bases.clamp(0, num_tokens - 1)
        physical_limit = compressed_index_key_pages * STEP5_PAGE_SIZE
        current = req_to_token[slots, safe_positions]
        query_valid = (
            (bases >= 0)
            & (bases < min(num_tokens, context_len))
            & (current >= STEP5_PAGE_SIZE)
            & (current < physical_limit)
        )
        offsets = torch.arange(region_size, device=device)
        logical = (safe_positions // region_size * region_size)[:, None] + offsets
        mapped = req_to_token[slots[:, None], logical.clamp_max(num_tokens - 1)]
        first = mapped[:, 0].to(torch.int64)
        block_valid = (
            (first >= STEP5_PAGE_SIZE)
            & (first < physical_limit)
            & (first.remainder(region_size) == 0)
            & (mapped == first[:, None] + offsets).all(1)
        )
        completes_block = bases.remainder(region_size) == region_size - 1
        admitted = (
            count_valid
            & (slots == raw_slots)
            & (uses[slots] == 1)
            & query_valid
            & (~completes_block | block_valid)
        )
        plan = cls(
            request_slots=slots,
            input_start_positions=safe_positions,
            input_start_rows=torch.arange(rows, dtype=torch.int64, device=device),
            input_lengths=torch.ones(rows, dtype=torch.int64, device=device),
            request_active=active,
            request_mapping_valid=admitted,
            request_count_valid=count_valid,
            compressed_key_write_indices=(
                (first - first.remainder(region_size)) // region_size
            )[None, :],
            num_input_tokens=rows,
        )
        query_slots = torch.where(active, raw_slots, 0).to(torch.int32)
        return plan, query_slots, safe_positions.to(torch.int32)

    def prefill_query_layout(self) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = self.input_lengths
        device = lengths.device
        token_requests = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=device),
            lengths,
            output_size=self.num_input_tokens,
        )
        requests = self.request_slots[token_requests].to(torch.int32)
        positions = (
            self.input_start_positions[token_requests]
            + torch.arange(self.num_input_tokens, device=device)
            - self.input_start_rows[token_requests]
        ).to(torch.int32)
        return requests, torch.where(requests != 0, positions, 0)


class Step5SelectionWorkspace(msgspec.Struct, frozen=True, kw_only=True):
    """Per-forward selection scratch, reused by every DSA layer in turn.

    packed/counts/lengths have one row per query; indexer_logits/ids hold one row tile.
    """

    packed: torch.Tensor
    counts: torch.Tensor
    lengths: torch.Tensor
    indexer_logits: torch.Tensor
    ids: torch.Tensor
    tiles: tuple[tuple[int, int], ...]
    use_deepselect: bool = False
    topk_lengths: torch.Tensor | None = None
    topk_ids: torch.Tensor | None = None

    @classmethod
    def allocate(
        cls,
        *,
        rows: int,
        score_width: int,
        max_score_width: int,
        row_to_req: torch.Tensor,
        positions: torch.Tensor,
        topk: int,
        region_size: int,
    ) -> Step5SelectionWorkspace:
        device = positions.device
        use_deepselect = score_width > topk
        if use_deepselect:
            score_width = deepselect_score_width(score_width)
            if score_width > deepselect_score_width(max_score_width):
                raise ValueError("Step5 score width exceeds the request capacity")
        capacity = min(rows, STEP5_SCORE_WORKSPACE_ELEMENTS // max(score_width, 1))
        workspace = cls(
            packed=torch.empty((rows, topk + 1), dtype=torch.int64, device=device),
            counts=torch.empty(rows, dtype=torch.int32, device=device),
            lengths=torch.empty(rows, dtype=torch.int32, device=device),
            indexer_logits=torch.empty(
                (capacity, score_width), dtype=torch.float32, device=device
            ),
            ids=torch.empty((capacity, topk + 1), dtype=torch.int32, device=device),
            use_deepselect=use_deepselect,
            topk_lengths=(
                torch.empty(rows, dtype=torch.int32, device=device)
                if use_deepselect
                else None
            ),
            topk_ids=(
                torch.empty((capacity, topk), dtype=torch.int32, device=device)
                if use_deepselect
                else None
            ),
            tiles=cls._tiles(rows=rows, capacity=capacity),
        )
        return workspace.for_rows(
            rows=rows,
            row_to_req=row_to_req,
            positions=positions,
            region_size=region_size,
        )

    def for_rows(
        self,
        *,
        rows: int,
        row_to_req: torch.Tensor,
        positions: torch.Tensor,
        region_size: int,
    ) -> Step5SelectionWorkspace:
        capacity = min(rows, self.indexer_logits.shape[0])
        topk_lengths = None
        if self.use_deepselect:
            topk_lengths = self.topk_lengths[:rows]
            topk_lengths.copy_(torch.where(row_to_req > 0, positions // region_size, 0))
        return Step5SelectionWorkspace(
            packed=self.packed[:rows],
            counts=self.counts[:rows],
            lengths=self.lengths[:rows],
            indexer_logits=self.indexer_logits[:capacity],
            ids=self.ids[:capacity],
            use_deepselect=self.use_deepselect,
            topk_lengths=topk_lengths,
            topk_ids=self.topk_ids[:capacity] if self.use_deepselect else None,
            tiles=self._tiles(rows=rows, capacity=capacity),
        )

    @staticmethod
    def _tiles(*, rows: int, capacity: int) -> tuple[tuple[int, int], ...]:
        return tuple(
            (start, min(start + capacity, rows))
            for start in range(0, rows, capacity or 1)
        )


class Step5IndexerMetadata(msgspec.Struct, frozen=True, kw_only=True):
    """Indexer metadata for one forward; R = compression request rows, Q = query rows."""

    # Borrowed references; storage and lifetime remain owned by the backend and pools.
    layer_states: dict[int, Step5LayerIndexerState]
    req_to_token: torch.Tensor
    compression: Step5CompressionPlan

    # Query/scoring layout, complete at construction (no later replace/binding).
    query_request_slots: torch.Tensor  # [Q], request slot for each query
    query_positions: torch.Tensor  # [Q], absolute logical query positions

    # Per-layer scratch: _admit overwrites both buffers for every request row,
    # so they must not be cached as input validity.
    compression_enabled: torch.Tensor  # bool [R], input AND layer-ring admission
    compression_success: torch.Tensor  # bool [R], success (including inert padding)
    selection_workspace: Step5SelectionWorkspace


def wrap_step5_attention_backend(runner, full_attn_backend: AttentionBackend):
    """Compose target sparse attention with the configured public SWA backend."""
    if get_spec().speculative_algorithm is not None:
        raise ValueError("Step5 sparse attention does not support speculative decoding")
    return Step5AttnBackend(
        swa_backend=full_attn_backend,
        kv_pool=runner.token_to_kv_pool,
        req_to_token_pool=runner.req_to_token_pool,
        context_len=runner.model_config.context_len,
        max_context_len=runner.model_config.hf_text_config.max_position_embeddings,
        profile=parse_step5_attention_profile(runner.model_config.hf_text_config),
    )


def _validate_config(*, kv_pool, context_len: int, max_context_len: int) -> None:
    if not is_cuda():
        raise ValueError("Step5 sparse attention requires NVIDIA CUDA")
    if not isinstance(kv_pool, SWAKVPool):
        raise ValueError("Step5 sparse attention requires the hybrid SWA KV pool")
    if kv_pool.page_size != STEP5_PAGE_SIZE:
        raise ValueError(f"Step5 indexer kernels require page size {STEP5_PAGE_SIZE}")
    if kv_pool.full_kv_pool.dtype != torch.bfloat16:
        raise ValueError("Step5 sparse attention requires BF16 KV cache")
    if kv_pool.post_capture_active:
        raise ValueError("Step5 indexer state requires an allocated full KV pool")
    if not is_deep_select_supported():
        raise ValueError("Step5 block selection requires a DeepSelect-capable GPU")
    if context_len > max_context_len:
        raise ValueError(f"Step5 context length must be at most {max_context_len}")


def _allocate_indexer_states(
    *, kv_pool: SWAKVPool, num_req_slots: int, region_size: int, index_head_dim: int
) -> dict[int, Step5LayerIndexerState]:
    full_pool = kv_pool.full_kv_pool
    device = full_pool.device
    num_layers = kv_pool.full_layer_nums
    num_pages = full_pool.size // kv_pool.page_size + 1
    ring_shape = (
        num_layers,
        num_req_slots,
        region_size,
        1,
        index_head_dim,
    )
    compressed_index_keys = torch.empty(
        (
            num_layers,
            num_pages,
            kv_pool.page_size // region_size,
            1,
            index_head_dim,
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
        (num_layers, num_req_slots, region_size),
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


class Step5AttnBackend(AttentionBackend):
    plans_padded_extend_rows = True

    def __init__(
        self,
        *,
        swa_backend: AttentionBackend,
        kv_pool: SWAKVPool,
        req_to_token_pool,
        context_len: int,
        max_context_len: int,
        profile: Step5AttentionProfile,
    ):
        _validate_config(
            kv_pool=kv_pool, context_len=context_len, max_context_len=max_context_len
        )
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
        self.region_size = profile.region_size
        self.topk = profile.topk
        self.full_num_pages = kv_pool.full_kv_pool.size // kv_pool.page_size + 1
        self.indexer_states = _allocate_indexer_states(
            kv_pool=kv_pool,
            num_req_slots=req_to_token_pool.req_to_token.shape[0],
            region_size=profile.region_size,
            index_head_dim=profile.index_head_dim,
        )
        self.dsa_layer_ids = frozenset(self.indexer_states)
        # Metadata storage only. CUDA graph objects belong to the public runner.
        self._cuda_graph_metadata: dict[int, Step5IndexerMetadata] = {}
        self._graph_valid_rows: torch.Tensor | None = None
        self._graph_workspace: Step5SelectionWorkspace | None = None
        self.forward_metadata: Step5IndexerMetadata | None = None

    def window_capacity_children(self) -> list:
        return [self.swa_backend]

    @property
    def needs_cpu_seq_lens(self) -> bool:
        return self.swa_backend.needs_cpu_seq_lens

    def shared_read_ends(self, fm: ForwardMode) -> SharedReadEnds:
        return SharedReadEnds.max_of(
            (SharedReadEnds.POST_REPLAY, self.swa_backend.shared_read_ends(fm))
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        self.forward_metadata = None
        mode = forward_batch.forward_mode
        if not (
            mode.is_decode_or_idle() or mode in (ForwardMode.EXTEND, ForwardMode.MIXED)
        ):
            raise ValueError(
                f"Step5 sparse attention does not support forward mode {mode.name}; "
                "speculative decoding is not supported"
            )
        if mode.is_idle():
            return
        if mode.is_decode():
            self.forward_metadata = self._build_eager_decode_metadata(forward_batch)
        else:
            self.forward_metadata = self._build_prefill_metadata(forward_batch)
        self.swa_backend.init_forward_metadata(forward_batch)

    def _build_indexer_metadata(
        self,
        *,
        plan: Step5CompressionPlan,
        row_to_req: torch.Tensor,
        positions: torch.Tensor,
        score_width: int,
        selection_workspace: Step5SelectionWorkspace | None = None,
    ) -> Step5IndexerMetadata:
        req_to_token = self.req_to_token_pool.req_to_token
        if selection_workspace is None:
            selection_workspace = Step5SelectionWorkspace.allocate(
                rows=positions.numel(),
                score_width=score_width,
                max_score_width=req_to_token.shape[1] // self.region_size,
                row_to_req=row_to_req,
                positions=positions,
                topk=self.topk,
                region_size=self.region_size,
            )
        return Step5IndexerMetadata(
            layer_states=self.indexer_states,
            req_to_token=req_to_token,
            compression=plan,
            query_request_slots=row_to_req,
            query_positions=positions,
            selection_workspace=selection_workspace,
            compression_enabled=torch.empty_like(plan.request_slots, dtype=torch.bool),
            compression_success=torch.empty_like(plan.request_slots, dtype=torch.bool),
        )

    def _build_prefill_metadata(self, fb: ForwardBatch) -> Step5IndexerMetadata:
        req_to_token = self.req_to_token_pool.req_to_token
        prefixes = list(fb.extend_prefix_lens_cpu)
        lengths = list(fb.extend_seq_lens_cpu)
        slots = fb.req_pool_indices
        padded_tokens = fb.input_ids.numel()
        if sum(lengths) != padded_tokens:
            # Prefill graphs and DP attention pad tokens without adding a request; give
            # that tail an ownerless padding row so it never extends a live frontier.
            prefixes.append(0)
            lengths.append(padded_tokens - sum(lengths))
            slots = torch.cat((slots, slots.new_zeros(1)))
        plan = Step5CompressionPlan.for_prefill(
            request_slots=slots,
            prefix_lengths=prefixes,
            append_lengths=lengths,
            req_to_token=req_to_token,
            compressed_index_key_pages=self.full_num_pages,
            region_size=self.region_size,
        )
        requests, positions = plan.prefill_query_layout()
        score_width = max(
            1,
            triton.cdiv(
                max((p + n for p, n in zip(prefixes, lengths)), default=0),
                self.region_size,
            ),
        )
        return self._build_indexer_metadata(
            plan=plan,
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
        self,
        fb: ForwardBatch,
        *,
        valid_rows: torch.Tensor,
        graph_workspace: Step5SelectionWorkspace | None = None,
    ) -> Step5IndexerMetadata:
        plan, row_to_req, positions = Step5CompressionPlan.for_decode(
            request_slots=fb.req_pool_indices,
            positions=fb.positions,
            valid_rows=valid_rows,
            req_to_token=self.req_to_token_pool.req_to_token,
            compressed_index_key_pages=self.full_num_pages,
            context_len=self.context_len,
            region_size=self.region_size,
        )
        return self._build_indexer_metadata(
            plan=plan,
            row_to_req=row_to_req,
            positions=positions,
            score_width=triton.cdiv(self.context_len, self.region_size),
            selection_workspace=(
                None
                if graph_workspace is None
                else graph_workspace.for_rows(
                    rows=positions.numel(),
                    row_to_req=row_to_req,
                    positions=positions,
                    region_size=self.region_size,
                )
            ),
        )

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ) -> None:
        bs = forward_batch.batch_size
        if in_capture:
            self._graph_valid_rows.zero_()
            # The in-graph hook binds the final capture's tensors after warmup.
            self._cuda_graph_metadata.pop(bs, None)
            self.forward_metadata = None
        else:
            self._graph_valid_rows.fill_(bs - forward_batch.num_padding)
            self.forward_metadata = self._cuda_graph_metadata[bs]
        self.swa_backend.init_forward_metadata_out_graph(
            forward_batch, in_capture=in_capture
        )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        self.forward_metadata = self._build_decode_metadata(
            forward_batch,
            valid_rows=self._graph_valid_rows,
            graph_workspace=self._graph_workspace,
        )
        self._cuda_graph_metadata[forward_batch.batch_size] = self.forward_metadata
        self.swa_backend.init_forward_metadata_in_graph(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        req_to_token = self.req_to_token_pool.req_to_token
        if (
            triton.cdiv(self.context_len, self.region_size) * self.region_size
            > req_to_token.shape[1]
        ):
            raise ValueError("Step5 graph context exceeds native request capacity")
        self._cuda_graph_metadata.clear()
        self.forward_metadata = None
        self._graph_valid_rows = torch.zeros(
            1, dtype=torch.int32, device=req_to_token.device
        )
        self._graph_workspace = Step5SelectionWorkspace.allocate(
            rows=max_bs,
            score_width=triton.cdiv(self.context_len, self.region_size),
            max_score_width=req_to_token.shape[1] // self.region_size,
            row_to_req=torch.zeros(
                max_bs, dtype=torch.int32, device=req_to_token.device
            ),
            positions=torch.zeros(
                max_bs, dtype=torch.int32, device=req_to_token.device
            ),
            topk=self.topk,
            region_size=self.region_size,
        )
        self.swa_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return self.swa_backend.get_cuda_graph_seq_len_fill_value()

    def get_indexer_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> Step5IndexerMetadata | None:
        return self.forward_metadata

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if layer.layer_id in self.dsa_layer_ids:
            return self._forward_dsa(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )
        return self.swa_backend.forward(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
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
            # SWA writes only the real rows; keep the padded tail finite
            # before residuals and subsequent KV writes.
            real_tokens = forward_batch._original_num_tokens
            if real_tokens is not None and real_tokens < output.shape[0]:
                output[real_tokens:].zero_()
        return output

    def _forward_dsa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        *,
        index_selection: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if save_kv_cache:
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
        return step5_dsa_sparse_attention(
            q,
            k_cache,
            v_cache,
            packed=packed,
            counts=counts,
            lengths=lengths,
            out=torch.empty_like(q),
            scale=layer.scaling,
            region_size=self.region_size,
        )
