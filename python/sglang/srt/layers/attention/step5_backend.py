"""Step5 attention backend, forward/graph metadata, workspace and execution."""

from __future__ import annotations

import copy

import numpy as np
import torch

from sglang.kernels.ops.attention.step5.fp8_kv import (
    step5_fp8_kv_dequant_pages,
    step5_fp8_kv_remap_pages,
    step5_fp8_prefill_pages,
)
from sglang.kernels.ops.attention.step5.scoring import _assert_group32_ownership
from sglang.kernels.ops.attention.step5.sparse_decode import (
    build_step5_sparse_decode_schedule,
    step5_sparse_decode_selected,
)
from sglang.kernels.ops.attention.step5.sparse_prefill import (
    step5_sparse_prefill_selected,
    use_sm100_sparse_prefill,
)
from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    SharedReadEnds,
)
from sglang.srt.layers.attention.step5.compressor import (
    build_prefill_compression_inputs,
)
from sglang.srt.layers.attention.step5.metadata import (
    Step5AttentionMetadata,
    Step5IndexerMetadata,
    Step5PrefillWorkspace,
    query_group_spans,
)
from sglang.srt.layers.attention.step5.region_selection import (
    allocate_selection_workspace,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.step5_kv_pool import Step5TokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils.common import async_h2d

STEP5_SWA_GRAPH_REQUEST_SLOTS = 8


def _allocate_attention_buffers(
    *, rows, attention_shape, work, decode, groups, region_bins, device
):
    """Allocate only this forward's capacities, before entering decoder layers.

    Scratch never escapes attention; all consumers run on the model stream before
    the next layer overwrites it. Concurrent forwards must own distinct metadata.
    During capture these allocations belong to the native graph's memory pool.
    Attention outputs and persistent KV/ring state are deliberately not included.
    """
    heads, head_dim = attention_shape
    partial = partial_lse = prefill = None
    if decode and work != rows:
        partial = torch.empty(
            (work, heads, head_dim), dtype=torch.float32, device=device
        )
        partial_lse = torch.empty((work, heads), dtype=torch.float32, device=device)
    if not decode:
        capacity = min(16 * 513, region_bins)
        sm100_buffers = {}
        if use_sm100_sparse_prefill():
            blocks = (capacity + 15) // 16
            sm100_buffers = dict(
                sm100_block_counts=torch.empty(
                    groups, dtype=torch.int32, device=device
                ),
                sm100_block_indices=torch.arange(
                    blocks + 2, dtype=torch.int32, device=device
                )
                .expand(groups, -1)
                .clone(),
                sm100_full_block_counts=torch.zeros(
                    groups, dtype=torch.int32, device=device
                ),
                sm100_region_words=torch.empty(
                    (groups, blocks, 128 // heads), dtype=torch.int32, device=device
                ),
                sm100_token_limits=torch.empty(
                    (groups, 128 // heads), dtype=torch.int32, device=device
                ),
            )
        prefill = Step5PrefillWorkspace(
            **sm100_buffers,
            logical_starts=torch.empty((rows, 513), dtype=torch.int32, device=device),
            physical_regions=torch.empty((rows, 513), dtype=torch.int32, device=device),
            union_physical=torch.empty(
                (groups, capacity), dtype=torch.int32, device=device
            ),
            union_logical=torch.empty(
                (groups, capacity), dtype=torch.int32, device=device
            ),
            membership=torch.empty(
                (groups, capacity), dtype=torch.int32, device=device
            ),
            union_counts=torch.empty(groups, dtype=torch.int32, device=device),
            bitset=torch.empty(
                (groups, (region_bins + 31) // 32), dtype=torch.int32, device=device
            ),
        )
    return dict(
        lse=(
            None
            if not decode and use_sm100_sparse_prefill()
            else torch.empty((rows, heads), dtype=torch.float32, device=device)
        ),
        partial=partial,
        partial_lse=partial_lse,
        prefill=prefill,
    )


def _build_attention_metadata(
    *,
    token_to_kv_pool,
    req_to_token,
    attention_shape,
    compression_inputs,
    row_to_req,
    positions,
    score_width,
    score_group_rows,
    query_start_loc,
    group_start,
    group_end,
    region_bins,
    decode,
    valid_rows,
    prefill_spans=None,
) -> Step5AttentionMetadata:
    """Assemble complete attention/indexer metadata once before layer execution.

    Decode split scheduling depends only on shape/device, not tensor contents.
    Eager calls own their workspace; captured calls use the graph memory pool.
    No request-index calculation or input validation belongs here.
    """
    rows = positions.numel()
    schedule = dict(
        decode_dynamic_splits=False,
        decode_split4_rows=0,
        decode_split2_rows=0,
        decode_work_rows=rows,
    )
    if decode:
        schedule = build_step5_sparse_decode_schedule(
            rows=rows,
            request_rows=query_start_loc.numel() - 1,
            device_index=positions.get_device(),
        )
    return Step5AttentionMetadata(
        query_start_loc=query_start_loc,
        group_start=group_start,
        group_end=group_end,
        region_bins=region_bins,
        decode=decode,
        valid_rows=valid_rows,
        indexer_metadata=Step5IndexerMetadata(
            token_to_kv_pool=token_to_kv_pool,
            req_to_token=req_to_token,
            **compression_inputs,
            query_request_slots=row_to_req,
            query_positions=positions,
            score_width=score_width,
            score_group_rows=score_group_rows,
            selection_workspace=allocate_selection_workspace(
                rows=rows,
                score_width=score_width,
                score_group_rows=score_group_rows,
                device=compression_inputs["request_slots"].device,
                prefill=not decode,
                max_score_width=req_to_token.shape[1] // 8,
                prefill_spans=prefill_spans,
                req_to_token=req_to_token,
                row_to_req=row_to_req,
                positions=positions,
            ),
            compression_enabled=torch.empty_like(
                compression_inputs["request_slots"], dtype=torch.bool
            ),
            compression_success=torch.empty_like(
                compression_inputs["request_slots"], dtype=torch.bool
            ),
        ),
        **schedule,
        **_allocate_attention_buffers(
            rows=rows,
            attention_shape=attention_shape,
            work=schedule["decode_work_rows"],
            decode=decode,
            groups=group_start.numel(),
            region_bins=region_bins,
            device=positions.device,
        ),
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


class Step5AttentionBackend(AttentionBackend):
    handles_padded_extend = True

    def __init__(
        self,
        *,
        swa_backend,
        kv_pool: Step5TokenToKVPool,
        req_to_token_pool,
        context_len: int,
        position_padding: int = 0,
        speculative_eagle_topk: int = 1,
        speculative_num_steps: int = 0,
        speculative_num_draft_tokens: int = 1,
    ):
        if not isinstance(kv_pool, Step5TokenToKVPool):
            raise ValueError("Step5 backend requires Step5TokenToKVPool")
        # Configuration is fixed for this backend's lifetime. The verify width
        # includes the initial target input; draft workers use native SWA and
        # do not interpret draft-extend rows through this target-only contract.
        if type(speculative_eagle_topk) is not int or speculative_eagle_topk != 1:
            raise ValueError("Step5 supports only chain speculation (topk=1)")
        width = speculative_num_draft_tokens
        if (
            type(width) is not int
            or width < 1
            or type(speculative_num_steps) is not int
            or speculative_num_steps + 1 != width
        ):
            raise ValueError(
                "Step5 chain requires positive input width and steps + 1 = width"
            )
        if width > kv_pool.index_ring_size - (kv_pool.compression_ratio - 1):
            raise ValueError("Step5 verify width exceeds ring retention capacity")
        if context_len < width:
            raise ValueError("Step5 context must contain the verify window")
        self._verify_width = width
        self.swa_backend = swa_backend
        self.token_to_kv_pool = kv_pool
        self.req_to_token_pool = req_to_token_pool
        self.context_len = context_len
        self.position_capacity = context_len + position_padding
        self.attention_shape = (swa_backend.num_attention_heads, swa_backend.head_dim)
        self.fp8_full_kv = getattr(kv_pool, "full_kv_fp8", False)
        self._fp8_prefill = None
        # Metadata storage only. CUDA graph objects belong to the public runner.
        self._cuda_graph_max_bs = 0
        self._cuda_graph_metadata = {}
        self.forward_metadata = None
        self.use_captured_forward_metadata_for_breakable_cuda_graph = (
            getattr(swa_backend, "fa_impl_ver", None) == 4
            and getattr(swa_backend, "use_sliding_window_kv_pool", False)
            and hasattr(swa_backend, "_init_full_cg_prefill_metadata")
        )
        self.max_breakable_prefill_graph_batch_size = (
            STEP5_SWA_GRAPH_REQUEST_SLOTS
            if self.use_captured_forward_metadata_for_breakable_cuda_graph
            else None
        )
        self.swa_graph_active = False
        self._swa_graph_buffers = None

    def init_forward_metadata_for_breakable_cuda_graph_capture(self, forward_batch):
        self.init_forward_metadata(forward_batch)
        self._init_swa_graph_metadata(forward_batch, in_capture=True)
        return self.forward_metadata

    def prepare_forward_metadata_for_breakable_cuda_graph_replay(
        self, capture_metadata, forward_batch, *, static_forward_batch=None
    ):
        self.init_forward_metadata(forward_batch)
        self._init_swa_graph_metadata(forward_batch, in_capture=False)

    def _init_swa_graph_metadata(self, forward_batch, *, in_capture):
        slots = STEP5_SWA_GRAPH_REQUEST_SLOTS
        bs = forward_batch.batch_size
        if bs > slots:
            raise ValueError("Step5 SWA graph batch exceeds its request slots")
        device = forward_batch.seq_lens.device
        if self._swa_graph_buffers is None:
            self._swa_graph_buffers = {
                "seq_lens": torch.zeros(slots, dtype=torch.int64, device=device),
                "extend_seq_lens": torch.zeros(slots, dtype=torch.int64, device=device),
                "extend_prefix_lens": torch.zeros(
                    slots, dtype=torch.int64, device=device
                ),
                "req_pool_indices": torch.zeros(
                    slots, dtype=torch.int64, device=device
                ),
                "seq_lens_cpu": torch.zeros(slots, dtype=torch.int64),
            }
        buffers = self._swa_graph_buffers
        for name in (
            "seq_lens",
            "extend_seq_lens",
            "extend_prefix_lens",
            "req_pool_indices",
        ):
            buffers[name][bs:].zero_()
            buffers[name][:bs].copy_(getattr(forward_batch, name)[:bs])
        buffers["seq_lens_cpu"].zero_()
        buffers["seq_lens_cpu"][:bs].copy_(forward_batch.seq_lens_cpu[:bs])
        view = copy.copy(forward_batch)
        view.batch_size = slots
        view.seq_lens = buffers["seq_lens"]
        view.seq_lens_cpu = buffers["seq_lens_cpu"]
        view.extend_seq_lens = buffers["extend_seq_lens"]
        view.extend_prefix_lens = buffers["extend_prefix_lens"]
        view.req_pool_indices = buffers["req_pool_indices"]
        self.swa_backend._init_full_cg_prefill_metadata(view, in_capture)
        self.swa_graph_active = True

    def forward_swa_in_graph(self, q, k, v, layer, forward_batch, **kwargs):
        output = self.swa_backend.forward(
            q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
        )
        output = output.view(q.shape[0], -1)
        valid = self.swa_backend.forward_metadata.cu_seqlens_q[-1:]
        rows = torch.arange(output.shape[0], device=output.device)
        return torch.where((rows < valid)[:, None], output, output.new_zeros(()))

    @property
    def needs_cpu_seq_lens(self):
        # FULL/indexer metadata uses native device lengths in eager and graph
        # paths. Preserve the wrapped SWA backend's requirements (e.g. sharding)
        # and let the common worker aggregate all target/draft backends.
        return getattr(self.swa_backend, "needs_cpu_seq_lens", True)

    @property
    def verify_mask(self):
        # The top-1 indexer path is causal and does not read a tree mask.
        # Reuse SWA's public mask contract, including its graph-sized buffer,
        # instead of making the common worker allocate a FULL mask per round.
        return self.swa_backend.verify_mask

    @property
    def max_context_len(self):
        # Public speculative mask allocation uses this bound when no host
        # length sum is available; it must never pull lengths back from CUDA.
        return self.context_len

    def shared_read_ends(self, fm):
        return SharedReadEnds.POST_REPLAY

    def update_verify_buffers_to_fill_after_draft(self, spec_info, cuda_graph_bs):
        # Chain positions are copied by the public runner before the in-graph
        # GPU plan runs. There is no draft-dependent tree mask to rebuild here.
        # Bucket/layout validation belongs to the metadata capture/replay entry.
        pass

    def init_forward_metadata(self, forward_batch):
        self.forward_metadata = None
        if forward_batch.forward_mode.is_idle():
            return
        if forward_batch.forward_mode.is_target_verify():
            if (
                forward_batch.spec_info is None
                or forward_batch.spec_info.draft_token_num != self._verify_width
            ):
                raise ValueError(
                    "Step5 verify width differs from configured chain width"
                )
        fb = forward_batch
        req_to_token = self.req_to_token_pool.req_to_token
        compressed_index_key_pages = self.token_to_kv_pool.full_num_pages
        verify = fb.forward_mode.is_target_verify()
        if verify or fb.forward_mode.is_decode():
            # Graph misses consume the same native device lengths, position and
            # FULL mapping as replay. Own scratch per forward so overlap cannot
            # overwrite an in-flight reader. Capacity is bounded by model context;
            # do not synchronize to obtain the current maximum sequence length.
            width = self._verify_width if verify else 1
            device = req_to_token.device
            valid_rows = torch.tensor(
                [
                    (
                        fb._original_batch_size
                        if fb._original_batch_size is not None
                        else fb.batch_size
                    )
                ],
                dtype=torch.int32,
                device=device,
            )
            indices = self._build_decode_verify_index_plan(
                request_slots=fb.req_pool_indices,
                positions=fb.positions,
                seq_lens=fb.seq_lens,
                out_cache_loc=fb.out_cache_loc,
                verify=verify,
                token_width=width,
                valid_rows=valid_rows,
                boundaries=torch.arange(
                    fb.batch_size + 1, dtype=torch.int32, device=device
                )
                * width,
                empty_groups=torch.empty(0, dtype=torch.int32, device=device),
            )
            self.forward_metadata = _build_attention_metadata(
                token_to_kv_pool=self.token_to_kv_pool,
                req_to_token=self.req_to_token_pool.req_to_token,
                attention_shape=self.attention_shape,
                **indices,
            )
        else:
            prefixes, lengths = _build_prefill_request_spans(fb)
            slots = fb.req_pool_indices
            if len(lengths) > fb.batch_size:
                slots = torch.cat((slots, slots.new_zeros(1)))
            compression_inputs = build_prefill_compression_inputs(
                request_slots=slots,
                prefix_lengths=prefixes,
                append_lengths=lengths,
                req_to_token=req_to_token,
                compressed_index_key_pages=compressed_index_key_pages,
            )
            device = req_to_token.device
            query_group = min(16, 128 // self.attention_shape[0])
            starts, ends, boundaries = query_group_spans(
                lengths=lengths, query_group=query_group
            )
            batch = len(lengths)
            totals = [p + n for p, n in zip(prefixes, lengths)][: fb.batch_size]
            staged64 = async_h2d(
                np.concatenate(
                    (
                        np.asarray(prefixes, dtype=np.int64),
                        np.asarray(lengths, dtype=np.int64),
                        boundaries[:-1],
                        np.asarray(totals, dtype=np.int64),
                    )
                ),
                dtype=torch.int64,
                device=device,
            )
            staged32 = async_h2d(
                np.concatenate(
                    (
                        boundaries,
                        starts,
                        ends,
                        np.asarray([fb.input_ids.numel()], dtype=np.int64),
                    )
                ),
                dtype=torch.int32,
                device=device,
            )
            spans = staged64[: 3 * batch].view(3, batch)
            totals_t = staged64[3 * batch :]
            query_start_loc = staged32[: len(boundaries)]
            group_start = staged32[len(boundaries) : len(boundaries) + len(starts)]
            group_end = staged32[
                len(boundaries) + len(starts) : len(boundaries)
                + len(starts)
                + len(ends)
            ]
            valid_rows = staged32[-1:]
            token_requests = torch.repeat_interleave(
                torch.arange(len(lengths), device=device),
                spans[1],
                output_size=int(boundaries[-1]),
            )
            requests = compression_inputs["request_slots"][token_requests].to(
                torch.int32
            )
            positions = (
                spans[0, token_requests]
                + torch.arange(int(boundaries[-1]), device=device)
                - spans[2, token_requests]
            ).to(torch.int32)
            positions = torch.where(requests != 0, positions, 0)
            torch._assert_async(
                ((requests == 0) | (fb.positions == positions)).all(),
                "Step5 positions differ from native packed spans",
            )
            torch._assert_async(
                (
                    fb.out_cache_loc == req_to_token[requests.long(), positions.long()]
                ).all(),
                "Step5 write locations differ from FULL mapping",
            )
            width = max(
                1,
                (max((p + n for p, n in zip(prefixes, lengths)), default=0) + 7) // 8,
            )
            if lengths and all(length % 32 == 0 for length in lengths):
                score_group_rows = 32
            elif lengths and all(length % 4 == 0 for length in lengths):
                score_group_rows = 4
            else:
                score_group_rows = 1
            if score_group_rows == 32:
                _assert_group32_ownership(requests)
            if self.fp8_full_kv:
                pages, page_map = step5_fp8_prefill_pages(
                    req_to_token,
                    fb.req_pool_indices,
                    totals_t,
                    totals,
                    self.token_to_kv_pool.full_num_pages,
                )
                scratch = torch.empty(
                    (2, pages.numel() * 64, 1, self.attention_shape[1]),
                    dtype=torch.bfloat16,
                    device=device,
                )
                self._fp8_prefill = (pages, page_map, scratch[0], scratch[1])
            self.forward_metadata = _build_attention_metadata(
                token_to_kv_pool=self.token_to_kv_pool,
                req_to_token=self.req_to_token_pool.req_to_token,
                attention_shape=self.attention_shape,
                compression_inputs=compression_inputs,
                row_to_req=requests,
                positions=positions,
                query_start_loc=query_start_loc,
                group_start=group_start,
                group_end=group_end,
                score_width=width,
                region_bins=1 << (width - 1).bit_length(),
                score_group_rows=score_group_rows,
                decode=False,
                valid_rows=valid_rows,
                prefill_spans=(
                    prefixes[: fb.batch_size],
                    lengths[: fb.batch_size],
                    compression_inputs["request_slots"][: fb.batch_size],
                ),
            )
        self.swa_backend.init_forward_metadata(forward_batch)

    def _build_decode_verify_index_plan(
        self,
        *,
        request_slots,
        positions,
        seq_lens,
        out_cache_loc,
        verify,
        token_width,
        valid_rows,
        boundaries,
        empty_groups,
    ):
        """Compute decode/verify indices and compression plans on the device.

        Shared by eager and capture; replay executes the captured GPU operations.
        This performs shape/admission checks but allocates no attention workspace
        and does not bind or cache a Step5AttentionMetadata object.

        Decode positions name the current token (seq_len - 1). Verify seq_lens
        name the accepted prefix and each query has its own causal boundary.
        """
        req_to_token = self.req_to_token_pool.req_to_token
        compressed_index_key_pages = self.token_to_kv_pool.full_num_pages
        context_len = self.position_capacity
        rows = request_slots.numel()
        if (
            rows < 1
            or positions.shape != (rows * token_width,)
            or seq_lens.shape != (rows,)
            or out_cache_loc.shape != (rows * token_width,)
        ):
            raise ValueError("Step5 token metadata shape differs from its batch/width")
        from sglang.kernels.ops.attention.step5.csa_metadata import (
            build_compression_metadata,
        )

        (
            compression_inputs,
            row_to_req,
            output_positions,
            token_valid_rows,
            native_valid,
        ) = build_compression_metadata(
            request_slots=request_slots,
            positions=seq_lens if verify else positions,
            valid_rows=valid_rows,
            req_to_token=req_to_token,
            compressed_index_key_pages=compressed_index_key_pages,
            token_width=token_width,
            verify=verify,
            native_positions=positions,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            context_len=context_len,
        )
        torch._assert_async(
            native_valid.all(),
            "Step5 graph position/FULL mapping differs from native batch",
        )
        if verify:
            group_start, group_end = boundaries[:rows], boundaries[1 : rows + 1]
            score_group_rows = 4 if token_width % 4 == 0 else 1
        else:
            group_start = group_end = empty_groups
            score_group_rows = 1
        score_width = (context_len + 7) // 8
        return dict(
            compression_inputs=compression_inputs,
            row_to_req=row_to_req,
            positions=output_positions,
            query_start_loc=boundaries[: rows + 1],
            group_start=group_start,
            group_end=group_end,
            score_width=score_width,
            region_bins=1 << (score_width - 1).bit_length(),
            score_group_rows=score_group_rows,
            decode=True,
            valid_rows=token_valid_rows,
        )

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        mode = forward_batch.forward_mode
        bs = forward_batch.batch_size
        spec_info = forward_batch.spec_info
        if mode.is_target_verify():
            if spec_info is None or spec_info.draft_token_num != self._verify_width:
                raise ValueError(
                    "Step5 verify width differs from initialized graph width"
                )
        elif mode.is_decode_or_idle() and spec_info is None:
            mode = ForwardMode.DECODE
        else:
            raise ValueError("Step5 graph requires decode/idle or target verify")
        if not 1 <= bs <= self._cuda_graph_max_bs:
            raise ValueError("Step5 graph batch exceeds initialized capacity")
        key = (mode, bs)
        if in_capture:
            self._graph_valid_rows.zero_()
            # The in-graph hook binds the final capture's tensors after warmup.
            self._cuda_graph_metadata.pop(key, None)
            self.forward_metadata = None
        else:
            # Resolve the bucket and validate the live count before any device write.
            if key not in self._cuda_graph_metadata:
                raise ValueError("Step5 graph metadata bucket has not been captured")
            metadata = self._cuda_graph_metadata[key]
            raw_bs = bs - forward_batch.num_padding
            if not 0 <= raw_bs <= bs:
                raise ValueError("Step5 live rows exceed graph capacity")
            self._graph_valid_rows.fill_(raw_bs)
            self.forward_metadata = metadata
        self.swa_backend.init_forward_metadata_out_graph(forward_batch, in_capture)

    def init_forward_metadata_in_graph(self, forward_batch):
        # The out-of-graph hook validates mode and the runner's fixed width.
        mode = forward_batch.forward_mode
        if mode.is_idle():
            mode = ForwardMode.DECODE
        key = (mode, forward_batch.batch_size)
        width = self._verify_width if mode.is_target_verify() else 1
        indices = self._build_decode_verify_index_plan(
            request_slots=forward_batch.req_pool_indices,
            positions=forward_batch.positions,
            seq_lens=forward_batch.seq_lens,
            out_cache_loc=forward_batch.out_cache_loc,
            verify=mode.is_target_verify(),
            token_width=width,
            valid_rows=self._graph_valid_rows,
            boundaries=self._graph_query_start_loc[width],
            empty_groups=self._graph_empty_groups,
        )
        self.forward_metadata = _build_attention_metadata(
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token_pool.req_to_token,
            attention_shape=self.attention_shape,
            **indices,
        )
        self._cuda_graph_metadata[key] = self.forward_metadata
        self.swa_backend.init_forward_metadata_in_graph(forward_batch)

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        if max_bs < 1 or max_num_tokens % max_bs:
            raise ValueError("Step5 graph requires a fixed token capacity per request")
        width = max_num_tokens // max_bs
        if width != self._verify_width:
            raise ValueError("Step5 graph width differs from configured chain width")
        req_to_token = self.req_to_token_pool.req_to_token
        if (
            not 1 <= self.context_len <= 1048576
            or (self.position_capacity + 7) // 8 * 8 > req_to_token.shape[1]
        ):
            raise ValueError("Step5 graph context exceeds native request capacity")
        self._cuda_graph_max_bs = max_bs
        self._cuda_graph_metadata.clear()
        self.forward_metadata = None
        device = req_to_token.device
        self._graph_valid_rows = torch.zeros(1, dtype=torch.int32, device=device)
        rows = torch.arange(max_bs + 1, dtype=torch.int32, device=device)
        widths = torch.arange(width + 1, dtype=torch.int32, device=device)
        self._graph_query_start_loc = widths[:, None] * rows[None, :]
        self._graph_empty_groups = torch.empty(0, dtype=torch.int32, device=device)
        self.swa_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def get_cuda_graph_seq_len_fill_value(self):
        return self.swa_backend.get_cuda_graph_seq_len_fill_value()

    def supports_swa_head_gate(self) -> bool:
        fn = getattr(self.swa_backend, "supports_swa_head_gate", None)
        return bool(fn()) if fn is not None else False

    def supports_dsa_head_gate(self) -> bool:
        metadata = self.forward_metadata
        return (
            metadata is not None
            and metadata.decode
            and metadata.decode_dynamic_splits
            and metadata.decode_work_rows == 64
            and self.attention_shape == (16, 192)
            and metadata.lse.shape == (4, 16)
            and metadata.partial.dtype == torch.float32
            and metadata.partial.is_cuda
            and torch.version.hip is None
            and torch.cuda.get_device_capability(metadata.partial.device) == (10, 3)
        )

    def swa_kv_write_target(self, layer):
        fn = getattr(self.swa_backend, "swa_kv_write_target", None)
        return fn(layer) if fn is not None else None

    def forward_decode(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        if layer.layer_id in self.token_to_kv_pool.dsa_layer_ids:
            return self._forward_dsa(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )
        return self.swa_backend.forward(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )

    def forward_extend(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        if layer.layer_id in self.token_to_kv_pool.dsa_layer_ids:
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

    def _fp8_dequantize_prefill(self, k_cache, v_cache, k_scale, v_scale, packed):
        pages, page_map, k_out, v_out = self._fp8_prefill
        step5_fp8_kv_dequant_pages(
            pages, k_cache, v_cache, k_scale, v_scale, k_out, v_out
        )
        packed_out = torch.empty_like(packed)
        step5_fp8_kv_remap_pages(packed, page_map, packed_out)
        return k_out, v_out, packed_out

    def get_indexer_metadata(self, layer_id: int, forward_batch: ForwardBatch):
        return self.forward_metadata.indexer_metadata

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
        index_stream=None,
        head_gate_getter=None,
    ):
        if self.forward_metadata is None:
            raise RuntimeError("Step5 backend metadata has not been initialized")
        if save_kv_cache:
            if k is None or v is None:
                raise ValueError("Step5 DSA KV store requires both K and V")
            self.token_to_kv_pool.set_kv_buffer(
                layer, KVWriteLoc(forward_batch.out_cache_loc), k, v
            )
        k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        metadata = self.forward_metadata
        softmax_scale = layer.scaling
        # Capture may defer the independent Indexer branch until the main KV
        # write is recorded. The fork is unchanged; record it before the join
        # so CUDA graph replay does not move the main chain onto a new lane.
        if callable(index_selection):
            index_selection = index_selection()
        if index_stream is not None:
            # Join after the independent main KV write, before reading selection.
            torch.cuda.current_stream().wait_stream(index_stream)
        head_gate = None
        if head_gate_getter is not None:
            head_gate = head_gate_getter()
            if head_gate is None:
                raise RuntimeError("Step5 side branch did not produce its head gate")
            head_gate.record_stream(torch.cuda.current_stream())
        packed, counts, lengths = index_selection
        kv_scales = {}
        if self.fp8_full_kv:
            k_scale, v_scale = self.token_to_kv_pool.get_kv_scale_buffer(layer.layer_id)
            if metadata.decode:
                kv_scales = dict(k_scale=k_scale, v_scale=v_scale)
            else:
                k_cache, v_cache, packed = self._fp8_dequantize_prefill(
                    k_cache, v_cache, k_scale, v_scale, packed
                )
        out = torch.empty_like(q)
        lse = metadata.lse
        fuse_head_gate = (
            head_gate is not None
            and self.supports_dsa_head_gate()
            and head_gate.dtype == out.dtype == torch.bfloat16
            and head_gate.device == out.device
            and head_gate.shape == (4, 16)
            and head_gate.is_contiguous()
        )
        if metadata.decode:
            step5_sparse_decode_selected(
                q,
                k_cache,
                v_cache,
                counts=counts,
                packed=packed,
                causal_lengths=lengths,
                query_start_loc=metadata.query_start_loc,
                valid_rows=metadata.valid_rows,
                out=out,
                lse=lse,
                scale=softmax_scale,
                dynamic_splits=metadata.decode_dynamic_splits,
                n_split4=metadata.decode_split4_rows,
                n_split2=metadata.decode_split2_rows,
                work_rows=metadata.decode_work_rows,
                partial=metadata.partial,
                partial_lse=metadata.partial_lse,
                **({"head_gate": head_gate} if fuse_head_gate else {}),
                **kv_scales,
            )
        else:
            workspace = metadata.prefill
            step5_sparse_prefill_selected(
                q,
                k_cache,
                v_cache,
                packed=packed,
                counts=counts,
                query_positions=metadata.indexer_metadata.query_positions,
                group_query_start=metadata.group_start,
                group_query_end=metadata.group_end,
                num_region_bins=metadata.region_bins,
                logical_starts=workspace.logical_starts,
                physical_regions=workspace.physical_regions,
                union_physical=workspace.union_physical,
                union_logical=workspace.union_logical,
                union_counts=workspace.union_counts,
                membership=workspace.membership,
                bitset=workspace.bitset,
                sm100_workspace=workspace if use_sm100_sparse_prefill() else None,
                out=out,
                lse=lse,
                scale=softmax_scale,
            )
        if head_gate is not None and not fuse_head_gate:
            heads, dim = self.attention_shape
            out = fused_sigmoid_mul(
                out.reshape(-1, heads * dim),
                head_gate.unsqueeze(-1).expand(-1, -1, dim),
                round_gate=True,
            ).view_as(out)
        return out
