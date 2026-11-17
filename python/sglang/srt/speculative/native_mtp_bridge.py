"""GPU-only DFlash verification -> native MTP draft-extend bridge."""

import logging
import os
from functools import partial

import torch
import triton
import triton.language as tl

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.speculative.dmtp_mtp_graph import DMTPMTPGraphRunner
from sglang.srt.speculative.dmtp_runtime import mtp_config, scoped_config
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.srt.speculative.mtp_append_pack import pack_commit
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


@triton.jit
def _stage(
    candidates,
    bonus,
    hidden,
    locations,
    reqs,
    prefix,
    counts,
    ids,
    features,
    slots,
    positions,
    mrope,
    seq,
    mapping,
    cache,
    H: tl.constexpr,
    CAP: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    PENDING: tl.constexpr,
    SCRATCH: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    L0: tl.constexpr,
    L1: tl.constexpr,
    H0: tl.constexpr,
    H1: tl.constexpr,
    H2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b, tile = tl.program_id(0), tl.program_id(1)
    p, n, req = tl.load(prefix + b), tl.load(counts + b), tl.load(reqs + b)
    j, d = tl.arange(0, 8), tile * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(
        hidden + b * H0 + j[:, None] * H1 + d[None, :] * H2,
        mask=(j[:, None] < n) & (d[None, :] < H),
        other=0,
    )
    tl.store(
        features + (b * 8 + j[:, None]) * H + d[None, :], value, mask=d[None, :] < H
    )
    old = tl.load(locations + b * L0 + j * L1, mask=j < n, other=0)
    tl.store(
        cache + old[:, None] * H + d[None, :],
        value,
        mask=(j[:, None] < n) & (d[None, :] < H),
    )
    if tile == 0:
        token = tl.load(candidates + b * C0 + (j + 1) * C1, mask=j < n - 1, other=0)
        token = tl.where(j == n - 1, tl.load(bonus + b), token)
        following = tl.load(locations + b * L0 + (j + 1) * L1, mask=j < n - 1, other=0)
        native = tl.where(j < n - 1, following, PENDING + req)
        native = tl.where(j < n, native, SCRATCH + req * 7 + j - 1)
        pos = p + j + 1
        tl.store(ids + b * 8 + j, token)
        tl.store(slots + b * 8 + j, native)
        tl.store(positions + b * 8 + j, pos)
        for k in tl.static_range(3):
            tl.store(mrope + k * CAP + b * 8 + j, pos)
        tl.store(seq + b, p + 8)
        tl.store(mapping + req * MAP_STRIDE + p + j, native, mask=j < n)
        tl.store(mapping + req * MAP_STRIDE + p - 1, tl.load(locations + b * L0))


def validate_bridge_proposals(controller):
    # Commit consumes accepted tokens/counts, not proposal probabilities. Sparse
    # selector q is owned by the verifier, so it does not change this KV bridge.
    if (
        not getattr(controller, "native_selector", False)
        or controller.trace_dir
        or controller.batch_trace
    ):
        raise ValueError(
            "GPU bridge requires native sampled proposals "
            "without per-request debug traces"
        )


class NativeMTPBridge:
    def __init__(self, native):
        self.native = native
        self.owner = native.owner
        self.controller = self.owner._mtp_first
        validate_bridge_proposals(self.controller)
        runner = native.worker.model_runner
        self.backend = DraftBackendFactory(runner, 1, 1).create_draft_extend_backend()
        if (
            runner.server_args.attention_backend == "triton"
            and os.getenv("SGLANG_MTP_PREFIX_QUERY_TILE", "0") == "1"
        ):
            self.backend.extend_attention_fwd = partial(
                self.backend.extend_attention_fwd, mtp_query_tile=True
            )
            print("MTP_QUERY_TILE block_m=16 block_n=unchanged", flush=True)
        from sglang.srt.speculative.mtp_anchor_fa3 import Fixed8FA3Metadata

        assert self.backend.fa_impl_ver == 3
        self.backend.mtp_fixed8_metadata = Fixed8FA3Metadata(self.backend)
        self.backend.init_forward_metadata_out_graph = self.backend.mtp_fixed8_metadata
        print("MTP_FA3_BACKEND", type(self.backend).__name__, flush=True)
        self.graph = DMTPMTPGraphRunner(native, self.backend)
        self.max_batch = self.graph.max_bs
        self.capacity = self.max_batch * 8
        self.mapping = runner.req_to_token_pool.req_to_token
        self.context_limit = self.mapping.shape[1]
        self.scratch = native.pending_base + native.target_mapping.req_to_token.shape[0]
        self.width = native.owner.target_worker.model_runner.model.lm_head.weight.shape[
            1
        ]
        device = self.mapping.device
        dtype = native.owner.target_worker.model_runner.model.lm_head.weight.dtype
        self.ids = torch.zeros(self.capacity, dtype=torch.int64, device=device)
        self.features = torch.zeros(
            self.capacity, self.width, dtype=dtype, device=device
        )
        self.slots = torch.zeros_like(self.ids)
        self.positions = torch.zeros_like(self.ids)
        self.mrope = torch.zeros(3, self.capacity, dtype=torch.int64, device=device)
        self.seq = torch.zeros(self.max_batch, dtype=torch.int64, device=device)
        self.reqs = torch.zeros_like(self.seq)
        self.accept = torch.ones(self.max_batch, dtype=torch.int32, device=device)
        self.correct = torch.zeros_like(self.accept)
        self.extend = torch.full_like(self.accept, 8)
        size = self.mapping.shape[0]
        self.hidden_by_req = torch.empty(size, self.width, dtype=dtype, device=device)
        self.anchor_by_req = torch.full((size,), -1, dtype=torch.int64, device=device)
        self.token_by_req = torch.full_like(self.anchor_by_req, -1)
        self.verified_batches = set()
        self.graph_audit_count = 0
        self.hits = self.fallbacks = 0
        self.batches = {bs: self._batch(bs) for bs in self.graph.capture_bs}

    def _batch(self, bs):
        n = bs * 8
        spec = EagleDraftExtendInput(
            hidden_states=self.features[:n],
            num_correct_drafts=self.correct[:bs],
            num_accept_tokens=self.accept[:bs],
        )
        return ForwardBatch(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            batch_size=bs,
            input_ids=self.ids[:n],
            req_pool_indices=self.reqs[:bs],
            seq_lens=self.seq[:bs],
            seq_lens_cpu=None,
            seq_lens_sum=None,
            out_cache_loc=self.slots[:n],
            positions=self.positions[:n],
            mrope_positions=self.mrope[:, :n],
            extend_num_tokens=n,
            extend_seq_lens=self.extend[:bs],
            extend_seq_lens_cpu=[8] * bs,
            extend_start_loc=torch.arange(bs, device=self.ids.device, dtype=torch.int32)
            * 8,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            spec_info=spec,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
        )

    def seed(self, reqs, hidden, anchors, tokens):
        self.hidden_by_req.index_copy_(0, reqs.long(), hidden)
        self.anchor_by_req.index_copy_(0, reqs.long(), anchors.long())
        self.token_by_req.index_copy_(0, reqs.long(), tokens.long())

    def t1(self, batch, block_ids):
        c = self.controller
        c.graph_used = {}
        c.req_indices = batch.req_pool_indices.long().clone()
        self.prefix = batch.seq_lens.long().clone()
        expected = self.anchor_by_req.index_select(0, c.req_indices)
        tokens = self.token_by_req.index_select(0, c.req_indices)
        torch._assert_async(
            ((expected == self.prefix) & (tokens == block_ids[:, 0])).all(),
            "MTP GPU request/anchor/cache state is not aligned",
        )
        c.sampling = c.validate_sampling(batch)
        c.hidden = self.hidden_by_req.index_select(0, c.req_indices)
        c.prepare_t1(block_ids)
        c.q1 = None

    def can_graph(self, bs, cpu_prefix_hint):
        return (
            bs in self.batches
            and cpu_prefix_hint is not None
            and cpu_prefix_hint.device.type == "cpu"
            and int(cpu_prefix_hint.max()) + 8 <= self.context_limit
        )

    def commit(
        self, candidates, accept_len, bonus, target_features, next_positions, locations
    ):
        c = self.controller
        bs = candidates.shape[0]
        counts = accept_len.to(torch.int32) + 1
        torch._assert_async(
            (
                (counts >= 1)
                & (counts <= 8)
                & (next_positions == self.prefix + counts)
                & (self.prefix >= 1)
            ).all(),
            "MTP GPU commit counts/positions mismatch",
        )
        hidden = c.target_hidden(bs * 8).view(bs, 8, -1)
        hint = self.owner._draft_seq_lens_cpu_buf[:bs]
        if not self.can_graph(bs, hint):
            updated = self._fallback(candidates, bonus, hidden, locations, counts)
            c.graph_used["mtp_append"] = False
        else:
            pool = self.native.worker.model_runner.token_to_kv_pool
            keys, values = pool.get_kv_buffer(pool.start_layer)
            pending = c.req_indices + self.native.pending_base
            keys.index_copy_(0, locations[:, 0].long(), keys[pending])
            values.index_copy_(0, locations[:, 0].long(), values[pending])
            self.reqs[:bs].copy_(c.req_indices)
            self.accept[:bs].copy_(counts)
            self.correct[:bs].copy_(counts - 1)
            _stage[(bs, triton.cdiv(self.width, 256))](
                candidates,
                bonus.contiguous(),
                hidden,
                locations,
                c.req_indices,
                self.prefix,
                counts,
                self.ids,
                self.features,
                self.slots,
                self.positions,
                self.mrope,
                self.seq,
                self.mapping,
                c.target_hidden_cache,
                H=self.width,
                CAP=self.capacity,
                MAP_STRIDE=self.mapping.stride(0),
                PENDING=self.native.pending_base,
                SCRATCH=self.scratch,
                C0=candidates.stride(0),
                C1=candidates.stride(1),
                L0=locations.stride(0),
                L1=locations.stride(1),
                H0=hidden.stride(0),
                H1=hidden.stride(1),
                H2=hidden.stride(2),
                BLOCK=256,
            )
            batch = self.batches[bs]
            result = self.graph.replay(batch).hidden_states
            self._check(batch, result, counts)
            last = torch.arange(bs, device=counts.device) * 8 + counts.long() - 1
            updated = result.index_select(0, last)
            c.graph_used["mtp_append"] = True
            self.hits += 1
        self.seed(c.req_indices, updated, next_positions, bonus)
        c.final_hidden = None

    def _check(self, batch, result, counts):
        bs = batch.batch_size
        if (
            bs in self.verified_batches
            and os.getenv("SGLANG_MTP_PREFIX_MTP_GRAPH_CHECK", "0") != "1"
        ):
            return
        buffers = self.graph.buffers
        n = bs * 8
        spec = EagleDraftExtendInput(
            hidden_states=buffers.hidden_states[:n],
            num_correct_drafts=buffers.num_correct_drafts[:bs],
            num_accept_tokens=buffers.num_accept_tokens[:bs],
        )
        spec.extend_seq_lens_tensor = buffers.extend_seq_lens[:bs]
        reference_batch = ForwardBatch(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            batch_size=bs,
            input_ids=buffers.input_ids[:n],
            req_pool_indices=buffers.req_pool_indices[:bs],
            seq_lens=buffers.seq_lens[:bs],
            seq_lens_cpu=None,
            seq_lens_sum=None,
            out_cache_loc=buffers.out_cache_loc[:n],
            positions=buffers.positions[:n],
            mrope_positions=buffers.mrope_positions[:, :n],
            extend_num_tokens=n,
            extend_seq_lens=buffers.extend_seq_lens[:bs],
            extend_seq_lens_cpu=[8] * bs,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            spec_info=spec,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
        )
        self.graph.bind(reference_batch)
        pool = self.native.worker.model_runner.token_to_kv_pool
        keys, values = pool.get_kv_buffer(pool.start_layer)
        slots = batch.out_cache_loc.long()
        before_hidden = result.clone()
        before_k, before_v = keys[slots].clone(), values[slots].clone()
        runner = self.native.worker.model_runner
        with scoped_config(**mtp_config(self.owner)), forward_context(
            ForwardContext(attn_backend=self.backend)
        ):
            self.backend.init_forward_metadata_out_graph(
                reference_batch, in_capture=True
            )
            set_dp_buffer_len(
                None, n, reference_batch.dp_padding_mode.is_max_len(), None
            )
            set_is_extend_in_batch(False)
            ref = runner.model.forward(
                reference_batch.input_ids, reference_batch.positions, reference_batch
            ).hidden_states
        mask = (torch.arange(8, device=counts.device)[None] < counts[:, None]).reshape(
            -1
        )
        match = (
            (before_hidden[mask] == ref[mask]).all()
            & (before_k[mask] == keys[slots][mask]).all()
            & (before_v[mask] == values[slots][mask]).all()
        )
        failed = get_tp_group().all_reduce((~match).to(torch.int32).reshape(1))
        if failed.item():
            raise RuntimeError(
                "Native MTP graph differs from same-layout eager hidden/KV"
            )
        self.verified_batches.add(bs)
        self.graph_audit_count += 1
        logging.getLogger(__name__).info(
            "DMTP_MTP_GRAPH_STRICT_PASS B=%d effective_hidden_exact=True effective_KV_exact=True",
            bs,
        )

    def _fallback(self, candidates, bonus, hidden, locations, counts):
        c = self.controller
        lengths, prefixes, reqs = (
            counts.tolist(),
            self.prefix.tolist(),
            c.req_indices.tolist(),
        )
        self.native.pending_positions.update(zip(reqs, prefixes))
        packed = pack_commit(
            candidates,
            bonus,
            hidden,
            locations,
            c.req_indices,
            lengths,
            prefixes,
            self.native.pending_base,
        )
        updated = self.native.append(
            packed.tokens,
            packed.hidden,
            c.req_indices,
            packed.cache_locations,
            prefixes,
            lengths,
            continuation=True,
            prepared=packed,
            request_ids=reqs,
        )
        c.target_hidden_cache.index_copy_(
            0, packed.cache_locations.long(), packed.hidden
        )
        self.fallbacks += 1
        return updated.index_select(0, packed.last_rows)

    def run(self, batch, prefixes, lengths):
        # Legacy packed prefill/boundary calls retain the existing eager path.
        return None
