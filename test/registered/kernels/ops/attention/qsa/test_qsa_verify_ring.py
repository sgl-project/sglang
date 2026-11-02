"""Speculative forwards must compress the same QSA index keys as decode and prefill.

A target-verify or MTP draft-extend forward stores the raw index key of every row
in the per-request pending ring before it compresses the groups those rows
complete, so the ring must keep every member a group reaches back for. This
drives the real indexer, pool and metadata (eager, and the CUDA-graph metadata
kernels) through prefill + verify with every acceptance pattern and through MTP
draft extend, and requires the compressed cache to be bit-equal to prefill +
decode and to one whole-sequence prefill.
"""

import random
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa.graph_metadata import launch_graph_metadata
from sglang.srt.layers.attention.qsa.metadata import QSAIndexerMetadata
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
    QwenSparseAttnMetadata,
)
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# Qwen3.8-Flash-Next index geometry and attention RoPE.
RATIO = 4
PAGE = 64
HEAD_DIM = 128
NUM_Q_HEADS = 4
QK_WIDTH = (NUM_Q_HEADS + 1) * HEAD_DIM
LAYER_ID = 3


@pytest.fixture(autouse=True)
def runtime_context():
    publish(ServerArgs(model_path="dummy"), role="test")
    try:
        yield
    finally:
        reset_context()


class _ProjectedRows(torch.nn.Module):
    """index_qk_proj stand-in: rows arrive already projected, so every forward
    stores bit-identical raw keys whatever its row count."""

    def forward(self, rows):
        return rows, None


def _indexer_config():
    return SimpleNamespace(
        indexer_n_heads=NUM_Q_HEADS,
        indexer_kv_heads=1,
        indexer_head_dim=HEAD_DIM,
        indexer_budget=2048,
        indexer_compress_ratio=RATIO,
        hidden_size=QK_WIDTH,
        rms_norm_eps=1e-6,
    )


def _pool(*, max_rows, num_requests, tokens):
    return QSATokenToKVPool(
        size=tokens * num_requests,
        dtype=torch.bfloat16,
        page_size=PAGE,
        head_num=1,
        head_dim=HEAD_DIM,
        full_attention_layer_ids=[LAYER_ID],
        device="cuda",
        mamba_pool=None,
        qsa_index_kv_heads=1,
        qsa_index_head_dim=HEAD_DIM,
        qsa_compress_ratio=RATIO,
        qsa_token_topk=2048,
        num_request_slots=num_requests + 1,
        qsa_max_rows_per_request=max_rows,
    )


class _Harness:
    """One QSA layer over a real pool; every forward goes through the backend's
    metadata builders and ``QSAIndexer._forward_impl``."""

    def __init__(self, *, max_rows, num_requests, max_len, seed=0):
        torch.manual_seed(seed)
        pages = -(-max_len // PAGE)
        self.pool = _pool(
            max_rows=max_rows, num_requests=num_requests, tokens=PAGE * pages
        )
        # Request r owns full pages 1 + (r - 1) * pages ...; page 0 is reserved.
        cols = torch.arange(pages * PAGE, dtype=torch.int32, device="cuda")
        rows = [torch.zeros_like(cols)] + [
            (1 + (r - 1) * pages + cols // PAGE) * PAGE + cols % PAGE
            for r in range(1, num_requests + 1)
        ]
        self.req_to_token = torch.stack(rows).contiguous()
        runner = SimpleNamespace(
            token_to_kv_pool=self.pool,
            device="cuda",
            is_draft_worker=False,
            req_to_token_pool=SimpleNamespace(req_to_token=self.req_to_token),
            model_config=SimpleNamespace(
                hf_config=_indexer_config(), context_len=pages * PAGE
            ),
        )
        self.backend = QwenSparseAttnBackend(runner)
        rotary = MRotaryEmbedding(
            head_size=256,
            rotary_dim=64,
            max_position_embeddings=pages * PAGE,
            base=10_000_000,
            is_neox_style=True,
            dtype=torch.bfloat16,
            mrope_section=[11, 11, 10],
            mrope_interleaved=True,
        )
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            self.indexer = QSAIndexer(
                _indexer_config(),
                layer_id=LAYER_ID,
                quant_config=None,
                rotary_emb=rotary,
            )
            # Device-only move: the fp32 cos/sin cache keeps its dtype.
            self.indexer.to(device="cuda")
        finally:
            torch.set_default_dtype(previous)
        self.indexer.index_qk_proj = _ProjectedRows()
        with torch.no_grad():
            for norm in (self.indexer.q_layernorm, self.indexer.k_layernorm):
                norm._weight_loader(
                    norm.weight,
                    torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.1,
                )
        self.max_pages = -(
            -(pages * PAGE // RATIO) // self.pool.qsa_compressed_page_size
        )

    @staticmethod
    def _rope(positions):
        return positions.long().unsqueeze(0).expand(3, -1).contiguous()

    def _run(self, forward_batch, rows, rope, metadata):
        with torch.no_grad():
            self.indexer._forward_impl(rows, rope, forward_batch, metadata)

    def prefill(self, requests, rows_by_request):
        """One extend of each request's first rows (positions 0..len-1)."""
        lengths = [rows.shape[0] for rows in rows_by_request]
        positions = torch.cat([torch.arange(n, device="cuda") for n in lengths])
        rope = self._rope(positions)
        lengths_t = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            batch_size=len(requests),
            req_pool_indices=torch.tensor(requests, dtype=torch.int32, device="cuda"),
            seq_lens=lengths_t,
            seq_lens_cpu=lengths_t.cpu(),
            extend_seq_lens=lengths_t,
            positions=positions,
            mrope_positions=rope,
            input_ids=torch.zeros(positions.numel(), dtype=torch.int32, device="cuda"),
            out_cache_loc=torch.cat(
                [self.req_to_token[r, :n].long() for r, n in zip(requests, lengths)]
            ),
            spec_info=None,
            _original_forward_mode=None,
        )
        metadata = self.backend._metadata_from_forward_batch(batch).indexer_metadata
        self._run(batch, torch.cat(rows_by_request), rope, metadata)

    def paged(self, requests, bases, rows_by_request, *, graph):
        """One decode (one row per request) or target-verify forward (W rows per
        request at positions base..base+W-1), eager or with CUDA-graph metadata."""
        width = rows_by_request[0].shape[0]
        verify = width > 1
        positions = torch.cat(
            [torch.arange(b, b + width, device="cuda") for b in bases]
        )
        rope = self._rope(positions)
        # TARGET_VERIFY carries the committed prefix; DECODE the length with its row.
        seq_lens = torch.tensor(
            bases if verify else [b + 1 for b in bases],
            dtype=torch.int32,
            device="cuda",
        )
        req_pool_indices = torch.tensor(requests, dtype=torch.int32, device="cuda")
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY if verify else ForwardMode.DECODE,
            batch_size=len(requests),
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            extend_seq_lens=None,
            positions=positions,
            mrope_positions=None,
            input_ids=torch.zeros(positions.numel(), dtype=torch.int32, device="cuda"),
            out_cache_loc=torch.cat(
                [
                    self.req_to_token[r, b : b + width].long()
                    for r, b in zip(requests, bases)
                ]
            ),
            spec_info=SimpleNamespace(draft_token_num=width, topk=1)
            if verify
            else None,
            _original_forward_mode=None,
        )
        if graph:
            # The backend's guard, then the GPU metadata kernels graph replay
            # launches, on persistent graph-shaped buffers.
            self.backend._require_chain_speculation(batch.forward_mode, batch.spec_info)
            metadata = self._graph_metadata(positions.numel())
            launch_graph_metadata(
                mode=1 if verify else 0,
                bs=len(requests),
                num_rows=positions.numel(),
                seq_lens=seq_lens,
                req_pool_indices=req_pool_indices,
                extend_lens=None,
                extend_len=width if verify else 0,
                num_padding=0,
                metadata=metadata,
                req_to_token=self.req_to_token,
                pool=self.pool,
            )
        else:
            metadata = self.backend._metadata_from_forward_batch(batch)
        self._run(batch, torch.cat(rows_by_request), rope, metadata.indexer_metadata)

    def draft_extend(self, requests, ends, rows_by_request, *, graph):
        """One MTP draft-extend forward (DRAFT_EXTEND_V2): request i's rows are its
        newest committed positions, ends[i] - len .. ends[i] - 1."""
        lengths = [rows.shape[0] for rows in rows_by_request]
        positions = torch.cat(
            [torch.arange(e - n, e, device="cuda") for e, n in zip(ends, lengths)]
        )
        rope = self._rope(positions)
        seq_lens = torch.tensor(ends, dtype=torch.int32, device="cuda")
        extend_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        req_pool_indices = torch.tensor(requests, dtype=torch.int32, device="cuda")
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            batch_size=len(requests),
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            extend_seq_lens=extend_lens,
            positions=positions,
            mrope_positions=None,
            input_ids=torch.zeros(positions.numel(), dtype=torch.int32, device="cuda"),
            out_cache_loc=torch.cat(
                [
                    self.req_to_token[r, e - n : e].long()
                    for r, e, n in zip(requests, ends, lengths)
                ]
            ),
            spec_info=SimpleNamespace(
                num_tokens_per_req=max(lengths), extend_seq_lens_cpu=lengths
            ),
            _original_forward_mode=None,
        )
        if graph:
            metadata = self._graph_metadata(positions.numel())
            launch_graph_metadata(
                mode=2,
                bs=len(requests),
                num_rows=positions.numel(),
                seq_lens=seq_lens,
                req_pool_indices=req_pool_indices,
                extend_lens=extend_lens,
                extend_len=0,
                num_padding=0,
                metadata=metadata,
                req_to_token=self.req_to_token,
                pool=self.pool,
            )
        else:
            metadata = self.backend._metadata_from_forward_batch(batch)
        self._run(batch, torch.cat(rows_by_request), rope, metadata.indexer_metadata)

    def _graph_metadata(self, num_rows):
        def zeros(*shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device="cuda")

        sequence_lengths = zeros(num_rows)
        row_req_pool_indices = zeros(num_rows)
        token_to_batch_idx = torch.arange(num_rows, dtype=torch.int32, device="cuda")
        token_slot_table = zeros(num_rows, 1)
        indexer = QSAIndexerMetadata(
            sequence_lengths=sequence_lengths,
            token_to_batch_idx=token_to_batch_idx,
            token_slot_table=token_slot_table,
            out_cache_loc=zeros(num_rows, dtype=torch.int64),
            token_to_kv_pool=self.pool,
            compress_ratio=RATIO,
            block_topk=512,
            req_pool_indices=row_req_pool_indices,
            is_cuda_graph=True,
            graph_write_locs=zeros(num_rows),
            graph_compressed_page_table=zeros(num_rows, self.max_pages),
            graph_compressed_lengths=zeros(num_rows),
            graph_prefix_lengths=zeros(num_rows),
            decode_logical_positions=zeros(num_rows),
            pending_ring_slots=zeros(num_rows, dtype=torch.int64),
            graph_ring_group_locs=zeros(num_rows, RATIO),
        )
        return QwenSparseAttnMetadata(
            sequence_lengths=sequence_lengths,
            token_to_batch_idx=token_to_batch_idx,
            token_slot_table=token_slot_table,
            indexer_metadata=indexer,
            row_req_pool_indices=row_req_pool_indices,
            is_cuda_graph=True,
        )

    def compressed(self, request, length):
        """The compressed keys of the request's complete groups, oldest first."""
        blocks = length // RATIO
        slots = self.req_to_token[request, : blocks * RATIO : RATIO].long() // RATIO
        return self.pool.get_qsa_compressed_k_buffer(LAYER_ID)[slots].clone()


def _token_rows(count, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        count, QK_WIDTH, device="cuda", dtype=torch.bfloat16, generator=generator
    )


def _run_speculative(*, width, prompt_lens, total_len, commits, graph):
    """Prefill each request, then verify ``width``-row windows (decode when 1) until
    every request committed ``total_len`` tokens; ``commits[i]`` yields request i's
    accepted row count per window (row 0, the bonus token, always commits).
    Rejected rows carry other tokens' keys, as rejected drafts do."""
    requests = list(range(1, len(prompt_lens) + 1))
    harness = _Harness(
        max_rows=width, num_requests=len(requests), max_len=total_len + width
    )
    truth = [_token_rows(total_len + width, seed=r) for r in requests]
    drafts = [_token_rows(total_len + width, seed=1000 + r) for r in requests]
    harness.prefill(requests, [t[:n] for t, n in zip(truth, prompt_lens)])
    committed = list(prompt_lens)
    while True:
        # A request leaves the batch once it committed total_len tokens.
        active = [i for i, length in enumerate(committed) if length < total_len]
        if not active:
            return harness, truth
        bases = [committed[i] for i in active]
        windows = []
        for i, base in zip(active, bases):
            commit = min(next(commits[i]), width, total_len - base)
            windows.append(
                torch.cat(
                    [
                        truth[i][base : base + commit],
                        drafts[i][base + commit : base + width],
                    ]
                )
            )
            committed[i] = base + commit
        harness.paged([requests[i] for i in active], bases, windows, graph=graph)


def _reference(truth, total_len, request_count):
    harness = _Harness(max_rows=1, num_requests=request_count, max_len=total_len + 8)
    harness.prefill(list(range(1, request_count + 1)), [t[:total_len] for t in truth])
    return harness


def _assert_same_groups(actual, expected, label):
    differing = (actual != expected).flatten(1).any(dim=1).nonzero().flatten()
    assert differing.numel() == 0, (
        f"{label}: {differing.numel()} of {actual.shape[0]} compressed groups differ, "
        f"first {differing[:8].tolist()}"
    )


def _always(value):
    while True:
        yield value


def _random_commits(width, seed):
    rng = random.Random(seed)
    while True:
        yield rng.randint(1, width)


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("prompt_len", [61, 62, 63, 64])
def test_full_acceptance_verify_matches_decode(graph, width, prompt_len):
    total = 160
    verified, truth = _run_speculative(
        width=width,
        prompt_lens=[prompt_len],
        total_len=total,
        commits=[_always(width)],
        graph=graph,
    )
    decoded, _ = _run_speculative(
        width=1,
        prompt_lens=[prompt_len],
        total_len=total,
        commits=[_always(1)],
        graph=graph,
    )
    reference = _reference(truth, total, 1)
    expected = reference.compressed(1, total)
    _assert_same_groups(decoded.compressed(1, total), expected, "decode vs prefill")
    _assert_same_groups(verified.compressed(1, total), expected, "verify vs prefill")


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("width", [4, 5, 8])
def test_partial_acceptance_batched_verify_matches_prefill(graph, width):
    """Accept counts vary per window and per request, so every window alignment and
    rejected-row overwrite is exercised; widths above the ratio complete two
    groups per window."""
    total, prompt_lens = 448, [129, 62, 67]
    verified, truth = _run_speculative(
        width=width,
        prompt_lens=prompt_lens,
        total_len=total,
        commits=[_random_commits(width, seed=7 * width + i) for i in range(3)],
        graph=graph,
    )
    reference = _reference(truth, total, 3)
    for request in (1, 2, 3):
        _assert_same_groups(
            verified.compressed(request, total),
            reference.compressed(request, total),
            f"request {request}",
        )


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_mtp_draft_extend_matches_prefill(graph):
    """The MTP draft's own QSA layer extends each request by its newly committed
    tokens (1..4 per round) through the same ring."""
    width, total, prompt_lens = 4, 448, [129, 62, 67]
    requests = [1, 2, 3]
    harness = _Harness(max_rows=width, num_requests=3, max_len=total + width)
    truth = [_token_rows(total + width, seed=r) for r in requests]
    harness.prefill(requests, [t[:n] for t, n in zip(truth, prompt_lens)])
    committed = list(prompt_lens)
    rounds = [random.Random(31 + r) for r in requests]
    while True:
        active = [i for i, length in enumerate(committed) if length < total]
        if not active:
            break
        rows = []
        for i in active:
            count = min(rounds[i].randint(1, width), total - committed[i])
            rows.append(truth[i][committed[i] : committed[i] + count])
            committed[i] += count
        harness.draft_extend(
            [requests[i] for i in active],
            [committed[i] for i in active],
            rows,
            graph=graph,
        )
    reference = _reference(truth, total, 3)
    for request in requests:
        _assert_same_groups(
            harness.compressed(request, total),
            reference.compressed(request, total),
            f"request {request}",
        )


@pytest.mark.parametrize("max_rows,ring", [(1, 4), (4, 8), (8, 16)])
def test_pool_allocates_the_ring(max_rows, ring):
    pool = _pool(max_rows=max_rows, num_requests=2, tokens=PAGE)
    assert pool.qsa_ring_size == ring
    assert pool.get_qsa_key_state_buffer(LAYER_ID).shape[0] == 3 * ring
    assert pool.qsa_rope_position_buffer.shape[0] == 3 * ring


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
