"""TOP DeepSeek-V4.1: a request resumed from a page-aligned radix hit runs its rows bit for bit as a cold prefill.

The ratio-2 and ratio-1 KV sources and the index layers carry state across forwards only in the pair ring (the even
token of a pair waiting for its odd partner) and in the paged compressed KV and index keys a hit shares. This drives
the TOP low-ratio program (``DeepseekV4AttnBackend._top_low_ratio_sources``: pooling, the KV and index-K writes, the
indexer's ``select``) on two real ``DeepSeekV4TokenToKVPool``s with the same weights and cache slots. In the cold pool
request A prefills the whole sequence. In the warm pool request C prefills the first two pages (what the radix tree
keeps), then request B (another req slot, its pair ring full of a previous occupant's rows) resumes there; then A and B
each decode one token. Every compressed KV and index-K byte, B's selections (request-local positions), its candidate
mask and its pair ring must equal A's. The control resumes B one token inside a pair, which a page size of whole pairs
(``validate_dsv41_radix``) rules out, and must differ. All comparisons are bitwise.
"""

import types
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

HIDDEN = 2048
Q_LORA = 1280
HEAD_DIM = 512
ROPE_DIM = 64
INDEX_HEADS = 32
INDEX_HEAD_DIM = 128
# Below the compressed positions a row sees, so every selection is sparse.
INDEX_TOPK = 128
CANDIDATE_BLOCKS, CANDIDATE_BLOCK = 4, 64
EPS = 1e-6
# The release's layer schedule and kv_source layers: 2 ratio-2 sources (2, 8), ratio-1 sources from 20.
RATIOS = [0] * 2 + [2] * 18 + [1] * 20
KV_SOURCE_LAYERS = (2, 8, 14, 20)
PAGE = 256
POOL_PAGES = 6
MAX_REQS = 16
CTX = 4 * PAGE
# A prefills SEQ tokens cold; B resumes from C's first HIT tokens (two pages). SEQ is odd so the decode step after it
# completes a pair from the ring.
SEQ, HIT = 3 * PAGE + 77, 2 * PAGE
REQ_A, REQ_B, REQ_C = 3, 7, 11
dev = "cuda"


def _bits(x):
    return x.contiguous().view(torch.uint8)


def _pool():
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    return DeepSeekV4TokenToKVPool(
        max_num_reqs=MAX_REQS,
        swa_size=POOL_PAGES * PAGE,
        c4_size=0,
        c128_size=0,
        c4_state_pool_size=0,
        c128_state_pool_size=0,
        page_size=PAGE,
        swa_page_size=128,
        dtype=torch.float8_e4m3fn,
        c4_state_dtype=torch.float32,
        c128_state_dtype=torch.float32,
        qk_nope_head_dim=HEAD_DIM - ROPE_DIM,
        qk_rope_head_dim=ROPE_DIM,
        indexer_head_dim=INDEX_HEAD_DIM,
        layer_num=len(RATIOS),
        device=dev,
        enable_memory_saver=False,
        compression_ratios=RATIOS,
        kv_source_layers=KV_SOURCE_LAYERS,
        full_size=POOL_PAGES * PAGE,
    )


def _indexer(g, *, owns_k, candidate_source, uses_candidates):
    """The V4.1 indexer's own methods on a stand-in holding random weights (ReplicatedLinear needs a TP group)."""
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
        DeepseekV41Indexer,
        RMSNorm,
    )
    from sglang.srt.true_on_policy.dsv41_attention import project_bf16

    def weight(*shape):
        return torch.randn(*shape, generator=g, device=dev).bfloat16() * 0.05

    wq_b = weight(INDEX_HEADS * INDEX_HEAD_DIM, Q_LORA)
    ns = types.SimpleNamespace(
        n_local_heads=INDEX_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        rope_head_dim=ROPE_DIM,
        index_topk=INDEX_TOPK,
        owns_k=owns_k,
        is_candidate_source=candidate_source,
        uses_candidates=uses_candidates,
        candidate_topk_blocks=CANDIDATE_BLOCKS,
        candidate_block_size=CANDIDATE_BLOCK,
        head_weight_scale=INDEX_HEAD_DIM**-0.5 * INDEX_HEADS**-0.5,
        top=True,
        wq_b=lambda q_lora: (project_bf16(q_lora, wq_b), None),
        weights_proj=types.SimpleNamespace(weight=weight(INDEX_HEADS, HIDDEN)),
        wk=types.SimpleNamespace(weight=weight(INDEX_HEAD_DIM, HEAD_DIM)),
        k_norm=RMSNorm(INDEX_HEAD_DIM, EPS).to(dev),
    )
    for name in (
        "forward_wk",
        "index_keys",
        "queries",
        "head_weights_raw",
        "head_weights",
    ):
        setattr(ns, name, types.MethodType(getattr(DeepseekV41Indexer, name), ns))
    return ns


def _layers(ratio, g):
    """The ratio's first KV source (owning its index keys) and the index layer after it reading them."""
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Compressor
    from sglang.srt.model_loader.utils import set_default_torch_dtype

    source_id = KV_SOURCE_LAYERS[0] if ratio == 2 else KV_SOURCE_LAYERS[-1]
    with set_default_torch_dtype(torch.bfloat16):
        compressor = DeepseekV41Compressor(
            HIDDEN, HEAD_DIM, ratio, EPS, fused_compress=False
        ).to(dev)
    with torch.no_grad():
        for param in compressor.parameters():
            param.copy_(torch.randn(param.shape, generator=g, device=dev) * 0.05)
    ang = torch.randn(CTX, ROPE_DIM // 2, generator=g, device=dev)
    freqs = torch.polar(torch.ones_like(ang), ang)
    # Ratio 1 also runs the candidate source and a consumer of its mask, as the release's ratio-1 layers do.
    source = types.SimpleNamespace(
        layer_id=source_id,
        compress_ratio=ratio,
        rope_head_dim=ROPE_DIM,
        freqs_cis=freqs,
        compressor=compressor,
        indexer=_indexer(
            g, owns_k=True, candidate_source=ratio == 1, uses_candidates=False
        ),
    )
    reader = types.SimpleNamespace(
        layer_id=source_id + 1,
        compress_ratio=ratio,
        rope_head_dim=ROPE_DIM,
        freqs_cis=freqs,
        compressor=None,
        indexer=_indexer(
            g, owns_k=False, candidate_source=False, uses_candidates=ratio == 1
        ),
    )
    return source, reader


class _Mode:
    def __init__(self, decode):
        self.decode = decode

    def is_decode(self):
        return self.decode

    def is_target_verify(self):
        return False

    def is_extend(self):
        return not self.decode


class _Engine:
    """The backend state the TOP low-ratio program reads, around one pool and one page table."""

    def __init__(self, ratio, seed):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        self.ratio = ratio
        self.pool = _pool()
        g = torch.Generator(device=dev).manual_seed(seed)
        self.layers = _layers(ratio, g)
        if ratio == 2:
            # A previous occupant's pair rows, so a row that reads a partner it does not own shows up.
            ring = self.pool.get_attention_compress_states(self.layers[0].layer_id)
            ring.kv_score_buffer.kv_score.copy_(
                torch.randn(
                    ring.kv_score_buffer.kv_score.shape, generator=g, device=dev
                )
            )
        self.x = torch.randn(SEQ + 1, HIDDEN, generator=g, device=dev).bfloat16()
        self.q_lora = torch.randn(SEQ + 1, Q_LORA, generator=g, device=dev).bfloat16()
        # Every request maps a position to the same cache slot (page 0 is the pool's padding page).
        self.req_to_token = torch.zeros(MAX_REQS, CTX, dtype=torch.int32, device=dev)
        self.req_to_token[:, : SEQ + 1] = torch.arange(
            PAGE, PAGE + SEQ + 1, dtype=torch.int32, device=dev
        )
        self.backend = types.SimpleNamespace(
            token_to_kv_pool=self.pool,
            req_to_token=self.req_to_token,
            top_candidates=None,
        )
        for name in (
            "_top_low_ratio_sources",
            "_low_ratio_compress_torch",
            "_low_ratio_pair_partners",
            "_low_ratio_write_group",
        ):
            setattr(
                self.backend,
                name,
                types.MethodType(getattr(DeepseekV4AttnBackend, name), self.backend),
            )

    def forward(self, req, start, end):
        """Positions [start, end) of request ``req`` as one extend (one decode row when end == start + 1 == SEQ + 1);
        returns each index layer's request-local selections and the candidate mask."""
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            _low_ratio_compression_metadata,
        )

        pos = torch.arange(start, end, device=dev)
        rows = pos.numel()
        loc = self.req_to_token[req, start:end].to(torch.int64)
        out_loc, _ = _low_ratio_compression_metadata(self.ratio, pos + 1, loc)
        page_indices = torch.full((rows, INDEX_TOPK), -1, dtype=torch.int32, device=dev)
        raw_indices = torch.full_like(page_indices, -1)
        core = types.SimpleNamespace(
            raw_out_loc=loc,
            c1_out_loc=out_loc,
            c2_out_loc=out_loc,
            sparse_page_indices=lambda ratio: page_indices,
            sparse_raw_indices=lambda ratio: raw_indices,
        )
        width = types.SimpleNamespace(max_compressed_seq_len=CTX // self.ratio)
        self.backend.forward_metadata = types.SimpleNamespace(
            core_metadata=core, c1_indexer_metadata=width, c2_indexer_metadata=width
        )
        decode = end == SEQ + 1 and rows == 1
        forward_batch = types.SimpleNamespace(
            forward_mode=_Mode(decode),
            req_pool_indices=torch.tensor([req], device=dev),
            extend_seq_lens=torch.tensor([rows], device=dev),
            seq_lens_cpu=[end],
        )
        selections = []
        for layer in self.layers:
            self.backend._top_low_ratio_sources(
                layer, self.x[start:end], self.q_lora[start:end], pos, forward_batch
            )
            selections.append(raw_indices.clone())
        mask = self.backend.top_candidates
        self.backend.top_candidates = None
        return selections, None if mask is None else mask.clone()

    def cache_bytes(self):
        """The source layer's compressed KV and index-K buffers past their first page, which holds the slots of the
        padding page that every row completing no group writes."""
        layer_id = self.layers[0].layer_id
        return (
            _bits(self.pool.get_extra_key_buffer(layer_id))[1:].clone(),
            _bits(self.pool.get_index_k_with_scale_buffer(layer_id))[1:].clone(),
        )

    def ring_rows(self, req, end):
        """The request's pair ring after position ``end - 1``, ordered by position."""
        ring = self.pool.get_attention_compress_states(self.layers[0].layer_id)
        pos = torch.arange(end - ring.ring_size, end, device=dev)
        return ring.kv_score_buffer.kv_score[
            req * ring.ring_size + pos % ring.ring_size
        ]


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestDsv41RadixRows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.runtime_context import get_context

        # The pool reads the published config bags (spec, for the pair-ring size).
        override = get_context().override_server_args(page_size=PAGE)
        override.install()
        cls.addClassCleanup(override.restore)

    def _resume(self, ratio, hit):
        engine = _Engine(ratio, seed=100 + ratio)
        cold = engine.forward(REQ_A, 0, SEQ)
        warm = engine.forward(REQ_B, hit, SEQ)
        return engine, cold, warm

    def _assert_same(self, ratio, step, cold, warm, rows):
        """A's ``rows`` of each selection and its candidate mask against B's, then every cache byte of both pools."""
        (cold_out, cold_mask), (warm_out, warm_mask) = cold[1], warm[1]
        for layer, (a, b) in enumerate(zip(cold_out, warm_out)):
            self.assertTrue(
                torch.equal(a[rows], b),
                f"{ratio=} {step} index layer {layer}: selections differ",
            )
        if cold_mask is not None:
            self.assertTrue(
                torch.equal(cold_mask[rows], warm_mask),
                f"{ratio=} {step}: candidate masks differ",
            )
        for name, a, b in zip(
            ("compressed KV", "index K"), cold[0].cache_bytes(), warm[0].cache_bytes()
        ):
            self.assertTrue(a.any(), f"{ratio=} {step}: no {name} written")
            self.assertTrue(
                torch.equal(a, b),
                f"{ratio=} {step}: {int((a != b).sum())} {name} bytes differ",
            )

    def test_page_aligned_hit_equals_cold_prefill(self):
        for ratio in (2, 1):
            with self.subTest(ratio=ratio):
                cold, warm = (
                    _Engine(ratio, seed=100 + ratio),
                    _Engine(ratio, seed=100 + ratio),
                )
                a = cold.forward(REQ_A, 0, SEQ)
                warm.forward(REQ_C, 0, HIT)
                b = warm.forward(REQ_B, HIT, SEQ)
                self.assertTrue((a[0][0][HIT:SEQ] >= 0).any(), "nothing selected")
                self._assert_same(
                    ratio, "prefill", (cold, a), (warm, b), slice(HIT, SEQ)
                )
                if ratio == 2:
                    self.assertTrue(
                        torch.equal(
                            cold.ring_rows(REQ_A, SEQ), warm.ring_rows(REQ_B, SEQ)
                        ),
                        "pair rings differ after the prefill",
                    )
                # The decode after it completes a pair from the ring.
                a = cold.forward(REQ_A, SEQ, SEQ + 1)
                b = warm.forward(REQ_B, SEQ, SEQ + 1)
                self._assert_same(ratio, "decode", (cold, a), (warm, b), slice(None))

    def test_hit_inside_a_pair_differs(self):
        """The control: resumed on a pair's odd token, B pools it with a partner from another occupant's ring."""
        cold, warm = _Engine(2, seed=102), _Engine(2, seed=102)
        cold.forward(REQ_A, 0, SEQ)
        warm.forward(REQ_C, 0, HIT - 1)
        warm.forward(REQ_B, HIT - 1, SEQ)
        self.assertFalse(
            all(
                torch.equal(a, b)
                for a, b in zip(cold.cache_bytes(), warm.cache_bytes())
            ),
            "a hit inside a pair matched the cold prefill",
        )


if __name__ == "__main__":
    unittest.main()
