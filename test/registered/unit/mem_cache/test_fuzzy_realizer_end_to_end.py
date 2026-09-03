"""End-to-end realization math on real tensors.

Donor K written at donor positions must, after realization, equal K
written directly at the target positions, and V must move unchanged,
using the same rotary helper the model uses. Covers both realizer
paths: contiguous (fresh slots, req_to_token repoint, protected-prefix
narrowing) and scattered segments (extend-slot displacement, per-segment
copy, surplus slot release). Mock-based tests cannot guard this: the
whole feature is the position math.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.mem_cache.fuzzy_match.fuzzy_match_provider import FuzzyMatchSegment
from sglang.srt.mem_cache.fuzzy_match.realizer import FuzzyKVRealizer
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.test_utils import CustomTestCase

HEAD_DIM = 64
NUM_HEADS = 2
NUM_LAYERS = 2
POOL_SIZE = 512
MAX_POS = 4096


def _cos_sin_cache() -> torch.Tensor:
    inv_freq = 1.0 / (
        10_000 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    freqs = torch.einsum(
        "i,j->ij", torch.arange(MAX_POS, dtype=torch.float32), inv_freq
    )
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _rotate_at(k_raw, positions, cache):
    cos, sin = cache.index_select(0, positions).chunk(2, dim=-1)
    return apply_rotary_emb(k_raw, cos, sin, True)


def _fake_pool() -> MHATokenToKVPool:
    pool = object.__new__(MHATokenToKVPool)
    pool.layer_num = NUM_LAYERS
    pool.k_buffer = [
        torch.zeros(POOL_SIZE, NUM_HEADS, HEAD_DIM) for _ in range(NUM_LAYERS)
    ]
    pool.v_buffer = [
        torch.zeros(POOL_SIZE, NUM_HEADS, HEAD_DIM) for _ in range(NUM_LAYERS)
    ]
    return pool


def _realizer(pool, req_to_token, freed):
    allocator = SimpleNamespace(
        get_kvcache=lambda: pool,
        free=lambda locs: freed.append(locs.clone()),
    )
    cache = _cos_sin_cache()
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        rotary_emb=SimpleNamespace(
                            cos_sin_cache=cache,
                            is_neox_style=True,
                            rotary_dim=HEAD_DIM,
                        )
                    )
                )
            ]
        )
    )
    realizer = FuzzyKVRealizer(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        token_to_kv_pool_allocator=allocator,
        model=model,
    )
    return realizer, cache


class TestContiguousRealization(CustomTestCase):
    def test_donor_kv_lands_at_target_positions(self):
        torch.manual_seed(3)
        n, exact = 8, 4
        pool = _fake_pool()
        req_to_token = torch.zeros(4, 64, dtype=torch.int64)
        freed = []
        realizer, cache = _realizer(pool, req_to_token, freed)

        donor_pos = torch.arange(100, 100 + n, dtype=torch.long)
        target_pos = torch.arange(exact, exact + n, dtype=torch.long)
        donor_locs = torch.arange(0, n, dtype=torch.long)
        realized_locs = torch.arange(64, 64 + n, dtype=torch.long)

        k_raw = torch.randn(n, NUM_HEADS, HEAD_DIM)
        v_ref = torch.randn(n, NUM_HEADS, HEAD_DIM)
        for layer in range(NUM_LAYERS):
            pool.k_buffer[layer][donor_locs] = _rotate_at(k_raw, donor_pos, cache)
            pool.v_buffer[layer][donor_locs] = v_ref

        req_to_token[1, target_pos] = donor_locs
        req = SimpleNamespace(
            kv=SimpleNamespace(
                req_pool_idx=1,
                cache_protected_len=exact + n,
                cache_fuzzy_matched_len=n,
                fuzzy_match_result=SimpleNamespace(
                    segments=None,
                    cached_start_pos=100,
                    layer_zero_mask=None,
                    quality_signals=None,
                ),
                fuzzy_realized_locs=realized_locs,
            ),
            prefix_indices=torch.arange(0, exact + n, dtype=torch.long),
        )

        realizer.realize(fuzzy_reqs=[req])

        expected_k = _rotate_at(k_raw, target_pos, cache)
        for layer in range(NUM_LAYERS):
            torch.testing.assert_close(
                pool.k_buffer[layer][realized_locs], expected_k, atol=1e-4, rtol=1e-4
            )
            torch.testing.assert_close(pool.v_buffer[layer][realized_locs], v_ref)
        self.assertTrue(torch.equal(req_to_token[1, target_pos], realized_locs))
        self.assertEqual(req.kv.cache_protected_len, exact)
        self.assertIsNone(req.kv.fuzzy_realized_locs)
        self.assertEqual(req.kv.cache_fuzzy_matched_len, 0)


class TestAlignedParaphraseRealization(CustomTestCase):
    def test_verified_paraphrase_is_copied_with_zero_delta(self):
        """A verified paraphrase served whole sits at the same positions as
        the exact prefix boundary (delta 0). It is different content, so it
        must still be copied into fresh request-owned slots instead of being
        skipped as exact-tree content: skipping left the request reading the
        donor's slots and inserting them into its own tree branch."""
        torch.manual_seed(5)
        n, exact = 8, 4
        pool = _fake_pool()
        req_to_token = torch.zeros(4, 64, dtype=torch.int64)
        freed = []
        realizer, cache = _realizer(pool, req_to_token, freed)

        pos = torch.arange(exact, exact + n, dtype=torch.long)
        donor_locs = torch.arange(0, n, dtype=torch.long)
        realized_locs = torch.arange(64, 64 + n, dtype=torch.long)
        k_raw = torch.randn(n, NUM_HEADS, HEAD_DIM)
        v_ref = torch.randn(n, NUM_HEADS, HEAD_DIM)
        for layer in range(NUM_LAYERS):
            pool.k_buffer[layer][donor_locs] = _rotate_at(k_raw, pos, cache)
            pool.v_buffer[layer][donor_locs] = v_ref

        req_to_token[1, pos] = donor_locs
        req = SimpleNamespace(
            kv=SimpleNamespace(
                req_pool_idx=1,
                cache_protected_len=exact + n,
                cache_fuzzy_matched_len=n,
                fuzzy_match_result=SimpleNamespace(
                    segments=None,
                    cached_start_pos=exact,
                    layer_zero_mask=None,
                    quality_signals=SimpleNamespace(confidence_tier="paraphrase_verified"),
                ),
                fuzzy_realized_locs=realized_locs,
            ),
            prefix_indices=torch.arange(0, exact + n, dtype=torch.long),
        )

        realizer.realize(fuzzy_reqs=[req])

        for layer in range(NUM_LAYERS):
            torch.testing.assert_close(
                pool.k_buffer[layer][realized_locs], _rotate_at(k_raw, pos, cache), atol=1e-4, rtol=1e-4
            )
            torch.testing.assert_close(pool.v_buffer[layer][realized_locs], v_ref)
        self.assertTrue(torch.equal(req_to_token[1, pos], realized_locs))
        self.assertEqual(freed, [])
        self.assertEqual(req.kv.cache_protected_len, exact)

    def test_aligned_exact_tier_is_still_skipped(self):
        n, exact = 8, 4
        pool = _fake_pool()
        req_to_token = torch.zeros(4, 64, dtype=torch.int64)
        freed = []
        realizer, _cache = _realizer(pool, req_to_token, freed)
        realized_locs = torch.arange(64, 64 + n, dtype=torch.long)
        req = SimpleNamespace(
            kv=SimpleNamespace(
                req_pool_idx=1,
                cache_protected_len=exact + n,
                cache_fuzzy_matched_len=n,
                fuzzy_match_result=SimpleNamespace(
                    segments=None,
                    cached_start_pos=exact,
                    layer_zero_mask=None,
                    quality_signals=SimpleNamespace(confidence_tier="exact"),
                ),
                fuzzy_realized_locs=realized_locs,
            ),
            prefix_indices=torch.arange(0, exact + n, dtype=torch.long),
        )
        realizer.realize(fuzzy_reqs=[req])
        self.assertEqual(len(freed), 1)
        self.assertTrue(torch.equal(freed[0], realized_locs))


class TestSegmentsRealization(CustomTestCase):
    def test_segments_displace_extend_slots_and_release_surplus(self):
        torch.manual_seed(5)
        n = 6
        pool = _fake_pool()
        req_to_token = torch.zeros(4, 64, dtype=torch.int64)
        freed = []
        realizer, cache = _realizer(pool, req_to_token, freed)

        target_pos = torch.arange(20, 20 + n, dtype=torch.long)
        donor_pos = torch.arange(300, 300 + n, dtype=torch.long)
        donor_locs = torch.arange(0, n, dtype=torch.long)
        displaced = torch.arange(200, 200 + n, dtype=torch.long)
        # One surplus slot beyond the segment: must be released, not leaked.
        realized_locs = torch.arange(64, 64 + n + 1, dtype=torch.long)

        k_raw = torch.randn(n, NUM_HEADS, HEAD_DIM)
        for layer in range(NUM_LAYERS):
            pool.k_buffer[layer][donor_locs] = _rotate_at(k_raw, donor_pos, cache)

        req_to_token[2, target_pos] = displaced
        segment = FuzzyMatchSegment(
            target_positions=target_pos,
            donor_positions=donor_pos,
            donor_kv_indices=donor_locs,
            length=n,
        )
        req = SimpleNamespace(
            kv=SimpleNamespace(
                req_pool_idx=2,
                cache_protected_len=0,
                cache_fuzzy_matched_len=n,
                fuzzy_match_result=SimpleNamespace(
                    segments=[segment],
                    cached_start_pos=0,
                    layer_zero_mask=None,
                ),
                fuzzy_realized_locs=realized_locs,
            ),
            prefix_indices=torch.arange(0, 32, dtype=torch.long),
        )

        realizer.realize(fuzzy_reqs=[req])

        expected_k = _rotate_at(k_raw, target_pos, cache)
        torch.testing.assert_close(
            pool.k_buffer[0][realized_locs[:n]], expected_k, atol=1e-4, rtol=1e-4
        )
        self.assertTrue(torch.equal(req_to_token[2, target_pos], realized_locs[:n]))
        self.assertTrue(any(torch.equal(f, displaced) for f in freed))
        self.assertTrue(
            any(f.numel() == 1 and f.item() == realized_locs[-1].item() for f in freed)
        )
        self.assertIsNone(req.kv.fuzzy_realized_locs)


if __name__ == "__main__":
    unittest.main()
