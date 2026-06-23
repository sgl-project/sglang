"""Parity tests: RustUnifiedRadixCache vs the Python UnifiedRadixCache."""

import unittest
from array import array

import torch
from test_unified_radix_cache_unittest import (
    CacheConfig,
    ComponentType,
    build_fixture,
)

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.rust_unified_radix_cache import RustUnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


# FULL at two page sizes, plus device-tier SWA and Mamba. SWA is page_size=1
# only (the fixture's SWA allocator only supports single-token alloc).
_CONFIGS = [
    CacheConfig(page_size=1),
    CacheConfig(page_size=16),
    CacheConfig(
        page_size=1,
        components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=4,
    ),
    CacheConfig(page_size=1, components=(ComponentType.FULL, ComponentType.MAMBA)),
]


class RustParitySuite:
    cfg: CacheConfig
    _rid: int = 0

    # ----- fixture + op helpers -----

    def _build_pair(self):
        ref = build_fixture(self.cfg)
        rust = build_fixture(self.cfg, tree_cls=RustUnifiedRadixCache)
        return ref, rust

    def _make_seq(self, start: int, num_pages: int) -> list[int]:
        page_size = self.cfg.page_size
        return list(range(start, start + num_pages * page_size))

    def _make_req(self, req_to_token_pool):
        req = Req(
            rid=self._rid,
            origin_input_text="",
            origin_input_ids=[],
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        self._rid += 1
        req_to_token_pool.alloc([req])
        return req

    def _insert(self, fixture, tokens):
        tree, allocator, req_to_token_pool = fixture
        key = RadixKey(array("q", tokens))
        value = allocator.alloc(len(tokens))
        self.assertIsNotNone(value, "allocator ran out of slots in the fixture")
        params = InsertParams(key=key, value=value[: len(key)])
        if self.cfg.has_mamba:
            req = self._make_req(req_to_token_pool)
            params.mamba_value = req.mamba_pool_idx.unsqueeze(0)
        return tree.insert(params)

    def _match_len(self, fixture, tokens) -> int:
        tree = fixture[0]
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
        return len(m.device_indices)

    def _assert_sizes_equal(self, ref, rust, msg=""):
        ref_tree, rust_tree = ref[0], rust[0]
        self.assertEqual(
            ref_tree.evictable_size(), rust_tree.evictable_size(), f"evictable {msg}"
        )
        self.assertEqual(
            ref_tree.protected_size(), rust_tree.protected_size(), f"protected {msg}"
        )
        self.assertEqual(
            ref_tree.total_size(), rust_tree.total_size(), f"total {msg}"
        )

    # ----- parity scenarios -----

    def test_insert_and_match_parity(self):
        ref, rust = self._build_pair()
        seq_a = self._make_seq(1, 2)
        seq_b = seq_a + self._make_seq(1000, 1)

        self._insert(ref, seq_a)
        self._insert(rust, seq_a)
        r_ins = self._insert(ref, seq_b)
        u_ins = self._insert(rust, seq_b)
        self.assertEqual(r_ins.prefix_len, u_ins.prefix_len)

        for probe in (
            seq_b,
            seq_a + self._make_seq(9000, 1),
            self._make_seq(5000, 2),
        ):
            self.assertEqual(
                self._match_len(ref, probe),
                self._match_len(rust, probe),
                f"match length mismatch for probe starting {probe[0]}",
            )
        self._assert_sizes_equal(ref, rust, "after inserts")
        ref[0].sanity_check()
        rust[0].sanity_check()

    def test_shared_prefix_split_parity(self):
        ref, rust = self._build_pair()
        base = self._make_seq(1, 2)
        branch_a = base + self._make_seq(100, 2)
        branch_b = base + self._make_seq(200, 2)

        for seq in (base, branch_a, branch_b):
            r = self._insert(ref, seq)
            u = self._insert(rust, seq)
            self.assertEqual(r.prefix_len, u.prefix_len, f"prefix_len for {seq[0]}")

        for probe in (branch_a, branch_b, base + self._make_seq(999, 1)):
            self.assertEqual(
                self._match_len(ref, probe), self._match_len(rust, probe)
            )
        self._assert_sizes_equal(ref, rust, "after split")
        ref[0].sanity_check()
        rust[0].sanity_check()

    def test_evict_parity(self):
        ref, rust = self._build_pair()
        seq_a = self._make_seq(1, 2)
        seq_b = self._make_seq(500, 2)
        for seq in (seq_a, seq_b):
            self._insert(ref, seq)
            self._insert(rust, seq)
        self._assert_sizes_equal(ref, rust, "before evict")

        r_ev = ref[0].evict(EvictParams(num_tokens=len(seq_a)))
        u_ev = rust[0].evict(EvictParams(num_tokens=len(seq_a)))
        self.assertEqual(r_ev.num_tokens_evicted, u_ev.num_tokens_evicted)
        self._assert_sizes_equal(ref, rust, "after evict")
        ref[0].sanity_check()
        rust[0].sanity_check()

    def test_lock_ref_parity(self):
        ref, rust = self._build_pair()
        seq = self._make_seq(1, 3)
        self._insert(ref, seq)
        self._insert(rust, seq)

        r_match = ref[0].match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
        u_match = rust[0].match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
        ref[0].inc_lock_ref(r_match.last_device_node)
        rust[0].inc_lock_ref(u_match.last_device_node)
        # Locked tokens are protected, not evictable, in both implementations.
        self.assertEqual(ref[0].protected_size(), rust[0].protected_size())
        self.assertEqual(ref[0].evictable_size(), rust[0].evictable_size())

        ref[0].dec_lock_ref(r_match.last_device_node)
        rust[0].dec_lock_ref(u_match.last_device_node)
        self._assert_sizes_equal(ref, rust, "after unlock")
        ref[0].sanity_check()
        rust[0].sanity_check()


# Generate one concrete TestCase per config: Test_<label>.
for _cfg in _CONFIGS:
    _name = f"TestRustParity_{_cfg.label}"
    globals()[_name] = type(
        _name, (RustParitySuite, CustomTestCase), {"cfg": _cfg}
    )

if __name__ == "__main__":
    unittest.main()
