"""PD DCP1 prefill -> DCP-N decode for DSA models (GLM-5.2) on ROCm.

The decode stripes the target latent KV and, on ROCm, the MTP draft latent KV
by position, while the index-K cache stays replicated over the widened slots.
These tests simulate the transfers on host byte arrays.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager, PrefillServerInfo
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.mem_cache.common import suffix_page_indices
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _mgr_stub(*, page_size, kv_item_lens, num_draft_entries):
    return SimpleNamespace(
        kv_args=SimpleNamespace(
            page_size=page_size,
            kv_item_lens=kv_item_lens,
            num_draft_entries=num_draft_entries,
        )
    )


class TestDraftLayoutPolicy(CustomTestCase):
    def test_missing_backend_policy_preserves_replicated_draft(self):
        from sglang.srt.configs import model_config
        from sglang.srt.disaggregation.common.conn import _decode_draft_shards_under_dcp

        with patch.object(
            model_config, "dsa_draft_shards_under_dcp", None, create=True
        ):
            self.assertFalse(_decode_draft_shards_under_dcp(4))

    def test_uses_backend_policy_when_available(self):
        from sglang.srt.configs import model_config
        from sglang.srt.disaggregation.common.conn import _decode_draft_shards_under_dcp

        config = object()
        with (
            patch.object(
                model_config,
                "dsa_draft_shards_under_dcp",
                return_value=True,
                create=True,
            ) as policy,
            patch(
                "sglang.srt.runtime_context.process_model_config",
                return_value=SimpleNamespace(hf_config=config),
            ),
        ):
            self.assertTrue(_decode_draft_shards_under_dcp(4))
            policy.assert_called_once_with(config, 4)

    def test_unported_backends_fail_closed_for_sharded_drafts(self):
        for backend in ("nixl", "mori"):
            with self.subTest(backend=backend):
                manager = SimpleNamespace(dcp_draft_sharded=lambda _: True)
                with self.assertRaisesRegex(NotImplementedError, "mooncake"):
                    CommonKVManager.reject_sharded_dcp_draft(manager, 4, backend)
                manager.dcp_draft_sharded = lambda _: False
                CommonKVManager.reject_sharded_dcp_draft(manager, 4, backend)


class TestShardedDraftItemLens(CustomTestCase):
    """T1: a DCP-sharded draft pages like the target, not over the widened page."""

    def test_sharded_draft_uses_target_page_scale(self):
        mgr = _mgr_stub(
            page_size=64, kv_item_lens=[64 * 576, 64 * 576], num_draft_entries=1
        )
        token_lens = CommonKVManager.prepare_dcp_token_item_lens(
            mgr, [64 * 576, 64 * 576], dst_dcp_size=4, draft_sharded=True
        )
        self.assertEqual(token_lens, [576, 576])

    def test_sharded_draft_rejects_widened_dst(self):
        mgr = _mgr_stub(
            page_size=64, kv_item_lens=[64 * 576, 64 * 576], num_draft_entries=1
        )
        with self.assertRaisesRegex(RuntimeError, "geometry differs at entry 1"):
            CommonKVManager.prepare_dcp_token_item_lens(
                mgr, [64 * 576, 4 * 64 * 576], dst_dcp_size=4, draft_sharded=True
            )

    def test_draft_sharded_follows_model_rule(self):
        mgr = _mgr_stub(page_size=64, kv_item_lens=[1, 1], num_draft_entries=1)
        target = "sglang.srt.disaggregation.common.conn._decode_draft_shards_under_dcp"
        with patch(target, return_value=True):
            self.assertTrue(CommonKVManager.dcp_draft_sharded(mgr, 4))
        with patch(target, return_value=False):
            self.assertFalse(CommonKVManager.dcp_draft_sharded(mgr, 4))
        no_draft = _mgr_stub(page_size=64, kv_item_lens=[1], num_draft_entries=0)
        with patch(target, return_value=True):
            self.assertFalse(CommonKVManager.dcp_draft_sharded(no_draft, 4))


class _HostMemory:
    """Byte regions at fake addresses; applies (src, dst, length) blocks."""

    def __init__(self):
        self.regions = []

    def alloc(self, nbytes: int) -> int:
        base = (len(self.regions) + 1) << 40
        self.regions.append((base, np.zeros(nbytes, dtype=np.uint8)))
        return base

    def view(self, addr: int, nbytes: int) -> np.ndarray:
        for base, buf in self.regions:
            if base <= addr < base + buf.size:
                off = addr - base
                assert off + nbytes <= buf.size, "block overruns its region"
                return buf[off : off + nbytes]
        raise AssertionError(f"unknown address {addr:#x}")

    def apply(self, blocks):
        for src, dst, length in blocks:
            self.view(dst, length)[:] = self.view(src, length)


def _fill_rows(mem, ptr, rows, row_bytes, values):
    for row, value in zip(rows, values):
        mem.view(ptr + int(row) * row_bytes, row_bytes)[:] = value % 251 + 1


def _relayout(
    test,
    *,
    num_tokens,
    src_pages,
    dst_pages,
    dcp,
    src_layer_ids,
    dst_layer_ids,
    prefill_draft_entries,
    draft_sharded,
    prefill_attn_tp_size,
    chunk_pages=None,
):
    """Send one prefill rank's KV to every decode DCP rank; check each rank holds
    exactly its owned positions in every target (and sharded draft) entry."""
    page, row = 4, 8  # tokens per page, bytes per token per entry
    n_rows_src = (max(src_pages) + 1) * page
    n_rows_dst = (max(dst_pages) + 1) * page
    src_rows = [src_pages[p // page] * page + p % page for p in range(num_tokens)]
    for rank in range(dcp):
        mem = _HostMemory()
        src_ptrs = [mem.alloc(n_rows_src * row) for _ in src_layer_ids]
        dst_ptrs = [mem.alloc(n_rows_dst * row) for _ in dst_layer_ids]
        for e, ptr in enumerate(src_ptrs):
            values = [p * 3 + src_layer_ids[e] for p in range(num_tokens)]
            _fill_rows(mem, ptr, src_rows, row, values)

        mgr = object.__new__(MooncakeKVManager)
        mgr.kv_args = SimpleNamespace(
            page_size=page,
            kv_data_ptrs=src_ptrs,
            kv_item_lens=[page * row] * len(src_layer_ids),
            kv_layer_ids=src_layer_ids,
            num_draft_entries=prefill_draft_entries,
            engine_rank=0,
        )
        mgr.attn_tp_size = prefill_attn_tp_size
        mgr.is_mla_backend = True
        mgr.enable_custom_mem_pool = False
        mgr.max_transfer_batch_indices = 4096
        mgr._transfer_data = lambda _session, blocks, _mem=mem: (_mem.apply(blocks), 0)[
            1
        ]

        start = 0
        for n in chunk_pages or [len(src_pages)]:
            ret = MooncakeKVManager.send_kvcache_dcp(
                mgr,
                "session",
                np.asarray(src_pages[start : start + n], dtype=np.int32),
                dst_ptrs,
                np.asarray(dst_pages, dtype=np.int32),
                dcp_token_item_lens=[row] * len(src_layer_ids),
                dst_dcp_size=dcp,
                dst_dcp_rank=rank,
                src_page_offset=start,
                decode_prefix_len=0,
                num_kv_tokens=min(n * page, num_tokens - start * page),
                executor=None,
                dst_layer_ids=dst_layer_ids,
                pack_buffer=None,
                dst_kv_item_lens=[page * row] * len(dst_layer_ids),
                dst_tp_rank=rank,
                dst_attn_tp_size=dcp,
                draft_sharded=draft_sharded,
            )
            test.assertEqual(ret, 0)
            start += n

        owned = [p for p in range(num_tokens) if p % dcp == rank]
        for j, (lid, ptr) in enumerate(zip(dst_layer_ids, dst_ptrs)):
            buf = mem.view(ptr, n_rows_dst * row).reshape(n_rows_dst, row)
            written = int(np.count_nonzero(buf[:, 0]))
            if lid not in src_layer_ids:
                test.assertEqual(written, 0, f"rank {rank} entry {lid} has no source")
                continue
            test.assertEqual(written, len(owned), f"rank {rank} entry {lid} row count")
            for p in owned:
                local = p // dcp
                dst_row = dst_pages[local // page] * page + local % page
                want = (p * 3 + lid) % 251 + 1
                test.assertTrue(
                    np.all(buf[dst_row] == want),
                    f"rank {rank} entry {lid} pos {p}: got {buf[dst_row][0]} want {want}",
                )


class TestShardedDraftRelayout(CustomTestCase):
    """T2: with a sharded draft, the draft rows follow the target's owner rule."""

    def _run(self, **kwargs):
        layer_ids = [0, 1, 78]  # two target layers + the NextN draft layer
        _relayout(
            self,
            dcp=2,
            src_layer_ids=layer_ids,
            dst_layer_ids=layer_ids,
            prefill_draft_entries=1,
            draft_sharded=True,
            prefill_attn_tp_size=2,
            **kwargs,
        )

    def test_one_virtual_page(self):
        self._run(num_tokens=8, src_pages=[5, 2], dst_pages=[3])

    def test_partial_last_page_and_chunks(self):
        self._run(
            num_tokens=21,
            src_pages=[9, 0, 4, 7, 1, 6],
            dst_pages=[2, 0, 5],
            chunk_pages=[2, 2, 2],
        )


class TestDpaPrefillFanOut(CustomTestCase):
    """The 22k shape on a DCP decode: one DP-attention prefill rank (attn TP1, no
    EAGLE) feeds all four TP4/DCP4 decode ranks."""

    def test_rank_mapping_fans_out_to_every_dcp_rank(self):
        info = PrefillServerInfo(
            attn_tp_size=1,
            attn_cp_size=1,
            dp_size=4,
            pp_size=1,
            page_size=64,
            kv_cache_dtype="fp8_e4m3",
            follow_bootstrap_room=True,
        )
        for engine_rank in range(4):
            mgr = SimpleNamespace(
                is_mla_backend=True,
                is_hybrid_mla_backend=False,
                attn_tp_size=4,
                attn_cp_size=1,
                attn_cp_rank=0,
                pp_size=1,
                pp_rank=0,
                enable_all_cp_ranks_for_transfer=False,
                kv_args=SimpleNamespace(engine_rank=engine_rank),
            )
            CommonKVManager._resolve_rank_mapping(mgr, info)
            self.assertEqual(info.target_tp_ranks, [0])
            self.assertEqual(info.required_dst_info_num, 4)
            self.assertEqual(info.required_prefill_response_num, 1)

    def test_target_only_prefill_to_dcp4_decode_with_draft(self):
        # Mode N: the prefill has no draft entry; the decode's NextN entry
        # (layer 78) must be skipped, not paired with a target layer.
        _relayout(
            self,
            num_tokens=45,
            src_pages=list(range(11, -1, -1)),
            dst_pages=[1, 0, 2],
            dcp=4,
            src_layer_ids=[0, 1],
            dst_layer_ids=[0, 1, 78],
            prefill_draft_entries=0,
            draft_sharded=False,
            prefill_attn_tp_size=1,
            chunk_pages=[5, 7],
        )


class TestDpaPrefillFanOutToDp2Dcp2(CustomTestCase):
    """E2 shape: the DP-attention prefill (attn TP1) feeds a decode with DP2
    attention over TP4, i.e. attn TP2 with DCP2 inside each decode DP group."""

    def test_rank_mapping_fans_out_to_both_dcp_ranks_of_a_group(self):
        info = PrefillServerInfo(
            attn_tp_size=1,
            attn_cp_size=1,
            dp_size=4,
            pp_size=1,
            page_size=64,
            kv_cache_dtype="fp8_e4m3",
            follow_bootstrap_room=True,
        )
        for engine_rank in range(4):
            mgr = SimpleNamespace(
                is_mla_backend=True,
                is_hybrid_mla_backend=False,
                attn_tp_size=2,
                attn_cp_size=1,
                attn_cp_rank=0,
                pp_size=1,
                pp_rank=0,
                enable_all_cp_ranks_for_transfer=False,
                kv_args=SimpleNamespace(engine_rank=engine_rank),
            )
            CommonKVManager._resolve_rank_mapping(mgr, info)
            self.assertEqual(info.target_tp_ranks, [0])
            self.assertEqual(info.required_dst_info_num, 2)
            self.assertEqual(info.required_prefill_response_num, 1)

    def test_target_only_prefill_to_dcp2_decode_with_draft(self):
        _relayout(
            self,
            num_tokens=45,
            src_pages=list(range(11, -1, -1)),
            dst_pages=[2, 0, 5, 1, 3, 4],
            dcp=2,
            src_layer_ids=[0, 1],
            dst_layer_ids=[0, 1, 78],
            prefill_draft_entries=0,
            draft_sharded=False,
            prefill_attn_tp_size=1,
            chunk_pages=[5, 7],
        )


class TestPpPrefillToDcpDecode(CustomTestCase):
    """PP prefill: each stage sends its own layer range; the unpipelined DCP4
    decode registers all 78 target layers plus the NextN draft (79 entries)."""

    STAGES = [
        (0, 20),
        (20, 40),
        (40, 60),
        (60, 78),
    ]  # SGLANG_PP_LAYER_PARTITION=20,20,20,18

    def test_each_stage_lands_on_its_layers(self):
        dst_layer_ids = list(range(79))
        for start, end in self.STAGES:
            with self.subTest(stage=(start, end)):
                _relayout(
                    self,
                    num_tokens=21,
                    src_pages=[3, 0, 5, 1, 2, 4],
                    dst_pages=[1, 0],
                    dcp=4,
                    src_layer_ids=list(range(start, end)),
                    dst_layer_ids=dst_layer_ids,
                    prefill_draft_entries=0,
                    draft_sharded=False,
                    prefill_attn_tp_size=1,
                )

    def test_index_k_state_slices_by_stage_offset(self):
        # StateType.DSA has no layer ids: the stage's entries map positionally from
        # prefill_start_layer; the last stage's draft index-K follows its target.
        dst = list(range(1000, 1079))
        for (start, end), n_draft in zip(self.STAGES, (0, 0, 0, 1)):
            n_src = end - start + n_draft
            mgr = object.__new__(MooncakeKVManager)
            mgr.kv_args = SimpleNamespace(
                prefill_start_layer=start,
                mla_compression_ratios=None,
                num_draft_entries=n_draft,
            )
            src = list(range(n_src))
            _, sliced, n = CommonKVManager.get_mla_kv_ptrs_with_pp(
                mgr, src, dst, state_type=StateType.DSA
            )
            self.assertEqual(n, n_src)
            self.assertEqual(sliced, dst[start : start + n_src])


class TestIndexKPagePairing(CustomTestCase):
    """T3: replicated index K pairs prefill 64-slot pages with decode index pages
    over the widened slots, through the generic DSA page path."""

    P, W = 64, 4

    def _simulate(self, seq_len, prefix_len, seed):
        P, W = self.P, self.W
        rng = np.random.default_rng(seed)
        n_src_pages = -(-seq_len // P)
        n_dst_vpages = -(-seq_len // (P * W))
        src_pages = rng.permutation(4 * n_src_pages + 3)[:n_src_pages] + 1
        dst_vpages = rng.permutation(4 * n_dst_vpages + 3)[:n_dst_vpages] + 1
        pos = np.arange(seq_len)
        src_slots = src_pages[pos // P] * P + pos % P
        dst_slots = dst_vpages[pos // (P * W)] * (P * W) + pos % (P * W)
        width = n_dst_vpages * P * W + P
        p_req = torch.zeros((1, width), dtype=torch.int64)
        d_req = torch.zeros((1, width), dtype=torch.int64)
        p_req[0, :seq_len] = torch.from_numpy(src_slots)
        d_req[0, :seq_len] = torch.from_numpy(dst_slots)

        # Prefill index rows: one row per 64-slot page, entry = position + 1.
        src_buf = np.zeros(((src_pages.max() + 1), P), dtype=np.int64)
        src_buf[src_slots // P, src_slots % P] = pos + 1
        # Decode replicated index K: rows by widened slot // 64.
        dst_buf = np.zeros(((dst_vpages.max() + 1) * W, P), dtype=np.int64)

        src_list = suffix_page_indices(p_req, 0, prefix_len, seq_len, P)
        dst_list = suffix_page_indices(d_req, 0, prefix_len, seq_len, P)
        self.assertEqual(len(src_list), len(dst_list))
        dst_buf[dst_list] = src_buf[src_list]

        got = dst_buf[dst_slots // P, dst_slots % P]
        want = np.where(pos >= prefix_len, pos + 1, 0)
        np.testing.assert_array_equal(got[prefix_len:], want[prefix_len:])

    def test_pairs_every_position(self):
        for seq_len in (1, 63, 64, 65, 255, 256, 257, 10000):
            for prefix_len in (0, 256):
                if prefix_len >= seq_len:
                    continue
                with self.subTest(seq_len=seq_len, prefix_len=prefix_len):
                    self._simulate(seq_len, prefix_len, seed=seq_len + prefix_len)


if __name__ == "__main__":
    unittest.main()
