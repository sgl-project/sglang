import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import _resolve_dflash_draft_kv_ring
from sglang.srt.disaggregation.utils import draft_kv_ring_row_pairs
from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.platforms import current_platform
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_draft_ring import (
    DFlashDraftKVRing,
    draft_kv_ring_paged_rows_per_req,
    draft_kv_ring_pool_rows,
    draft_kv_ring_rows_per_req,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HOOK_MODULE = "sglang.srt.arg_groups.speculative_hook"
SWITCH = envs.SGLANG_ENABLE_DFLASH_DRAFT_KV_RING
SWA_MLA_DRAFT = {
    "num_hidden_layers": 5,
    "dflash_config": {"attention_mode": "mla"},
    "layer_types": ["sliding_attention"] * 5,
    "sliding_window": 1024,
}
SWA_MHA_DRAFT = {
    "num_hidden_layers": 5,
    "layer_types": ["sliding_attention"] * 5,
    "sliding_window": 2048,
}


@contextlib.contextmanager
def _switch(value=True):
    """``value=None`` leaves the switch unset, so its default applies."""
    with SWITCH.override(True if value is None else value):
        if value is None:
            SWITCH.clear()
        yield


def _args(**overrides) -> ServerArgs:
    kwargs = dict(
        model_path="dummy",
        speculative_algorithm="DFLASH",
        speculative_draft_model_path="draft",
        speculative_num_draft_tokens=8,
        speculative_draft_attention_backend="cutedsl_mla",
        dcp_size=8,
    )
    kwargs.update(overrides)
    return ServerArgs(**kwargs)


class TestResolveDraftKVRing(CustomTestCase):
    def _resolve(self, args, draft_cfg=SWA_MLA_DRAFT) -> bool:
        """``draft_cfg=None``: the decision must not load the draft config."""
        load = patch(
            f"{HOOK_MODULE}._load_draft_hf_config",
            return_value=draft_cfg,
            side_effect=None if draft_cfg else AssertionError("draft config loaded"),
        )
        with load:
            _resolve_dflash_draft_kv_ring(args)
        return resolution_result(args, "speculative_dflash_draft_kv_ring")

    def _assert_rejected(self, args, draft_cfg=SWA_MLA_DRAFT):
        with self.assertRaisesRegex(ValueError, "SGLANG_ENABLE_DFLASH_DRAFT_KV_RING=1"):
            self._resolve(args, draft_cfg=draft_cfg)

    def test_off_by_default_without_loading_the_draft(self):
        with _switch(None):
            self.assertFalse(self._resolve(_args(), draft_cfg=None))

    def test_on_for_a_sliding_mla_draft(self):
        for backend in ("cutedsl_mla", "trtllm_mla"):
            with self.subTest(backend=backend), _switch():
                args = _args(speculative_draft_attention_backend=backend)
                self.assertTrue(self._resolve(args))

    def test_on_under_pd_over_mooncake(self):
        with _switch():
            args = _args(
                disaggregation_mode="decode", disaggregation_transfer_backend="mooncake"
            )
            self.assertTrue(self._resolve(args))

    def test_rejected_by_launch_flags_without_loading_the_draft(self):
        cases = {
            "prefill over nixl": _args(
                disaggregation_mode="prefill", disaggregation_transfer_backend="nixl"
            ),
            "decode host receive": _args(
                disaggregation_mode="decode",
                disaggregation_decode_host_receive_threshold=0.5,
            ),
            "unified": _args(enable_unified_memory=True),
            "flashinfer": _args(speculative_draft_attention_backend="flashinfer"),
        }
        for name, args in cases.items():
            with self.subTest(name), _switch():
                self._assert_rejected(args, draft_cfg=None)

    def test_rejected_for_drafts_it_does_not_fit(self):
        cases = {
            "mixed": dict(
                SWA_MLA_DRAFT,
                layer_types=["sliding_attention"] * 4 + ["full_attention"],
            ),
            "not mla": dict(SWA_MLA_DRAFT, dflash_config={}),
        }
        for name, draft_cfg in cases.items():
            with self.subTest(name), _switch():
                self._assert_rejected(_args(), draft_cfg=draft_cfg)

    def test_rejected_under_dcp_when_the_draft_does_not_run_dcp(self):
        # Ascend drafts run without DCP, so their rows are not DCP-collapsed.
        with (
            _switch(),
            patch("sglang.srt.configs.model_config.is_npu", return_value=True),
        ):
            self._assert_rejected(_args(dcp_size=8))
            self.assertTrue(self._resolve(_args(dcp_size=1)))

    def test_explicit_off_is_kept(self):
        with _switch(False):
            self.assertFalse(self._resolve(_args(), draft_cfg=None))


def _mha_args(**overrides) -> ServerArgs:
    kwargs = dict(speculative_draft_attention_backend="trtllm_mha", dcp_size=1)
    kwargs.update(overrides)
    return _args(**kwargs)


class TestResolveMHADraftKVRing(CustomTestCase):
    def _resolve(self, args, draft_cfg=SWA_MHA_DRAFT, cuda=True) -> bool:
        with patch.object(current_platform, "is_cuda", return_value=cuda):
            return TestResolveDraftKVRing._resolve(self, args, draft_cfg=draft_cfg)

    def _assert_rejected(self, args, draft_cfg=SWA_MHA_DRAFT, cuda=True):
        with self.assertRaisesRegex(ValueError, "SGLANG_ENABLE_DFLASH_DRAFT_KV_RING=1"):
            self._resolve(args, draft_cfg=draft_cfg, cuda=cuda)

    def test_off_by_default(self):
        with _switch(None):
            self.assertFalse(self._resolve(_mha_args(), draft_cfg=None))

    def test_on_for_a_sliding_mha_draft_on_trtllm_mha_and_fa4(self):
        cases = {
            "trtllm_mha": _mha_args(),
            "fa4": _mha_args(speculative_draft_attention_backend="fa4"),
            "pd prefill over mooncake": _mha_args(
                disaggregation_mode="prefill",
                disaggregation_transfer_backend="mooncake",
            ),
            "pd decode over mooncake": _mha_args(
                disaggregation_mode="decode",
                disaggregation_transfer_backend="mooncake",
            ),
        }
        for name, args in cases.items():
            with self.subTest(name), _switch():
                self.assertTrue(self._resolve(args))

    def test_mha_rejected_by_launch_flags_without_loading_the_draft(self):
        cases = {
            "dcp": _mha_args(dcp_size=8),
            "attn cp": _mha_args(attn_cp_size=2),
            "dp attention": _mha_args(attn_dp_size=2),
            "draft window": _mha_args(speculative_draft_window_size=512),
            "fa3": _mha_args(speculative_draft_attention_backend="fa3"),
        }
        for name, args in cases.items():
            with self.subTest(name), _switch():
                self._assert_rejected(args, draft_cfg=None)
        with self.subTest("not cuda"), _switch():
            self._assert_rejected(_mha_args(), draft_cfg=None, cuda=False)

    def test_mha_rejected_for_drafts_it_does_not_fit(self):
        cases = {
            "mixed": dict(
                SWA_MHA_DRAFT,
                layer_types=["sliding_attention"] * 4 + ["full_attention"],
            ),
            "mla": SWA_MLA_DRAFT,
        }
        for name, draft_cfg in cases.items():
            with self.subTest(name), _switch():
                self._assert_rejected(_mha_args(), draft_cfg=draft_cfg)


WINDOW_LEFT = 1023
BLOCK = 8
_NO_MEM_POOL = SimpleNamespace(maybe_get_custom_mem_pool=lambda: None)


def _ring(dcp, req_to_token_rows=4, dcp_rank=0, pool=_NO_MEM_POOL):
    return DFlashDraftKVRing(
        rows_per_req=draft_kv_ring_rows_per_req(
            window_left=WINDOW_LEFT, block_size=BLOCK, dcp_size=dcp
        ),
        dcp_size=dcp,
        req_to_token_rows=req_to_token_rows,
        device="cpu",
        window_left=WINDOW_LEFT,
        dcp_rank=dcp_rank,
        pool=pool,
    )


class TestDraftKVRingLayout(CustomTestCase):
    def test_window_and_block_never_share_a_row(self):
        for dcp in (1, 8):
            with self.subTest(dcp=dcp):
                ring = _ring(dcp)
                p = 123457
                # The block's rows and every row a query in that block can read.
                keys = torch.arange(p - WINDOW_LEFT, p + BLOCK)
                locs = ring.locs(torch.full_like(keys, 2), keys)
                self.assertEqual(locs.unique().numel(), keys.numel())

    def test_extend_keeps_only_the_readable_tail(self):
        ring = _ring(8)
        lens = torch.tensor([3000, 40])
        prefix = torch.tensor([0, 100])
        positions = torch.cat(
            [torch.arange(int(a), int(a + n)) for a, n in zip(prefix, lens)]
        )
        locs = ring.extend_locs(
            req_pool_indices=torch.tensor([1, 2]),
            positions=positions,
            extend_lens=lens,
            seq_ends=prefix + lens,
        )
        first, second = locs[:3000], locs[3000:]
        kept = first != 0
        self.assertEqual(int(kept.sum()), ring.span)
        self.assertTrue(bool(kept[-ring.span :].all()))
        self.assertEqual(first[kept].unique().numel(), ring.span)
        self.assertTrue(bool((second != 0).all()))

    def test_the_floor_resets_when_the_slot_is_allocated_again(self):
        pool = ReqToTokenPool(
            size=3, max_context_len=8, device="cpu", enable_memory_saver=False
        )
        ring = _ring(1)
        slot = pool.alloc_rows(1)[0]
        req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=slot))
        ring.record_floors([req], [100], pool.req_generation)
        ring.record_floors([req], [3000], pool.req_generation)
        self.assertEqual(int(ring.min_pos[slot]), 100)
        # The same request gets the same slot back, as after a retraction.
        pool.free_rows([slot])
        self.assertEqual(pool.alloc_rows(1), [slot])
        ring.record_floors([req], [5], pool.req_generation)
        self.assertEqual(int(ring.min_pos[slot]), 5)


PAGE = 64
MHA_WINDOW_LEFT = 2047


def _paged_ring(req_to_token_rows=4, block=BLOCK):
    return DFlashDraftKVRing(
        rows_per_req=draft_kv_ring_paged_rows_per_req(
            window_left=MHA_WINDOW_LEFT, block_size=block, page_size=PAGE
        ),
        dcp_size=1,
        req_to_token_rows=req_to_token_rows,
        device="cpu",
        window_left=MHA_WINDOW_LEFT,
        dcp_rank=0,
        pool=_NO_MEM_POOL,
        page_size=PAGE,
    )


class TestPagedDraftKVRing(CustomTestCase):
    def test_geometry_is_whole_pages(self):
        for block in (1, 8, 16):
            rows = draft_kv_ring_paged_rows_per_req(
                window_left=MHA_WINDOW_LEFT, block_size=block, page_size=PAGE
            )
            self.assertEqual(rows % PAGE, 0)
            self.assertGreaterEqual(rows, MHA_WINDOW_LEFT + 1 + block + PAGE)
        # The pool's sink is one page, so the last request's last row still fits.
        ring = _paged_ring(req_to_token_rows=6)
        pool_rows = draft_kv_ring_pool_rows(
            req_to_token_rows=6, rows_per_req=ring.rows_per_req, page_size=PAGE
        )
        self.assertEqual(pool_rows, PAGE + 6 * ring.rows_per_req)
        last = ring.locs(torch.tensor([5]), torch.tensor([ring.rows_per_req - 1]))
        self.assertEqual(int(last), pool_rows - 1)

    def test_page_table_reads_the_rows_locs_write(self):
        ring = _paged_ring()
        width = 200
        cols = (torch.arange(width) % ring.pages_per_req).to(torch.int32)
        reqs = torch.tensor([3, 0, 2])
        table = torch.empty((3, width), dtype=torch.int32)
        ring.fill_page_table(reqs, cols, table)
        ring.min_pos[reqs] = torch.tensor([0, 777, 5000], dtype=torch.int32)
        for i, (req, floor) in enumerate(zip(reqs.tolist(), [0, 777, 5000])):
            for end in (floor + 1, floor + 2100, floor + 9000):
                keys = torch.arange(max(floor, end - MHA_WINDOW_LEFT - BLOCK), end)
                locs = ring.locs(torch.full_like(keys, req), keys)
                rel = keys - floor
                self.assertTrue(
                    torch.equal(locs // PAGE, table[i, rel // PAGE].to(torch.int64))
                )
                self.assertTrue(torch.equal(locs % PAGE, rel % PAGE))

    def test_window_and_block_never_share_a_page(self):
        ring = _paged_ring()
        req = torch.tensor([1])
        for floor in (0, 13, 64):
            ring.min_pos[req] = floor
            for prefix in range(floor, floor + 3 * PAGE + 5, 7):
                prefix += 10_000
                keys = torch.arange(
                    max(floor, prefix - MHA_WINDOW_LEFT), prefix + BLOCK
                )
                locs = ring.locs(req.expand_as(keys), keys)
                logical = (keys - floor) // PAGE
                physical = locs // PAGE
                self.assertEqual(logical.unique().numel(), physical.unique().numel())
                self.assertEqual(locs.unique().numel(), keys.numel())
                # Page 0 is the sink, no request row lands there.
                self.assertTrue(bool((physical >= 1).all()))

    def test_relative_seq_lens_floor_and_clamp(self):
        ring = _paged_ring()
        ring.min_pos[torch.tensor([1, 2])] = torch.tensor([100, 9], dtype=torch.int32)
        seq_lens = torch.tensor([150, 5, 40], dtype=torch.int64)
        rel = ring.relative_seq_lens(torch.tensor([1, 2, 0]), seq_lens)
        self.assertEqual(rel.tolist(), [50, 0, 40])
        self.assertEqual(rel.dtype, torch.int64)

    def test_extend_starts_at_the_floor(self):
        ring = _paged_ring()
        reqs = torch.tensor([2])
        prefix, n = 1000, 300
        ring.min_pos[reqs] = prefix
        positions = torch.arange(prefix, prefix + n)
        locs = ring.extend_locs(
            req_pool_indices=reqs,
            positions=positions,
            extend_lens=torch.tensor([n]),
            seq_ends=torch.tensor([prefix + n]),
        )
        base = PAGE + 2 * ring.rows_per_req
        self.assertEqual(locs.tolist(), list(range(base, base + n)))

    def test_mla_ring_keeps_its_layout(self):
        for dcp in (1, 8):
            ring = _ring(dcp)
            self.assertFalse(ring.paged)
            ring.min_pos.fill_(500)
            req = torch.tensor([3] * 40)
            pos = torch.arange(4000, 4040)
            expected = (
                1 + 3 * ring.rows_per_req + (pos // dcp) % ring.rows_per_req
            ) * dcp + pos % dcp
            self.assertTrue(torch.equal(ring.locs(req, pos), expected))

    def test_paged_ring_rejects_dcp_and_partial_pages(self):
        for rows, dcp in ((PAGE * 3 + 1, 1), (PAGE * 3, 2)):
            with self.subTest(rows=rows, dcp=dcp), self.assertRaises(ValueError):
                DFlashDraftKVRing(
                    rows_per_req=rows,
                    dcp_size=dcp,
                    req_to_token_rows=2,
                    device="cpu",
                    window_left=MHA_WINDOW_LEFT,
                    dcp_rank=0,
                    pool=_NO_MEM_POOL,
                    page_size=PAGE,
                )


class TestDraftKVRingPDTransfer(CustomTestCase):
    def _received(self, prefill_dcp, decode_dcp, seq_len):
        received = {}
        for rank in range(decode_dcp):
            src_rank = rank if prefill_dcp == decode_dcp else 0
            prefill = _ring(prefill_dcp, req_to_token_rows=8, dcp_rank=src_rank)
            decode = _ring(decode_dcp, req_to_token_rows=8, dcp_rank=rank)
            src, dst = draft_kv_ring_row_pairs(
                prefill.transfer_rows(3, seq_len).tolist(),
                decode.transfer_rows(5, seq_len).tolist(),
            )
            first = max(0, seq_len - WINDOW_LEFT - 1)
            positions = [p for p in range(first, seq_len) if p % decode_dcp == rank]
            received[rank] = (positions, src, dst, prefill, decode)
        return received

    def test_each_decode_rank_gets_the_window_positions_it_owns(self):
        for prefill_dcp, decode_dcp in ((1, 8), (8, 8)):
            for seq_len in (5, WINDOW_LEFT + 1, 5000):
                with self.subTest(prefill=prefill_dcp, decode=decode_dcp, L=seq_len):
                    received = self._received(prefill_dcp, decode_dcp, seq_len)
                    for positions, src, dst, prefill, decode in received.values():
                        pos = torch.tensor(positions, dtype=torch.int64)
                        # The rows each side's own locs give those positions.
                        self.assertEqual(
                            src,
                            (
                                prefill.locs(torch.full_like(pos, 3), pos)
                                // prefill_dcp
                            ).tolist(),
                        )
                        self.assertEqual(
                            dst,
                            (
                                decode.locs(torch.full_like(pos, 5), pos) // decode_dcp
                            ).tolist(),
                        )

    def test_buffer_infos_address_one_ring_row_per_item(self):
        page_size = 64
        pool = SimpleNamespace(
            page_size=page_size,
            maybe_get_custom_mem_pool=lambda: None,
            get_contiguous_buf_infos=lambda: (
                [1000, 2000],
                [page_size * 576 * 10, page_size * 576 * 10],
                [page_size * 576, page_size * 576],
            ),
        )
        ring = _ring(8, pool=pool)
        self.assertEqual(
            ring.ring_buf_infos(),
            ([1000, 2000], [page_size * 5760] * 2, [576, 576]),
        )
        ptrs, lens, item_lens = ring.floor_buf_infos()
        self.assertEqual(ptrs, [ring.min_pos.data_ptr()])
        self.assertEqual((lens, item_lens), ([4 * 4], [4]))


class TestPagedDraftKVRingPDTransfer(CustomTestCase):
    def test_decode_reads_what_prefill_wrote(self):
        page, window_left, block = 4, 15, 4
        rows = draft_kv_ring_paged_rows_per_req(
            window_left=window_left, block_size=block, page_size=page
        )

        def ring():
            return DFlashDraftKVRing(
                rows_per_req=rows,
                dcp_size=1,
                req_to_token_rows=8,
                device="cpu",
                window_left=window_left,
                dcp_rank=0,
                pool=_NO_MEM_POOL,
                page_size=page,
            )

        for floor, seq_len in ((0, 7), (0, 40), (13, 14), (13, 50), (6, 300)):
            with self.subTest(floor=floor, seq_len=seq_len):
                prefill, decode = ring(), ring()
                pool_rows = page + 8 * rows
                src_pool = torch.full((pool_rows,), -1, dtype=torch.int64)
                dst_pool = torch.full((pool_rows,), -2, dtype=torch.int64)
                prefill.min_pos[3] = floor
                decode.min_pos[5] = 999
                positions = torch.arange(floor, seq_len)
                src_pool[prefill.locs(torch.full_like(positions, 3), positions)] = (
                    positions
                )
                src, dst = draft_kv_ring_row_pairs(
                    prefill.transfer_rows(3, seq_len).tolist(),
                    decode.transfer_rows(5, seq_len).tolist(),
                )
                self.assertEqual(len(src), rows)
                dst_pool[dst] = src_pool[src]
                decode.min_pos[5] = prefill.min_pos[3]
                keys = torch.arange(max(floor, seq_len - window_left), seq_len)
                got = dst_pool[decode.locs(torch.full_like(keys, 5), keys)]
                self.assertTrue(torch.equal(got, keys))
                cols = (torch.arange(128) % decode.pages_per_req).to(torch.int32)
                table = torch.empty((1, 128), dtype=torch.int32)
                decode.fill_page_table(torch.tensor([5]), cols, table)
                rel = keys - floor
                read = table[0, rel // page].to(torch.int64) * page + rel % page
                self.assertTrue(torch.equal(dst_pool[read], keys))


if __name__ == "__main__":
    unittest.main()
