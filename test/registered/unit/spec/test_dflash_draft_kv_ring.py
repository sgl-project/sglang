import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import _resolve_dflash_draft_kv_ring
from sglang.srt.disaggregation.utils import draft_kv_ring_row_pairs
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_draft_ring import (
    DFlashDraftKVRing,
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


@contextlib.contextmanager
def _switch(value=None):
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

    def test_on_by_default_for_a_sliding_mla_draft(self):
        for backend in ("cutedsl_mla", "trtllm_mla"):
            with self.subTest(backend=backend), _switch():
                args = _args(speculative_draft_attention_backend=backend)
                self.assertTrue(self._resolve(args))
                # The decision lives in the declaration, not in the environment.
                self.assertFalse(SWITCH.is_set())

    def test_on_under_pd_over_mooncake(self):
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode), _switch():
                args = _args(
                    disaggregation_mode=mode, disaggregation_transfer_backend="mooncake"
                )
                self.assertTrue(self._resolve(args))

    def test_off_by_launch_flags_without_loading_the_draft(self):
        cases = {
            "prefill over nixl": _args(
                disaggregation_mode="prefill", disaggregation_transfer_backend="nixl"
            ),
            "decode over mori": _args(
                disaggregation_mode="decode", disaggregation_transfer_backend="mori"
            ),
            "decode host receive": _args(
                disaggregation_mode="decode",
                disaggregation_decode_host_receive_threshold=0.5,
            ),
            "unified": _args(enable_unified_memory=True),
            "flashinfer": _args(speculative_draft_attention_backend="flashinfer"),
            "compact window": _args(speculative_draft_window_size=4096),
        }
        for name, args in cases.items():
            with self.subTest(name), _switch():
                self.assertFalse(self._resolve(args, draft_cfg=None))

    def test_off_for_drafts_it_does_not_fit(self):
        cases = {
            "full attention": dict(SWA_MLA_DRAFT, layer_types=["full_attention"] * 5),
            "mixed": dict(
                SWA_MLA_DRAFT,
                layer_types=["sliding_attention"] * 4 + ["full_attention"],
            ),
            "not mla": dict(SWA_MLA_DRAFT, dflash_config={}),
        }
        for name, draft_cfg in cases.items():
            with self.subTest(name), _switch():
                self.assertFalse(self._resolve(_args(), draft_cfg=draft_cfg))

    def test_off_under_dcp_when_the_draft_does_not_run_dcp(self):
        # Ascend drafts run without DCP, so their rows are not DCP-collapsed.
        with (
            _switch(),
            patch("sglang.srt.configs.model_config.is_npu", return_value=True),
        ):
            self.assertFalse(self._resolve(_args(dcp_size=8)))
            self.assertTrue(self._resolve(_args(dcp_size=1)))

    def test_explicit_on_where_not_served_is_an_error(self):
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode), _switch(True):
                args = _args(
                    disaggregation_mode=mode, disaggregation_transfer_backend="nixl"
                )
                with self.assertRaises(ValueError):
                    self._resolve(args, draft_cfg=None)

    def test_explicit_off_is_kept_and_logged(self):
        with (
            _switch(False),
            self.assertLogs(HOOK_MODULE, level="INFO") as logs,
        ):
            self.assertFalse(self._resolve(_args(), draft_cfg=None))
        self.assertIn("ring: off", "\n".join(logs.output))


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
                self.assertGreaterEqual(ring.span, WINDOW_LEFT + 1 + BLOCK)

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

    def test_a_new_request_with_a_reused_rid_resets_the_floor(self):
        class Req:
            def __init__(self):
                self.rid = "same"
                self.kv = SimpleNamespace(req_pool_idx=2)

        ring = _ring(1)
        first = Req()
        ring.record_floors([first], [100])
        ring.record_floors([first], [3000])
        self.assertEqual(int(ring.min_pos[2]), 100)
        ring.record_floors([Req()], [5])
        self.assertEqual(int(ring.min_pos[2]), 5)


class TestDraftKVRingPDTransfer(CustomTestCase):
    """Prefill and decode build position-matched ``transfer_rows`` for their own
    req pool slot; Mooncake copies the pairs both ranks own."""

    def _received(self, prefill_dcp, decode_dcp, seq_len):
        """{decode rank: [(position, prefill row, decode row)]} one prefill rank
        per decode rank: the same rank under equal DCP, else prefill rank 0."""
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
        for prefill_dcp, decode_dcp in ((1, 1), (1, 8), (8, 8)):
            for seq_len in (5, WINDOW_LEFT + 1, 5000):
                with self.subTest(prefill=prefill_dcp, decode=decode_dcp, L=seq_len):
                    covered = []
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
                        covered += positions
                    window = range(max(0, seq_len - WINDOW_LEFT - 1), seq_len)
                    self.assertEqual(sorted(covered), list(window))

    def test_reset_floor_empties_the_window(self):
        ring = _ring(8)
        ring.min_pos[2] = 7
        ring.reset_floor(2, 4096)
        self.assertEqual(ring.min_pos.tolist(), [0, 0, 4096, 0])

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


if __name__ == "__main__":
    unittest.main()
