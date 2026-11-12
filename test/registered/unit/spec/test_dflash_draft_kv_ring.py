import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import _resolve_dflash_draft_kv_ring
from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
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

    def test_off_by_launch_flags_without_loading_the_draft(self):
        cases = {
            "prefill": _args(disaggregation_mode="prefill"),
            "unified": _args(enable_unified_memory=True),
            "flashinfer": _args(speculative_draft_attention_backend="flashinfer"),
        }
        for name, args in cases.items():
            with self.subTest(name), _switch():
                self.assertFalse(self._resolve(args, draft_cfg=None))

    def test_off_for_drafts_it_does_not_fit(self):
        cases = {
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
        with _switch(True), self.assertRaises(ValueError):
            self._resolve(_args(disaggregation_mode="prefill"), draft_cfg=None)

    def test_explicit_off_is_kept(self):
        with _switch(False):
            self.assertFalse(self._resolve(_args(), draft_cfg=None))


WINDOW_LEFT = 1023
BLOCK = 8


def _ring(dcp, req_to_token_rows=4):
    return DFlashDraftKVRing(
        rows_per_req=draft_kv_ring_rows_per_req(
            window_left=WINDOW_LEFT, block_size=BLOCK, dcp_size=dcp
        ),
        dcp_size=dcp,
        req_to_token_rows=req_to_token_rows,
        device="cpu",
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


if __name__ == "__main__":
    unittest.main()
