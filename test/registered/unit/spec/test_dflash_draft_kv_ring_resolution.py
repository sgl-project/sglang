"""DFLASH draft KV ring resolution and budget.

The ring is resolved once at startup into speculative_dflash_draft_kv_ring:
on by default for an all-sliding MLA draft on trtllm_mla / cutedsl_mla, off
(with a log line) where it is not served, and an explicit
SGLANG_ENABLE_DFLASH_DRAFT_KV_RING=1 there is an error. When on, the ring is
charged as a fixed pool.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import _resolve_dflash_draft_kv_ring
from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_draft_ring import (
    draft_kv_ring_pool_rows,
    draft_kv_ring_rows_per_req,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HOOK_MODULE = "sglang.srt.arg_groups.speculative_hook"
KVC_MODULE = "sglang.srt.mem_cache.kv_cache_configurator"
SWITCH = envs.SGLANG_ENABLE_DFLASH_DRAFT_KV_RING
SWA_MLA_DRAFT = {
    "num_hidden_layers": 5,
    "dflash_config": {"attention_mode": "mla"},
    "layer_types": ["sliding_attention"] * 5,
    "sliding_window": 1024,
}


@contextlib.contextmanager
def _switch(value=None):
    """The ring switch at ``value``, or unset (the default) for None."""
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

    def test_off_by_launch_flags_without_loading_the_draft(self):
        cases = {
            "prefill": _args(disaggregation_mode="prefill"),
            "decode": _args(disaggregation_mode="decode"),
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
                with self.assertRaises(ValueError):
                    self._resolve(_args(disaggregation_mode=mode), draft_cfg=None)

    def test_explicit_off_is_kept(self):
        with _switch(False):
            self.assertFalse(self._resolve(_args(), draft_cfg=None))


def _kvc(*, is_draft_worker=False, cell_size=2880, attn_dp_size=1):
    return SimpleNamespace(
        is_draft_worker=is_draft_worker,
        spec_aux_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=SWA_MLA_DRAFT)
        ),
        dflash_draft_cell_size_per_token=cell_size,
        dflash_draft_loc_space_scale=1,
        attn_dp_size=attn_dp_size,
        _req_pool_rows=lambda n: n,
    )


class TestDraftKVRingBudget(CustomTestCase):
    def _ring_bytes(
        self, kvc, *, ring=True, dcp_size=8, max_running_requests=48, page_size=1
    ):
        with (
            patch(
                f"{KVC_MODULE}.get_spec",
                return_value=SimpleNamespace(
                    speculative_num_draft_tokens=8,
                    speculative_dflash_draft_kv_ring=ring,
                ),
            ),
            patch(
                f"{KVC_MODULE}.get_parallel",
                return_value=SimpleNamespace(attn_dcp_size=dcp_size),
            ),
            patch(
                f"{KVC_MODULE}.get_schedule",
                return_value=SimpleNamespace(
                    max_running_requests=max_running_requests, page_size=page_size
                ),
            ),
        ):
            return KVCacheConfigurator._dflash_draft_kv_ring_bytes(kvc)

    def test_ring_is_charged_as_its_draft_pool(self):
        for dcp_size, page_size, dp in ((8, 1, 1), (1, 64, 1), (8, 64, 2)):
            with self.subTest(dcp=dcp_size, page=page_size, dp=dp):
                rows_per_req = draft_kv_ring_rows_per_req(
                    window_left=1023, block_size=8, dcp_size=dcp_size
                )
                # The draft pool spans every req_to_token row (48 / dp + 1)
                # plus the pool's padding page.
                rows = draft_kv_ring_pool_rows(
                    req_pool_rows=48 // dp + 1,
                    rows_per_req=rows_per_req,
                    page_size=page_size,
                )
                self.assertEqual(
                    self._ring_bytes(
                        _kvc(attn_dp_size=dp), dcp_size=dcp_size, page_size=page_size
                    ),
                    (rows + page_size) * 2880,
                )

    def test_nothing_is_charged_without_a_ring(self):
        cases = {
            "draft worker": (_kvc(is_draft_worker=True), True),
            "unresolved draft bytes": (_kvc(cell_size=None), True),
            "ring off": (_kvc(), False),
        }
        for name, (kvc, ring) in cases.items():
            with self.subTest(name):
                self.assertEqual(self._ring_bytes(kvc, ring=ring), 0)


if __name__ == "__main__":
    unittest.main()
