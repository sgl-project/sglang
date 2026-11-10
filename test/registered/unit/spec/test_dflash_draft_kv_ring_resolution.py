"""SGLANG_DFLASH_DRAFT_KV_RING resolution and budget.

The ring is resolved once at startup: on by default for an all-sliding MLA
draft on trtllm_mla / cutedsl_mla, off (with a log line) where it is not
served, and an explicit 1 there is an error. When on, the target's per-token
budget carries no draft row and the ring is charged as a fixed pool.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.arg_groups.speculative_hook import _resolve_dflash_draft_kv_ring
from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.model_executor.pool_configurator import _has_dflash_draft_kv
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
FLAG = "SGLANG_DFLASH_DRAFT_KV_RING"
SWA_MLA_DRAFT = {
    "num_hidden_layers": 5,
    "dflash_config": {"attention_mode": "mla"},
    "layer_types": ["sliding_attention"] * 5,
    "sliding_window": 1024,
}


def _env(flag=None):
    env = {k: v for k, v in os.environ.items() if k != FLAG}
    if flag is not None:
        env[FLAG] = flag
    return patch.dict(os.environ, env, clear=True)


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
        with patch(f"{HOOK_MODULE}._load_draft_hf_config", return_value=draft_cfg):
            _resolve_dflash_draft_kv_ring(args)
        return envs.SGLANG_DFLASH_DRAFT_KV_RING.get()

    def test_on_by_default_for_a_sliding_mla_draft(self):
        for backend in ("cutedsl_mla", "trtllm_mla"):
            with self.subTest(backend=backend), _env():
                args = _args(speculative_draft_attention_backend=backend)
                self.assertTrue(self._resolve(args))

    def test_off_by_default_where_not_served(self):
        cases = {
            "prefill": _args(disaggregation_mode="prefill"),
            "decode": _args(disaggregation_mode="decode"),
            "unified": _args(enable_unified_memory=True),
            "flashinfer": _args(speculative_draft_attention_backend="flashinfer"),
        }
        for name, args in cases.items():
            with self.subTest(name), _env():
                self.assertFalse(self._resolve(args))
        with _env():
            mixed = dict(SWA_MLA_DRAFT, layer_types=["full_attention"] * 5)
            self.assertFalse(self._resolve(_args(), draft_cfg=mixed))

    def test_explicit_on_where_not_served_is_an_error(self):
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode), _env("1"):
                with self.assertRaises(ValueError):
                    self._resolve(_args(disaggregation_mode=mode))

    def test_explicit_off_is_kept(self):
        with _env("0"):
            self.assertFalse(self._resolve(_args()))


def _kvc(*, is_draft_worker=False, dflash=True, cell_size=2880, attn_dp_size=1):
    spec_algorithm = MagicMock()
    spec_algorithm.is_dflash.return_value = dflash
    spec_algorithm.is_dflash_family.return_value = True
    return SimpleNamespace(
        is_draft_worker=is_draft_worker,
        spec_algorithm=spec_algorithm,
        spec_aux_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=SWA_MLA_DRAFT)
        ),
        dflash_draft_cell_size_per_token=cell_size,
        dflash_draft_loc_space_scale=1,
        attn_dp_size=attn_dp_size,
        _req_pool_rows=lambda n: n,
    )


class TestDraftKVRingBudget(CustomTestCase):
    def _ring_bytes(self, kvc, *, dcp_size=8, max_running_requests=48, page_size=1):
        with (
            patch(
                f"{KVC_MODULE}.get_spec",
                return_value=SimpleNamespace(speculative_num_draft_tokens=8),
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
            with self.subTest(dcp=dcp_size, page=page_size, dp=dp), _env():
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
            "draft worker": (_kvc(is_draft_worker=True), None),
            "dspark": (_kvc(dflash=False), None),
            "unresolved draft bytes": (_kvc(cell_size=None), None),
            "ring off": (_kvc(), "0"),
        }
        for name, (kvc, flag) in cases.items():
            with self.subTest(name), _env(flag):
                self.assertEqual(self._ring_bytes(kvc), 0)

    def test_per_token_draft_charge_follows_the_ring(self):
        with _env():
            self.assertFalse(_has_dflash_draft_kv(_kvc()))
            # DSPARK has no ring and keeps its per-token draft rows.
            self.assertTrue(_has_dflash_draft_kv(_kvc(dflash=False)))
            self.assertFalse(_has_dflash_draft_kv(_kvc(is_draft_worker=True)))
        with _env("0"):
            self.assertTrue(_has_dflash_draft_kv(_kvc()))


if __name__ == "__main__":
    unittest.main()
