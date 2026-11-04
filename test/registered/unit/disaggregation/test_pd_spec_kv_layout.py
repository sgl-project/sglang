from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import asyncio
import dataclasses
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    PrefillServerInfo,
)
from sglang.srt.disaggregation.utils import get_spec_kv_layout
from sglang.test.test_utils import CustomTestCase, published_topology

PAGE = 64
DSPARK = {
    "algorithm": "DSPARK",
    "num_draft_tokens": 4,
    "num_steps": 3,
    "dspark_block_size": 4,
    "dflash_block_size": None,
}
SENTINEL_QUERY = dict.fromkeys(
    ("prefill_dp_rank", "prefill_cp_rank", "target_tp_rank", "target_pp_rank"), "-1"
)


def decode_manager(spec_kv_layout):
    manager = CommonKVManager.__new__(CommonKVManager)
    manager.prefill_info_table = {}
    manager.kv_args = SimpleNamespace(page_size=PAGE, engine_rank=0)
    manager.kv_cache_dtype_str = "bf16"
    manager.kv_layout = None
    manager.dsv41_spec_layout = None
    manager.spec_kv_layout = spec_kv_layout
    manager.attn_tp_size = manager.attn_cp_size = manager.pp_size = 1
    manager.dcp_size = 1
    manager.attn_cp_rank = manager.pp_rank = 0
    manager.is_mla_backend = manager.is_hybrid_mla_backend = False
    return manager


def prefill_info(spec_kv_layout):
    info = PrefillServerInfo(
        attn_tp_size=1,
        attn_cp_size=1,
        dp_size=1,
        pp_size=1,
        page_size=PAGE,
        kv_cache_dtype="bf16",
        follow_bootstrap_room=True,
        spec_kv_layout=spec_kv_layout,
    )
    payload = dataclasses.asdict(info)
    if spec_kv_layout is None:
        # What an older bootstrap server, or one with no spec, returns.
        payload.pop("spec_kv_layout")
    response = MagicMock(status_code=200)
    response.json.return_value = payload
    return response


class TestSpecKVLayout(CustomTestCase):
    def test_only_prefill_injected_draft_kv_publishes_a_layout(self):
        for algorithm, expected in (
            (None, None),
            ("EAGLE", None),
            ("NEXTN", None),
            ("DSPARK", {**DSPARK, "num_draft_tokens": 5, "dspark_block_size": 5}),
            (
                "DFLASH",
                {
                    **DSPARK,
                    "algorithm": "DFLASH",
                    "dspark_block_size": None,
                    "dflash_block_size": 8,
                },
            ),
        ):
            with self.subTest(algorithm=algorithm):
                kwargs = dict(speculative_num_draft_tokens=4, speculative_num_steps=3)
                if algorithm == "DSPARK":
                    kwargs.update(
                        speculative_num_draft_tokens=5, speculative_dspark_block_size=5
                    )
                if algorithm == "DFLASH":
                    kwargs.update(speculative_dflash_block_size=8)
                with published_topology(speculative_algorithm=algorithm, **kwargs):
                    self.assertEqual(get_spec_kv_layout(), expected)

    def test_decode_compares_layout_before_recording_the_prefill(self):
        cases = {
            "no spec on either side": (None, None, None),
            "matched DSpark": (DSPARK, DSPARK, None),
            "decode-only EAGLE": (None, None, None),
            "DSpark decode, prefill without spec or older": (DSPARK, None, "mismatch"),
            "DSpark prefill, decode without spec": (None, DSPARK, "mismatch"),
            "different block size": (
                DSPARK,
                {**DSPARK, "dspark_block_size": 5},
                "mismatch",
            ),
        }
        for name, (local, remote, error) in cases.items():
            with self.subTest(name):
                manager = decode_manager(local)
                with patch(
                    "sglang.srt.disaggregation.common.conn.requests.get",
                    return_value=prefill_info(remote),
                ):
                    if error:
                        with self.assertRaisesRegex(
                            RuntimeError, "Speculative KV layout mismatch"
                        ):
                            manager.try_ensure_parallel_info("prefill:8998")
                        self.assertEqual(manager.prefill_info_table, {})
                    else:
                        self.assertTrue(
                            manager.try_ensure_parallel_info("prefill:8998")
                        )


class TestSpecKVLayoutBootstrap(CustomTestCase):
    def setUp(self):
        with patch.object(CommonKVBootstrapServer, "run"):
            self.server = CommonKVBootstrapServer("127.0.0.1", 0)
        self.row = dict(
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_cp_size=1,
            attn_cp_rank=0,
            attn_dp_size=1,
            attn_dp_rank=0,
            pp_size=2,
            pp_rank=0,
            system_dp_size=1,
            system_dp_rank=0,
            rank_ip="10.0.0.1",
            rank_port=17000,
            page_size=PAGE,
            kv_cache_dtype="bf16",
        )

    def register(self, **fields):
        request = SimpleNamespace(json=AsyncMock(return_value={**self.row, **fields}))
        return asyncio.run(self.server._handle_route_put(request))

    def route(self):
        response = asyncio.run(
            self.server._handle_route_get(SimpleNamespace(query=SENTINEL_QUERY))
        )
        return response.status, json.loads(response.text)

    def test_ranks_must_agree_and_old_decodes_never_see_the_field(self):
        self.assertEqual(self.register(spec_kv_layout=DSPARK).status, 200)
        self.assertEqual(self.register(pp_rank=1).status, 400)
        self.assertEqual(
            self.register(pp_rank=1, spec_kv_layout={**DSPARK, "num_steps": 1}).status,
            400,
        )
        self.assertEqual(self.register(pp_rank=1, spec_kv_layout=DSPARK).status, 200)
        status, payload = self.route()
        self.assertEqual(status, 200)
        self.assertEqual(payload["spec_kv_layout"], DSPARK)

    def test_no_spec_prefill_omits_the_field(self):
        self.assertEqual(self.register().status, 200)
        self.assertEqual(self.register(pp_rank=1).status, 200)
        status, payload = self.route()
        self.assertEqual(status, 200)
        self.assertNotIn("spec_kv_layout", payload)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
