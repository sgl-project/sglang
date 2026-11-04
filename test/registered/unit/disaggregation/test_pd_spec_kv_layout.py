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
    "num_draft_entries": 6,
    "dspark_block_size": 4,
    "dflash_block_size": None,
}
SENTINEL_QUERY = dict.fromkeys(
    ("prefill_dp_rank", "prefill_cp_rank", "target_tp_rank", "target_pp_rank"), "-1"
)


def layout(algorithm, *, num_draft_entries=6, num_steps=3, num_draft_tokens=4):
    kv_args = SimpleNamespace(num_draft_entries=num_draft_entries)
    with published_topology(
        speculative_algorithm=algorithm,
        speculative_num_steps=num_steps,
        speculative_num_draft_tokens=num_draft_tokens,
        speculative_dspark_block_size=4,
        speculative_dflash_block_size=None,
    ):
        return get_spec_kv_layout(kv_args)


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
        payload.pop("spec_kv_layout")
    response = MagicMock(status_code=200)
    response.json.return_value = payload
    return response


def bootstrap_server():
    with patch.object(CommonKVBootstrapServer, "run"):
        return CommonKVBootstrapServer("127.0.0.1", 0)


def register(server, row, **fields):
    request = SimpleNamespace(json=AsyncMock(return_value={**row, **fields}))
    return asyncio.run(server._handle_route_put(request))


def route(server, *, opt_in=False):
    query = dict(SENTINEL_QUERY)
    if opt_in:
        query["spec_kv_layout"] = "1"
    response = asyncio.run(server._handle_route_get(SimpleNamespace(query=query)))
    return response.status, json.loads(
        response.text
    ) if response.status == 200 else None


class TestSpecKVLayout(CustomTestCase):
    def test_only_prefill_injected_draft_kv_publishes_a_layout(self):
        for algorithm in (None, "EAGLE", "NEXTN"):
            with self.subTest(algorithm=algorithm):
                self.assertEqual(layout(algorithm), {})
        self.assertEqual(layout("DSPARK"), DSPARK)
        self.assertEqual(layout("DSPARK", num_steps=8, num_draft_tokens=16), DSPARK)
        self.assertEqual(layout("DSPARK", num_draft_entries=0), None)
        self.assertEqual(
            layout("DFLASH"),
            {
                "algorithm": "DFLASH",
                "num_draft_entries": 6,
                "dspark_block_size": 4,
                "dflash_block_size": None,
            },
        )

    def test_decode_compares_only_known_layouts(self):
        cases = {
            "ordinary PD or decode-only EAGLE": ({}, {}, None),
            "matched DSpark": (DSPARK, DSPARK, None),
            "legacy prefill": (DSPARK, None, None),
            "DSpark decode, target-only prefill": (DSPARK, {}, "mismatch"),
            "DSpark prefill, target-only decode": ({}, DSPARK, "mismatch"),
            "different draft entries": (
                DSPARK,
                {**DSPARK, "num_draft_entries": 4},
                "mismatch",
            ),
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
                ) as get:
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
                    self.assertIn("spec_kv_layout=1", get.call_args.args[0])


class TestSpecKVLayoutBootstrap(CustomTestCase):
    def setUp(self):
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

    def test_draftless_ranks_and_old_decodes_are_compatible(self):
        for order in ((None, DSPARK), (DSPARK, None)):
            with self.subTest(order=order):
                server = bootstrap_server()
                for pp_rank, spec_kv_layout in enumerate(order):
                    status = register(
                        server, self.row, pp_rank=pp_rank, spec_kv_layout=spec_kv_layout
                    ).status
                    self.assertEqual(status, 200)
                status, payload = route(server)
                self.assertEqual(status, 200)
                self.assertNotIn("spec_kv_layout", payload)
                _, payload = route(server, opt_in=True)
                self.assertEqual(payload["spec_kv_layout"], DSPARK)

    def test_known_layouts_must_agree_across_ranks(self):
        server = bootstrap_server()
        self.assertEqual(register(server, self.row, spec_kv_layout=DSPARK).status, 200)
        incompatible = {**DSPARK, "num_draft_entries": 7}
        self.assertEqual(
            register(server, self.row, pp_rank=1, spec_kv_layout=incompatible).status,
            400,
        )
        self.assertEqual(route(server)[0], 503)

    def test_no_spec_prefill_publishes_an_empty_layout_only_on_opt_in(self):
        server = bootstrap_server()
        self.assertEqual(register(server, self.row, spec_kv_layout={}).status, 200)
        self.assertEqual(
            register(server, self.row, pp_rank=1, spec_kv_layout={}).status, 200
        )
        self.assertNotIn("spec_kv_layout", route(server)[1])
        self.assertEqual(route(server, opt_in=True)[1]["spec_kv_layout"], {})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
