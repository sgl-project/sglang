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
}
SENTINEL_QUERY = dict.fromkeys(
    ("prefill_dp_rank", "prefill_cp_rank", "target_tp_rank", "target_pp_rank"), "-1"
)


def layout(algorithm, *, num_draft_entries=6, num_draft_tokens=5, block_size_flag=4):
    kv_args = SimpleNamespace(num_draft_entries=num_draft_entries)
    with published_topology(
        speculative_algorithm=algorithm,
        speculative_num_draft_tokens=num_draft_tokens,
        speculative_dspark_block_size=block_size_flag,
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


def prefill_info(spec_kv_layout, **extra):
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
    payload.update(extra)
    response = MagicMock(status_code=200)
    response.json.return_value = payload
    return response


def bootstrap_server():
    with patch.object(CommonKVBootstrapServer, "run"):
        return CommonKVBootstrapServer("127.0.0.1", 0)


def register(server, row, **fields):
    request = SimpleNamespace(json=AsyncMock(return_value={**row, **fields}))
    return asyncio.run(server._handle_route_put(request))


def route(server):
    query = dict(SENTINEL_QUERY)
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
        self.assertEqual(layout("DSPARK", block_size_flag=None), DSPARK)
        self.assertEqual(
            layout("DSPARK", num_draft_tokens=9, block_size_flag=8), DSPARK
        )
        self.assertEqual(layout("DSPARK", num_draft_entries=0), None)
        self.assertEqual(
            layout("DFLASH", num_draft_tokens=4, block_size_flag=None),
            {**DSPARK, "algorithm": "DFLASH"},
        )

    def test_decode_compares_only_published_layouts(self):
        cases = {
            "ordinary PD or decode-only EAGLE": ({}, None, None),
            "matched DSpark": (DSPARK, DSPARK, None),
            "legacy, target-only, or draftless prefill": (DSPARK, None, None),
            "DSpark prefill, draftless decode PP stage": (None, DSPARK, None),
            "DSpark prefill, target-only decode": ({}, DSPARK, "mismatch"),
            "different draft entries": (
                DSPARK,
                {**DSPARK, "num_draft_entries": 4},
                "mismatch",
            ),
            "different algorithm": (
                DSPARK,
                {**DSPARK, "algorithm": "DFLASH"},
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
                    self.assertNotIn("spec_kv_layout", get.call_args.args[0])

    def test_decode_ignores_fields_it_does_not_know(self):
        manager = decode_manager({})
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=prefill_info(None, future_field={"v": 2}),
        ):
            self.assertTrue(manager.try_ensure_parallel_info("prefill:8998"))
        self.assertEqual(len(manager.prefill_info_table), 1)


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

    def test_draftless_ranks_are_compatible(self):
        cases = [
            ((None, DSPARK), DSPARK),
            ((DSPARK, None), DSPARK),
            ((None, None), None),
        ]
        for order, expected in cases:
            with self.subTest(order=order):
                server = bootstrap_server()
                for pp_rank, spec_kv_layout in enumerate(order):
                    status = register(
                        server, self.row, pp_rank=pp_rank, spec_kv_layout=spec_kv_layout
                    ).status
                    self.assertEqual(status, 200)
                status, payload = route(server)
                self.assertEqual(status, 200)
                self.assertEqual(payload.get("spec_kv_layout"), expected)
                self.assertEqual("spec_kv_layout" in payload, expected is not None)

    def test_known_layouts_must_agree_across_ranks(self):
        server = bootstrap_server()
        self.assertEqual(register(server, self.row, spec_kv_layout=DSPARK).status, 200)
        incompatible = {**DSPARK, "num_draft_entries": 7}
        self.assertEqual(
            register(server, self.row, pp_rank=1, spec_kv_layout=incompatible).status,
            400,
        )
        self.assertEqual(route(server)[0], 503)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
