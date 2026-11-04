from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qsl, urlsplit

import pytest

from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
)
from sglang.srt.disaggregation.utils import get_spec_kv_layout
from sglang.test.test_utils import CustomTestCase

PAGE = 64
LEGACY = object()
SENTINEL_QUERY = dict.fromkeys(
    ("prefill_dp_rank", "prefill_cp_rank", "target_tp_rank", "target_pp_rank"), "-1"
)


def layout(algorithm, *, num_draft_entries=6, draft_item_len=2048 * PAGE):
    kv_args = SimpleNamespace(
        num_draft_entries=num_draft_entries,
        kv_item_lens=[576 * PAGE] * 4 + [draft_item_len] * num_draft_entries,
    )
    with patch(
        "sglang.srt.disaggregation.utils.get_spec",
        return_value=SimpleNamespace(speculative_algorithm=algorithm),
    ):
        return get_spec_kv_layout(kv_args)


DSPARK = layout("DSPARK")


def bootstrap_with(stage_layouts):
    with patch.object(CommonKVBootstrapServer, "run"):
        server = CommonKVBootstrapServer("127.0.0.1", 0)
    for pp_rank, spec_kv_layout in stage_layouts:
        row = dict(
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_cp_size=1,
            attn_cp_rank=0,
            attn_dp_size=1,
            attn_dp_rank=0,
            pp_size=2,
            pp_rank=pp_rank,
            system_dp_size=1,
            system_dp_rank=0,
            rank_ip="10.0.0.1",
            rank_port=17000 + pp_rank,
            page_size=PAGE,
            kv_cache_dtype="auto",
        )
        if spec_kv_layout is not LEGACY:
            row["spec_kv_layout"] = spec_kv_layout
        response = asyncio.run(
            server._handle_route_put(SimpleNamespace(json=AsyncMock(return_value=row)))
        )
        assert response.status == 200
    return server


def sentinel(server, query):
    response = asyncio.run(server._handle_route_get(SimpleNamespace(query=query)))
    assert response.status == 200
    return json.loads(response.text)


def decode_manager(spec_kv_layout):
    """An MLA decode at attention TP 2, against the prefill's attention TP 1."""
    manager = CommonKVManager.__new__(CommonKVManager)
    manager.prefill_info_table = {}
    manager.kv_args = SimpleNamespace(page_size=PAGE, engine_rank=0)
    manager.kv_cache_dtype_str = "auto"
    manager.spec_kv_layout = spec_kv_layout
    manager.attn_tp_size = 2
    manager.attn_cp_size = manager.pp_size = manager.dcp_size = 1
    manager.attn_cp_rank = manager.pp_rank = 0
    manager.is_mla_backend = True
    manager.is_hybrid_mla_backend = False
    return manager


class TestSpecKvLayout(CustomTestCase):
    def test_only_dflash_family_publishes_draft_kv(self):
        for algorithm in (None, "EAGLE", "NEXTN", "STANDALONE"):
            with self.subTest(algorithm=algorithm):
                self.assertEqual(layout(algorithm), {})
        self.assertEqual(DSPARK, {"algorithm": "DSPARK", "num_draft_entries": 6})
        self.assertEqual(layout("DFLASH")["algorithm"], "DFLASH")
        # Draft heads are resliced across TP/DCP, so per-rank widths are not compared.
        self.assertEqual(layout("DSPARK", draft_item_len=4 * PAGE), DSPARK)
        # A prefill PP stage without the draft worker has no say.
        self.assertIsNone(layout("DSPARK", num_draft_entries=0))

    def test_old_decode_does_not_receive_the_layout(self):
        server = bootstrap_with([(0, DSPARK), (1, DSPARK)])
        self.assertNotIn("spec_kv_layout", sentinel(server, SENTINEL_QUERY))
        self.assertEqual(
            sentinel(server, {**SENTINEL_QUERY, "spec_kv_layout": "1"})[
                "spec_kv_layout"
            ],
            DSPARK,
        )

    def test_decode_checks_prefill_draft_kv_before_recording(self):
        target_only, eagle = layout(None), layout("EAGLE")
        cases = (
            ("dspark pair", None, DSPARK, DSPARK, True),
            (
                "eagle decode, target-only prefill",
                target_only,
                target_only,
                eagle,
                True,
            ),
            ("legacy prefill", LEGACY, LEGACY, DSPARK, True),
            (
                "dspark decode, target-only prefill",
                target_only,
                target_only,
                DSPARK,
                False,
            ),
            ("target-only decode, dspark prefill", None, DSPARK, target_only, False),
            ("dspark decode, dflash prefill", None, layout("DFLASH"), DSPARK, False),
            (
                "different draft model",
                None,
                layout("DSPARK", num_draft_entries=4),
                DSPARK,
                False,
            ),
        )
        for name, first_stage, last_stage, decode_layout, compatible in cases:
            for order in ((0, 1), (1, 0)):
                with self.subTest(name, registration_order=order):
                    stages = {0: first_stage, 1: last_stage}
                    server = bootstrap_with([(rank, stages[rank]) for rank in order])
                    manager = decode_manager(decode_layout)

                    def get(url, **kwargs):
                        query = dict(parse_qsl(urlsplit(url).query))
                        self.assertEqual(query["spec_kv_layout"], "1")
                        response = MagicMock(status_code=200)
                        response.json.return_value = sentinel(server, query)
                        return response

                    with patch(
                        "sglang.srt.disaggregation.common.conn.requests.get",
                        side_effect=get,
                    ):
                        if compatible:
                            self.assertTrue(
                                manager.try_ensure_parallel_info("prefill:8998")
                            )
                        else:
                            with self.assertRaisesRegex(
                                RuntimeError, "Speculative draft KV layout mismatch"
                            ):
                                manager.try_ensure_parallel_info("prefill:8998")
                            self.assertEqual(manager.prefill_info_table, {})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
