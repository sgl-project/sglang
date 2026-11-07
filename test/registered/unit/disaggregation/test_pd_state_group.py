"""Unit tests for the PD state-group pairing RFC (disaggregation/common/state_group).

A linear-attention state group may span several attention-DP groups (e.g. GDN
TP64 over attention TP8 x DP16). These tests pin the per-room edge plan, the
bootstrap wire field that carries the prefill's state TP, and the decode-side
refusal while the plan is not wired into the receiver.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


import asyncio
import json
import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    PrefillServerInfo,
)
from sglang.srt.disaggregation.common.state_group import (
    StateGroupLayout,
    plan_state_group_room,
    prefill_sender_id,
    tp_targets,
)
from sglang.test.test_utils import CustomTestCase


class TestStateGroupPlan(CustomTestCase):
    def test_stock_layout_reduces_to_attention_tp_mapping(self):
        # Prefill TP4 -> decode TP8, no attention DP, state sharded like
        # attention: _resolve_rank_mapping gives every decode rank one sender
        # and every prefill rank required_dst_info_num = 8 // 4 = 2.
        plan = plan_state_group_room(
            StateGroupLayout(attn_tp=4, attn_dp=1),
            StateGroupLayout(attn_tp=8, attn_dp=1),
            prefill_dp_rank=0,
            decode_dp_rank=0,
        )
        self.assertEqual(plan.decode_participants, list(range(8)))
        self.assertEqual(plan.senders, {d: [d // 2] for d in range(8)})
        self.assertEqual(plan.required_dst_info_num, {p: 2 for p in range(4)})
        self.assertEqual(plan.kv_senders, plan.state_senders)

    def test_stock_attention_dp_stays_inside_the_routed_groups(self):
        plan = plan_state_group_room(
            StateGroupLayout(attn_tp=2, attn_dp=4),
            StateGroupLayout(attn_tp=2, attn_dp=4),
            prefill_dp_rank=3,
            decode_dp_rank=1,
        )
        self.assertEqual(plan.senders, {2: [6], 3: [7]})
        self.assertEqual(plan.required_dst_info_num, {6: 1, 7: 1})

    def test_state_group_spanning_attention_dp(self):
        # Prefill attention TP8 x DP16 with GDN TP16; decode attention TP8 x
        # DP16 with GDN TP64. Room routed to prefill DP5 and decode DP11.
        prefill = StateGroupLayout(attn_tp=8, attn_dp=16, state_tp=16)
        decode = StateGroupLayout(attn_tp=8, attn_dp=16, state_tp=64)
        plan = plan_state_group_room(prefill, decode, 5, 11)

        # Decode: attention group 11 = ranks 88..95, inside state group 1 =
        # ranks 64..127, so all 64 state ranks participate.
        self.assertEqual(plan.decode_participants, list(range(64, 128)))
        # Prefill: state group 5 // 2 = 2 -> ranks 32..47 (attention DP 4 and 5),
        # although only DP5 (40..47) holds the request's KV.
        self.assertEqual(plan.prefill_participants, list(range(32, 48)))
        self.assertEqual(sorted(plan.kv_senders), list(range(88, 96)))
        for j in range(8):
            self.assertEqual(plan.kv_senders[88 + j], [40 + j])
        # State TP16 -> TP64: decode state rank i reads prefill state rank i // 4.
        for i, d in enumerate(range(64, 128)):
            self.assertEqual(plan.state_senders[d], [32 + i // 4])

        # Decode ranks of the attention group receive KV and state from
        # different prefill ranks; the others only state.
        self.assertEqual(plan.required_prefill_response_num(88), 2)
        self.assertEqual(plan.required_prefill_response_num(64), 1)
        # A KV sender also serves 4 decode state ranks; DP4 ranks only state.
        self.assertEqual(plan.required_dst_info_num[40], 5)
        self.assertEqual(plan.required_dst_info_num[32], 4)
        # Every metadata message is counted on both sides.
        self.assertEqual(
            sum(plan.required_dst_info_num.values()),
            sum(len(s) for s in plan.senders.values()),
        )

    def test_wider_prefill_state_group_gathers(self):
        plan = plan_state_group_room(
            StateGroupLayout(attn_tp=2, attn_dp=4, state_tp=8),
            StateGroupLayout(attn_tp=4, attn_dp=1),
            prefill_dp_rank=2,
            decode_dp_rank=0,
        )
        self.assertEqual(plan.state_senders[1], [2, 3])
        self.assertEqual(plan.kv_senders[1], [4])
        self.assertEqual(plan.senders[1], [2, 3, 4])

    def test_tp_targets_matches_rank_mapping_rule(self):
        self.assertEqual(tp_targets(4, 4, 3), [3])
        self.assertEqual([tp_targets(2, 8, r) for r in range(8)][::3], [[0], [0], [1]])
        self.assertEqual(tp_targets(8, 2, 1), [4, 5, 6, 7])
        with self.assertRaises(ValueError):
            tp_targets(3, 4, 0)

    def test_layout_validation(self):
        with self.assertRaises(ValueError):
            StateGroupLayout(attn_tp=8, attn_dp=16, state_tp=12)  # not a multiple
        with self.assertRaises(ValueError):
            StateGroupLayout(attn_tp=8, attn_dp=2, state_tp=32)  # exceeds TP x DP
        self.assertEqual(StateGroupLayout(8, 16, 64).dp_span, 8)
        self.assertEqual(StateGroupLayout(8, 16).dp_span, 1)

    def test_prefill_sender_id(self):
        stock = StateGroupLayout(attn_tp=4, attn_dp=2)
        # Unchanged from _prefill_unique_rank when the group does not span.
        self.assertEqual(
            [prefill_sender_id(stock, dp, tp) for dp in (0, 1) for tp in range(4)],
            [0, 1, 2, 3, 0, 1, 2, 3],
        )
        spanning = StateGroupLayout(attn_tp=8, attn_dp=16, state_tp=16)
        ids = [prefill_sender_id(spanning, dp, tp) for dp in (4, 5) for tp in range(8)]
        self.assertEqual(sorted(ids), list(range(16)))


class TestStateGroupWire(CustomTestCase):
    def test_prefill_server_info_parses_with_and_without_field(self):
        base = dict(
            attn_tp_size=8,
            attn_cp_size=1,
            dp_size=16,
            pp_size=1,
            page_size=1,
            kv_cache_dtype="auto",
            follow_bootstrap_room=True,
        )
        self.assertIsNone(PrefillServerInfo(**base).state_tp_size)
        self.assertEqual(
            PrefillServerInfo(**base, state_tp_size="16").state_tp_size, 16
        )

    def test_decode_refuses_spanning_state_group(self):
        check = CommonKVManager._check_state_group_supported
        info = SimpleNamespace(attn_tp_size=8, state_tp_size=None)
        check(SimpleNamespace(attn_tp_size=8, state_group_tp_size=None), info)
        check(SimpleNamespace(attn_tp_size=8, state_group_tp_size=8), info)
        with self.assertRaisesRegex(NotImplementedError, "decode state_tp=64"):
            check(SimpleNamespace(attn_tp_size=8, state_group_tp_size=64), info)
        with self.assertRaisesRegex(NotImplementedError, "prefill state_tp=16"):
            check(
                SimpleNamespace(attn_tp_size=8, state_group_tp_size=None),
                SimpleNamespace(attn_tp_size=8, state_tp_size=16),
            )


class _Request:
    def __init__(self, method, data=None, query=None):
        self.method = method
        self._data = data
        self.query = query or {}

    async def json(self):
        return self._data


def _bootstrap_server():
    """A CommonKVBootstrapServer without its HTTP thread."""
    server = object.__new__(CommonKVBootstrapServer)
    server.lock = asyncio.Lock()
    for name in (
        "pp_size",
        "attn_tp_size",
        "attn_cp_size",
        "dp_size",
        "page_size",
        "kv_cache_dtype",
        "kv_layout",
        "state_tp_size",
        "dsv41_spec_layout",
        "follow_bootstrap_room",
        "enable_dsa_cache_layer_split",
        "enable_kv_cache_sharding",
        "prefill_http_port",
    ):
        setattr(server, name, None)
    server.decode_allocation_policy = "early"
    server.prefill_port_table = {}
    server.room_to_dp_rank = {}
    server._registered_count = 0
    return server


def _row(tp_rank, **extra):
    return {
        "attn_tp_size": 2,
        "attn_tp_rank": tp_rank,
        "attn_cp_size": 1,
        "attn_cp_rank": 0,
        "attn_dp_size": 1,
        "attn_dp_rank": 0,
        "pp_size": 1,
        "pp_rank": 0,
        "system_dp_size": 1,
        "system_dp_rank": 0,
        "rank_ip": "127.0.0.1",
        "rank_port": 10000 + tp_rank,
        "page_size": 1,
        "kv_cache_dtype": "auto",
        **extra,
    }


_SERVER_INFO_QUERY = {
    "prefill_dp_rank": "-1",
    "prefill_cp_rank": "-1",
    "target_tp_rank": "-1",
    "target_pp_rank": "-1",
}


class TestStateGroupBootstrapServer(CustomTestCase):
    def _server_info(self, server):
        resp = asyncio.run(
            server._handle_route_get(_Request("GET", query=_SERVER_INFO_QUERY))
        )
        self.assertEqual(resp.status, 200)
        return json.loads(resp.text)

    def test_field_omitted_by_default(self):
        server = _bootstrap_server()
        for r in range(2):
            asyncio.run(server._handle_route_put(_Request("PUT", _row(r))))
        self.assertNotIn("state_tp_size", self._server_info(server))

    def test_field_served_when_registered(self):
        server = _bootstrap_server()
        for r in range(2):
            asyncio.run(
                server._handle_route_put(_Request("PUT", _row(r, state_tp_size=4)))
            )
        self.assertEqual(self._server_info(server)["state_tp_size"], 4)

    def test_inconsistent_field_rejected(self):
        server = _bootstrap_server()
        asyncio.run(server._handle_route_put(_Request("PUT", _row(0, state_tp_size=4))))
        resp = asyncio.run(server._handle_route_put(_Request("PUT", _row(1))))
        self.assertEqual(resp.status, 400)


if __name__ == "__main__":
    unittest.main()
