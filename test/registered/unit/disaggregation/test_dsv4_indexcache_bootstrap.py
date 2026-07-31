import asyncio
import json
import unittest

from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_DESCRIPTOR = {
    "dsv4_index_cache_layout_signature": "sigA",
    "dsv4_index_cache_producer_layer_ids": [2, 6],
}


class _FakeRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class _FakeGetRequest:
    def __init__(self, wants_descriptor):
        self.query = {
            "prefill_dp_rank": "-1",
            "prefill_cp_rank": "-1",
            "target_tp_rank": "-1",
            "target_pp_rank": "-1",
        }
        if wants_descriptor:
            self.query["dsv4_indexcache_desc"] = "1"


def _base_payload(**overrides):
    payload = {
        "attn_tp_size": 1,
        "attn_tp_rank": 0,
        "attn_cp_size": 1,
        "attn_cp_rank": 0,
        "attn_dp_size": 1,
        "attn_dp_rank": 0,
        "pp_size": 1,
        "pp_rank": 0,
        "system_dp_size": 1,
        "system_dp_rank": 0,
        "rank_ip": "127.0.0.1",
        "rank_port": 10000,
        "page_size": 64,
        "kv_cache_dtype": "auto",
        "load_balance_method": "follow_bootstrap_room",
        "dsv4_index_cache_layout_signature": None,
        "dsv4_index_cache_producer_layer_ids": None,
    }
    payload.update(overrides)
    return payload


class TestBootstrapIndexCacheDescriptorConsistency(CustomTestCase):
    def _new_server(self):
        server = CommonKVBootstrapServer.__new__(CommonKVBootstrapServer)
        server.lock = asyncio.Lock()
        server.attn_tp_size = None
        server.attn_cp_size = None
        server.dp_size = None
        server.pp_size = None
        server.page_size = None
        server.kv_cache_dtype = None
        server.follow_bootstrap_room = None
        server.enable_dsa_cache_layer_split = None
        server.prefill_http_port = None
        server.dsv4_index_cache_layout_signature = None
        server.dsv4_index_cache_producer_layer_ids = None
        server._dsv4_descriptor_seen = False
        server.prefill_port_table = {}
        server.room_to_dp_rank = {}
        server._registered_count = 0
        return server

    def _put(self, server, **overrides):
        return asyncio.run(
            server._handle_route_put(_FakeRequest(_base_payload(**overrides)))
        )

    def test_matching_descriptors_and_non_v4_ranks_register(self):
        cases = [("descriptor", _DESCRIPTOR), ("non_v4", {})]
        for name, descriptor in cases:
            with self.subTest(name=name):
                server = self._new_server()
                responses = [
                    self._put(
                        server,
                        attn_tp_rank=rank,
                        rank_port=10000 + rank,
                        **descriptor,
                    )
                    for rank in range(2)
                ]
                self.assertEqual(
                    [response.status for response in responses], [200, 200]
                )
                self.assertEqual(server._registered_count, 2)
                self.assertEqual(server._dsv4_descriptor_seen, bool(descriptor))

    def test_descriptor_mismatches_are_rejected(self):
        cases = [
            ("descriptor_then_missing", _DESCRIPTOR, {}),
            ("missing_then_descriptor", {}, _DESCRIPTOR),
            (
                "layout",
                _DESCRIPTOR,
                {
                    "dsv4_index_cache_layout_signature": "sigB",
                    "dsv4_index_cache_producer_layer_ids": [2, 6],
                },
            ),
            (
                "producers",
                _DESCRIPTOR,
                {
                    "dsv4_index_cache_layout_signature": "sigA",
                    "dsv4_index_cache_producer_layer_ids": [2, 8],
                },
            ),
        ]
        for name, first, second in cases:
            with self.subTest(name=name):
                server = self._new_server()
                first_response = self._put(
                    server,
                    attn_tp_rank=0,
                    rank_port=10000,
                    **first,
                )
                second_response = self._put(
                    server,
                    attn_tp_rank=1,
                    rank_port=10001,
                    **second,
                )
                self.assertEqual(first_response.status, 200)
                self.assertEqual(second_response.status, 400)
                self.assertEqual(server._registered_count, 1)

    def test_get_descriptor_capability_matrix(self):
        for has_descriptor in (False, True):
            for wants_descriptor in (False, True):
                with self.subTest(
                    has_descriptor=has_descriptor,
                    wants_descriptor=wants_descriptor,
                ):
                    server = self._new_server()
                    descriptor = _DESCRIPTOR if has_descriptor else {}
                    self._put(server, **descriptor)
                    response = asyncio.run(
                        server._handle_route_get(
                            _FakeGetRequest(wants_descriptor=wants_descriptor)
                        )
                    )
                    self.assertEqual(response.status, 200)
                    body = json.loads(response.body.decode())
                    keys = (
                        "dsv4_index_cache_layout_signature",
                        "dsv4_index_cache_producer_layer_ids",
                    )
                    if wants_descriptor:
                        expected = ("sigA", [2, 6]) if has_descriptor else (None, None)
                        self.assertEqual(tuple(body[key] for key in keys), expected)
                    else:
                        for key in keys:
                            self.assertNotIn(key, body)


if __name__ == "__main__":
    unittest.main()
