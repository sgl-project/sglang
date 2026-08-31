import struct
import unittest
from types import SimpleNamespace

import msgspec
from sglang.srt.managers.rust_server import RustServer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeServer:
    def __init__(self):
        self.embedding_calls = []
        self.errors = []

    def push_embedding(self, header, data):
        self.embedding_calls.append((bytes(header), bytes(data)))
        return True

    def push_error(self, rid, message):
        self.errors.append((rid, message))
        return True


class TestEmbeddingBridge(CustomTestCase):
    def _bridge(self):
        bridge = RustServer.__new__(RustServer)
        bridge.server = _FakeServer()
        return bridge

    def test_dense_batch_is_one_columnar_f32_call(self):
        bridge = self._bridge()
        bridge.push_embedding(
            SimpleNamespace(
                rids=["a", "b"],
                finished_reasons=[{"type": "stop"}, {"type": "stop"}],
                embeddings=[[0.5, -2.0], [7.0]],
                prompt_tokens=[3, 4],
            )
        )
        self.assertEqual(len(bridge.server.embedding_calls), 1)
        header, data = bridge.server.embedding_calls[0]
        self.assertEqual(
            msgspec.msgpack.decode(header),
            [
                ["a", "b"],
                [{"type": "stop"}, {"type": "stop"}],
                [3, 4],
                [2, 1],
            ],
        )
        self.assertEqual(struct.unpack("<3f", data), (0.5, -2.0, 7.0))

    def test_sparse_nested_and_scalar_outputs_fail_explicitly(self):
        bridge = self._bridge()
        bridge.push_embedding(
            SimpleNamespace(
                rids=["sparse", "nested", "scalar"],
                finished_reasons=[None, None, None],
                embeddings=[{1: 0.5}, [[1.0]], 0.25],
                prompt_tokens=[1, 1, 1],
            )
        )
        self.assertEqual(len(bridge.server.errors), 3)
        self.assertEqual(bridge.server.embedding_calls, [])


if __name__ == "__main__":
    unittest.main()
