"""Unit tests for native MM scheduler-input tensor wrapping."""

import os
import unittest
from unittest.mock import patch

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.rust_server import MmProcessorHost  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNativeMmDrainAdapter(CustomTestCase):
    def setUp(self):
        self.host = MmProcessorHost.__new__(MmProcessorHost)
        self.host._native = {
            "feature_dim": 6,
            "image_token_id": 10,
            "vision_start_token_id": 11,
            "vision_end_token_id": 12,
            "video_token_id": 13,
        }

    def build(self, adapter=None):
        features = np.arange(30, dtype=np.float32)
        output = (adapter or self.host).build_native_mm(
            (
                features,
                [(1, 2, 2), (1, 1, 1)],
                [101, 202],
                [(2, 5), (8, 8)],
                np.arange(30, dtype=np.int64),
                -3,
            )
        )
        return output, features

    def test_wraps_and_slices_native_buffers(self):
        output, features = self.build()
        self.assertEqual(
            [tuple(item.feature.shape) for item in output.mm_items], [(4, 6), (1, 6)]
        )
        self.assertEqual([item.hash for item in output.mm_items], [101, 202])
        self.assertEqual(
            [item.offsets for item in output.mm_items], [[(2, 5)], [(8, 8)]]
        )
        self.assertEqual(tuple(output.mrope_positions.shape), (3, 10))
        self.assertEqual(output.mrope_position_delta.item(), -3)
        self.assertEqual(
            (output.im_start_id, output.im_token_id, output.im_end_id), (11, 10, 12)
        )
        features[0] = 99
        self.assertEqual(output.mm_items[0].feature[0, 0].item(), 99)

    def test_optional_pad_values_use_precomputed_hashes(self):
        from sglang.srt.managers.schedule_batch import _compute_pad_value

        # The whole point of worker-precomputed hashes is that the scheduler
        # loop never runs hash_feature — make any call a hard failure.
        with (
            patch.dict(os.environ, {"SGLANG_MM_PRECOMPUTE_HASH": "1"}),
            patch(
                "sglang.srt.managers.mm_utils.hash_feature",
                side_effect=AssertionError("scheduler loop must not hash features"),
            ),
        ):
            output, _ = self.build()
        self.assertEqual(
            [item.pad_value for item in output.mm_items],
            [_compute_pad_value(101), _compute_pad_value(202)],
        )

    def test_standalone_client_adapter_matches_host(self):
        """``RustServer.drain`` duck-types ``build_native_mm`` across the
        in-process host and the standalone client; the client must produce
        the same wrapping from the handshake-provided native dict (guards a
        regression where the client wires the dict wrong — no other test
        covers its adapter path)."""
        from types import SimpleNamespace

        from sglang.srt.managers.standalone_mm_host import StandaloneMmClient

        client = StandaloneMmClient(
            ipc_name="ipc:///dev/null",
            proc=SimpleNamespace(is_alive=lambda: True),
            spec="spec",
            native=dict(self.host._native),
        )
        host_out, _ = self.build()
        client_out, _ = self.build(adapter=client)
        for name in ("im_token_id", "im_start_id", "im_end_id", "video_token_id"):
            self.assertEqual(getattr(client_out, name), getattr(host_out, name))
        self.assertEqual(
            [tuple(i.feature.shape) for i in client_out.mm_items],
            [tuple(i.feature.shape) for i in host_out.mm_items],
        )
        self.assertEqual(
            [i.hash for i in client_out.mm_items],
            [i.hash for i in host_out.mm_items],
        )
        self.assertEqual(
            client_out.mrope_position_delta.item(),
            host_out.mrope_position_delta.item(),
        )


if __name__ == "__main__":
    unittest.main()
