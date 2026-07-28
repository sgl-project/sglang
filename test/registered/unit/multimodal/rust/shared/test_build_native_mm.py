"""``NativeMmHost.build_native_mm`` (managers/rust_server.py): the drain-time
wrapping contracts — tensors are zero-copy views over the Rust-owned buffers,
and pad values come from worker-precomputed hashes (the scheduler loop must
never hash features). Synthetic buffers only, so this runs without the Rust
extension (unlike the qwen parity suite)."""

import os
import unittest
from unittest.mock import patch

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.rust_server import NativeMmHost  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestBuildNativeMm(CustomTestCase):
    def setUp(self):
        self.host = NativeMmHost.__new__(NativeMmHost)
        self.host._native = {
            "feature_dim": 6,
            "image_token_id": 10,
            "vision_start_token_id": 11,
            "vision_end_token_id": 12,
            "video_token_id": 13,
        }

    GRIDS = [(1, 2, 2), (1, 1, 1)]
    HASHES = [101, 202]
    OFFSETS = [(2, 5), (8, 8)]

    def build(self, shm_names=None):
        features = np.arange(30, dtype=np.float32)
        output = self.host.build_native_mm(
            (
                None if shm_names else features,
                shm_names,
                self.GRIDS,
                self.HASHES,
                self.OFFSETS,
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


class TestBuildNativeMmShm(TestBuildNativeMm):
    """The shm entry shape (TP>1): features arrive as named POSIX segments the
    Rust worker wrote; each item becomes a ``ShmPointerMMData`` stub whose
    ``materialize()`` yields the item's slice — and unlinks, transferring the
    cleanup duty exactly once."""

    def setUp(self):
        super().setUp()
        self._segments = []

    def tearDown(self):
        # Defensive: unlink anything a failing test left behind.
        for shm in self._segments:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass

    def _park(self, features):
        from multiprocessing import shared_memory

        names, row = [], 0
        for t, h, w in self.GRIDS:
            n = t * h * w
            payload = features[row * 6 : (row + n) * 6].tobytes()
            shm = shared_memory.SharedMemory(create=True, size=len(payload))
            shm.buf[:] = payload
            self._segments.append(shm)
            names.append(shm.name)
            row += n
        return names

    def build(self, shm_names=None):
        features = np.arange(30, dtype=np.float32)
        names = self._park(features)
        output = self.host.build_native_mm(
            (
                None,
                names,
                self.GRIDS,
                self.HASHES,
                self.OFFSETS,
                np.arange(30, dtype=np.int64),
                -3,
            )
        )
        return output, features

    def test_wraps_and_slices_native_buffers(self):
        import torch

        from sglang.srt.managers.mm_utils import ShmPointerMMData

        output, features = self.build()
        for item in output.mm_items:
            self.assertIsInstance(item.feature, ShmPointerMMData)
        # The stub is a zero-copy view over the segment until materialized.
        self.assertEqual(
            [tuple(item.feature.shape) for item in output.mm_items], [(4, 6), (1, 6)]
        )
        self.assertEqual(
            [item.feature.precomputed_hash for item in output.mm_items], self.HASHES
        )
        tensors = [item.feature.materialize() for item in output.mm_items]
        expected = torch.from_numpy(features).reshape(-1, 6)
        self.assertTrue(torch.equal(tensors[0], expected[:4]))
        self.assertTrue(torch.equal(tensors[1], expected[4:]))
        # materialize() unlinked: the names must be gone.
        from multiprocessing import shared_memory

        for item in output.mm_items:
            with self.assertRaises(FileNotFoundError):
                shared_memory.SharedMemory(name=item.feature.shm_name)

    def test_optional_pad_values_use_precomputed_hashes(self):
        super().test_optional_pad_values_use_precomputed_hashes()


if __name__ == "__main__":
    unittest.main()
