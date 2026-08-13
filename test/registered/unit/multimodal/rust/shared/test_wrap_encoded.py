"""``MmSpec.wrap_encoded`` (managers/rust_server.py): the drain-time wrapping
contracts. Features arrive as one named POSIX segment per item; the
``stub_broadcast`` policy decides whether an item stays a ``ShmPointerMMData``
stub or rank 0 takes ownership at the drain. Synthetic buffers — no Rust
extension needed."""

import os
import unittest
from multiprocessing import shared_memory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.rust_server import MmSpec  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestWrapEncoded(CustomTestCase):
    """``stub_broadcast=True`` (single-node TP): each item becomes a
    ``ShmPointerMMData`` stub whose ``materialize()`` yields that item's
    slice — and unlinks, taking the cleanup duty exactly once."""

    stub_broadcast = True

    def setUp(self):
        # feature_dim == 3 * temporal_patch_size * patch_size**2 == 6.
        self.spec = MmSpec(
            family="qwen_vl",
            image_token_id=10,
            patch_size=1,
            merge_size=1,
            temporal_patch_size=2,
            min_pixels=1,
            max_pixels=1 << 30,
            image_mean=(0.0, 0.0, 0.0),
            image_std=(1.0, 1.0, 1.0),
            resample="aten_u8",
            vision_start_token_id=11,
            vision_end_token_id=12,
            video_token_id=13,
            stub_broadcast=self.stub_broadcast,
        )
        self._segments = []

    def tearDown(self):
        # Defensive: unlink anything a failing test left behind.
        for shm in self._segments:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass

    GRIDS = [(1, 2, 2), (1, 1, 1)]
    HASHES = [101, 202]
    OFFSETS = [(2, 5), (8, 8)]

    def _park(self, features):
        """The worker's transport: one POSIX segment per item, holding that
        item's `t*h*w` rows."""
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

    def build(self):
        features = np.arange(30, dtype=np.float32)
        output = self.spec.wrap_encoded(
            SimpleNamespace(  # the shape of Rust's MmEncodedResult
                shm_names=self._park(features),
                grids=self.GRIDS,
                hashes=self.HASHES,
                offsets=self.OFFSETS,
                mrope=np.arange(30, dtype=np.int64),
                mrope_delta=-3,
            ),
        )
        return output, features

    def assert_scalars(self, output):
        self.assertEqual([item.hash for item in output.mm_items], self.HASHES)
        self.assertEqual(
            [item.offsets for item in output.mm_items], [[(2, 5)], [(8, 8)]]
        )
        self.assertEqual(tuple(output.mrope_positions.shape), (3, 10))
        self.assertEqual(output.mrope_position_delta.item(), -3)
        self.assertEqual(
            (output.im_start_id, output.im_token_id, output.im_end_id), (11, 10, 12)
        )

    def test_wraps_and_slices_native_buffers(self):
        import torch

        from sglang.srt.managers.mm_utils import ShmPointerMMData

        output, features = self.build()
        self.assert_scalars(output)
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
        for item in output.mm_items:
            with self.assertRaises(FileNotFoundError):
                shared_memory.SharedMemory(name=item.feature.shm_name)

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


class TestWrapEncodedMaterialized(TestWrapEncoded):
    """``stub_broadcast=False`` (rank 1, multinode, skip_tokenizer_init): the
    drain hands the scheduler plain tensors — zero-copy views over the
    segments' pages, already unlinked."""

    stub_broadcast = False

    def test_wraps_and_slices_native_buffers(self):
        import torch

        output, features = self.build()
        self.assert_scalars(output)
        for item in output.mm_items:
            self.assertIsInstance(item.feature, torch.Tensor)
        expected = torch.from_numpy(features).reshape(-1, 6)
        self.assertTrue(torch.equal(output.mm_items[0].feature, expected[:4]))
        self.assertTrue(torch.equal(output.mm_items[1].feature, expected[4:]))
        # Zero-copy: the feature aliases the segment's pages (write through
        # the creator's still-open mapping, observe through the tensor).
        self._segments[0].buf[:4] = np.float32(99).tobytes()
        self.assertEqual(output.mm_items[0].feature[0, 0].item(), 99)
        # Rank 0 took ownership: the names are gone while the mappings — and
        # the pages — live on with the tensors.
        for shm in self._segments:
            self.assertFalse(os.path.exists(os.path.join("/dev/shm", shm.name)))
        self.assertEqual(output.mm_items[1].feature[0, 0].item(), 24)


if __name__ == "__main__":
    unittest.main()
