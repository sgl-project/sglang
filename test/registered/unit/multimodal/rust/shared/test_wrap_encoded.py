"""``RustMmProcessor.wrap_encoded``: the drain-time
wrapping contracts — tensors are zero-copy views over the Rust-owned buffers, and
pad values come from worker-precomputed hashes, since the scheduler loop must
never hash features. Synthetic buffers, so this needs no Rust extension."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.rust_server.multimodal import (  # noqa: E402
    RustMmProcessor,
    RustMmSpec,
)

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestWrapEncoded(CustomTestCase):
    def setUp(self):
        # feature_dim == 3 * temporal_patch_size * patch_size**2 == 6.
        self.spec = RustMmSpec(
            family="qwen_vl",
            feature_shm=False,
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
        )

    GRIDS = [(1, 2, 2), (1, 1, 1)]
    HASHES = [101, 202]
    OFFSETS = [(2, 5), (8, 8)]

    def transport(self, features):
        """Inline: each item's features ride their own shaped numpy array
        (views of one backing array here, so a write shows through both)."""
        return {
            "mm.feature.0": features[:24].reshape(4, 6),
            "mm.feature.1": features[24:].reshape(1, 6),
        }

    def meta(self):
        """The `mm.meta` sidecar as the Rust worker encodes it."""
        return np.frombuffer(
            msgspec.msgpack.encode(
                {
                    "items": [
                        {
                            "modality": "image",
                            "hash": item_hash,
                            "offsets": [list(offset)],
                            "model_specific_data": {"image_grid_thw": list(grid)},
                        }
                        for grid, item_hash, offset in zip(
                            self.GRIDS, self.HASHES, self.OFFSETS
                        )
                    ],
                    "token_ids": None,
                    "mrope_delta": -3,
                }
            ),
            dtype=np.uint8,
        )

    def build(self):
        features = np.arange(30, dtype=np.float32)
        buffers = {  # the `mm.*` buffers of one `IngressRequest`
            "mm.mrope": np.arange(30, dtype=np.int64).reshape(3, 10),
            "mm.meta": self.meta(),
            **self.transport(features),
        }
        output = RustMmProcessor.wrap_encoded(self.spec, buffers)
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


class FakeShmBuffer:
    """The Rust extension's ``ShmBuffer`` contract: it owns the unlink until
    ``release()`` hands it to the stubs, or ``discard()`` unlinks now."""

    def __init__(self, name, shape):
        self.name = name
        self.dtype = "float32"
        self.shape = shape
        self.released = False
        self.discarded = False

    def release(self):
        self.released = True

    def discard(self):
        from multiprocessing import shared_memory

        self.discarded = True
        try:
            handle = shared_memory.SharedMemory(name=self.name)
        except FileNotFoundError:
            return
        handle.close()
        handle.unlink()


def segment_exists(name) -> bool:
    from multiprocessing import shared_memory

    try:
        handle = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return False
    handle.close()
    return True


class TestWrapEncodedShm(TestWrapEncoded):
    """The shm entry shape (TP>1): features arrive as named POSIX segments, and
    each item becomes a ``ShmPointerMMData`` stub whose ``materialize()`` yields
    that item's slice — and unlinks, taking the cleanup duty exactly once."""

    def setUp(self):
        super().setUp()
        self._segments = []
        self._buffers = []

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

    def transport(self, features):
        """Shm: the worker placed each item's slice in its own segment; the
        buffer is the `ShmBuffer` stub naming it."""
        self._buffers = [
            FakeShmBuffer(name, (n, 6))
            for name, n in zip(
                self._park(features), [t * h * w for t, h, w in self.GRIDS]
            )
        ]
        return {f"mm.feature.{i}": b for i, b in enumerate(self._buffers)}

    def test_wraps_and_slices_native_buffers(self):
        import torch

        from sglang.srt.managers.mm_utils import ShmPointerMMData

        output, features = self.build()
        for item in output.mm_items:
            self.assertIsInstance(item.feature, ShmPointerMMData)
        # The wrapper never decides ownership: nothing released or discarded
        # here, and the segments are still there for the drain loop to settle.
        self.assertFalse(any(b.released or b.discarded for b in self._buffers))
        self.assertTrue(all(segment_exists(b.name) for b in self._buffers))
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

    def test_partial_wrap_failure_closes_stubs_and_decides_nothing(self):
        """Item 1 fails after item 0's stub exists: the stub is closed, and the
        wrapper neither releases nor discards; that is the drain loop's call."""
        features = np.arange(30, dtype=np.float32)
        meta = msgspec.msgpack.decode(self.meta().tobytes())
        del meta["items"][1]["model_specific_data"]["image_grid_thw"]
        buffers = {
            "mm.mrope": np.arange(30, dtype=np.int64).reshape(3, 10),
            "mm.meta": np.frombuffer(msgspec.msgpack.encode(meta), dtype=np.uint8),
            **self.transport(features),
        }
        with self.assertRaises(KeyError):
            RustMmProcessor.wrap_encoded(self.spec, buffers)
        self.assertFalse(any(b.released or b.discarded for b in self._buffers))


class TestDrainShmOwnership(TestWrapEncodedShm):
    """``RustServer.drain`` must not leak a segment on any rejection path: a
    header that never decodes (the buffers are never looked at) and a request
    whose wrapping fails part-way both end with every segment unlinked and the
    client told."""

    # Reuse the shm fixture only; the inherited cases already ran above.
    test_wraps_and_slices_native_buffers = None
    test_optional_pad_values_use_precomputed_hashes = None
    test_partial_wrap_failure_closes_stubs_and_decides_nothing = None

    def _server(self, header, buffers):
        from sglang.srt.rust_server.server import RustServer

        errors = []
        fake = SimpleNamespace(
            recv_requests=lambda limit: [
                SimpleNamespace(header=header, buffers=list(buffers.items()))
            ],
            push_error=lambda rid, msg: errors.append((rid, msg)),
        )
        return RustServer(server=fake, http_port=0, mm_spec=self.spec), errors

    @staticmethod
    def _header(rid):
        """A minimal generate header, as the Rust api_server encodes it."""
        from sglang.srt.managers.io_struct import (
            TokenizedGenerateReqInput,
            msgpack_encode,
        )
        from sglang.srt.sampling.sampling_params import SamplingParams

        return msgpack_encode(
            TokenizedGenerateReqInput(
                rid=rid,
                input_text="t",
                input_ids=None,  # rides the `input_ids` buffer, as from Rust
                input_embeds=None,
                mm_inputs=None,
                token_type_ids=None,
                sampling_params=SamplingParams(),
                return_logprob=False,
                logprob_start_len=0,
                top_logprobs_num=0,
                token_ids_logprob=None,
                stream=False,
            )
        )

    def _request_buffers(self, meta=None):
        features = np.arange(30, dtype=np.float32)
        return {
            "mm.mrope": np.arange(30, dtype=np.int64).reshape(3, 10),
            "mm.meta": self.meta() if meta is None else meta,
            **self.transport(features),
        }

    def test_malformed_header_discards_segments(self):
        server, errors = self._server(b"\xc1", self._request_buffers())
        self.assertEqual(server.drain(8), [])
        self.assertFalse(any(segment_exists(b.name) for b in self._buffers))
        self.assertFalse(any(b.released for b in self._buffers))

    def test_wrap_failure_rejects_request_and_discards_segments(self):
        meta = msgspec.msgpack.decode(self.meta().tobytes())
        del meta["items"][1]["model_specific_data"]["image_grid_thw"]
        header = self._header("r1")
        server, errors = self._server(
            header,
            self._request_buffers(
                np.frombuffer(msgspec.msgpack.encode(meta), dtype=np.uint8)
            ),
        )
        self.assertEqual(server.drain(8), [], "rejected, not handed to the scheduler")
        self.assertEqual([rid for rid, _ in errors], ["r1"])
        self.assertFalse(any(segment_exists(b.name) for b in self._buffers))
        self.assertFalse(any(b.released for b in self._buffers))

    def test_admitted_request_keeps_segments_for_receivers(self):
        from sglang.srt.managers.mm_utils import ShmPointerMMData

        header = self._header("r2")
        server, errors = self._server(header, self._request_buffers())
        drained = server.drain(8)
        self.assertEqual(errors, [], msg=str(errors))
        (obj,) = drained
        self.assertTrue(all(b.released for b in self._buffers))
        # Still there until a receiver materializes: nothing unlinked early.
        self.assertTrue(all(segment_exists(b.name) for b in self._buffers))
        for item in obj.mm_inputs.mm_items:
            self.assertIsInstance(item.feature, ShmPointerMMData)
            item.feature.materialize()
        self.assertFalse(any(segment_exists(b.name) for b in self._buffers))

    def test_external_wrapper_never_touches_ownership(self):
        """An external package's `_wrap_mm_result` builds stubs from the
        segment names and knows nothing about release/discard. Its segments
        must still be there for the TP receivers after `drain` returns, and
        must be unlinked when its wrapping fails."""
        from sglang.srt.managers.mm_utils import ShmPointerMMData
        from sglang.srt.rust_server.server import RustServer

        class PackageServer(RustServer):
            fail = False

            def _wrap_mm_result(self, buffers):
                if self.fail:
                    raise RuntimeError("package wrapper failed")
                stubs = []
                for i in range(2):
                    b = buffers[f"mm.feature.{i}"]
                    stub = ShmPointerMMData.__new__(ShmPointerMMData)
                    stub.__setstate__(
                        {"shm_name": b.name, "shape": b.shape, "dtype": None}
                    )
                    stubs.append(stub)
                return stubs  # opaque to drain; only ownership matters here

        for fail in (False, True):
            with self.subTest(fail=fail):
                server, errors = self._server(
                    self._header("pkg"), self._request_buffers()
                )
                server.__class__ = PackageServer
                server.fail = fail
                drained = server.drain(8)
                if fail:
                    self.assertEqual(drained, [])
                    self.assertEqual([rid for rid, _ in errors], ["pkg"])
                    self.assertTrue(all(b.discarded for b in self._buffers))
                    self.assertFalse(any(segment_exists(b.name) for b in self._buffers))
                else:
                    self.assertEqual(len(drained), 1)
                    self.assertTrue(all(b.released for b in self._buffers))
                    self.assertTrue(all(segment_exists(b.name) for b in self._buffers))
                    for stub in drained[0].mm_inputs:
                        stub.close_and_unlink()
                self._segments, self._buffers = [], []
                self.setUp()


if __name__ == "__main__":
    unittest.main()
