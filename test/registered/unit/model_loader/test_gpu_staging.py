"""Unit tests for GPU staging of safetensors checkpoints (gpu_staging.py)."""

import json
import os
import struct
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import safetensors
import safetensors.torch
import torch

import sglang.srt.model_loader.gpu_staging as gpu_staging
import sglang.srt.model_loader.loader as loader_mod
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.model_loader.gpu_staging import GpuStagingReader
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.weight_utils import (
    CheckpointTensorReaders,
    safetensors_weights_iterator,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

# Small sizes so that a test file has several ranges, several reads per range
# and tensors over the size limit.
_SMALL = dict(_CHUNK_BYTES=4096, _MAX_TENSOR_BYTES=16384, _READ_BYTES=1024)


def _checkpoint_tensors():
    g = torch.Generator().manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, generator=g)

    def randint(low, high, *shape, dtype):
        return torch.randint(low, high, shape, generator=g).to(dtype)

    tensors = {
        "bf16": rand(37, 64).to(torch.bfloat16),
        "f16": rand(3, 5, 7).to(torch.float16),
        "f32": rand(1000),
        "f64": rand(9).to(torch.float64),
        "i8": randint(-128, 128, 13, dtype=torch.int8),
        "u8": randint(0, 256, 7, dtype=torch.uint8),
        "i16": randint(-1000, 1000, 2, 3, dtype=torch.int16),
        "i32": randint(-(2**31), 2**31 - 1, 11, dtype=torch.int32),
        "i64": randint(-(2**40), 2**40, 5, dtype=torch.int64),
        "bool": torch.tensor([True, False, True]),
        "scalar": torch.tensor(3.5),
        "fp8_e4m3": rand(33, 17).to(torch.float8_e4m3fn),
        "fp8_e5m2": rand(5).to(torch.float8_e5m2),
        "over_chunk": rand(3000),  # 12000 bytes: a range of its own
        "oversize": rand(5000),  # 20000 bytes: stays a mapped view
        "empty": torch.empty(0, 4),
    }
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is not None and e8m0 in safetensors.torch._TYPES.values():
        tensors["fp8_e8m0"] = randint(0, 255, 64, dtype=torch.uint8).view(e8m0)
    return tensors


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def _byte_ranges(path):
    with open(path, "rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return {k: [8 + size + o for o in v["data_offsets"]] for k, v in header.items()}


@unittest.skipUnless(torch.cuda.is_available(), "GPU staging needs CUDA")
class TestGpuStagingReader(CustomTestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, "model.safetensors")
        self.tensors = _checkpoint_tensors()
        safetensors.torch.save_file(self.tensors, self.path)
        small = patch.multiple(gpu_staging, **_SMALL)
        small.start()
        self.addCleanup(small.stop)
        self.reader = GpuStagingReader()
        self.addCleanup(self.reader.close)

    def tearDown(self):
        self.tmpdir.cleanup()

    def assert_matches_safe_open(self, got, path=None):
        with safetensors.safe_open(path or self.path, framework="pt") as f:
            for name, t in got:
                ref = f.get_tensor(name)
                with self.subTest(name=name):
                    self.assertEqual(t.dtype, ref.dtype)
                    self.assertEqual(t.shape, ref.shape)
                    self.assertTrue(torch.equal(_bytes(t), _bytes(ref)))

    def test_staged_tensors_match_safe_open(self):
        got = list(self.reader.read_file(self.path))
        self.assertEqual(sorted(n for n, _ in got), sorted(self.tensors))
        self.assert_matches_safe_open(got)
        on_gpu = {n for n, t in got if t.is_cuda}
        self.assertEqual(on_gpu, set(self.tensors) - {"oversize", "empty"})
        self.assertEqual(self.reader.staged_tensors, len(on_gpu))
        self.assertEqual(self.reader.mapped_tensors, 2)

    def test_skip_is_applied_before_reading(self):
        skipped = {"bf16", "over_chunk", "oversize"}
        asked = {}

        def skip(name, nbytes):
            asked[name] = nbytes
            return name in skipped

        reads = []
        pread_into = gpu_staging._pread_into

        def record(fd, buf, pos):
            reads.append((pos, pos + len(buf)))
            pread_into(fd, buf, pos)

        with patch.object(gpu_staging, "_pread_into", side_effect=record):
            got = list(self.reader.read_file(self.path, skip))
        self.assertEqual({n for n, _ in got}, set(self.tensors) - skipped)
        self.assert_matches_safe_open(got)
        ranges = _byte_ranges(self.path)
        self.assertEqual(asked, {n: end - begin for n, (begin, end) in ranges.items()})
        for name in skipped:
            begin, end = ranges[name]
            for start, stop in reads:
                self.assertFalse(start < end and begin < stop, (name, start, stop))

    def test_read_errors_reach_the_loader(self):
        for name, target, effect, error in (
            ("os error", "_pread_into", OSError("I/O error"), OSError),
            ("short read", "os.preadv", lambda *_: 0, EOFError),
        ):
            with self.subTest(name):
                target = target if "." in target else f"{gpu_staging.__name__}.{target}"
                with patch(target, side_effect=effect), self.assertRaises(error):
                    list(self.reader.read_file(self.path))
                self.assertFalse(self._producer_alive())
        # The reader still works after an error.
        self.assert_matches_safe_open(list(self.reader.read_file(self.path)))

    def test_buffers_are_reused_safely_while_the_loader_lags(self):
        # One range per tensor, many more ranges than buffers.
        tensors = {f"w{i:03d}": torch.randn(1024) for i in range(40)}
        path = os.path.join(self.tmpdir.name, "many.safetensors")
        safetensors.torch.save_file(tensors, path)
        got = []
        for name, t in self.reader.read_file(path):
            got.append((name, t))  # A loader may keep what it is given.
            if hasattr(torch.cuda, "_sleep"):
                # Delay the copies out of the next buffer on this stream.
                torch.cuda._sleep(1_000_000)
        torch.cuda.synchronize()
        self.assertEqual(len({t.data_ptr() for _, t in got}), len(tensors))
        self.assert_matches_safe_open(got, path)

    def test_closing_early_stops_the_reads(self):
        weights = self.reader.read_file(self.path)
        next(weights)
        weights.close()
        self.assertFalse(self._producer_alive())
        self.assert_matches_safe_open(list(self.reader.read_file(self.path)))

    @staticmethod
    def _producer_alive():
        return any(t.name == "gpu-staging" for t in threading.enumerate())


@unittest.skipUnless(torch.cuda.is_available(), "GPU staging needs CUDA")
class TestGpuStagingIterator(CustomTestCase):
    @patch("torch.distributed.is_initialized", return_value=False)
    def test_matches_the_default_iterator(self, _):
        g = torch.Generator().manual_seed(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i in range(3):
                paths.append(os.path.join(tmpdir, f"model-{i}.safetensors"))
                tensors = {
                    f"layers.{i}.w{j}": torch.randn(64 * (j + 1), generator=g)
                    for j in range(20)
                }
                safetensors.torch.save_file(tensors, paths[-1])
            # Readers skip what a rank does not read (expert parallelism); their
            # prefetch needs torch.distributed.
            readers = CheckpointTensorReaders(
                rank_of=lambda name, nbytes: None if name.endswith("w3") else (0, 1)
            )
            for kwargs in ({"prefetch": True}, {"readers": readers}):
                with self.subTest(readers="readers" in kwargs):
                    expected = dict(safetensors_weights_iterator(paths, **kwargs))
                    with patch.multiple(gpu_staging, **_SMALL):
                        staged = dict(
                            safetensors_weights_iterator(
                                paths, gpu_staging=True, **kwargs
                            )
                        )
                    self.assertEqual(sorted(staged), sorted(expected))
                    for name, t in staged.items():
                        self.assertTrue(t.is_cuda)
                        self.assertTrue(torch.equal(_bytes(t), _bytes(expected[name])))


class TestGpuStagingConfig(CustomTestCase):
    def setUp(self):
        loader_mod._log_gpu_staging_off.cache_clear()

    def test_extra_config_validation(self):
        DefaultModelLoader(LoadConfig(model_loader_extra_config={"gpu_staging": True}))
        for load_format, extra in (
            (LoadFormat.AUTO, {"gpu_staging": "true"}),
            (LoadFormat.FASTSAFETENSORS, {"gpu_staging": True}),
            (LoadFormat.FLASH_RL, {"gpu_staging": True}),
        ):
            with self.subTest(load_format=load_format, extra=extra):
                with self.assertRaises(ValueError):
                    DefaultModelLoader(
                        LoadConfig(
                            load_format=load_format, model_loader_extra_config=extra
                        )
                    )

    def _dispatch(self, extra, tp_size=4, ep_size=1, model_config=True, draft=False):
        """Returns (staged, the 'no effect' log message or None)."""
        loader = DefaultModelLoader(LoadConfig(model_loader_extra_config=extra))
        source = DefaultModelLoader.Source(
            "/dummy",
            None,
            fall_back_to_pt=False,
            model_config=(
                SimpleNamespace(is_draft_model=draft, hf_config=None)
                if model_config
                else None
            ),
        )
        server_args = SimpleNamespace(
            weight_loader_disable_mmap=False,
            weight_loader_prefetch_checkpoints=True,
            weight_loader_prefetch_num_threads=4,
            weight_loader_drop_cache_after_load=False,
        )
        with (
            patch.object(
                DefaultModelLoader,
                "_prepare_weights",
                return_value=("/dummy", ["f.safetensors"], True),
            ),
            patch.object(
                loader_mod, "maybe_add_mtp_safetensors", side_effect=lambda f, *_: f
            ),
            patch.object(loader_mod, "get_model", return_value=server_args),
            patch.object(
                loader_mod,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=tp_size, moe_ep_size=ep_size),
            ),
            patch.object(loader_mod, "is_cuda", return_value=True),
            patch.object(
                loader_mod,
                "buffered_multi_thread_safetensors_weights_iterator",
                return_value=iter([]),
            ) as buffered,
            patch.object(
                loader_mod, "safetensors_weights_iterator", return_value=iter([])
            ) as single,
            patch.object(loader_mod, "rank0_log") as log,
        ):
            list(loader._get_weights_iterator(source))
        if single.called:
            staged = single.call_args.kwargs["gpu_staging"]
            self.assertTrue(single.call_args.kwargs["prefetch"])
        else:
            buffered.assert_called_once()
            staged = False
        self.assertLessEqual(log.call_count, 1)
        return staged, log.call_args.args[0] if log.called else None

    def test_staging_replaces_the_multithreaded_loader(self):
        for extra in (
            {"gpu_staging": True},
            {"gpu_staging": True, "enable_multithread_load": True},
        ):
            with self.subTest(extra=extra):
                self.assertEqual(self._dispatch(extra), (True, None))

    def test_staging_is_off_without_gain(self):
        on = {"gpu_staging": True, "enable_multithread_load": True}
        for kwargs, reason in (
            (dict(tp_size=1), "without tensor parallelism"),
            (dict(ep_size=4), "with expert parallelism"),
            (dict(draft=True), None),
            (dict(model_config=False), None),  # e.g. a LoRA adapter
        ):
            with self.subTest(**kwargs):
                staged, message = self._dispatch(on, **kwargs)
                self.assertFalse(staged)
                if reason is None:
                    self.assertIsNone(message)
                else:
                    self.assertIn(reason, message)
        self.assertEqual(self._dispatch({}), (False, None))


if __name__ == "__main__":
    unittest.main()
