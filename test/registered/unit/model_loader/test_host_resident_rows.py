"""Positioned row reads of host-resident safetensors weights."""

import errno
import os
import random
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import safe_open, save_file

import sglang.srt.layers.engram as engram
import sglang.srt.model_loader.weight_utils as weight_utils
from sglang.srt.environ import envs
from sglang.srt.model_loader.weight_utils import (
    SafetensorsRowSource,
    host_resident_weights_iterator,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_ROWS, DIM = 37, 64  # 37 rows split unevenly over 2, 3 and 4 ranks.
WEIGHT = "layers.1.engram.embed.weight"
SCALE = "layers.1.engram.embed.scale"
FP32_TABLE = "layers.1.engram.embed.fp32"
OTHER = "layers.1.engram.wkv.weight"
HOST_RESIDENT = (WEIGHT, SCALE, FP32_TABLE)
SENTINEL = 0xA5


class _FakeHostTable:
    """_HostTable without the memfd, the huge pages and cudaHostRegister."""

    def __init__(self, layout, nbytes, name, group):
        self.layout = layout
        self.dirty = False
        self.bytes = torch.full((nbytes,), SENTINEL, dtype=torch.uint8)
        self.device_ptr = self.bytes.data_ptr()


def _random_bytes(shape, dtype, generator):
    raw = torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
    return raw.view(dtype)


class TestHostResidentRows(CustomTestCase):
    def setUp(self):
        super().setUp()
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = os.path.join(folder.name, "model-00002.safetensors")
        g = torch.Generator().manual_seed(0)
        self.tensors = {
            WEIGHT: _random_bytes((NUM_ROWS, DIM), torch.float8_e4m3fn, g),
            SCALE: _random_bytes(
                (NUM_ROWS, DIM // engram.FP8_BLOCK_SIZE), torch.float8_e8m0fnu, g
            ),
            # Multi-byte elements, so small chunks split elements across reads.
            FP32_TABLE: torch.randn(NUM_ROWS, 5, generator=g),
            OTHER: torch.randn(8, 16, generator=g),
        }
        save_file(self.tensors, self.path)
        self.items = dict(host_resident_weights_iterator([self.path], (".embed.",)))
        with safe_open(self.path, framework="pt") as f:
            self.mmap = {name: f.get_tensor(name) for name in self.tensors}

    def assertBytesEqual(self, actual, expected, msg=None):
        self.assertEqual(actual.dtype, expected.dtype, msg)
        self.assertTrue(
            torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), msg
        )

    def test_iterator_yields_row_sources_for_matching_names(self):
        self.assertCountEqual(self.items, self.tensors)
        for name in HOST_RESIDENT:
            source = self.items[name]
            self.assertIsInstance(source, SafetensorsRowSource)
            self.assertEqual(source.shape, tuple(self.tensors[name].shape))
            self.assertEqual(source.dtype, self.tensors[name].dtype)
        self.assertBytesEqual(self.items[OTHER], self.tensors[OTHER])

    def test_random_row_ranges_match_mmap(self):
        rng = random.Random(0)
        for name in HOST_RESIDENT:
            full = self.mmap[name]
            for _ in range(30):
                start = rng.randrange(NUM_ROWS + 1)
                rows = rng.randrange(NUM_ROWS - start + 1)
                chunk_bytes = rng.choice([1, 3, 8, 77, 1 << 20])
                num_threads = rng.choice([1, 3, 8])
                dst = torch.empty((rows, *full.shape[1:]), dtype=full.dtype)
                self.items[name].read_rows_into(
                    dst, start, num_threads=num_threads, chunk_bytes=chunk_bytes
                )
                self.assertBytesEqual(
                    dst,
                    full[start : start + rows],
                    f"{name} rows {start}:{start + rows}, chunk {chunk_bytes}",
                )

    def test_short_reads_and_failed_cache_advice(self):
        real_preadv = os.preadv

        def short_preadv(fd, buffers, offset):
            return real_preadv(fd, [buffers[0][:5]], offset)

        def failing_fadvise(*_args):
            raise OSError(errno.ENOSYS, "posix_fadvise")

        dst = torch.empty(10, 5)
        with (
            patch.object(weight_utils.os, "preadv", short_preadv),
            patch.object(
                weight_utils.os, "posix_fadvise", failing_fadvise, create=True
            ),
            patch.object(weight_utils.os, "POSIX_FADV_DONTNEED", 4, create=True),
        ):
            self.items[FP32_TABLE].read_rows_into(
                dst, 3, drop_page_cache=True, num_threads=3, chunk_bytes=64
            )
        self.assertBytesEqual(dst, self.mmap[FP32_TABLE][3:13])

    def test_page_cache_eviction_only_for_per_rank_tables(self):
        for layout, expected in ((None, False), ("per_rank", True), ("shared", False)):
            calls = []
            with (
                patch.object(
                    weight_utils.os,
                    "posix_fadvise",
                    lambda *args: calls.append(args),
                    create=True,
                ),
                patch.object(weight_utils.os, "POSIX_FADV_DONTNEED", 4, create=True),
            ):
                self._load_engram(2, 1, layout, self.items)
            self.assertEqual(bool(calls), expected, layout)

    def test_truncated_file_raises(self):
        path = self.path + ".copy"
        shutil.copyfile(self.path, path)
        source = dict(host_resident_weights_iterator([path], (".embed.",)))[WEIGHT]
        os.truncate(path, source.offset + 2 * source.row_bytes)

        dst = torch.empty(2, DIM, dtype=torch.float8_e4m3fn)
        source.read_rows_into(dst, 0)
        self.assertBytesEqual(dst, self.tensors[WEIGHT][:2])
        with self.assertRaises(EOFError):
            source.read_rows_into(torch.empty_like(dst), 1)

    def _load_engram(self, tp_size, tp_rank, layout, weights):
        """Load WEIGHT and SCALE into one rank's EngramEmbedding."""
        parallel = SimpleNamespace(tp_size=tp_size, tp_rank=tp_rank, tp_group=None)
        with (
            patch.object(engram, "get_parallel", return_value=parallel),
            patch.object(engram, "_HostTable", _FakeHostTable),
            envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(layout is not None),
            envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.override(layout or "shared"),
        ):
            embed = engram.EngramEmbedding(NUM_ROWS, DIM, layer_id=1)
        for param, name in ((embed.weight, WEIGHT), (embed.scale, SCALE)):
            param.weight_loader(param, weights[name])
        return embed

    def test_engram_shards_match_mmap_path(self):
        # Device rows, and both host-table layouts with weight and scale
        # carved out of one host buffer.
        for layout in (None, "per_rank", "shared"):
            for tp_size in (1, 2, 3, 4):
                for tp_rank in range(tp_size):
                    msg = f"{layout=} {tp_size=} {tp_rank=}"
                    via_mmap = self._load_engram(tp_size, tp_rank, layout, self.mmap)
                    via_rows = self._load_engram(tp_size, tp_rank, layout, self.items)
                    rows = slice(via_rows.row_start, via_rows.row_start + via_rows.rows)
                    for name in ("weight", "scale"):
                        actual = getattr(via_rows, name).data
                        self.assertBytesEqual(actual, getattr(via_mmap, name).data, msg)
                        own = actual[rows] if layout == "shared" else actual
                        expected = self.tensors[WEIGHT if name == "weight" else SCALE]
                        self.assertBytesEqual(own, expected[rows], msg)
                        if layout == "shared":
                            # Other ranks' rows are left untouched.
                            others = torch.ones(NUM_ROWS, dtype=torch.bool)
                            others[rows] = False
                            untouched = actual[others].view(torch.uint8) == SENTINEL
                            self.assertTrue(untouched.all(), msg)
                    if layout is not None:
                        self.assertTrue(via_rows.host_table.dirty, msg)

    def test_rejects_mismatched_destinations(self):
        source = self.items[WEIGHT]
        for dst, row_start in (
            (torch.empty(3, DIM, dtype=torch.float8_e5m2), 0),  # same size, wrong dtype
            (torch.empty(3, DIM - 1, dtype=torch.float8_e4m3fn), 0),
            (torch.empty(4, DIM, dtype=torch.float8_e4m3fn), NUM_ROWS - 3),
            (torch.empty(DIM, 3, dtype=torch.float8_e4m3fn).t(), 0),
        ):
            with self.subTest(shape=tuple(dst.shape), row_start=row_start):
                with self.assertRaises(ValueError):
                    source.read_rows_into(dst, row_start)


if __name__ == "__main__":
    unittest.main()
