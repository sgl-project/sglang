"""Single-NPU PLE pinned/file lookups against a CPU embedding reference.

Run: python3 test/manual/ascend/test_npu_ple_offload.py -v
No model weights, UVA configuration, or custom host-mapping operators needed.

Progress goes to stdout with elapsed time. A test running longer than
SGLANG_TEST_PLE_TIMEOUT_S (default 300) dumps every thread's stack and exits
the process instead of hanging.
"""

import faulthandler
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401
from torch import nn

from sglang.srt.hardware_backend.npu.graph_runner.host_offload_graph import (
    NPUHostOffloadGraph,
)
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)
from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding
from sglang.srt.models.qwen4_exp_ple_table import PLE_FILE_PREFETCH_MIN_ROWS

# Arbitrary; every test here takes seconds when healthy.
TIMEOUT_S = float(os.environ.get("SGLANG_TEST_PLE_TIMEOUT_S", "300"))
_START = time.monotonic()


def log(message: str) -> None:
    print(f"[ple-test +{time.monotonic() - _START:8.2f}s] {message}", flush=True)


class RecordingPrefetcher:
    """Stands in for PleFilePrefetcher: records enqueue calls, starts no thread."""

    def __init__(self):
        self.calls = []

    def enqueue(self, flat_ids, *, vocab_start=0, vocab_end=None):
        self.calls.append((flat_ids.clone(), vocab_start, vocab_end))
        return True

    def close(self):
        pass


class TestNpuPleOffload(unittest.TestCase):
    def setUp(self):
        if not torch.npu.is_available():
            self.skipTest("NPU not available")
        name = self._testMethodName
        log(f"{name}: start")
        # A hang inside C code never returns to Python, so only a watchdog helps.
        faulthandler.dump_traceback_later(TIMEOUT_S, exit=True)
        self.addCleanup(faulthandler.cancel_dump_traceback_later)
        self.addCleanup(log, f"{name}: cleanup done")
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def make_embedding(self, backend, dtype, dim, *, start=0, end=None, vocab=8):
        """A host-table embedding holding rows [start, end) of a vocab-row table."""
        end = vocab if end is None else end
        log(
            f"make_embedding: {backend} {dtype} dim={dim} "
            f"rows [{start}, {end}) of {vocab}"
        )
        # The real constructor builds the table on meta before offloading.
        weight = nn.Parameter(
            torch.empty((end - start, dim), dtype=dtype, device="meta"),
            requires_grad=False,
        )
        weight.output_dim = 0
        shard = VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index=start,
            padded_org_vocab_end_index=end,
            padded_added_vocab_start_index=vocab,
            padded_added_vocab_end_index=vocab,
            org_vocab_start_index=start,
            org_vocab_end_index=end,
            added_vocab_start_index=vocab,
            added_vocab_end_index=vocab,
        )
        source = SimpleNamespace(
            weight=weight,
            quant_config=None,
            enable_tp=True,
            use_attn_tp_group=False,
            tp_size=1,
            num_embeddings=vocab,
            org_vocab_size=vocab,
            padding_size=1,
            num_added_embeddings=0,
            use_presharded_weights=False,
            org_vocab_size_padded=vocab,
            num_embeddings_padded=vocab,
            shard_indices=shard,
            embedding_dim=dim,
            weight_scale=torch.tensor([0.5], device=self.device),
            quant_method=UnquantizedEmbeddingMethod(),
            num_embeddings_per_partition=end - start,
            num_org_embeddings_per_partition=end - start,
            num_added_embeddings_per_partition=0,
        )
        table_dir = Path(tempfile.mkdtemp(dir=self.directory))
        embedding = Qwen4ExpPinnedHostEmbedding(
            source, backend=backend, table_dir=str(table_dir)
        )
        # Keep the mapped storage alive until its background workers stop.
        self.addCleanup(self.close_embedding, embedding)
        log(f"  table allocated, files: {sorted(os.listdir(table_dir))}")
        rows = (
            ((torch.arange(vocab * dim, dtype=torch.float32) % 31 - 15) / 4)
            .reshape(vocab, dim)
            .to(dtype)
        )
        ptr = embedding.weight.data_ptr()
        embedding.weight_loader(embedding.weight, rows)
        log("  rows loaded")
        self.assertEqual(embedding.weight.data_ptr(), ptr)
        self.assertEqual(embedding.weight.device.type, "cpu")
        self.assertEqual(embedding.weight.dtype, dtype)
        self.assertEqual(embedding.weight.is_pinned(), backend == "pinned")
        self.assertIs(embedding.weight_scale, source.weight_scale)
        return embedding, rows, table_dir

    @staticmethod
    def close_embedding(embedding):
        for worker in (embedding._file_prefetcher, embedding._file_rss_trimmer):
            if worker is not None:
                worker.close()
        log("  embedding workers closed")

    @staticmethod
    def use_recorder(embedding):
        if embedding._file_prefetcher is not None:
            embedding._file_prefetcher.close()
        recorder = RecordingPrefetcher()
        embedding._file_prefetcher = recorder
        return recorder

    def lookup(self, embedding, ids, **kwargs):
        log(f"  gather: ids {tuple(ids.shape)} {ids.dtype}")
        actual = embedding.gather(ids, **kwargs)
        log("  gather returned, synchronizing")
        torch.npu.synchronize()
        log("  synchronized")
        return actual

    @staticmethod
    def reference(rows, ids, start, end):
        """bf16 rows for ids in [start, end), zeros elsewhere."""
        ids = ids.cpu().long()
        valid = (ids >= start) & (ids < end)
        expected = torch.zeros((*ids.shape, rows.shape[1]), dtype=torch.bfloat16)
        expected[valid] = rows.to(torch.bfloat16)[ids[valid]]
        return expected

    def check(self, actual, expected):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        log("  matches reference")

    def test_pinned_and_file_gather(self):
        for backend in ("pinned", "file"):
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                for dim in (7, 64, 257):
                    for id_dtype in (torch.int32, torch.int64):
                        with self.subTest(
                            backend=backend, dtype=dtype, dim=dim, ids=id_dtype
                        ):
                            embedding, rows, _ = self.make_embedding(
                                backend, dtype, dim
                            )
                            ids = torch.tensor(
                                [[7, 0, 3], [3, 1, 6]],
                                dtype=id_dtype,
                                device=self.device,
                            )
                            actual = self.lookup(embedding, ids)
                            self.check(actual, self.reference(rows, ids, 0, 8))

    def test_shard_boundaries_duplicates_and_output_buffer(self):
        # Two disjoint TP shards without an HCCL process group; their sum is the table.
        ids = torch.tensor([[-1, 0, 3], [4, 7, 8], [7, 4, 99]], device=self.device)
        for backend in ("pinned", "file"):
            with self.subTest(backend=backend):
                combined = torch.zeros((*ids.shape, 13), dtype=torch.bfloat16)
                for start, end in ((0, 4), (4, 8)):
                    embedding, rows, _ = self.make_embedding(
                        backend, torch.bfloat16, 13, start=start, end=end
                    )
                    # Keep the NaN sentinel without requiring an NPU fill kernel
                    # just to prepare the buffer whose overwrite we are testing.
                    log("  preparing NaN output buffer on CPU, copying to NPU")
                    out = torch.full(
                        (*ids.shape, 13),
                        float("nan"),
                        dtype=torch.bfloat16,
                        device="cpu",
                    ).to(self.device)
                    log("  NaN output buffer ready")
                    actual = self.lookup(embedding, ids, out=out)
                    self.assertIs(actual, out)
                    self.check(actual, self.reference(rows, ids, start, end))
                    combined += actual.cpu()
                torch.testing.assert_close(
                    combined, self.reference(rows, ids, 0, 8), rtol=0, atol=0
                )
                log("  shard sum matches the full table")

    def test_file_directory_persistence_and_updated_rows(self):
        embedding, rows, directory = self.make_embedding("file", torch.bfloat16, 13)
        files = list(directory.iterdir())
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].stat().st_size, rows.numel() * rows.element_size())
        reopened = torch.from_file(
            str(files[0]), shared=True, size=rows.numel(), dtype=rows.dtype
        ).reshape_as(rows)
        torch.testing.assert_close(reopened, rows, rtol=0, atol=0)
        log("  file holds the loaded rows")
        ids = torch.tensor([2, 5, 2], device=self.device)
        first = self.lookup(embedding, ids).cpu()
        # A write through another mapping of the same file must be visible.
        reopened[2].fill_(42)
        second = self.lookup(embedding, ids).cpu()
        self.assertFalse(torch.equal(first[0], second[0]))
        torch.testing.assert_close(second, reopened[[2, 5, 2]], rtol=0, atol=0)
        log("  lookup sees the updated file contents")

    def test_file_gather_hands_host_ids_to_prefetcher(self):
        """The NPU gather must hint the file prefetcher with host ids and its shard."""
        embedding, rows, _ = self.make_embedding(
            "file", torch.bfloat16, 13, start=4, end=8
        )
        recorder = self.use_recorder(embedding)
        for id_dtype in (torch.int32, torch.int64):
            with self.subTest(ids=id_dtype):
                recorder.calls.clear()
                ids = torch.tensor(
                    [[1, 5], [7, 5]], dtype=id_dtype, device=self.device
                )
                actual = self.lookup(embedding, ids)
                self.assertEqual(len(recorder.calls), 1)
                hinted_ids, vocab_start, vocab_end = recorder.calls[0]
                log(
                    f"  prefetcher got {hinted_ids.device} "
                    f"ids [{vocab_start}, {vocab_end})"
                )
                # Both input dtypes reach the prefetcher as host int64 IDs.
                self.assertEqual(hinted_ids.device.type, "cpu")
                self.assertEqual(hinted_ids.dtype, torch.int64)
                self.assertEqual(hinted_ids.tolist(), [1, 5, 7, 5])
                self.assertEqual((vocab_start, vocab_end), (4, 8))
                self.check(actual, self.reference(rows, ids, 4, 8))

    def test_prefill_sized_lookup(self):
        """A prefill-sized lookup, with the real file prefetcher and its thread."""
        tokens, heads, vocab, dim = 2048, 16, 4096, 160
        start, end = vocab // 4, vocab
        generator = torch.Generator().manual_seed(0)
        ids_cpu = torch.randint(0, vocab, (tokens, heads), generator=generator)
        self.assertGreaterEqual(ids_cpu.numel(), PLE_FILE_PREFETCH_MIN_ROWS)
        for backend in ("pinned", "file"):
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                with self.subTest(backend=backend, dtype=dtype):
                    embedding, rows, _ = self.make_embedding(
                        backend, dtype, dim, start=start, end=end, vocab=vocab
                    )
                    prefetcher = embedding._file_prefetcher
                    if backend == "file":
                        self.assertIsNotNone(
                            prefetcher, "SGLANG_QWEN4_PLE_FILE_PREFETCH is off"
                        )
                    queued = self.log_prefetcher_calls(prefetcher)
                    ids = ids_cpu.to(self.device)
                    log(f"  live threads: {threading.active_count()}")
                    actual = self.lookup(embedding, ids)
                    self.check(actual, self.reference(rows, ids_cpu, start, end))
                    if backend == "pinned":
                        self.assertIsNone(prefetcher)
                        continue
                    self.assertEqual(queued, [True])
                    log("  waiting for the prefetch worker")
                    prefetcher._pool.shutdown(wait=True)
                    log(f"  prefetch worker done, threads: {threading.active_count()}")

    @staticmethod
    def log_prefetcher_calls(prefetcher):
        """Log around each real enqueue; returns the list its results land in."""
        results = []
        if prefetcher is None:
            return results
        enqueue = prefetcher.enqueue

        def logged_enqueue(*args, **kwargs):
            log("  prefetcher.enqueue: enter")
            queued = enqueue(*args, **kwargs)
            log(f"  prefetcher.enqueue: queued={queued}")
            results.append(queued)
            return queued

        prefetcher.enqueue = logged_enqueue
        return results

    def test_side_stream_lookup_into_reused_buffer(self):
        # As in Qwen4ExpPLELayer: a side stream refills one buffer, main waits.
        for backend in ("pinned", "file"):
            with self.subTest(backend=backend):
                embedding, rows, _ = self.make_embedding(
                    backend, torch.bfloat16, 257
                )
                stream = torch.npu.Stream(device=self.device)
                out = torch.empty((128, 257), dtype=torch.bfloat16, device=self.device)
                for offset in range(4):
                    # Input construction is not under test; keep the H2D producer
                    # on the main stream and the lookup on the dependent side stream.
                    ids_cpu = (torch.arange(128, device="cpu") + offset) % 8
                    ids = ids_cpu.to(self.device)
                    stream.wait_stream(torch.npu.current_stream())
                    ids.record_stream(stream)
                    log(f"  side-stream gather {offset}")
                    with torch.npu.stream(stream):
                        actual = embedding.gather(ids, out=out)
                    torch.npu.current_stream().wait_stream(stream)
                    self.assertIs(actual, out)
                    self.check(actual, self.reference(rows, ids_cpu, 0, 8))

    def test_empty_input(self):
        for backend in ("pinned", "file"):
            with self.subTest(backend=backend):
                embedding, _, _ = self.make_embedding(backend, torch.bfloat16, 7)
                ids = torch.empty((0, 3), dtype=torch.int64, device=self.device)
                actual = self.lookup(embedding, ids)
                self.assertEqual(actual.shape, (0, 3, 7))

    def test_graph_replays_host_lookup_between_segments(self):
        for backend in ("pinned", "file"):
            with self.subTest(backend=backend):
                embedding, rows, _ = self.make_embedding(backend, torch.bfloat16, 7)
                recorder = self.use_recorder(embedding)
                # Graph inputs and outputs are prepared on the CPU, as elsewhere
                # in this file. Inside the graph only D2D copies run: segment 0
                # produces the IDs, segment 1 consumes the staged rows.
                ids_in = torch.zeros((2, 2), dtype=torch.int32).to(self.device)
                ids = torch.zeros((2, 2), dtype=torch.int32).to(self.device)
                result = torch.full((2, 2, 7), float("nan"), dtype=torch.bfloat16)
                result = result.to(self.device)

                graph = NPUHostOffloadGraph(torch.npu)
                log("  capturing")
                with graph.capture(stream=torch.npu.Stream()):
                    ids.copy_(ids_in)
                    result.copy_(embedding.gather(ids))
                self.assertEqual(graph.num_segments, 2)
                self.assertEqual(graph.num_host_lookups, 1)
                # Capture must not read IDs that no segment has produced yet.
                self.assertEqual(recorder.calls, [])

                for step, ids_cpu in enumerate(
                    (torch.tensor([[7, 0], [5, 5]]), torch.tensor([[1, 2], [6, 3]]))
                ):
                    ids_in.copy_(ids_cpu.to(torch.int32))
                    # A single update broadcasts to every segment, including
                    # those without dispatch records, as replay_with_input_update
                    # does for attention seq_lens.
                    graph.update([{"actual_seq_lengths_kv": [1]}])
                    log(f"  replay {step}")
                    graph.replay()
                    torch.npu.synchronize()
                    self.check(result, self.reference(rows, ids_cpu, 0, 8))
                    self.assertTrue(
                        torch.equal(recorder.calls[-1][0], ids_cpu.reshape(-1))
                    )
                self.assertEqual(len(recorder.calls), 2)

    def test_raw_graph_capture_rejected_before_host_row_selection(self):
        embedding, _, _ = self.make_embedding("file", torch.bfloat16, 7)
        recorder = self.use_recorder(embedding)
        for id_dtype in (torch.int32, torch.int64):
            with self.subTest(ids=id_dtype):
                ids = torch.tensor([1], dtype=id_dtype, device=self.device)
                # A plain NPUGraph capture would freeze capture-time rows.
                with patch("torch.npu.is_current_stream_capturing", return_value=True):
                    with self.assertRaisesRegex(
                        RuntimeError, "requires NPUHostOffloadGraph"
                    ):
                        embedding.gather(ids)
        # Rejected before any host work: nothing reached the prefetcher.
        self.assertEqual(recorder.calls, [])
        log("  raw capture rejected before host row selection")

    def test_graph_update_follows_segment_dispatch_records(self):
        class FakeSegment:
            def __init__(self, records):
                self.graph_dispatch_mode = SimpleNamespace(
                    graph_dispatch_records=[None] * records
                )
                self.updates = []

            def update(self, cpu_update_input):
                self.updates.append(cpu_update_input)

        segments = [FakeSegment(2), FakeSegment(0), FakeSegment(1)]
        graph = NPUHostOffloadGraph(torch.npu)
        graph._segments = segments

        graph.update([{"seq": 0}])
        self.assertEqual([seg.updates[-1] for seg in segments], [[{"seq": 0}]] * 3)

        # Per-op lists (EAGLE draft steps) are split in capture order.
        graph.update([{"seq": 0}, {"seq": 1}, {"seq": 2}])
        self.assertEqual(
            [seg.updates[-1] for seg in segments],
            [[{"seq": 0}, {"seq": 1}], [], [{"seq": 2}]],
        )

        with self.assertRaisesRegex(ValueError, "update count"):
            graph.update([{"seq": 0}, {"seq": 1}])


if __name__ == "__main__":
    unittest.main()
