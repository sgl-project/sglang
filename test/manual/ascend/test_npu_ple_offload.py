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
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401
from torch import nn

from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)
from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding, Qwen4ExpPLELayer

# Arbitrary; every test here takes seconds when healthy.
TIMEOUT_S = float(os.environ.get("SGLANG_TEST_PLE_TIMEOUT_S", "300"))
_START = time.monotonic()


def log(message: str) -> None:
    print(f"[ple-test +{time.monotonic() - _START:8.2f}s] {message}", flush=True)


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
        self.assertTrue(embedding.npu_uva_enabled)
        self.assertEqual(dtype, torch.bfloat16)
        self.assertIsNone(embedding._file_prefetcher)
        return embedding, rows, table_dir

    @staticmethod
    def close_embedding(embedding):
        for worker in (embedding._file_prefetcher, embedding._file_rss_trimmer):
            if worker is not None:
                worker.close()
        owner = getattr(embedding, "_npu_uva_table", None)
        if owner is not None:
            owner.close()
        log("  embedding workers and UVA registration closed")

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
            for dim in (7, 64, 257):
                for id_dtype in (torch.int32, torch.int64):
                    with self.subTest(backend=backend, dim=dim, ids=id_dtype):
                        embedding, rows, _ = self.make_embedding(
                            backend, torch.bfloat16, dim
                        )
                        ids = torch.tensor(
                            [[7, 0, 3], [3, 1, 6]], dtype=id_dtype, device=self.device
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

    def test_prefill_sized_lookup(self):
        """A large lookup stays on-device without CPU file prefetch hints."""
        tokens, heads, vocab, dim = 2048, 16, 4096, 160
        start, end = vocab // 4, vocab
        generator = torch.Generator().manual_seed(0)
        ids_cpu = torch.randint(0, vocab, (tokens, heads), generator=generator)
        for backend in ("pinned", "file"):
            with self.subTest(backend=backend):
                embedding, rows, _ = self.make_embedding(
                    backend, torch.bfloat16, dim, start=start, end=end, vocab=vocab
                )
                self.assertIsNone(embedding._file_prefetcher)
                actual = self.lookup(embedding, ids_cpu.to(self.device))
                self.check(actual, self.reference(rows, ids_cpu, start, end))

    def test_side_stream_lookup_into_reused_buffer(self):
        # As in Qwen4ExpPLELayer: a side stream refills one buffer, main waits.
        for backend in ("pinned", "file"):
            with self.subTest(backend=backend):
                embedding, rows, _ = self.make_embedding(backend, torch.bfloat16, 257)
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

    def test_uva_plain_graph_reads_updated_ids_and_host_rows(self):
        for backend in ("pinned", "file"):
            for id_dtype in (torch.int32, torch.int64):
                with self.subTest(backend=backend, ids=id_dtype):
                    start, end, dim = 4, 8, 13
                    embedding, rows, _ = self.make_embedding(
                        backend, torch.bfloat16, dim, start=start, end=end
                    )
                    ids = torch.tensor(
                        [4, 7, 5, -1, 8, 5], dtype=id_dtype, device=self.device
                    )
                    result = torch.empty(
                        (6, dim), dtype=torch.bfloat16, device=self.device
                    )
                    embedding.gather(ids, out=result)
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        embedding.gather(ids, out=result)
                    for step, pattern in enumerate(
                        ([4, 7, 5, -1, 8, 5], [6, 5, 7, 6, 0, 99])
                    ):
                        ids_cpu = torch.tensor(pattern, dtype=id_dtype)
                        updated = (rows.float() + (step + 1) * 32).to(rows.dtype)
                        embedding.weight.copy_(updated[start:end])
                        ids.copy_(ids_cpu)
                        graph.replay()
                        torch.npu.synchronize()
                        self.check(result, self.reference(updated, ids_cpu, start, end))

    def test_uva_unavailable_fails_construction(self):
        for backend in ("pinned", "file"):
            with self.subTest(backend=backend), patch(
                "sglang.srt.hardware_backend.npu.ple.create_ple_uva_table",
                side_effect=RuntimeError(f"NPU PLE backend={backend} requires UVA"),
            ), patch(
                "sglang.srt.models.qwen4_exp.make_ple_file_prefetcher"
            ) as prefetch, patch(
                "sglang.srt.models.qwen4_exp.make_ple_file_rss_trimmer"
            ) as trim:
                with self.assertRaisesRegex(RuntimeError, "requires UVA"):
                    self.make_embedding(backend, torch.bfloat16, 13)
                prefetch.assert_not_called()
                trim.assert_not_called()

    def test_fp8_rejected_before_table_allocation(self):
        for backend in ("pinned", "file"):
            with self.subTest(backend=backend), patch(
                "sglang.srt.models.qwen4_exp.allocate_ple_host_table"
            ) as allocate:
                with self.assertRaisesRegex(NotImplementedError, "bfloat16"):
                    self.make_embedding(backend, torch.float8_e4m3fn, 13)
                allocate.assert_not_called()

    def test_prefetch_side_stream_inside_plain_graph(self):
        """Exercise the real start/consume methods, with no capture-time skip."""
        for backend in ("pinned", "file"):
            with self.subTest(backend=backend):
                embedding, rows, _ = self.make_embedding(
                    backend, torch.bfloat16, 13, start=4, end=8
                )
                layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
                nn.Module.__init__(layer)
                layer.ple_embed_dim = 13
                layer._prefetch_stream = torch.npu.Stream(device=self.device)
                layer._graph_prefetch_buffers = {}
                layer._eager_prefetch_buffer = None
                layer._prefetch_state = None
                layer.ple_embedding = SimpleNamespace(
                    ngram_embedding=embedding,
                    ngram_heads=1,
                    gather_dp_tokens=False,
                    compute_ngram_ids=lambda batch: batch.ids,
                    _prepare_embedding_lookup=lambda ids, forward_batch, physical: (
                        ids,
                        ids.shape[0],
                    ),
                    _finish_embedding_lookup=lambda values, semantic, forward_batch, physical: values,
                )
                ids = torch.tensor(
                    [[4], [7], [5], [-1], [8], [5]],
                    dtype=torch.int64,
                    device=self.device,
                )
                batch = SimpleNamespace(ids=ids, physical_tokens=6)
                forward_batch = SimpleNamespace(input_ids=ids)

                def forward():
                    layer.start_prefetch(batch, forward_batch)
                    self.assertIsNotNone(layer._prefetch_state)
                    return layer._consume_prefetched_embeddings(forward_batch)

                with patch(
                    "sglang.srt.models.qwen4_exp.get_is_capture_mode", return_value=True
                ):
                    forward()
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        result = forward()
                    for step, pattern in enumerate(
                        ([7, 4, 5, 8, -1, 5], [6, 7, 4, 6, 0, 99])
                    ):
                        ids_cpu = torch.tensor(pattern).reshape(6, 1)
                        updated = (rows.float() + (step + 1) * 16).to(rows.dtype)
                        embedding.weight.copy_(updated[4:8])
                        ids.copy_(ids_cpu)
                        graph.replay()
                        torch.npu.synchronize()
                        expected = (
                            self.reference(updated, ids_cpu, 4, 8).reshape(6, 13) * 0.5
                        )
                        self.check(result, expected.to(result.dtype))


if __name__ == "__main__":
    unittest.main()
