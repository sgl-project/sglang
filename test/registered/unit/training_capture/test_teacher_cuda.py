"""Raw teacher values and full-vocabulary normalization on CUDA."""

import math
import unittest

import torch
from sglang.srt.training_capture.protocol import ContractError
from sglang.srt.training_capture.teacher import capture_teacher, warmup_teacher_capture
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTeacherCuda(CustomTestCase):
    def check_flashinfer(self, raw, vocab, indices=None):
        scores = raw[:, :vocab]
        if indices is not None:
            scores = scores[indices]
        rows = capture_teacher(raw, vocab, indices, topk_backend="flashinfer")
        ids = rows.token_ids.long()
        expected = scores.topk(128, dim=-1).values.float()
        torch.testing.assert_close(
            rows.logits, expected, rtol=0, atol=0, equal_nan=True
        )
        torch.testing.assert_close(
            rows.logits, scores.gather(1, ids).float(), rtol=0, atol=0, equal_nan=True
        )
        ordered = ids.sort(-1).values
        self.assertTrue((ordered[:, 1:] != ordered[:, :-1]).all())
        torch.testing.assert_close(
            rows.logsumexp,
            scores.float().double().logsumexp(-1).float(),
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )
        return rows

    def test_flashinfer_padding_selection_dtypes_and_owned_values(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16, torch.float64):
            for vocab in (128, 32768, 151936):
                for indices in (None, [], [2], [3, 0, 3, 1]):
                    with self.subTest(dtype=dtype, vocab=vocab, indices=indices):
                        raw = torch.randn(
                            8, (vocab + 17) * 2, device="cuda", dtype=dtype
                        )[::2, ::2]
                        raw[:, vocab:] = 10000
                        rows = self.check_flashinfer(raw, vocab, indices)
                        saved = [
                            value.clone()
                            for value in (rows.token_ids, rows.logits, rows.logsumexp)
                        ]
                        raw.fill_(float("nan"))
                        for value, expected in zip(
                            (rows.token_ids, rows.logits, rows.logsumexp), saved
                        ):
                            torch.testing.assert_close(value, expected, rtol=0, atol=0)

    def test_flashinfer_ties_and_nonfinite_values(self):
        raw = torch.zeros(7, 32769, device="cuda")
        raw[1].fill_(-10000)
        raw[2].fill_(10000)
        raw[3].fill_(-float("inf"))
        raw[4, 100] = float("inf")
        raw[5, 200] = float("nan")
        raw[6, :200] = 10
        rows = self.check_flashinfer(raw, 32769)
        expected = torch.arange(128, device="cuda", dtype=torch.int32).expand(4, -1)
        torch.testing.assert_close(rows.token_ids[:4], expected, rtol=0, atol=0)
        torch.testing.assert_close(rows.token_ids[6], expected[0], rtol=0, atol=0)
        self.assertFalse(torch.isfinite(rows.logsumexp[3:6]).any())

    def test_flashinfer_graph_and_concurrent_stream_workspaces(self):
        vocab = 151936
        warmup_teacher_capture(vocab, "cuda", topk_backend="flashinfer")
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        inputs = [torch.randn(8, vocab + 17, device="cuda") for _ in streams]
        graphs, outputs = [], []
        for raw, stream in zip(inputs, streams):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    capture_teacher(raw, vocab, topk_backend="flashinfer")
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                rows = capture_teacher(raw, vocab, topk_backend="flashinfer")
            graphs.append(graph)
            outputs.append(rows)
        for iteration in range(4):
            expected = []
            eager = []
            for raw, graph, stream in zip(inputs, graphs, streams):
                with torch.cuda.stream(stream):
                    raw.normal_().add_(iteration * 10)
                    expected.append(raw[:, :vocab].clone())
                    torch.cuda._sleep(1000000)
                    graph.replay()
                    eager.append(capture_teacher(raw, vocab, topk_backend="flashinfer"))
                    raw.fill_(float("nan"))
            for stream in streams:
                stream.synchronize()
            for rows, scores in zip(outputs + eager, expected + expected):
                torch.testing.assert_close(
                    rows.logits, scores.topk(128).values, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    rows.logits, scores.gather(1, rows.token_ids.long()), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    rows.logsumexp,
                    scores.double().logsumexp(-1).float(),
                    rtol=1e-6,
                    atol=1e-6,
                )

    def check_rows(self, raw, vocab, indices=None):
        scores = raw[:, :vocab]
        if indices is not None:
            scores = scores.index_select(
                0,
                (
                    torch.tensor(indices, dtype=torch.long, device=raw.device)
                    if isinstance(indices, list)
                    else indices
                ),
            )
        expected_values, expected_ids = scores.topk(128, dim=-1)
        # The public contract normalizes FP32 scores, including FP64 fallback.
        expected_lse = scores.float().double().logsumexp(-1).float()
        rows = capture_teacher(raw, vocab, indices)
        self.assertEqual(rows.token_ids.dtype, torch.int32)
        self.assertEqual(rows.logits.dtype, torch.float32)
        self.assertEqual(rows.logsumexp.dtype, torch.float32)
        torch.testing.assert_close(rows.token_ids, expected_ids.int(), rtol=0, atol=0)
        torch.testing.assert_close(
            rows.logits, expected_values.float(), rtol=0, atol=0, equal_nan=True
        )
        torch.testing.assert_close(
            rows.logsumexp, expected_lse, rtol=1e-6, atol=1e-6, equal_nan=True
        )
        return rows

    def test_dtypes_padding_and_strided_rows_and_columns(self):
        torch.manual_seed(31)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for vocab in (128, 16385, 151936):
                for strided in (False, True):
                    with self.subTest(dtype=dtype, vocab=vocab, strided=strided):
                        raw = torch.randn(
                            8, (vocab + 17) * 2, device="cuda", dtype=dtype
                        )
                        raw = raw[::2, ::2] if strided else raw[:4, : vocab + 17]
                        raw[:, vocab:] = 10000
                        self.check_rows(raw, vocab)

    def test_selected_duplicate_and_empty_rows_and_float64_fallback(self):
        for dtype in (torch.float32, torch.float64):
            raw = torch.randn(4, 513, device="cuda", dtype=dtype)
            for indices in ([3, 0, 3, 1], []):
                with self.subTest(dtype=dtype, indices=indices):
                    self.check_rows(
                        raw,
                        511,
                        torch.tensor(indices, device="cuda", dtype=torch.int64),
                    )
            self.check_rows(raw[:0], 511)

    def test_offsets_masked_values_and_nonfinite_rows(self):
        vocab = 32769
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            raw = torch.zeros(9, vocab, device="cuda", dtype=dtype)
            raw[1].fill_(10000)
            raw[2].fill_(-10000)
            raw[3, 100] = 10000
            raw[4].fill_(-math.log(vocab))
            raw[5].fill_(-float("inf"))
            raw[5, 17] = 2.5
            raw[6].fill_(-float("inf"))
            raw[7, 100] = float("inf")
            raw[8, 200] = float("nan")
            with self.subTest(dtype=dtype):
                self.check_rows(raw, vocab)

    def test_host_row_selection_and_source_reuse(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for indices in ([], [2], [1, 2, 3], [3, 0, 3, 1]):
                with self.subTest(dtype=dtype, indices=indices):
                    raw = torch.randn(8, 1030, device="cuda", dtype=dtype)[::2, ::2]
                    rows = self.check_rows(raw, 511, indices)
                    saved = (
                        rows.token_ids.clone(),
                        rows.logits.clone(),
                        rows.logsumexp.clone(),
                    )
                    raw.fill_(float("nan"))
                    for actual, expected in zip(
                        (rows.token_ids, rows.logits, rows.logsumexp), saved
                    ):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for indices in ([-1], [4], [1, 4], [True], [1.0]):
            with self.subTest(invalid=indices), self.assertRaises(ContractError):
                capture_teacher(torch.zeros(4, 128, device="cuda"), 128, indices)

    def test_warmed_graph_replay_and_owned_outputs(self):
        vocab = 151936
        warmup_teacher_capture(vocab, "cuda")
        raw = torch.randn(4, vocab + 64, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                capture_teacher(raw, vocab)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            rows = capture_teacher(raw, vocab)
        for offset in (0.0, 9.0):
            raw.copy_(torch.randn_like(raw) + offset)
            expected = raw[:, :vocab].clone()
            graph.replay()
            raw.fill_(-float("inf"))
            torch.testing.assert_close(
                rows.logits, expected.gather(1, rows.token_ids.long()), rtol=0, atol=0
            )
            torch.testing.assert_close(
                rows.logsumexp,
                expected.double().logsumexp(-1).float(),
                rtol=1e-6,
                atol=1e-6,
            )

    def test_host_contiguous_rows_in_graph(self):
        raw = torch.randn(8, 16400, device="cuda")
        warmup_teacher_capture(16385, "cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            rows = capture_teacher(raw, 16385, [2, 3, 4])
        for offset in (0.0, 9.0):
            raw.copy_(torch.randn_like(raw) + offset)
            expected = raw[2:5, :16385].clone()
            graph.replay()
            raw.fill_(float("nan"))
            torch.testing.assert_close(
                rows.logits, expected.gather(1, rows.token_ids.long()), rtol=0, atol=0
            )
            torch.testing.assert_close(
                rows.logsumexp,
                expected.double().logsumexp(-1).float(),
                rtol=1e-6,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
