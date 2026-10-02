"""Raw teacher values and full-vocabulary normalization on CUDA."""

import math
import unittest

import torch
from sglang.srt.training_capture.teacher import capture_teacher, warmup_teacher_capture
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTeacherCuda(CustomTestCase):
    def check_rows(self, raw, vocab, indices=None):
        scores = raw[:, :vocab]
        if indices is not None:
            scores = scores.index_select(0, indices)
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


if __name__ == "__main__":
    unittest.main()
