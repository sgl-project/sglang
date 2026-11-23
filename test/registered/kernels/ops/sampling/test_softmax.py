from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# SM100 exercises the JIT softmax kernel; non-SM100 CUDA and ROCm exercise the
# fused in-place Triton temperature softmax kernel.
register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=10, stage="base-b", runner_config="1-gpu-small-amd")

import unittest
from itertools import product

import torch

from sglang.kernels.ops.sampling import softmax
from sglang.kernels.ops.sampling.softmax_triton import temperature_softmax
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestTemperatureSoftmax(CustomTestCase):
    @staticmethod
    def temperatures(rows: int) -> torch.Tensor:
        return torch.linspace(
            0.5, 1.5, rows, dtype=torch.float32, device="cuda"
        ).unsqueeze(1)

    def assert_matches_torch(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor | None,
        *,
        rows_per_request: int = 1,
        inplace: bool = False,
    ):
        scaled_logits = logits.float()
        if temperatures is not None:
            t = (
                temperatures.repeat_interleave(rows_per_request, dim=0)
                if rows_per_request > 1
                else temperatures
            )
            scaled_logits = scaled_logits / t
        expected = torch.softmax(scaled_logits, dim=-1)

        if inplace:
            buf = logits.clone()
            actual = softmax(
                buf,
                temperatures=temperatures,
                rows_per_request=rows_per_request,
                out=buf,
            )
            self.assertIs(actual, buf)
        else:
            actual = softmax(
                logits,
                temperatures=temperatures,
                rows_per_request=rows_per_request,
            )

        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)

    def test_dense_logits(self):
        generator = torch.Generator(device="cuda").manual_seed(0)
        for rows, vocab_size in (
            (0, 1024),
            (1, 0),
            (1, 1024),
            (1, 32000),
            (1, 131072),
            (4, 154880),
            (64, 131072),
            (128, 152064),
            (1, 262144),
            (7, 65537),
            (32, 128257),
            (33, 4096),
        ):
            with self.subTest(rows=rows, vocab_size=vocab_size):
                logits = torch.randn(
                    (rows, vocab_size),
                    dtype=torch.float32,
                    device="cuda",
                    generator=generator,
                )
                self.assert_matches_torch(logits, None)
                self.assert_matches_torch(logits, self.temperatures(rows))
                self.assert_matches_torch(logits, self.temperatures(rows), inplace=True)

    def test_rows_per_request_and_low_precision_inputs(self):
        generator = torch.Generator(device="cuda").manual_seed(1)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for rows_per_request, reqs, vocab_size in (
                (4, 3, 152064),
                (4, 24, 128256),
                (2, 4, 4096),
            ):
                with self.subTest(
                    dtype=dtype, rows_per_request=rows_per_request, reqs=reqs
                ):
                    logits = torch.randn(
                        (reqs * rows_per_request, vocab_size),
                        dtype=dtype,
                        device="cuda",
                        generator=generator,
                    )
                    temps = self.temperatures(reqs)
                    self.assert_matches_torch(
                        logits,
                        temps,
                        rows_per_request=rows_per_request,
                        inplace=(dtype == torch.float32),
                    )

    def test_inplace_large_batch_has_zero_transient_allocation(self):
        """In-place softmax on large batches must not allocate temporary (B, V) buffers."""
        rows, vocab_size = 128, 152064
        logits = torch.randn((rows, vocab_size), dtype=torch.float32, device="cuda")
        temps = self.temperatures(rows)
        # Warm up kernel compilation / allocator state.
        softmax(logits, temps, out=logits)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
        softmax(logits, temps, out=logits)
        torch.cuda.synchronize()
        peak_extra = torch.cuda.max_memory_allocated() - base_mem
        self.assertEqual(peak_extra, 0)

    def test_noncontiguous_logits_fall_back(self):
        logits = torch.randn((4096, 3), dtype=torch.float32, device="cuda").T
        self.assert_matches_torch(logits, self.temperatures(logits.shape[0]))

    def test_misaligned_view_falls_back(self):
        # Row stride and vocab size are vector multiples, but the base address is not.
        logits = torch.randn((2, 1032), dtype=torch.float32, device="cuda")[:, 1:1025]
        self.assert_matches_torch(logits, None)
        self.assert_matches_torch(logits, self.temperatures(2), inplace=True)

    def test_sanitized_nan_sentinels(self):
        # SGLANG_SANITIZE_NAN_LOGITS maps +-inf / NaN to +-1e30.
        logits = torch.randn((4, 32000), dtype=torch.float32, device="cuda")
        logits[0, 7] = 1e30
        logits[1, 7] = 1e30
        logits[1, 100:200] = -1e30
        logits[2, :] = -1e30
        logits[3, 31999] = 1e30
        self.assert_matches_torch(logits, None)
        self.assert_matches_torch(logits, self.temperatures(4), inplace=True)
        # Small temperature must not overflow +-1e30 before subtracting raw max.
        small_t = torch.full((4, 1), 1e-4, dtype=torch.float32, device="cuda")
        expected = torch.softmax(
            (logits - logits.amax(dim=-1, keepdim=True)) / small_t, dim=-1
        )
        buf = logits.clone()
        actual = softmax(buf, temperatures=small_t, out=buf)
        self.assertIs(actual, buf)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)

        # Even T=1e-9 (where logits / T overflows fp32 max) stays finite in Triton
        # because raw max is subtracted before scaling.
        tiny_t = torch.full((4, 1), 1e-9, dtype=torch.float32, device="cuda")
        expected_tiny = torch.softmax(
            (logits - logits.amax(dim=-1, keepdim=True)) / tiny_t, dim=-1
        )
        buf_tiny = logits.clone()
        actual_tiny = temperature_softmax(buf_tiny, temperatures=tiny_t, out=buf_tiny)
        self.assertIs(actual_tiny, buf_tiny)
        torch.testing.assert_close(actual_tiny, expected_tiny, rtol=2e-5, atol=1e-7)

    def test_sparse_grammar_mask_with_fully_masked_splits(self):
        for vocab_size, rows in product((65537, 131072), (1, 3, 16, 32, 64)):
            with self.subTest(vocab_size=vocab_size, rows=rows):
                allowed = torch.tensor(
                    [0, vocab_size // 2, vocab_size - 1],
                    device="cuda",
                    dtype=torch.long,
                )
                logits = torch.full(
                    (rows, vocab_size),
                    -float("inf"),
                    dtype=torch.float32,
                    device="cuda",
                )
                values = torch.arange(
                    rows * allowed.numel(), dtype=torch.float32, device="cuda"
                ).reshape(rows, allowed.numel())
                values = values.remainder(7).sub_(3)
                logits[:, allowed] = values
                self.assert_matches_torch(logits, self.temperatures(rows), inplace=True)

    def test_triton_all_neg_inf_row_matches_torch_nan(self):
        for rows in (2, 64):
            with self.subTest(rows=rows):
                logits = torch.full(
                    (rows, 32000), -float("inf"), dtype=torch.float32, device="cuda"
                )
                actual = temperature_softmax(
                    logits, temperatures=self.temperatures(rows)
                )
                self.assertTrue(torch.isnan(actual).all())


if __name__ == "__main__":
    unittest.main()
