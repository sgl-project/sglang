import unittest
from unittest import mock

import torch

from sglang.kernels.ops.memory.req_to_token_pool import (
    GatherReqToTokenPool,
    WriteReqToTokenPool,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton parity")
class TestGatherReqToTokenPool(unittest.TestCase):

    def test_empty_batch_matches_all_implementations(self):
        """Empty batches return without launching and preserve the pool."""
        req_to_token = self._make_pool(rows=2, columns=8)
        req_to_token_before = req_to_token.clone()
        empty_cpu = torch.empty((0,), dtype=torch.int64)
        empty_device = empty_cpu.cuda()

        for output_dtype in (torch.int32, torch.int64):
            arguments = dict(
                req_pool_indices=empty_device,
                req_pool_indices_cpu=empty_cpu,
                prefix_lens=empty_device,
                prefix_lens_cpu=empty_cpu,
                seq_lens=empty_device,
                seq_lens_cpu=empty_cpu,
                extend_lens=empty_device,
                extend_lens_cpu=empty_cpu,
                extend_num_tokens=0,
                out_dtype=output_dtype,
            )
            vanilla = GatherReqToTokenPool.vanilla(req_to_token, **arguments)
            triton_output = GatherReqToTokenPool.triton(req_to_token, **arguments)

            self.assertEqual(vanilla.shape, (0,))
            self.assertEqual(triton_output.shape, (0,))
            self.assertEqual(vanilla.dtype, output_dtype)
            self.assertEqual(triton_output.dtype, output_dtype)
            self.assertTrue(torch.equal(req_to_token, req_to_token_before))

    def test_triton_vanilla_and_scalar_oracle_match_ragged_cases(self):
        """Triton, vanilla, and scalar gathers match across ragged edge cases."""
        cases = (
            ([1], [3], [4]),
            ([0], [5], [5]),
            ([3, 0, 4, 1], [2, 7, 0, 11], [7, 7, 9, 14]),
            ([2], [9], [266]),
            ([4], [13], [526]),
        )
        req_to_token = self._make_pool(rows=6, columns=600)

        for output_dtype in (torch.int32, torch.int64):
            for req_pool_indices, prefix_lens, seq_lens in cases:
                with self.subTest(
                    output_dtype=output_dtype,
                    req_pool_indices=req_pool_indices,
                    prefix_lens=prefix_lens,
                    seq_lens=seq_lens,
                ):
                    req_to_token_before = req_to_token.clone()
                    arguments = self._make_arguments(
                        req_pool_indices=req_pool_indices,
                        prefix_lens=prefix_lens,
                        seq_lens=seq_lens,
                        output_dtype=output_dtype,
                    )
                    vanilla = GatherReqToTokenPool.vanilla(req_to_token, **arguments)
                    triton_output = GatherReqToTokenPool.triton(
                        req_to_token, **arguments
                    )
                    oracle = self._scalar_oracle(
                        req_to_token,
                        req_pool_indices=req_pool_indices,
                        prefix_lens=prefix_lens,
                        seq_lens=seq_lens,
                        output_dtype=output_dtype,
                    )

                    self.assertTrue(torch.equal(triton_output, vanilla))
                    self.assertTrue(torch.equal(triton_output, oracle))
                    self.assertEqual(triton_output.dtype, output_dtype)
                    self.assertTrue(torch.equal(req_to_token, req_to_token_before))

    @staticmethod
    def _make_pool(*, rows: int, columns: int) -> torch.Tensor:
        return torch.arange(
            rows * columns,
            dtype=torch.int32,
            device="cuda",
        ).view(rows, columns)

    @staticmethod
    def _make_arguments(
        *,
        req_pool_indices: list[int],
        prefix_lens: list[int],
        seq_lens: list[int],
        output_dtype: torch.dtype,
    ) -> dict[str, object]:
        req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
        prefix_lens_cpu = torch.tensor(prefix_lens, dtype=torch.int64)
        seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        extend_lens_cpu = seq_lens_cpu - prefix_lens_cpu
        return dict(
            req_pool_indices=req_pool_indices_cpu.cuda(),
            req_pool_indices_cpu=req_pool_indices_cpu,
            prefix_lens=prefix_lens_cpu.cuda(),
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens_cpu.cuda(),
            seq_lens_cpu=seq_lens_cpu,
            extend_lens=extend_lens_cpu.cuda(),
            extend_lens_cpu=extend_lens_cpu,
            extend_num_tokens=int(extend_lens_cpu.sum().item()),
            out_dtype=output_dtype,
        )

    @staticmethod
    def _scalar_oracle(
        req_to_token: torch.Tensor,
        *,
        req_pool_indices: list[int],
        prefix_lens: list[int],
        seq_lens: list[int],
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        req_to_token_cpu = req_to_token.cpu()
        values: list[int] = []
        for req_pool_index, prefix_len, seq_len in zip(
            req_pool_indices, prefix_lens, seq_lens
        ):
            for token_position in range(prefix_len, seq_len):
                values.append(int(req_to_token_cpu[req_pool_index, token_position]))
        return torch.tensor(values, dtype=output_dtype, device=req_to_token.device)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton parity")
class TestWriteReqToTokenPool(unittest.TestCase):

    def test_empty_batch_returns_without_launching(self):
        """Empty batches return before pointer materialization or kernel launch."""
        req_to_token = torch.full(
            (2, 8),
            -7,
            dtype=torch.int32,
            device="cuda",
        )
        req_to_token_before = req_to_token.clone()
        empty_cpu = torch.empty((0,), dtype=torch.int64)
        empty_device = empty_cpu.cuda()

        with mock.patch(
            "sglang.kernels.ops.memory.req_to_token_pool._write_req_to_token_pool_kernel"
        ) as kernel:
            WriteReqToTokenPool.triton(
                req_to_token,
                req_pool_indices=empty_device,
                req_pool_indices_cpu=empty_cpu,
                prefix_lens=empty_device,
                prefix_lens_cpu=empty_cpu,
                seq_lens=empty_device,
                seq_lens_cpu=empty_cpu,
                extend_lens=empty_device,
                extend_lens_cpu=empty_cpu,
                prefix_tensors=[],
                out_cache_loc=torch.empty((0,), dtype=torch.int64, device="cuda"),
            )

        kernel.__getitem__.assert_not_called()
        self.assertTrue(torch.equal(req_to_token, req_to_token_before))

    def test_triton_vanilla_and_scalar_oracle_match_ragged_cases(self):
        """Triton, vanilla, and scalar writes match across ragged edge cases."""
        cases = (
            ([1], [[]], [[101, 102, 103]]),
            ([2], [[201, 202, 203]], [[]]),
            (
                [1, 4, 2, 5],
                [[], [301, 302], list(range(1000, 1257)), [401]],
                [[501, 502], [], list(range(2000, 2513)), [601, 602, 603]],
            ),
            (
                [3],
                [list(range(3000, 3513))],
                [list(range(4000, 4513))],
            ),
        )

        for out_dtype in (torch.int32, torch.int64):
            for req_pool_indices, prefixes, extensions in cases:
                with self.subTest(
                    out_dtype=out_dtype,
                    req_pool_indices=req_pool_indices,
                    prefix_lens=[len(prefix) for prefix in prefixes],
                    extend_lens=[len(extension) for extension in extensions],
                ):
                    vanilla_pool = torch.full(
                        (7, 1100),
                        -7,
                        dtype=torch.int32,
                        device="cuda",
                    )
                    triton_pool = vanilla_pool.clone()
                    arguments = self._make_write_arguments(
                        req_pool_indices=req_pool_indices,
                        prefixes=prefixes,
                        extensions=extensions,
                        out_dtype=out_dtype,
                    )

                    WriteReqToTokenPool.vanilla(vanilla_pool, **arguments)
                    WriteReqToTokenPool.triton(triton_pool, **arguments)
                    oracle = self._write_scalar_oracle(
                        rows=7,
                        columns=1100,
                        req_pool_indices=req_pool_indices,
                        prefixes=prefixes,
                        extensions=extensions,
                    )

                    self.assertTrue(torch.equal(triton_pool, vanilla_pool))
                    self.assertTrue(torch.equal(triton_pool, oracle))
                    untouched_rows = sorted(set(range(7)) - set(req_pool_indices))
                    self.assertTrue(torch.all(triton_pool[untouched_rows] == -7).item())
                    for req_pool_index, prefix, extension in zip(
                        req_pool_indices, prefixes, extensions
                    ):
                        seq_len = len(prefix) + len(extension)
                        self.assertTrue(
                            torch.all(
                                triton_pool[req_pool_index, seq_len:] == -7
                            ).item()
                        )

    @staticmethod
    def _make_write_arguments(
        *,
        req_pool_indices: list[int],
        prefixes: list[list[int]],
        extensions: list[list[int]],
        out_dtype: torch.dtype,
    ) -> dict[str, object]:
        req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
        prefix_lens_cpu = torch.tensor(
            [len(prefix) for prefix in prefixes],
            dtype=torch.int64,
        )
        extend_lens_cpu = torch.tensor(
            [len(extension) for extension in extensions],
            dtype=torch.int64,
        )
        seq_lens_cpu = prefix_lens_cpu + extend_lens_cpu
        return dict(
            req_pool_indices=req_pool_indices_cpu.cuda(),
            req_pool_indices_cpu=req_pool_indices_cpu,
            prefix_lens=prefix_lens_cpu.cuda(),
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens_cpu.cuda(),
            seq_lens_cpu=seq_lens_cpu,
            extend_lens=extend_lens_cpu.cuda(),
            extend_lens_cpu=extend_lens_cpu,
            prefix_tensors=[
                torch.tensor(prefix, dtype=torch.int64, device="cuda")
                for prefix in prefixes
            ],
            out_cache_loc=torch.tensor(
                [value for extension in extensions for value in extension],
                dtype=out_dtype,
                device="cuda",
            ),
        )

    @staticmethod
    def _write_scalar_oracle(
        *,
        rows: int,
        columns: int,
        req_pool_indices: list[int],
        prefixes: list[list[int]],
        extensions: list[list[int]],
    ) -> torch.Tensor:
        expected = torch.full((rows, columns), -7, dtype=torch.int32)
        for req_pool_index, prefix, extension in zip(
            req_pool_indices, prefixes, extensions
        ):
            for token_position, value in enumerate(prefix):
                expected[req_pool_index, token_position] = value
            for extension_offset, value in enumerate(extension):
                expected[req_pool_index, len(prefix) + extension_offset] = value
        return expected.cuda()


if __name__ == "__main__":
    unittest.main()
