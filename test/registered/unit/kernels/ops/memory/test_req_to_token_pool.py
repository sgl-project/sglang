import unittest

import torch

from sglang.kernels.ops.memory.req_to_token_pool import GatherReqToTokenPool
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


if __name__ == "__main__":
    unittest.main()
