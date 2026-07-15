import unittest

import torch

from sglang.kernels.ops.memory.req_to_token_pool import WriteReqToTokenPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton parity")
class TestWriteReqToTokenPool(CustomTestCase):

    def test_empty_batch_writes_nothing(self) -> None:
        """Empty batches launch a zero-program grid and leave the pool untouched."""
        empty_cpu = torch.empty((0,), dtype=torch.int64)
        empty_device = empty_cpu.cuda()

        for implementation in (WriteReqToTokenPool.triton, WriteReqToTokenPool.vanilla):
            with self.subTest(implementation=implementation.__name__):
                req_to_token = torch.full((2, 8), -7, dtype=torch.int32, device="cuda")
                req_to_token_before = req_to_token.clone()

                implementation(
                    req_to_token,
                    req_pool_indices=empty_device,
                    req_pool_indices_cpu=empty_cpu,
                    prefix_lens=empty_device,
                    prefix_lens_cpu=empty_cpu,
                    alloc_starts=empty_device,
                    alloc_starts_cpu=empty_cpu,
                    alloc_ends=empty_device,
                    alloc_ends_cpu=empty_cpu,
                    prefix_tensors=[],
                    out_cache_loc=torch.empty((0,), dtype=torch.int64, device="cuda"),
                )

                self.assertTrue(torch.equal(req_to_token, req_to_token_before))

    def test_host_resident_empty_prefix_tensor_is_accepted(self) -> None:
        """ChunkCache hands back a CPU empty prefix tensor; both backends and the validator take it."""
        req_to_token = torch.full((3, 16), -7, dtype=torch.int32, device="cuda")
        lens_cpu = torch.tensor([0], dtype=torch.int64)
        extend_lens_cpu = torch.tensor([3], dtype=torch.int64)
        chunk_cache_prefix = torch.empty((0,), dtype=torch.int64)
        self.assertEqual(chunk_cache_prefix.device.type, "cpu")
        arguments = dict(
            req_pool_indices=torch.tensor([1], dtype=torch.int64).cuda(),
            req_pool_indices_cpu=torch.tensor([1], dtype=torch.int64),
            prefix_lens=lens_cpu.cuda(),
            prefix_lens_cpu=lens_cpu,
            alloc_starts=lens_cpu.cuda(),
            alloc_starts_cpu=lens_cpu,
            alloc_ends=extend_lens_cpu.cuda(),
            alloc_ends_cpu=extend_lens_cpu,
            prefix_tensors=[chunk_cache_prefix],
            out_cache_loc=torch.tensor([11, 12, 13], dtype=torch.int64, device="cuda"),
        )

        for implementation in (WriteReqToTokenPool.triton, WriteReqToTokenPool.vanilla):
            with self.subTest(implementation=implementation.__name__):
                implementation(req_to_token, **arguments)

                self.assertEqual(req_to_token[1, :3].tolist(), [11, 12, 13])

        self.assertEqual(
            WriteReqToTokenPool._validate_inputs(req_to_token, **arguments), 1
        )

    def test_validate_inputs_returns_batch_size_and_rejects_malformed_batch(
        self,
    ) -> None:
        """The opt-in validator accepts a well-formed batch and rejects a short prefix."""
        arguments = self._make_write_arguments(
            req_pool_indices=[1, 4],
            prefixes=[[301, 302], []],
            extensions=[[501], [601, 602]],
            out_dtype=torch.int64,
        )
        pool = torch.full((7, 1100), -7, dtype=torch.int32, device="cuda")

        self.assertEqual(WriteReqToTokenPool._validate_inputs(pool, **arguments), 2)

        arguments["prefix_tensors"][0] = arguments["prefix_tensors"][0][:1]
        with self.assertRaises(AssertionError):
            WriteReqToTokenPool._validate_inputs(pool, **arguments)

    def test_triton_vanilla_and_scalar_oracle_match_ragged_cases(self) -> None:
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
            alloc_starts=prefix_lens_cpu.cuda(),
            alloc_starts_cpu=prefix_lens_cpu,
            alloc_ends=seq_lens_cpu.cuda(),
            alloc_ends_cpu=seq_lens_cpu,
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
