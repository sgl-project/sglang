import unittest
from contextlib import AbstractContextManager
from unittest.mock import patch

import torch

from sglang.kernels.ops.memory.req_to_token_pool import (
    AssignExtendCacheLocs,
    WriteReqToTokenPool,
)
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
                    new_loc=torch.empty((0,), dtype=torch.int64, device="cuda"),
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
            new_loc=torch.tensor([11, 12, 13], dtype=torch.int64, device="cuda"),
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

    def test_gap_below_alloc_start_is_left_untouched(self) -> None:
        """alloc_start > prefix_len means the gap is already published; rewriting it would corrupt live slots."""
        for implementation in (WriteReqToTokenPool.triton, WriteReqToTokenPool.vanilla):
            with self.subTest(implementation=implementation.__name__):
                req_to_token = torch.full((3, 32), -7, dtype=torch.int32, device="cuda")
                req_to_token[1, 4:10] = 777
                arguments = self._make_interval_arguments(
                    req_pool_indices=[1],
                    prefixes=[[301, 302, 303, 304]],
                    alloc_starts=[10],
                    alloc_ends=[16],
                )

                implementation(req_to_token, **arguments)

                self.assertTrue(torch.all(req_to_token[1, 4:10] == 777).item())
                self.assertEqual(req_to_token[1, :4].tolist(), [301, 302, 303, 304])
                self.assertEqual(
                    req_to_token[1, 10:16].tolist(),
                    arguments["new_loc"].tolist(),
                )

    def test_padding_above_seq_len_is_written(self) -> None:
        """The page-aligned tail past the logical seq_len is what makes req_to_token[0:kv_allocated_len] fully populated."""
        for implementation in (WriteReqToTokenPool.triton, WriteReqToTokenPool.vanilla):
            with self.subTest(implementation=implementation.__name__):
                req_to_token = torch.full((5, 32), -7, dtype=torch.int32, device="cuda")
                arguments = self._make_interval_arguments(
                    req_pool_indices=[0, 3],
                    prefixes=[[901, 902], []],
                    alloc_starts=[2, 0],
                    alloc_ends=[8, 16],
                )
                new_loc = arguments["new_loc"]

                implementation(req_to_token, **arguments)

                self.assertEqual(req_to_token[0, 2:8].tolist(), new_loc[0:6].tolist())
                self.assertEqual(req_to_token[3, 0:16].tolist(), new_loc[6:22].tolist())

    def test_triton_and_vanilla_agree_on_gap_plus_padding_batches(self) -> None:
        """The two backends must not drift once the write domain detaches from (prefix_len, seq_len)."""
        cases = (
            ([2], [[201]], [5], [9]),
            ([1, 4], [[], [301, 302]], [0, 7], [6, 7]),
            ([0, 3, 5], [[101, 102], [], [401]], [9, 4, 1], [20, 4, 17]),
        )

        for req_pool_indices, prefixes, alloc_starts, alloc_ends in cases:
            with self.subTest(alloc_starts=alloc_starts, alloc_ends=alloc_ends):
                vanilla_pool = torch.full((7, 32), -7, dtype=torch.int32, device="cuda")
                triton_pool = vanilla_pool.clone()
                arguments = self._make_interval_arguments(
                    req_pool_indices=req_pool_indices,
                    prefixes=prefixes,
                    alloc_starts=alloc_starts,
                    alloc_ends=alloc_ends,
                )

                WriteReqToTokenPool.vanilla(vanilla_pool, **arguments)
                WriteReqToTokenPool.triton(triton_pool, **arguments)

                self.assertTrue(torch.equal(triton_pool, vanilla_pool))
                self.assertEqual(
                    WriteReqToTokenPool._validate_inputs(triton_pool, **arguments),
                    len(req_pool_indices),
                )

    @staticmethod
    def _make_interval_arguments(
        *,
        req_pool_indices: list[int],
        prefixes: list[list[int]],
        alloc_starts: list[int],
        alloc_ends: list[int],
    ) -> dict[str, object]:
        req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
        prefix_lens_cpu = torch.tensor(
            [len(prefix) for prefix in prefixes], dtype=torch.int64
        )
        alloc_starts_cpu = torch.tensor(alloc_starts, dtype=torch.int64)
        alloc_ends_cpu = torch.tensor(alloc_ends, dtype=torch.int64)
        total = int((alloc_ends_cpu - alloc_starts_cpu).sum().item())
        return dict(
            req_pool_indices=req_pool_indices_cpu.cuda(),
            req_pool_indices_cpu=req_pool_indices_cpu,
            prefix_lens=prefix_lens_cpu.cuda(),
            prefix_lens_cpu=prefix_lens_cpu,
            alloc_starts=alloc_starts_cpu.cuda(),
            alloc_starts_cpu=alloc_starts_cpu,
            alloc_ends=alloc_ends_cpu.cuda(),
            alloc_ends_cpu=alloc_ends_cpu,
            prefix_tensors=[
                torch.tensor(prefix, dtype=torch.int64, device="cuda")
                for prefix in prefixes
            ],
            new_loc=torch.arange(5000, 5000 + total, dtype=torch.int64, device="cuda"),
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
            new_loc=torch.tensor(
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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton parity")
class TestAssignExtendCacheLocs(CustomTestCase):

    def _make_pool(self) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(0)
        pool = torch.randint(0, 100000, (8, 64), dtype=torch.int32, generator=generator)
        return pool.cuda()

    def _gather_oracle(
        self,
        req_to_token: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
        start_offset_cpu: torch.Tensor,
        end_offset_cpu: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                req_to_token[
                    int(req_pool_indices_cpu[index]),
                    int(start_offset_cpu[index]) : int(end_offset_cpu[index]),
                ].to(torch.int64)
                for index in range(req_pool_indices_cpu.shape[0])
            ]
        )

    def test_vanilla_matches_triton_across_ragged_ranges(self) -> None:
        """The vanilla path is the only implementation on HPU/MPS; it must not drift from Triton."""
        cases = (
            ([3], [4], [9]),
            ([1, 2], [0, 5], [7, 5]),
            ([3, 1, 0, 5], [4, 0, 11, 6], [37, 7, 11, 60]),
        )

        for req_pool_indices, starts, ends in cases:
            with self.subTest(starts=starts, ends=ends):
                req_to_token = self._make_pool()
                req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
                start_offset_cpu = torch.tensor(starts, dtype=torch.int64)
                end_offset_cpu = torch.tensor(ends, dtype=torch.int64)
                total = int((end_offset_cpu - start_offset_cpu).sum().item())

                triton_out = torch.empty(total, dtype=torch.int64, device="cuda")
                AssignExtendCacheLocs.triton(
                    req_to_token,
                    req_pool_indices=req_pool_indices_cpu.cuda(),
                    start_offset=start_offset_cpu.cuda(),
                    end_offset=end_offset_cpu.cuda(),
                    out_cache_loc=triton_out,
                    batch_size=len(req_pool_indices),
                )
                vanilla_out = torch.empty(total, dtype=torch.int32, device="cuda")
                AssignExtendCacheLocs.vanilla(
                    req_to_token,
                    req_pool_indices_cpu=req_pool_indices_cpu,
                    start_offset_cpu=start_offset_cpu,
                    end_offset_cpu=end_offset_cpu,
                    out_cache_loc=vanilla_out,
                )
                oracle = self._gather_oracle(
                    req_to_token, req_pool_indices_cpu, start_offset_cpu, end_offset_cpu
                )

                self.assertTrue(torch.equal(vanilla_out.to(torch.int64), triton_out))
                self.assertTrue(torch.equal(triton_out, oracle))

    def test_vanilla_leaves_req_to_token_untouched(self) -> None:
        """A gather is a pure read; a stray write would corrupt the pool it reads from."""
        req_to_token = self._make_pool()
        before = req_to_token.clone()

        AssignExtendCacheLocs.vanilla(
            req_to_token,
            req_pool_indices_cpu=torch.tensor([2, 4], dtype=torch.int64),
            start_offset_cpu=torch.tensor([1, 3], dtype=torch.int64),
            end_offset_cpu=torch.tensor([40, 60], dtype=torch.int64),
            out_cache_loc=torch.empty(96, dtype=torch.int32, device="cuda"),
        )

        self.assertTrue(torch.equal(req_to_token, before))

    def test_execute_falls_back_to_vanilla_and_returns_int64_off_the_whitelist(
        self,
    ) -> None:
        """HPU/MPS satisfy no _is_* flag; they used to fall off the dispatch and get None."""
        req_to_token = self._make_pool()
        req_pool_indices_cpu = torch.tensor([1, 3], dtype=torch.int64)
        start_offset_cpu = torch.tensor([0, 2], dtype=torch.int64)
        end_offset_cpu = torch.tensor([4, 6], dtype=torch.int64)

        with self._off_whitelist():
            out_cache_loc = AssignExtendCacheLocs.execute(
                req_to_token,
                req_pool_indices=req_pool_indices_cpu.cuda(),
                start_offset=start_offset_cpu.cuda(),
                end_offset=end_offset_cpu.cuda(),
                batch_size=2,
                out_tokens=8,
                device=torch.device("cuda"),
                ragged=False,
                req_pool_indices_cpu=req_pool_indices_cpu,
                start_offset_cpu=start_offset_cpu,
                end_offset_cpu=end_offset_cpu,
            )

        self.assertIsNotNone(out_cache_loc)
        self.assertEqual(out_cache_loc.dtype, torch.int64)
        oracle = self._gather_oracle(
            req_to_token, req_pool_indices_cpu, start_offset_cpu, end_offset_cpu
        )
        self.assertTrue(torch.equal(out_cache_loc[: oracle.numel()], oracle))

    def test_execute_without_host_mirrors_fails_loud_off_the_whitelist(self) -> None:
        """Missing mirrors must assert at the branch entry, not fail as a NoneType downstream."""
        req_to_token = self._make_pool()

        with self._off_whitelist():
            with self.assertRaises(AssertionError):
                AssignExtendCacheLocs.execute(
                    req_to_token,
                    req_pool_indices=torch.tensor([1], dtype=torch.int64).cuda(),
                    start_offset=torch.tensor([0], dtype=torch.int64).cuda(),
                    end_offset=torch.tensor([4], dtype=torch.int64).cuda(),
                    batch_size=1,
                    out_tokens=4,
                    device=torch.device("cuda"),
                    ragged=False,
                )

    def test_ragged_entry_never_reaches_the_npu_kernel(self) -> None:
        """cache_loc_update has only ever been handed equal-length ranges; ragged is unproven."""
        req_to_token = self._make_pool()
        req_pool_indices_cpu = torch.tensor([1, 3], dtype=torch.int64)
        start_offset_cpu = torch.tensor([0, 2], dtype=torch.int64)
        end_offset_cpu = torch.tensor([4, 9], dtype=torch.int64)

        with self._off_whitelist(npu=True):
            with patch.object(AssignExtendCacheLocs, "npu") as npu_kernel:
                out_cache_loc = AssignExtendCacheLocs.execute(
                    req_to_token,
                    req_pool_indices=req_pool_indices_cpu.cuda(),
                    start_offset=start_offset_cpu.cuda(),
                    end_offset=end_offset_cpu.cuda(),
                    batch_size=2,
                    out_tokens=11,
                    device=torch.device("cuda"),
                    ragged=True,
                    req_pool_indices_cpu=req_pool_indices_cpu,
                    start_offset_cpu=start_offset_cpu,
                    end_offset_cpu=end_offset_cpu,
                )

        npu_kernel.assert_not_called()
        oracle = self._gather_oracle(
            req_to_token, req_pool_indices_cpu, start_offset_cpu, end_offset_cpu
        )
        self.assertTrue(torch.equal(out_cache_loc, oracle))

    def test_equal_length_entry_does_reach_the_npu_kernel(self) -> None:
        """Guards the other half: the gate must not degrade to skipping the kernel outright."""
        req_to_token = self._make_pool()

        with self._off_whitelist(npu=True):
            with patch.object(AssignExtendCacheLocs, "npu") as npu_kernel:
                AssignExtendCacheLocs.execute(
                    req_to_token,
                    req_pool_indices=torch.tensor([1, 3], dtype=torch.int64).cuda(),
                    start_offset=torch.tensor([0, 2], dtype=torch.int64).cuda(),
                    end_offset=torch.tensor([4, 6], dtype=torch.int64).cuda(),
                    batch_size=2,
                    out_tokens=8,
                    device=torch.device("cuda"),
                    ragged=False,
                )

        npu_kernel.assert_called_once()

    def test_a_ragged_range_set_declared_equal_length_is_rejected(self) -> None:
        """The host mirrors make the ragged flag checkable; a false claim would reach the NPU kernel."""
        req_to_token = self._make_pool()
        req_pool_indices_cpu = torch.tensor([1, 3], dtype=torch.int64)
        start_offset_cpu = torch.tensor([0, 2], dtype=torch.int64)
        end_offset_cpu = torch.tensor([4, 9], dtype=torch.int64)

        with self.assertRaises(AssertionError):
            AssignExtendCacheLocs.execute(
                req_to_token,
                req_pool_indices=req_pool_indices_cpu.cuda(),
                start_offset=start_offset_cpu.cuda(),
                end_offset=end_offset_cpu.cuda(),
                batch_size=2,
                out_tokens=16,
                device=torch.device("cuda"),
                ragged=False,
                req_pool_indices_cpu=req_pool_indices_cpu,
                start_offset_cpu=start_offset_cpu,
                end_offset_cpu=end_offset_cpu,
            )

    def test_ragged_ranges_overrunning_the_output_are_rejected(self) -> None:
        """A caller sizing out_tokens too small would have the kernel write past the buffer."""
        req_to_token = self._make_pool()
        req_pool_indices_cpu = torch.tensor([1, 3], dtype=torch.int64)
        start_offset_cpu = torch.tensor([0, 2], dtype=torch.int64)
        end_offset_cpu = torch.tensor([4, 9], dtype=torch.int64)

        with self.assertRaises(AssertionError):
            AssignExtendCacheLocs.execute(
                req_to_token,
                req_pool_indices=req_pool_indices_cpu.cuda(),
                start_offset=start_offset_cpu.cuda(),
                end_offset=end_offset_cpu.cuda(),
                batch_size=2,
                out_tokens=10,
                device=torch.device("cuda"),
                ragged=True,
                req_pool_indices_cpu=req_pool_indices_cpu,
                start_offset_cpu=start_offset_cpu,
                end_offset_cpu=end_offset_cpu,
            )

    def test_ragged_output_may_be_sized_above_what_the_ranges_cover(self) -> None:
        """Graph-captured verify sizes the buffer to a static upper bound, not the live total."""
        req_to_token = self._make_pool()
        req_pool_indices_cpu = torch.tensor([1, 3], dtype=torch.int64)
        start_offset_cpu = torch.tensor([0, 2], dtype=torch.int64)
        end_offset_cpu = torch.tensor([4, 9], dtype=torch.int64)

        out_cache_loc = AssignExtendCacheLocs.execute(
            req_to_token,
            req_pool_indices=req_pool_indices_cpu.cuda(),
            start_offset=start_offset_cpu.cuda(),
            end_offset=end_offset_cpu.cuda(),
            batch_size=2,
            out_tokens=16,
            device=torch.device("cuda"),
            ragged=True,
        )

        oracle = self._gather_oracle(
            req_to_token, req_pool_indices_cpu, start_offset_cpu, end_offset_cpu
        )
        self.assertEqual(out_cache_loc.numel(), 16)
        self.assertTrue(torch.equal(out_cache_loc[:11], oracle))

    @staticmethod
    def _off_whitelist(npu: bool = False) -> AbstractContextManager:
        module = "sglang.kernels.ops.memory.req_to_token_pool"
        return patch.multiple(
            module,
            _is_cuda=False,
            _is_hip=False,
            _is_musa=False,
            _is_xpu=False,
            _is_npu=npu,
            _is_cpu=False,
        )


if __name__ == "__main__":
    unittest.main()
