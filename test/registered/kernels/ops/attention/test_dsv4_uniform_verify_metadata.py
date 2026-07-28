import types
import unittest

import torch

from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
    ExpandUniformVerify,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _reference_uniform_verify(
    *,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    num_draft_tokens: int,
):
    seq_lens_i64 = seq_lens.to(torch.int64)
    seq_lens_extended = seq_lens_i64 + num_draft_tokens
    seq_lens_casual = seq_lens_i64[:, None] + torch.arange(
        1,
        num_draft_tokens + 1,
        dtype=torch.int64,
        device=seq_lens.device,
    )
    idx_to_req_repeated = torch.arange(
        seq_lens.shape[0],
        dtype=torch.int32,
        device=seq_lens.device,
    ).repeat_interleave(num_draft_tokens)
    return types.SimpleNamespace(
        seq_lens_extended=seq_lens_extended,
        seq_lens_casual=seq_lens_casual.flatten().to(torch.int32),
        req_pool_indices_repeated=req_pool_indices[idx_to_req_repeated],
    )


class TestDSV4UniformVerifyMetadata(CustomTestCase):
    _RESULT_FIELDS = (
        "seq_lens_extended",
        "seq_lens_casual",
        "req_pool_indices_repeated",
    )

    def assert_result_equal(self, actual, expected):
        for field in self._RESULT_FIELDS:
            actual_tensor = getattr(actual, field)
            expected_tensor = getattr(expected, field)
            self.assertEqual(actual_tensor.dtype, expected_tensor.dtype)
            self.assertEqual(actual_tensor.shape, expected_tensor.shape)
            self.assertTrue(
                torch.equal(actual_tensor, expected_tensor),
                f"{field} mismatch: {actual_tensor=} {expected_tensor=}",
            )

    def test_torch_triton_parity(self):
        generator = torch.Generator(device="cuda").manual_seed(42)
        for bs, num_draft_tokens in ((0, 4), (1, 4), (7, 1), (64, 6), (129, 8)):
            for seq_lens_dtype in (torch.int32, torch.int64):
                for req_pool_dtype in (torch.int32, torch.int64):
                    seq_lens = torch.randint(
                        0,
                        1_048_576 - num_draft_tokens,
                        (bs,),
                        dtype=seq_lens_dtype,
                        device="cuda",
                        generator=generator,
                    )
                    req_pool_indices = torch.randperm(
                        bs + 17, device="cuda", generator=generator
                    )[:bs].to(req_pool_dtype)
                    expected = _reference_uniform_verify(
                        req_pool_indices=req_pool_indices,
                        seq_lens=seq_lens,
                        num_draft_tokens=num_draft_tokens,
                    )
                    for implementation in (
                        ExpandUniformVerify.torch,
                        ExpandUniformVerify.triton,
                    ):
                        with self.subTest(
                            bs=bs,
                            num_draft_tokens=num_draft_tokens,
                            seq_lens_dtype=seq_lens_dtype,
                            req_pool_dtype=req_pool_dtype,
                            implementation=implementation.__name__,
                        ):
                            actual = implementation(
                                req_pool_indices=req_pool_indices,
                                seq_lens=seq_lens,
                                num_draft_tokens=num_draft_tokens,
                            )
                            self.assert_result_equal(actual, expected)

    def test_int32_boundary_semantics(self):
        int32_max = torch.iinfo(torch.int32).max
        num_draft_tokens = 4
        seq_lens = torch.tensor(
            [int32_max - 1, int32_max - num_draft_tokens],
            dtype=torch.int32,
            device="cuda",
        )
        req_pool_indices = torch.tensor([7, 3], dtype=torch.int64, device="cuda")
        expected = _reference_uniform_verify(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            num_draft_tokens=num_draft_tokens,
        )

        # Extended lengths are promoted before addition, while causal lengths
        # intentionally materialize as int32 for their immediate consumer.
        self.assertEqual(
            expected.seq_lens_extended.tolist(),
            [int32_max + 3, int32_max],
        )
        for implementation in (
            ExpandUniformVerify.torch,
            ExpandUniformVerify.triton,
        ):
            actual = implementation(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                num_draft_tokens=num_draft_tokens,
            )
            self.assert_result_equal(actual, expected)

    def test_strided_inputs(self):
        bs, num_draft_tokens = 17, 4
        seq_lens = torch.arange(3, 3 + bs * 2, dtype=torch.int32, device="cuda")[::2]
        req_pool_indices = torch.arange(
            100, 100 + bs * 2, dtype=torch.int64, device="cuda"
        )[::2]
        expected = _reference_uniform_verify(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            num_draft_tokens=num_draft_tokens,
        )
        actual = ExpandUniformVerify.triton(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            num_draft_tokens=num_draft_tokens,
        )
        self.assert_result_equal(actual, expected)

    def test_cpu_dispatch(self):
        seq_lens = torch.tensor([0, 127, 1_000_000], dtype=torch.int64)
        req_pool_indices = torch.tensor([11, 3, 8], dtype=torch.int64)
        expected = _reference_uniform_verify(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            num_draft_tokens=4,
        )
        actual = ExpandUniformVerify.execute(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            num_draft_tokens=4,
        )
        self.assert_result_equal(actual, expected)

    def test_cuda_graph_replay_reads_live_inputs(self):
        num_draft_tokens = 4
        seq_lens = torch.tensor([17, 1003, 0], dtype=torch.int32, device="cuda")
        req_pool_indices = torch.tensor([8, 2, 11], dtype=torch.int64, device="cuda")

        # Compile before capture.
        ExpandUniformVerify.triton(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            num_draft_tokens=num_draft_tokens,
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = ExpandUniformVerify.triton(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                num_draft_tokens=num_draft_tokens,
            )

        for replay_seq_lens, replay_req_pool_indices in (
            ([1_000_000, 3, 127], [5, 4, 3]),
            ([0, 0, 0], [12, 10, 9]),
            ([31, 65_535, 511], [1, 7, 6]),
        ):
            seq_lens.copy_(
                torch.tensor(
                    replay_seq_lens,
                    dtype=seq_lens.dtype,
                    device=seq_lens.device,
                )
            )
            req_pool_indices.copy_(
                torch.tensor(
                    replay_req_pool_indices,
                    dtype=req_pool_indices.dtype,
                    device=req_pool_indices.device,
                )
            )
            graph.replay()
            expected = _reference_uniform_verify(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                num_draft_tokens=num_draft_tokens,
            )
            self.assert_result_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
