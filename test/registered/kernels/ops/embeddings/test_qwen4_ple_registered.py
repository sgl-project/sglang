"""Small host registrations exercise boundary gathers and storage ownership."""

import gc
import os
import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.cuda is not None,
    "requires CUDA host registration",
)
class TestRegisteredPleTable(CustomTestCase):
    def setUp(self):
        self.env = mock.patch.dict(
            os.environ, {"SGLANG_QWEN4_PLE_HOST_REGISTER_CHUNK_MB": "1"}
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_partial_registration_failure_rolls_back(self):
        import cuda.bindings.runtime as cuda_rt

        from sglang.srt.models.qwen4_exp_ple_table import allocate_ple_host_table

        register = cuda_rt.cudaHostRegister
        calls = []

        def fail_second(ptr, size, flags):
            calls.append(ptr)
            if len(calls) == 2:
                return (cuda_rt.cudaError_t.cudaErrorInvalidValue,)
            return register(ptr, size, flags)

        with (
            mock.patch.object(cuda_rt, "cudaHostRegister", side_effect=fail_second),
            mock.patch.object(
                cuda_rt, "cudaHostUnregister", wraps=cuda_rt.cudaHostUnregister
            ) as unregister,
        ):
            with self.assertRaisesRegex(RuntimeError, "PLE cudaHostRegister failed"):
                allocate_ple_host_table((8193, 160), torch.bfloat16, "pinned")
            self.assertEqual(len(calls), 2)
            unregister.assert_called_once_with(calls[0])

    def test_parameter_view_keeps_boundary_gathers_alive(self):
        import cuda.bindings.runtime as cuda_rt
        import triton

        from sglang.srt.models.qwen4_exp import _gather_ple_embedding_from_pinned_kernel
        from sglang.srt.models.qwen4_exp_ple_table import allocate_ple_host_table

        table = allocate_ple_host_table((8193, 160), torch.bfloat16, "pinned")
        ranges = table._sglang_ple_registered_ranges
        self.assertEqual(len(ranges), 3)
        self.assertTrue(all(size <= 1024**2 for _, size in ranges))
        ptr = table.data_ptr()
        rows = {0, 8192}
        for base, _ in ranges[1:]:
            boundary = (base - ptr) // 320
            rows.update([boundary - 1, boundary, boundary + 1])
        rows = sorted(rows)
        values = (torch.tensor(rows)[:, None] % 127 + torch.arange(160)[None, :]).to(
            torch.bfloat16
        )
        table[rows] = values
        view = torch.nn.Parameter(table, requires_grad=False).detach().view(-1, 160)
        del table
        gc.collect()
        ids = torch.tensor([999, *[r + 1000 for r in rows], 9193], device="cuda")
        output = torch.empty(ids.numel(), 160, dtype=torch.bfloat16, device="cuda")
        expected = torch.zeros_like(output)
        expected[1:-1] = values.to("cuda")

        def gather():
            _gather_ple_embedding_from_pinned_kernel[(ids.numel(),)](
                ptr,
                ids,
                output,
                embedding_dim=160,
                tp_vocab_start=1000,
                tp_vocab_end=9193,
                is_fp8=False,
                BLOCK_D=triton.next_power_of_2(160),
            )

        gather()
        self.assertTrue(torch.equal(output, expected))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gather()
        for _ in range(2):
            output.fill_(-1)
            graph.replay()
            self.assertTrue(torch.equal(output, expected))
        del graph
        with mock.patch.object(
            cuda_rt, "cudaHostUnregister", wraps=cuda_rt.cudaHostUnregister
        ) as unregister:
            del view
            gc.collect()
            self.assertEqual(
                [c.args[0] for c in unregister.call_args_list],
                [p for p, _ in reversed(ranges)],
            )


if __name__ == "__main__":
    unittest.main()
