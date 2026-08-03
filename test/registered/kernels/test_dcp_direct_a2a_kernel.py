"""Multi-rank correctness tests for the direct DCP symmetric-memory A2A."""

from __future__ import annotations

import contextlib
import gc
import os
import sys
import unittest

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.ops.attention.dcp_kernels import _lse_weighted_combine_cpu
from sglang.srt.layers.dcp import DirectSymmA2AWorkspace
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import CustomTestCase


register_cuda_ci(
    est_time=180,
    stage="base-b-kernel-unit",
    runner_config="4-gpu-b200",
)


class TestDirectDCPA2AKernel(CustomTestCase):
    """Compare peer-dispatch + LSE combine with the existing CPU reference."""

    max_num_tokens = 7
    heads_per_rank = 2
    head_dim = 64

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("direct DCP A2A requires CUDA")

        cls.local_rank = int(os.environ["LOCAL_RANK"])
        cls.world_size = int(os.environ["WORLD_SIZE"])
        if cls.world_size not in (2, 4):
            raise unittest.SkipTest("this test covers 2-rank and 4-rank DCP")

        torch.cuda.set_device(cls.local_rank)
        dist.init_process_group(backend="gloo")
        cls.cp_group = ps.init_world_group(
            ranks=list(range(cls.world_size)),
            local_rank=cls.local_rank,
            backend="nccl",
        )
        ps._WORLD = cls.cp_group
        cls.device = torch.device(f"cuda:{cls.local_rank}")
        cls.workspaces = []

    @classmethod
    def tearDownClass(cls):
        with contextlib.suppress(Exception):
            if dist.is_initialized():
                dist.barrier()
        if hasattr(cls, "workspaces"):
            cls.workspaces.clear()
        gc.collect()
        with contextlib.suppress(Exception):
            if dist.is_initialized():
                ps.destroy_distributed_environment()
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()

    def _new_workspace(self, dtype: torch.dtype) -> DirectSymmA2AWorkspace:
        workspace = DirectSymmA2AWorkspace(
            cp_group=self.cp_group,
            device=self.device,
            max_num_tokens=self.max_num_tokens,
            heads_per_rank=self.heads_per_rank,
            head_dim=self.head_dim,
            dtype=dtype,
            num_ubatches=1,
        )
        self.workspaces.append(workspace)
        return workspace

    def _fill_inputs(
        self,
        output: torch.Tensor,
        lse: torch.Tensor,
        *,
        seed: int,
        empty_shards: bool,
    ) -> None:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed + self.local_rank * 1009)
        output.copy_(
            torch.randn(
                output.shape,
                dtype=output.dtype,
                device=self.device,
                generator=generator,
            )
        )
        lse.copy_(
            torch.randn(
                lse.shape,
                dtype=torch.float32,
                device=self.device,
                generator=generator,
            )
            * 3.0
        )
        # seq_len=1 leaves every rank except rank 0 with an empty KV shard.
        # Empty attention shards may contain NaNs; the kernel must not read
        # them after their -inf LSE turns the combine weight into zero.
        if empty_shards and self.local_rank > 0:
            output.fill_(float("nan"))
            lse.fill_(float("-inf"))

    def _reference(
        self,
        output: torch.Tensor,
        lse: torch.Tensor,
        *,
        is_lse_base_on_e: bool,
    ) -> torch.Tensor:
        gathered_output = [torch.empty_like(output) for _ in range(self.world_size)]
        gathered_lse = [torch.empty_like(lse) for _ in range(self.world_size)]
        dist.all_gather(gathered_output, output, group=self.cp_group.device_group)
        dist.all_gather(gathered_lse, lse, group=self.cp_group.device_group)

        head_start = self.local_rank * self.heads_per_rank
        head_end = head_start + self.heads_per_rank
        rank_outputs = torch.stack(
            [item[:, head_start:head_end] for item in gathered_output]
        )
        rank_lses = torch.stack([item[:, head_start:head_end] for item in gathered_lse])
        # The production kernel deliberately skips output reads for zero-weight
        # (non-finite LSE) shards. Sanitize the reused CPU reference the same way
        # so NaN payloads from empty attention shards do not produce NaN * 0.
        rank_outputs = torch.where(
            torch.isfinite(rank_lses).unsqueeze(-1),
            rank_outputs,
            torch.zeros_like(rank_outputs),
        )
        return _lse_weighted_combine_cpu(
            rank_outputs.cpu(),
            rank_lses.cpu(),
            is_lse_base_on_e=is_lse_base_on_e,
        )

    def _assert_result(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        dtype: torch.dtype,
    ) -> None:
        atol = 2e-2 if dtype == torch.float16 else 3e-2
        torch.testing.assert_close(
            actual.float().cpu(), expected.float(), atol=atol, rtol=2e-2
        )

    def _run_case(
        self,
        workspace: DirectSymmA2AWorkspace,
        *,
        dtype: torch.dtype,
        is_lse_base_on_e: bool,
        use_graph: bool,
        empty_shards: bool,
    ) -> None:
        total_heads = self.world_size * self.heads_per_rank
        output = torch.empty(
            self.max_num_tokens,
            total_heads,
            self.head_dim,
            dtype=dtype,
            device=self.device,
        )
        lse = torch.empty(
            self.max_num_tokens,
            total_heads,
            dtype=torch.float32,
            device=self.device,
        )
        combined = torch.empty(
            self.max_num_tokens,
            self.heads_per_rank,
            self.head_dim,
            dtype=dtype,
            device=self.device,
        )

        self._fill_inputs(output, lse, seed=17, empty_shards=empty_shards)
        expected = self._reference(output, lse, is_lse_base_on_e=is_lse_base_on_e)

        if not use_graph:
            workspace.lse_reduce(
                output,
                lse,
                is_lse_base_on_e=is_lse_base_on_e,
                output=combined,
            )
            torch.cuda.synchronize()
            self._assert_result(combined, expected, dtype)
            return

        torch.cuda.synchronize()
        dist.barrier(group=self.cp_group.cpu_group)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            workspace.lse_reduce(
                output,
                lse,
                is_lse_base_on_e=is_lse_base_on_e,
                output=combined,
            )
        graph.replay()
        torch.cuda.synchronize()
        dist.barrier(group=self.cp_group.cpu_group)
        self._assert_result(combined, expected, dtype)

        # Change data while preserving captured addresses, then replay twice.
        # This exercises both parity slots selected by the in-graph epoch.
        for replay_index in range(2):
            self._fill_inputs(
                output,
                lse,
                seed=101 + replay_index,
                empty_shards=empty_shards,
            )
            expected = self._reference(output, lse, is_lse_base_on_e=is_lse_base_on_e)
            graph.replay()
            torch.cuda.synchronize()
            dist.barrier(group=self.cp_group.cpu_group)
            self._assert_result(combined, expected, dtype)

    def test_correctness_matrix(self):
        for dtype in (torch.float16, torch.bfloat16):
            workspace = self._new_workspace(dtype)
            for is_lse_base_on_e in (True, False):
                for use_graph in (False, True):
                    for empty_shards in (False, True):
                        with self.subTest(
                            world_size=self.world_size,
                            dtype=dtype,
                            is_lse_base_on_e=is_lse_base_on_e,
                            use_graph=use_graph,
                            empty_shards=empty_shards,
                        ):
                            self._run_case(
                                workspace,
                                dtype=dtype,
                                is_lse_base_on_e=is_lse_base_on_e,
                                use_graph=use_graph,
                                empty_shards=empty_shards,
                            )

    def test_all_empty_shards_return_zero(self):
        workspace = self._new_workspace(torch.bfloat16)
        total_heads = self.world_size * self.heads_per_rank
        output = torch.full(
            (1, total_heads, self.head_dim),
            float("nan"),
            dtype=torch.bfloat16,
            device=self.device,
        )
        lse = torch.full(
            (1, total_heads),
            float("-inf"),
            dtype=torch.float32,
            device=self.device,
        )

        combined = workspace.lse_reduce(output, lse, is_lse_base_on_e=False)
        torch.cuda.synchronize()
        dist.barrier(group=self.cp_group.cpu_group)

        torch.testing.assert_close(
            combined.float().cpu(),
            torch.zeros_like(combined, dtype=torch.float32).cpu(),
            atol=0,
            rtol=0,
        )


if __name__ == "__main__":
    if "_IS_TEST_MULTIGPU_SGLANG_JIT_KERNEL" in os.environ:
        pytest_args = ["-x" if arg == "-f" else arg for arg in sys.argv[1:]]
        raise SystemExit(
            pytest.main([__file__, "-o", "faulthandler_timeout=300", *pytest_args])
        )
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4), timeout=600)
