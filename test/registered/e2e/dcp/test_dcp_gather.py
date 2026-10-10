"""Real NCCL coverage for direct DCP prefix layouts and their consumers."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed import parallel_state as ps
from sglang.srt.layers.dcp.comm import (
    all_gather_kv_cache_for_dcp,
    all_gather_kv_cache_for_mha_chunk_extend,
    all_gather_kv_cache_for_mha_extend,
    all_gather_kv_cache_for_mla_extend,
)
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import CustomTestCase, publish_build_topology

register_cuda_ci(est_time=30, stage="base-b", runner_config="2-gpu-large")


class TestDcpGather(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        rank, world, local_rank = (
            int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
        )
        torch.cuda.set_device(local_rank)
        ps.set_custom_all_reduce(False)
        ps.init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method="env://",
        )
        publish_build_topology(tp_size=world, dcp_size=world, world_rank=rank)
        ps.initialize_model_parallel()

    @classmethod
    def tearDownClass(cls):
        torch.cuda.synchronize()
        ps.destroy_model_parallel()
        ps.destroy_distributed_environment()
        reset_context()

    def test_ragged_consumers(self):
        for dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ):
            for lengths, starts in (
                ([4096], [0]),
                ([2731, 0, 19], [3, 0, 17]),
                ([131072, 1024, 0], [0, 0, 0]),
                ([1, 0, 1], [0, 0, 0]),
                ([0, 0, 0], [0, 0, 0]),
            ):
                with self.subTest(dtype=dtype, lengths=lengths, starts=starts):
                    self._check_consumers(dtype, lengths, starts)

    def test_portable_direct_layout(self):
        # Exercise the non-CUDA copy implementation against the same shard
        # oracle, including unaligned starts and requests with no local rows.
        with patch("sglang.srt.layers.dcp.comm._is_hip", True):
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                for lengths, starts in (
                    ([4096], [0]),
                    ([19, 0, 1], [3, 0, 17]),
                    ([19, 0, 1], [0, 0, 0]),
                ):
                    with self.subTest(dtype=dtype, lengths=lengths, starts=starts):
                        self._check_consumers(dtype, lengths, starts)

    def _check_consumers(self, dtype, lengths, starts):
        parallel = get_parallel()
        generator = torch.Generator().manual_seed(731)
        rows = [
            torch.randn(n, 1, 576, generator=generator).to(dtype).float()
            for n in lengths
        ]
        local = [
            x[(parallel.dcp_rank - start) % parallel.dcp_size :: parallel.dcp_size]
            for x, start in zip(rows, starts)
        ]
        local = torch.cat(local).to(device="cuda", dtype=dtype)
        expected = torch.cat(rows).cuda()
        k, pe = local.split([512, 64], dim=-1)
        lens, offsets = torch.tensor(lengths), torch.tensor(starts)
        combined_prefix = local.new_empty(sum(lengths), 1, 576)
        all_gather_kv_cache_for_dcp(
            k, pe, lens, offsets, output=combined_prefix.split([512, 64], dim=-1)
        )
        actual = all_gather_kv_cache_for_mha_chunk_extend(
            k.squeeze(1), pe, lens, offsets
        )
        torch.testing.assert_close(combined_prefix.float(), expected, rtol=0, atol=0)
        torch.testing.assert_close(
            actual[0].float(), expected[:, 0, :512], rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual[1].float(), expected[..., 512:], rtol=0, atol=0
        )
        self.assertTrue(all(x.is_contiguous() for x in actual))

        if any(starts):
            return
        # The cache reader is a boundary here; exercise the real merge/layout
        # consumers with exact local KV, including ranks owning no prefix rows.
        pool = SimpleNamespace(
            get_mla_kv_buffer=lambda *args, dst_dtype=None: tuple(
                x.to(dst_dtype or dtype) for x in (k, pe)
            )
        )
        extend_lens = [2, 1, 3][: len(lengths)]
        suffix = (
            torch.randn(sum(extend_lens), 1, 576, generator=generator)
            .cuda()
            .to(torch.bfloat16)
        )
        combined = torch.empty(
            sum(lengths) + sum(extend_lens), 1, 576, device="cuda", dtype=dtype
        )
        all_gather_kv_cache_for_mla_extend(
            pool,
            None,
            lengths,
            None,
            sum(lengths),
            combined,
            512,
            suffix[..., :512],
            suffix[..., 512:],
        )
        torch.testing.assert_close(
            combined.float(),
            torch.cat([expected, suffix.to(dtype).float()]),
            rtol=0,
            atol=0,
        )
        outputs = all_gather_kv_cache_for_mha_extend(
            pool,
            None,
            None,
            lengths,
            extend_lens,
            suffix[:, 0, :512],
            suffix[..., 512:],
        )
        expected_mha = torch.cat(
            [
                part
                for pair in zip(
                    expected.to(torch.bfloat16).split(lengths),
                    suffix.split(extend_lens),
                )
                for part in pair
            ]
        )
        torch.testing.assert_close(outputs[0], expected_mha[:, 0, :512], rtol=0, atol=0)
        torch.testing.assert_close(outputs[1], expected_mha[..., 512:], rtol=0, atol=0)


if __name__ == "__main__":
    if "RANK" in os.environ:
        unittest.main()
    else:
        multigpu_pytest_main(__name__, __file__, num_gpus=(2,))
