"""Regression coverage for GLM-4.7-Flash DCP prefix metadata (#50574)."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.kv_indices import (
    create_chunked_prefix_cache_kv_indices,
)
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteForCausalLM
from sglang.srt.runtime_context import get_context, get_device, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "DCP planner uses CUDA kernels")
class TestGlm4MoeLiteDcpPrefix(CustomTestCase):
    def test_prefix_hit_produces_rank_local_metadata(self):
        # Two requests: [4 cached, 2 new] and [2 cached, 3 new] tokens.
        req_to_token = torch.tensor(
            [[20, 21, 22, 23, 0, 0], [40, 41, 0, 0, 0, 0]],
            dtype=torch.int32,
            device="cuda",
        )
        prefix_lens = torch.tensor([4, 2], dtype=torch.int32, device="cuda")
        extend_lens = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([6, 5], dtype=torch.int32, device="cuda")
        req_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        model = Glm4MoeLiteForCausalLM.__new__(Glm4MoeLiteForCausalLM)
        runner_ids = RunnerIds.__new__(RunnerIds)
        runner_ids.is_translating = False  # static KV pool
        backend = SimpleNamespace(runner_ids=runner_ids)

        with get_context().override_server_args(), get_device().override(device="cuda"):
            for rank in (0, 1):
                with (
                    self.subTest(rank=rank),
                    get_parallel().override(
                        dcp_enabled=True,
                        dcp_size=2,
                        dcp_rank=rank,
                        attn_dcp_size=2,
                    ),
                    forward_context(ForwardContext(attn_backend=backend)),
                ):
                    metadata = model.prepare_context_parallel_metadata_for_dcp(
                        seq_lens=seq_lens,
                        extend_prefix_lens=prefix_lens,
                        extend_prefix_lens_cpu=[4, 2],
                        extend_seq_lens=extend_lens,
                        req_pool_indices=req_indices,
                        req_to_token=req_to_token,
                        seq_lens_sum=11,
                        kv_buffer_shape=torch.Size([128, 1, 6]),
                        kv_cache_dtype=torch.float32,
                        kv_cache_device="cuda",
                        create_chunked_prefix_cache_kv_indices_fn=(
                            create_chunked_prefix_cache_kv_indices
                        ),
                    )

                    self.assertEqual(metadata.dcp_extend_prefix_lens_sum, 6)
                    self.assertEqual(metadata.dcp_kv_buffer.shape, (11, 1, 6))
                    self.assertEqual(metadata.dcp_kv_indptr.tolist(), [0, 6, 11])
                    self.assertEqual(
                        metadata.dcp_kv_indices.tolist(),
                        [0, 1, 2, 3, 6, 7, 4, 5, 8, 9, 10],
                    )
                    self.assertEqual(
                        metadata.dcp_local_prefix_kv_indices.tolist(), [10, 11, 20]
                    )


if __name__ == "__main__":
    unittest.main()
