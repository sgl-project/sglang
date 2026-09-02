import unittest

import torch

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.parallel_state import (
    get_moe_ep_group,
    initialize_model_parallel,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.nccl import (
    NcclCombineInput,
    NcclDispatcher,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput


class TestNcclDispatcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_distributed_environment(
            world_size=-1,
            rank=-1,
            local_rank=-1,
            backend="nccl",
        )
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            expert_model_parallel_size=world_size,
        )

    @classmethod
    def tearDownClass(cls):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def _create_dispatcher(self, hidden_size: int) -> NcclDispatcher:
        world_size = torch.distributed.get_world_size()
        return NcclDispatcher(
            group=get_moe_ep_group().device_group,
            moe_runner_config=MoeRunnerConfig(
                num_experts=world_size * 4,
                num_local_experts=4,
                num_fused_shared_experts=0,
                hidden_size=hidden_size,
                top_k=2,
                params_dtype=torch.bfloat16,
            ),
        )

    def test_uneven_dispatch_combine(self):
        world_size = torch.distributed.get_world_size()
        if world_size < 2:
            self.skipTest("requires at least two ranks")

        rank = torch.distributed.get_rank()
        num_tokens, hidden_size, top_k = 257, 128, 2
        token = torch.arange(num_tokens, device="cuda")
        feature = torch.arange(hidden_size, device="cuda")
        hidden_states = (
            rank + token[:, None] / 512.0 + feature[None, :] / 4096.0
        ).to(torch.bfloat16)

        destinations = torch.stack(
            [
                torch.where(
                    token % 10 < 7,
                    torch.zeros_like(token),
                    (rank + token) % world_size,
                ),
                (rank + token + 1) % world_size,
            ],
            dim=1,
        )
        local_experts = torch.stack(
            [(token + rank + slot) % 4 for slot in range(top_k)], dim=1
        )
        topk_ids = (destinations * 4 + local_experts).to(torch.int32)
        topk_weights = torch.tensor(
            [2.0 / 3.0, 1.0 / 3.0], dtype=torch.float32, device="cuda"
        ).expand(num_tokens, -1)
        topk_output = StandardTopKOutput(topk_weights, topk_ids, None)

        dispatcher = self._create_dispatcher(hidden_size)
        dispatch_output = dispatcher.dispatch(hidden_states, topk_output)
        local_ids = dispatch_output.topk_output.topk_ids
        self.assertEqual(local_ids.dtype, torch.int32)
        self.assertTrue(torch.all((local_ids >= 0) & (local_ids < 4)).item())

        scales = 1.0 + local_ids[:, 0].float() / 8.0
        local_output = (
            dispatch_output.hidden_states.float()
            * dispatch_output.topk_output.topk_weights[:, 0, None]
            * scales[:, None]
        )
        combined = dispatcher.combine(
            NcclCombineInput(local_output, dispatch_output.route_handle)
        )

        reference = torch.zeros_like(combined)
        for slot in range(top_k):
            scale = 1.0 + local_experts[:, slot].float() / 8.0
            reference += (
                hidden_states.float()
                * topk_weights[:, slot, None]
                * scale[:, None]
            )
        torch.testing.assert_close(combined, reference, rtol=1e-5, atol=1e-5)

    def test_empty_source_rank(self):
        world_size = torch.distributed.get_world_size()
        if world_size < 2:
            self.skipTest("requires at least two ranks")

        rank = torch.distributed.get_rank()
        hidden_size = 16
        num_tokens = 0 if rank == 1 else 8
        hidden_states = torch.full(
            (num_tokens, hidden_size),
            rank + 1.0,
            dtype=torch.bfloat16,
            device="cuda",
        )
        target = (rank + 1) % world_size
        topk_ids = torch.full(
            (num_tokens, 1), target * 4, dtype=torch.int32, device="cuda"
        )
        topk_weights = torch.ones(
            (num_tokens, 1), dtype=torch.float32, device="cuda"
        )
        dispatcher = self._create_dispatcher(hidden_size)
        dispatch_output = dispatcher.dispatch(
            hidden_states, StandardTopKOutput(topk_weights, topk_ids, None)
        )
        combined = dispatcher.combine(
            NcclCombineInput(
                dispatch_output.hidden_states.float(), dispatch_output.route_handle
            )
        )
        self.assertEqual(combined.shape, (num_tokens, hidden_size))
        if num_tokens:
            torch.testing.assert_close(
                combined,
                hidden_states.float(),
                rtol=0,
                atol=0,
            )

    def test_all_ranks_empty(self):
        world_size = torch.distributed.get_world_size()
        if world_size < 2:
            self.skipTest("requires at least two ranks")

        hidden_states = torch.empty(
            (0, 16), dtype=torch.bfloat16, device="cuda"
        )
        topk_ids = torch.empty((0, 1), dtype=torch.int32, device="cuda")
        topk_weights = torch.empty((0, 1), dtype=torch.float32, device="cuda")
        dispatcher = self._create_dispatcher(hidden_size=16)
        dispatched = dispatcher.dispatch(
            hidden_states, StandardTopKOutput(topk_weights, topk_ids, None)
        )
        combined = dispatcher.combine(
            NcclCombineInput(dispatched.hidden_states, dispatched.route_handle)
        )
        self.assertEqual(combined.shape, (0, 16))

    def test_deterministic_combine_repeatability(self):
        """Duplicate top-k rows must combine bitwise-identically on CUDA."""
        hidden_size, num_input_tokens = 32, 257
        num_records = num_input_tokens * 8
        hidden_states = torch.randn(
            (num_records, hidden_size), dtype=torch.bfloat16, device="cuda"
        )
        token_indices = torch.arange(
            num_records, dtype=torch.int64, device="cuda"
        ) % num_input_tokens
        route_order = torch.arange(num_records, dtype=torch.int64, device="cuda")

        first = NcclDispatcher._deterministic_combine(
            hidden_states, token_indices, num_input_tokens, route_order
        )
        second = NcclDispatcher._deterministic_combine(
            hidden_states, token_indices, num_input_tokens, route_order
        )
        torch.testing.assert_close(first, second, rtol=0, atol=0)

        # A different physical expert map changes returned chunk order. The
        # canonical route key must nevertheless reproduce the same sum.
        permutation = torch.randperm(num_records, device="cuda")
        remapped = NcclDispatcher._deterministic_combine(
            hidden_states.index_select(0, permutation),
            token_indices.index_select(0, permutation),
            num_input_tokens,
            route_order.index_select(0, permutation),
        )
        torch.testing.assert_close(first, remapped, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
