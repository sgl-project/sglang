import unittest

import torch

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.parallel_state import (
    get_tp_group,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import set_dp_buffer_len
from sglang.srt.layers.moe.token_dispatcher.flashinfer import FlashinferDispatcher
from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase


class TestFlashinferDispatcher(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        server_args = ServerArgs(model_path="dummy")
        server_args.moe_runner_backend = "flashinfer_cutlass"
        server_args.moe_a2a_backend = "flashinfer"
        set_global_server_args_for_scheduler(server_args)
        initialize_moe_config(server_args)

        init_distributed_environment(
            world_size=-1,  # Auto-detect from environment
            rank=-1,  # Auto-detect from environment
            local_rank=-1,  # Auto-detect from environment
            backend="nccl",
        )
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
        initialize_model_parallel(
            tensor_model_parallel_size=world_size, expert_model_parallel_size=world_size
        )

    @classmethod
    def tearDownClass(cls):
        # Clean up distributed environment
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def create_dispatcher(
        self, router_topk=2, num_experts=8, num_local_experts=4, hidden_size=128
    ):
        """Helper to create dispatcher instance"""
        return FlashinferDispatcher(
            group=get_tp_group().device_group,
            router_topk=router_topk,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=torch.bfloat16,
        )

    def test_dispatch_basic(self):
        """Test basic dispatch functionality"""
        num_tokens = 16
        hidden_size = 128
        router_topk = 1  # Single expert per token for simplicity
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        num_experts = world_size
        num_local_experts = 1  # One expert per rank

        set_dp_buffer_len(
            global_dp_buffer_len=num_tokens * world_size,
            local_dp_buffer_len=num_tokens,
            dp_max_padding=True,
            global_num_tokens=None,
        )

        # Create tokens with rank number
        hidden_states = torch.full(
            (num_tokens, hidden_size), 100.0 + rank, dtype=torch.bfloat16, device="cuda"
        )

        # Route all tokens from rank i to expert (i+1) % world_size
        target_rank = (rank + 1) % world_size
        target_expert = target_rank  # Since we have 1 expert per rank

        topk_ids = torch.full(
            (num_tokens, router_topk), target_expert, dtype=torch.int32, device="cuda"
        )
        topk_weights = torch.ones(
            (num_tokens, router_topk), dtype=torch.float32, device="cuda"
        )

        from sglang.srt.layers.moe.topk import StandardTopKOutput

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights, topk_ids=topk_ids, router_logits=None
        )

        torch.distributed.barrier()
        dispatcher = self.create_dispatcher(
            router_topk=router_topk,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
        )
        dispatcher.set_quant_config({"input_global_scale": None})

        dispatch_output = dispatcher.dispatch(hidden_states, topk_output)
        received_hidden_states = dispatch_output.hidden_states
        self.assertEqual(dispatch_output.hidden_states_scale, None)

        # Expected: we should receive tokens from rank (rank - 1) % world_size
        expected_source_rank = (rank - 1 + world_size) % world_size

        # Verify we received the right number of tokens
        self.assertEqual(
            received_hidden_states.shape[0],
            num_tokens * world_size,
            f"Should receive {num_tokens * world_size} tokens",
        )

        # Verify tokens came from the expected source
        self.assertTrue(
            torch.all(
                received_hidden_states[
                    expected_source_rank
                    * num_tokens : (expected_source_rank + 1)
                    * num_tokens
                ]
                == 100.0 + expected_source_rank
            )
        )
        self.assertTrue(
            torch.all(
                received_hidden_states[: expected_source_rank * num_tokens] == 0.0
            )
        )
        self.assertTrue(
            torch.all(
                received_hidden_states[(expected_source_rank + 1) * num_tokens :] == 0.0
            )
        )

    def test_dispatch_with_empty_tokens(self):
        """Test dispatch when there are no tokens (edge case)"""
        # This tests the dummy token handling
        num_tokens = 16
        hidden_size = 1
        router_topk = 1  # Single expert per token for simplicity
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        num_experts = world_size
        num_local_experts = 1  # One expert per rank

        set_dp_buffer_len(
            global_dp_buffer_len=num_tokens * world_size,
            local_dp_buffer_len=num_tokens,
            dp_max_padding=False,
            global_num_tokens=[16, 0, 16, 16],
        )

        # Route all tokens from rank i to expert (i+1) % world_size
        target_rank = (rank + 1) % world_size
        target_expert = target_rank  # Since we have 1 expert per rank

        # Create tokens with rank number, rank 1 has no tokens
        if rank == 1:
            hidden_states = torch.empty(
                0, hidden_size, dtype=torch.bfloat16, device="cuda"
            )
            topk_ids = torch.empty(0, router_topk, dtype=torch.int32, device="cuda")
            topk_weights = torch.empty(
                0, router_topk, dtype=torch.float32, device="cuda"
            )
        else:
            hidden_states = torch.full(
                (num_tokens, hidden_size),
                100.0 + rank,
                dtype=torch.bfloat16,
                device="cuda",
            )
            topk_ids = torch.full(
                (num_tokens, router_topk),
                target_expert,
                dtype=torch.int32,
                device="cuda",
            )
            topk_weights = torch.ones(
                (num_tokens, router_topk), dtype=torch.float32, device="cuda"
            )

        from sglang.srt.layers.moe.topk import StandardTopKOutput

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights, topk_ids=topk_ids, router_logits=None
        )

        dispatcher = self.create_dispatcher(
            router_topk=router_topk,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
        )
        dispatcher.set_quant_config({"input_global_scale": None})

        dispatch_output = dispatcher.dispatch(hidden_states, topk_output)
        received_hidden_states = dispatch_output.hidden_states

        # Expected: we should receive tokens from rank (rank - 1) % world_size
        expected_source_rank = (rank - 1 + world_size) % world_size

        # Verify we received the right number of tokens
        self.assertEqual(
            received_hidden_states.shape[0],
            num_tokens * world_size,
            f"Should receive {num_tokens * world_size} tokens",
        )

        # Verify tokens came from the expected source
        if rank == 2:
            # Rank 2 should receive no tokens since rank 1 was empty
            self.assertTrue(
                torch.all(received_hidden_states == 0.0),
                "Rank should receive no tokens",
            )
        else:
            self.assertTrue(
                torch.all(
                    received_hidden_states[
                        expected_source_rank
                        * num_tokens : (expected_source_rank + 1)
                        * num_tokens
                    ]
                    == 100.0 + expected_source_rank
                ),
                "Rank {rank} should receive tokens from the expected source {expected_source_rank}",
            )
            self.assertTrue(
                torch.all(
                    received_hidden_states[: expected_source_rank * num_tokens] == 0.0
                ),
                "Rank should receive no tokens from previous ranks",
            )
            self.assertTrue(
                torch.all(
                    received_hidden_states[(expected_source_rank + 1) * num_tokens :]
                    == 0.0
                ),
                "Rank should receive no tokens from next ranks",
            )

    def test_dispatch_with_fp4_quantization(self):
        """Test dispatch with FP4 quantization enabled"""
        num_tokens = 128
        hidden_size = 128
        router_topk = 1  # Single expert per token for simplicity
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        num_experts = world_size
        num_local_experts = 1  # One expert per rank

        set_dp_buffer_len(
            global_dp_buffer_len=num_tokens * world_size,
            local_dp_buffer_len=num_tokens,
            dp_max_padding=True,
            global_num_tokens=None,
        )

        # Create tokens with random values
        hidden_states = torch.randn(
            (num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda"
        )

        # Route all tokens from rank i to expert (i+1) % world_size
        target_rank = (rank + 1) % world_size
        target_expert = target_rank  # Since we have 1 expert per rank

        topk_ids = torch.full(
            (num_tokens, router_topk), target_expert, dtype=torch.int32, device="cuda"
        )
        topk_weights = torch.ones(
            (num_tokens, router_topk), dtype=torch.float32, device="cuda"
        )

        from sglang.srt.layers.moe.topk import StandardTopKOutput

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights, topk_ids=topk_ids, router_logits=None
        )

        dispatcher = self.create_dispatcher(
            router_topk=router_topk,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
        )
        # Set input global scale to enable FP4 quantization
        input_global_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        dispatcher.set_quant_config({"input_global_scale": input_global_scale})

        dispatch_output = dispatcher.dispatch(hidden_states, topk_output)

        self.assertEqual(
            dispatch_output.hidden_states.shape,
            (num_tokens * world_size, hidden_size // 2),
        )
        self.assertEqual(dispatch_output.hidden_states.dtype, torch.uint8)

        self.assertNotEqual(dispatch_output.hidden_states_scale, None)
        self.assertEqual(
            dispatch_output.hidden_states_scale.numel(),
            num_tokens * world_size * (hidden_size // 16),
        )
        self.assertEqual(dispatch_output.hidden_states_scale.dtype, torch.uint8)


if __name__ == "__main__":
    """
    Usage
    torchrun --nproc_per_node=4 test_flashinfer_dispatcher.py
    """
    unittest.main()
