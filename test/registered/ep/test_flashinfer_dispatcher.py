"""FlashinferDispatcher over FlashInfer's NVLink one-sided all-to-all on 4 GPUs.

Run with ``python test_flashinfer_dispatcher.py``; it relaunches itself under
torchrun on 4 GPUs. ``torchrun --nproc_per_node=4 test_flashinfer_dispatcher.py``
also works. The NVFP4 and MXFP8 dispatch paths need Blackwell (SM100+).
"""

import os
import unittest

import torch

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention, set_dp_buffer_len
from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
    FlashinferCombineInput,
    FlashinferDispatcher,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.runtime_context import get_context, publish
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import CustomTestCase, publish_build_topology

register_cuda_ci(est_time=240, stage="base-c", runner_config="4-gpu-b200")

IS_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


@unittest.skipUnless(IS_BLACKWELL, "Requires Blackwell (SM100+) GPUs")
class TestFlashinferDispatcher(CustomTestCase):
    """Runs FlashinferDispatcher over FlashInfer's NVLink one-sided all-to-all.

    The ranks run DP attention with attention TP 1 (attn_dp_size == tp_size ==
    ep_size), which is how the dispatcher is deployed: each rank is its own
    attention DP rank and dispatches its own batch, so a per-rank
    global_num_tokens list describes what each rank sends. Rank i sends its
    tokens to expert (i + 1) % world_size, which is the only expert on rank
    (i + 1) % world_size.
    """

    @classmethod
    def setUpClass(cls):
        # Dist-init first: world_size (and so the tp/ep width ServerArgs must
        # carry) is only known after it, and init_distributed_environment
        # itself reads no published config.
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

        parallel = dict(tp_size=world_size, ep_size=world_size, attn_dp_size=world_size)
        server_args = ServerArgs(model_path="dummy", **parallel)
        server_args.moe_runner_backend = "flashinfer_cutlass"
        server_args.moe_a2a_backend = "flashinfer_nvlink_one_sided"
        cls.server_args = server_args
        set_global_server_args_for_scheduler(server_args)
        publish(server_args, role="scheduler")
        initialize_moe_config()

        publish_build_topology(world_rank=rank, **parallel)
        initialize_model_parallel()
        initialize_dp_attention(server_args)

    @classmethod
    def tearDownClass(cls):
        destroy_model_parallel()
        destroy_distributed_environment()

    def setUp(self):
        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        # Rank i sends to rank i + 1, so this rank receives from rank i - 1.
        self.target_expert = (self.rank + 1) % self.world_size
        self.source_rank = (self.rank - 1) % self.world_size

    def create_dispatcher(self, router_topk=1, hidden_size=128):
        """Build a dispatcher with one expert per rank. Its workspace is
        released when the test ends."""
        dispatcher = FlashinferDispatcher(
            group=get_tp_group().device_group,
            router_topk=router_topk,
            num_experts=self.world_size,
            num_local_experts=1,
            hidden_size=hidden_size,
            params_dtype=torch.bfloat16,
        )
        self.addCleanup(self._destroy_dispatcher, dispatcher)
        return dispatcher

    @staticmethod
    def _destroy_dispatcher(dispatcher):
        # Instances with the same geometry share one refcounted workspace, and
        # the last destroy() frees it. Every rank must have finished its last
        # combine on it first.
        torch.cuda.synchronize()
        torch.distributed.barrier()
        dispatcher.moe_a2a.destroy()
        if dispatcher.sp_prefill_moe_a2a is not None:
            dispatcher.sp_prefill_moe_a2a.destroy()

    def set_dispatch_type(self, dispatch_type):
        get_context().override(
            "test_flashinfer_dispatcher",
            flashinfer_a2a_dispatch_type=dispatch_type,
        )

    def set_tokens_per_rank(self, tokens_per_rank):
        set_dp_buffer_len(
            global_dp_buffer_len=sum(tokens_per_rank),
            local_dp_buffer_len=tokens_per_rank[self.rank],
            dp_max_padding=False,
            global_num_tokens=list(tokens_per_rank),
        )

    def make_inputs(self, num_tokens, hidden_size, router_topk=1, fill=None):
        """Tokens for this rank, all routed to self.target_expert."""
        if fill is None:
            hidden_states = torch.randn(
                (num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda"
            )
        else:
            hidden_states = torch.full(
                (num_tokens, hidden_size), fill, dtype=torch.bfloat16, device="cuda"
            )
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones(
                (num_tokens, router_topk), dtype=torch.float32, device="cuda"
            ),
            topk_ids=torch.full(
                (num_tokens, router_topk),
                self.target_expert,
                dtype=torch.int32,
                device="cuda",
            ),
            router_logits=None,
        )
        return hidden_states, topk_output

    def assert_received_only_from(self, dispatch_output, dispatcher, tokens_per_rank):
        """Rows from this rank's source carry its tokens and this rank's
        expert; every other row is padding, marked with the invalid expert id.
        Padding hidden states are not cleared, so only the ids are checked."""
        max_tokens = max(tokens_per_rank)
        num_from_source = tokens_per_rank[self.source_rank]
        topk_ids = dispatch_output.topk_output.topk_ids.view(
            self.world_size, max_tokens, -1
        )
        expected = torch.full_like(topk_ids, dispatcher.invalid_token_expert_id)
        expected[self.source_rank, :num_from_source] = self.rank
        self.assertTrue(
            torch.equal(topk_ids, expected),
            f"rank {self.rank}: received expert ids do not match the routing",
        )
        return num_from_source

    def combine(self, dispatcher, expert_output):
        return dispatcher.combine(FlashinferCombineInput(hidden_states=expert_output))

    def run_bf16_round_trip(self, tokens_per_rank):
        """Dispatch BF16 tokens filled with 100 + rank, then combine the
        received rows unchanged, which must give back each rank's input."""
        self.set_dispatch_type("bf16")
        hidden_size = 128
        self.set_tokens_per_rank(tokens_per_rank)
        hidden_states, topk_output = self.make_inputs(
            tokens_per_rank[self.rank], hidden_size, fill=100.0 + self.rank
        )

        dispatcher = self.create_dispatcher(hidden_size=hidden_size)
        dispatcher.set_quant_config({"input_global_scale": None})
        torch.distributed.barrier()
        dispatch_output = dispatcher.dispatch(hidden_states, topk_output)

        received = dispatch_output.hidden_states
        self.assertIsNone(dispatch_output.hidden_states_scale)
        self.assertEqual(
            received.shape, (max(tokens_per_rank) * self.world_size, hidden_size)
        )
        num_from_source = self.assert_received_only_from(
            dispatch_output, dispatcher, tokens_per_rank
        )
        start = self.source_rank * max(tokens_per_rank)
        self.assertTrue(
            torch.all(
                received[start : start + num_from_source] == 100.0 + self.source_rank
            ),
            f"rank {self.rank}: expected the tokens of rank {self.source_rank}",
        )

        # With top-k 1 and unit weights, combine returns each token unchanged.
        combined = self.combine(dispatcher, received.clone())
        self.assertTrue(torch.equal(combined, hidden_states))

    def test_dispatch_basic(self):
        """BF16: each rank receives exactly its predecessor's tokens."""
        self.run_bf16_round_trip([16] * self.world_size)

    def test_dispatch_with_empty_tokens(self):
        """BF16 with rank 1 sending nothing: its receiver gets only padding."""
        tokens_per_rank = [16] * self.world_size
        tokens_per_rank[1] = 0
        self.run_bf16_round_trip(tokens_per_rank)

    def run_quantized_round(self, dispatch_type, quant_config, tokens_per_rank):
        """Dispatch, check what arrived, and combine zeros to close the round."""
        self.set_dispatch_type(dispatch_type)
        hidden_size = 128
        self.set_tokens_per_rank(tokens_per_rank)
        hidden_states, topk_output = self.make_inputs(
            tokens_per_rank[self.rank], hidden_size
        )

        dispatcher = self.create_dispatcher(hidden_size=hidden_size)
        dispatcher.set_quant_config(quant_config)
        torch.distributed.barrier()
        dispatch_output = dispatcher.dispatch(hidden_states, topk_output)
        self.assert_received_only_from(dispatch_output, dispatcher, tokens_per_rank)

        num_received = max(tokens_per_rank) * self.world_size
        expert_output = torch.zeros(
            (num_received, hidden_size), dtype=torch.bfloat16, device="cuda"
        )
        combined = self.combine(dispatcher, expert_output)
        self.assertEqual(combined.shape, hidden_states.shape)
        self.assertTrue(torch.all(combined == 0))
        return dispatch_output, num_received, hidden_size

    def test_dispatch_with_fp4_quantization(self):
        """NVFP4: packed hidden states and their block scales arrive."""
        input_global_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        dispatch_output, num_received, hidden_size = self.run_quantized_round(
            "nvfp4",
            {"input_global_scale": input_global_scale},
            [128] * self.world_size,
        )
        self.assertEqual(
            dispatch_output.hidden_states.shape, (num_received, hidden_size // 2)
        )
        self.assertEqual(dispatch_output.hidden_states.dtype, torch.uint8)
        self.assertIsNotNone(dispatch_output.hidden_states_scale)
        self.assertEqual(
            dispatch_output.hidden_states_scale.numel(),
            num_received * (hidden_size // 16),
        )
        self.assertEqual(dispatch_output.hidden_states_scale.dtype, torch.uint8)

    def assert_mxfp8_output(self, dispatch_output, num_received, hidden_size):
        self.assertEqual(
            dispatch_output.hidden_states.shape, (num_received, hidden_size)
        )
        self.assertEqual(dispatch_output.hidden_states.dtype, torch.float8_e4m3fn)
        self.assertEqual(dispatch_output.output_dtype, torch.bfloat16)
        self.assertIsNotNone(dispatch_output.hidden_states_scale)
        self.assertEqual(
            dispatch_output.hidden_states_scale.shape,
            (num_received, hidden_size // 32),
        )
        self.assertEqual(dispatch_output.hidden_states_scale.dtype, torch.uint8)
        self.assertEqual(dispatch_output.topk_output.topk_ids.shape, (num_received, 1))
        self.assertEqual(dispatch_output.topk_output.topk_ids.dtype, torch.int32)

    def test_dispatch_with_mxfp8_quantization(self):
        """MXFP8: float8_e4m3fn hidden states with uint8 block scales arrive."""
        self.assert_mxfp8_output(
            *self.run_quantized_round(
                "mxfp8",
                {"input_global_scale": None, "use_mxfp8": True},
                [128] * self.world_size,
            )
        )

    def test_dispatch_with_mxfp8_quantization_and_empty_rank(self):
        """All ranks must contribute the same payload dtypes, including empty ranks."""
        tokens_per_rank = [16] * self.world_size
        tokens_per_rank[1] = 0
        self.assert_mxfp8_output(
            *self.run_quantized_round(
                "mxfp8",
                {"input_global_scale": None, "use_mxfp8": True},
                tokens_per_rank,
            )
        )


if __name__ == "__main__":
    # Inside a torchrun worker (relaunched below, or started by hand), run the
    # tests on this rank; otherwise relaunch this file under torchrun.
    if "LOCAL_RANK" in os.environ:
        unittest.main()
    else:
        multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
