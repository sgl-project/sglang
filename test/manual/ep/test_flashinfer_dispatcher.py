"""Manual multi-GPU tests for the FlashInfer EP token dispatcher.

Needs 4 NVLink-connected GPUs of compute capability 10.0 or 10.3, the only ones
FlashInfer builds its Lamport transport for; elsewhere test_dispatch_basic and
test_dispatch_above_lamport_limit fail their preconditions. The other tests do
not check which transport they take.
"""

import unittest
from typing import NamedTuple
from unittest import mock

import torch

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention, set_dp_buffer_len
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
    FlashinferCombineInput,
    FlashinferDispatcher,
    FlashinferDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend, initialize_moe_config
from sglang.srt.runtime_context import get_context, get_parallel, get_server_args
from sglang.test.test_utils import CustomTestCase, publish_build_topology

HIDDEN_SIZE = 128
EMPTY_RANK = 1
# One token short of the full ranks, so its block at the receiver ends in padding.
SHORT_RANK = 2
# bf16 holds integers up to 256 exactly; a mark travels back as two base-256 digits.
MARK_RADIX = 256
# The rank each BF16 round sends to, as an offset from the sending rank; 0 is itself.
ROUND_OFFSETS = (1, 2, 3, 1, 0)


def assert_exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class Round(NamedTuple):
    """What one dispatch and combine produced, copied out of the workspace."""

    round_index: int
    dispatch_output: FlashinferDispatchOutput
    combined: torch.Tensor
    marks: torch.Tensor
    workspace_payload: bool


def mark_columns(marks):
    """A mark as the expert returns it: low part, then high part."""
    return marks % MARK_RADIX, torch.div(marks, MARK_RADIX, rounding_mode="floor")


def uneven_token_counts(world_size, num_tokens):
    """Token counts as DP ranks with uneven batches have them: EMPTY_RANK has no
    tokens, SHORT_RANK has num_tokens - 1 and the others num_tokens."""
    counts = [num_tokens] * world_size
    counts[EMPTY_RANK] = 0
    counts[SHORT_RANK] = num_tokens - 1
    return counts


class TestFlashinferDispatcher(CustomTestCase):
    # Marks never repeat across rounds, and every rank uses the same ones,
    # so a combine output read from memory an earlier round freed fails.
    marks_used = 0

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
        cls.world_size = torch.distributed.get_world_size()
        cls.rank = torch.distributed.get_rank()
        torch.cuda.set_device(cls.rank % torch.cuda.device_count())

        # The ordinary geometry for this backend, attention DP over the whole TP group,
        # so global_num_tokens holds one token count per EP rank.
        publish_build_topology(
            world_rank=cls.rank,
            tp_size=cls.world_size,
            ep_size=cls.world_size,
            attn_dp_size=cls.world_size,
            moe_a2a_backend="flashinfer",
        )
        initialize_model_parallel()
        initialize_dp_attention(get_server_args())
        initialize_moe_config()

    @classmethod
    def tearDownClass(cls):
        destroy_model_parallel()
        destroy_distributed_environment()

    def _callTestMethod(self, method):
        # No retry: a rank that retries alone re-enters collectives its peers have left.
        unittest.TestCase._callTestMethod(self, method)

    def create_dispatcher(self, dispatch_type, quant_config, runner_backend=None):
        """Helper to create dispatcher instance"""
        get_context().override(
            "test_flashinfer_dispatcher",
            flashinfer_a2a_dispatch_type=dispatch_type,
        )
        # Without a runner backend the dispatcher builds this config itself.
        moe_runner_config = None
        if runner_backend is not None:
            moe_runner_config = MoeRunnerConfig(
                num_experts=self.world_size,
                num_local_experts=1,
                num_fused_shared_experts=0,
                hidden_size=HIDDEN_SIZE,
                top_k=1,
                runner_backend=runner_backend,
            )
        # One expert per rank, so expert i lives on rank i.
        dispatcher = FlashinferDispatcher(
            group=get_parallel().tp_group.device_group,
            router_topk=1,
            num_experts=self.world_size,
            num_local_experts=1,
            hidden_size=HIDDEN_SIZE,
            params_dtype=torch.bfloat16,
            moe_runner_config=moe_runner_config,
        )
        dispatcher.set_quant_config(quant_config)
        self.addCleanup(self.destroy_dispatcher, dispatcher)
        return dispatcher

    @staticmethod
    def destroy_dispatcher(dispatcher):
        # Every rank must finish its combine before the workspace is freed.
        torch.cuda.synchronize()
        torch.distributed.barrier()
        dispatcher.moe_a2a.destroy()

    def target(self, round_index):
        return (self.rank + ROUND_OFFSETS[round_index]) % self.world_size

    def source(self, round_index):
        return (self.rank - ROUND_OFFSETS[round_index]) % self.world_size

    def run_round(self, dispatcher, hidden_states, global_num_tokens, round_index=0):
        """Each token carries a mark, new in each round, in its routing weight and
        in columns 0 and 1 of its row of hidden_states, which this overwrites.
        Its expert returns the mark in columns 0 and 1 and its own rank + 1
        elsewhere.

        Checks run later: a rank raising between rounds would leave its peers
        inside the next round's collectives.
        """
        set_dp_buffer_len(
            global_dp_buffer_len=sum(global_num_tokens),
            local_dp_buffer_len=global_num_tokens[self.rank],
            dp_max_padding=False,
            global_num_tokens=global_num_tokens,
        )
        num_tokens = hidden_states.shape[0]
        target_rank = self.target(round_index)
        # fp32 routing weights hold these marks exactly.
        first_mark = type(self).marks_used + 1
        type(self).marks_used += max(global_num_tokens)
        marks = (torch.arange(num_tokens, device="cuda") + first_mark).float()
        # In the payload too, so a check can pair each payload row with its weight.
        hidden_states[:, 0], hidden_states[:, 1] = mark_columns(marks)
        topk_output = StandardTopKOutput(
            topk_weights=marks.unsqueeze(1),
            topk_ids=torch.full(
                (num_tokens, 1), target_rank, dtype=torch.int32, device="cuda"
            ),
            router_logits=None,
        )
        dispatch_output = dispatcher.dispatch(hidden_states, topk_output)
        # The checks read copies: these views are valid only until the combine,
        # and unmapped once the workspace is freed.
        received = dispatch_output._replace(
            hidden_states=dispatch_output.hidden_states.clone(),
            hidden_states_scale=(
                None
                if dispatch_output.hidden_states_scale is None
                else dispatch_output.hidden_states_scale.clone()
            ),
            topk_output=dispatch_output.topk_output._replace(
                topk_ids=dispatch_output.topk_output.topk_ids.clone(),
                topk_weights=dispatch_output.topk_output.topk_weights.clone(),
            ),
            moe_output=None,
        )
        # A cutlass runner's expert writes into the combine buffer, read in place.
        expert_output = dispatch_output.moe_output
        if expert_output is None:
            expert_output = torch.empty(
                (received.hidden_states.shape[0], HIDDEN_SIZE),
                dtype=torch.bfloat16,
                device="cuda",
            )
        # Padding rows carry stale weights; neither combine sums their output.
        low, high = mark_columns(received.topk_output.topk_weights[:, 0])
        expert_output.fill_(self.rank + 1.0)
        expert_output[:, 0] = low
        expert_output[:, 1] = high
        # The fence combine skips staging only for a payload at the buffer's address;
        # record the address it receives.
        buffer_address = dispatcher.moe_a2a.get_combine_input_buffer(
            expert_output.dtype
        ).data_ptr()
        received_payloads = []
        combine = dispatcher.moe_a2a.combine

        def recording_combine(payload, **kwargs):
            received_payloads.append(payload.data_ptr())
            return combine(payload, **kwargs)

        with mock.patch.object(dispatcher.moe_a2a, "combine", recording_combine):
            combined = dispatcher.combine(FlashinferCombineInput(expert_output))
        return Round(
            round_index=round_index,
            dispatch_output=received,
            combined=combined,
            marks=marks,
            workspace_payload=received_payloads == [buffer_address],
        )

    def check_round(self, dispatcher, round_, global_num_tokens):
        """Check a finished round: whether its combine was handed the workspace's
        combine buffer, the expert ids it received, and what its own tokens
        brought back from their expert."""
        self.assertEqual(round_.workspace_payload, dispatcher.payload_in_workspace)
        # A rank receives its source's tokens in that source's rows;
        # the rest is padding, marked with num_experts, these runners' invalid id.
        source_rank = self.source(round_.round_index)
        expected_topk_ids = torch.full(
            (self.world_size, max(global_num_tokens), 1),
            dispatcher.num_experts,
            dtype=torch.int32,
            device="cuda",
        )
        expected_topk_ids[source_rank, : global_num_tokens[source_rank]] = self.rank
        assert_exact(
            actual=round_.dispatch_output.topk_output.topk_ids,
            expected=expected_topk_ids.flatten(0, 1),
        )
        target_rank = self.target(round_.round_index)
        expected = torch.full(
            (round_.marks.numel(), HIDDEN_SIZE),
            target_rank + 1.0,
            dtype=torch.bfloat16,
            device="cuda",
        )
        expected[:, 0], expected[:, 1] = mark_columns(round_.marks)
        assert_exact(actual=round_.combined, expected=expected)
        return round_.dispatch_output

    def one_round(self, dispatcher, hidden_states, global_num_tokens):
        return self.check_round(
            dispatcher=dispatcher,
            round_=self.run_round(
                dispatcher=dispatcher,
                hidden_states=hidden_states,
                global_num_tokens=global_num_tokens,
            ),
            global_num_tokens=global_num_tokens,
        )

    def check_bf16_dispatch(self, dispatcher, global_num_tokens):
        """Five rounds, as serving runs one per MoE layer. The Lamport transport
        alternates two expert-id sets, so rounds 2 and 3 reuse the sets of rounds
        0 and 1, now filled from other sources. Round 3 also goes back to round
        0's target, so it rereads the payload, weight and combine rows that round
        0 used. Round 4 sends each rank's tokens to itself, which Lamport handles
        without its landing rows. No two rounds send the same values, so a round
        that reads another's leftovers fails."""
        # Offsets 1, 2 and 3 must name three different peers.
        self.assertGreaterEqual(self.world_size, 4)

        def value(round_index, source):
            # Unique per round and source, and small enough to be exact in bf16.
            return 100.0 + self.world_size * round_index + source

        rounds = [
            self.run_round(
                dispatcher=dispatcher,
                hidden_states=torch.full(
                    (global_num_tokens[self.rank], HIDDEN_SIZE),
                    value(round_index=round_index, source=self.rank),
                    dtype=torch.bfloat16,
                    device="cuda",
                ),
                global_num_tokens=global_num_tokens,
                round_index=round_index,
            )
            for round_index in range(len(ROUND_OFFSETS))
        ]
        tokens_per_rank = max(global_num_tokens)
        for round_ in rounds:
            with self.subTest(round=round_.round_index):
                source_rank = self.source(round_.round_index)
                dispatch_output = self.check_round(
                    dispatcher=dispatcher,
                    round_=round_,
                    global_num_tokens=global_num_tokens,
                )
                self.assertIsNone(dispatch_output.hidden_states_scale)
                self.assertEqual(dispatch_output.hidden_states.dtype, torch.bfloat16)
                self.assertEqual(
                    dispatch_output.hidden_states.shape,
                    (self.world_size * tokens_per_rank, HIDDEN_SIZE),
                )
                rows = slice(global_num_tokens[source_rank])
                received = dispatch_output.hidden_states.view(
                    self.world_size, tokens_per_rank, HIDDEN_SIZE
                )[source_rank][rows]
                weights = dispatch_output.topk_output.topk_weights.view(
                    self.world_size, tokens_per_rank
                )[source_rank][rows]
                # A row's payload and its weight must carry one token's mark.
                expected = torch.full_like(
                    received, value(round_index=round_.round_index, source=source_rank)
                )
                expected[:, 0], expected[:, 1] = mark_columns(weights)
                assert_exact(actual=received, expected=expected)

    def test_dispatch_basic(self):
        """Small BF16 rounds, which FlashInfer sends over Lamport."""
        dispatcher = self.create_dispatcher(
            dispatch_type="bf16", quant_config={"input_global_scale": None}
        )
        num_tokens = 16
        self.assertGreaterEqual(dispatcher.moe_a2a.lamport_max_tokens, num_tokens)
        self.check_bf16_dispatch(
            dispatcher=dispatcher, global_num_tokens=[num_tokens] * self.world_size
        )

    def test_dispatch_above_lamport_limit(self):
        """BF16 above FlashInfer's Lamport limit takes the fence path, whose combine
        stages the expert output unless it is already in the workspace."""
        dispatcher = self.create_dispatcher(
            dispatch_type="bf16",
            quant_config={"input_global_scale": None},
            runner_backend=MoeRunnerBackend.FLASHINFER_CUTLASS,
        )
        # The cutlass runner is what moves the combine payload into the workspace.
        self.assertTrue(dispatcher.payload_in_workspace)
        # Without Lamport the limit is 0, so this would be a one-token round.
        self.assertTrue(dispatcher.moe_a2a.lamport_enabled)
        num_tokens = dispatcher.moe_a2a.lamport_max_tokens + 1
        self.check_bf16_dispatch(
            dispatcher=dispatcher, global_num_tokens=[num_tokens] * self.world_size
        )

    def test_dispatch_with_empty_tokens(self):
        """A rank with no tokens still takes part in the round, and a rank with
        fewer tokens than the round's maximum leaves padding after them."""
        dispatcher = self.create_dispatcher(
            dispatch_type="bf16", quant_config={"input_global_scale": None}
        )
        self.check_bf16_dispatch(
            dispatcher=dispatcher,
            global_num_tokens=uneven_token_counts(
                world_size=self.world_size, num_tokens=16
            ),
        )

    def test_dispatch_with_fp4_quantization(self):
        """NVFP4 dispatch packs two values per byte and carries a block scale,
        also from a rank without tokens."""
        num_tokens = 128
        global_num_tokens = uneven_token_counts(
            world_size=self.world_size, num_tokens=num_tokens
        )
        # Set input global scale to enable FP4 quantization
        input_global_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        dispatcher = self.create_dispatcher(
            dispatch_type="nvfp4",
            quant_config={"input_global_scale": input_global_scale},
        )
        hidden_states = torch.randn(
            (global_num_tokens[self.rank], HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device="cuda",
        )
        dispatch_output = self.one_round(
            dispatcher=dispatcher,
            hidden_states=hidden_states,
            global_num_tokens=global_num_tokens,
        )

        self.assertEqual(
            dispatch_output.hidden_states.shape,
            (num_tokens * self.world_size, HIDDEN_SIZE // 2),
        )
        self.assertEqual(dispatch_output.hidden_states.dtype, torch.uint8)

        self.assertIsNotNone(dispatch_output.hidden_states_scale)
        self.assertEqual(
            dispatch_output.hidden_states_scale.shape,
            (num_tokens * self.world_size, HIDDEN_SIZE // 16),
        )
        self.assertEqual(dispatch_output.hidden_states_scale.dtype, torch.uint8)

    def test_dispatch_with_mxfp8_quantization(self):
        """MXFP8 dispatch sends FP8 rows, a block scale and the routing ids."""
        num_tokens = 128
        dispatcher = self.create_dispatcher(
            dispatch_type="mxfp8",
            quant_config={"input_global_scale": None, "use_mxfp8": True},
        )
        hidden_states = torch.randn(
            (num_tokens, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
        )
        dispatch_output = self.one_round(
            dispatcher=dispatcher,
            hidden_states=hidden_states,
            global_num_tokens=[num_tokens] * self.world_size,
        )

        self.assertEqual(
            dispatch_output.hidden_states.shape,
            (num_tokens * self.world_size, HIDDEN_SIZE),
        )
        self.assertEqual(dispatch_output.hidden_states.dtype, torch.float8_e4m3fn)
        self.assertEqual(dispatch_output.output_dtype, torch.bfloat16)

        self.assertIsNotNone(dispatch_output.hidden_states_scale)
        self.assertEqual(
            dispatch_output.hidden_states_scale.shape,
            (num_tokens * self.world_size, HIDDEN_SIZE // 32),
        )
        self.assertEqual(dispatch_output.hidden_states_scale.dtype, torch.uint8)

    def test_dispatch_with_mxfp8_quantization_and_empty_rank(self):
        """All ranks must contribute the same payload dtypes, including empty ranks."""
        num_tokens = 16
        global_num_tokens = uneven_token_counts(
            world_size=self.world_size, num_tokens=num_tokens
        )
        dispatcher = self.create_dispatcher(
            dispatch_type="mxfp8",
            quant_config={"input_global_scale": None, "use_mxfp8": True},
        )
        hidden_states = torch.randn(
            (global_num_tokens[self.rank], HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device="cuda",
        )
        dispatch_output = self.one_round(
            dispatcher=dispatcher,
            hidden_states=hidden_states,
            global_num_tokens=global_num_tokens,
        )

        self.assertEqual(dispatch_output.hidden_states.dtype, torch.float8_e4m3fn)
        self.assertEqual(dispatch_output.hidden_states_scale.dtype, torch.uint8)
        self.assertEqual(
            dispatch_output.hidden_states.shape,
            (num_tokens * self.world_size, HIDDEN_SIZE),
        )
        self.assertEqual(
            dispatch_output.hidden_states_scale.shape,
            (num_tokens * self.world_size, HIDDEN_SIZE // 32),
        )


if __name__ == "__main__":
    """
    Usage
    torchrun --nproc_per_node=4 test_flashinfer_dispatcher.py
    """
    unittest.main()
