import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
    FlashinferDispatcher,
    _trtllm_output_in_workspace_eligible,
    _WorkspaceNamespace,
)
from sglang.srt.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopKConfig,
)
from sglang.srt.layers.moe.utils import FlashinferA2ADispatchType, MoeRunnerBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _RecordingCommunication:
    def __init__(self):
        self.combine_input_buffer = None

    def get_combine_input_buffer(self, dtype):
        rows = self.hidden_states.shape[0]
        self.combine_input_buffer = torch.zeros((rows, 128), dtype=dtype)
        return self.combine_input_buffer

    def combine(self, expert_output):
        self.combined_in_place = (
            expert_output.data_ptr() == self.combine_input_buffer.data_ptr()
        )
        return expert_output.clone()

    def dispatch(
        self,
        hidden_states,
        topk_ids,
        topk_weights=None,
        *,
        hidden_states_scale=None,
        **_kwargs,
    ):
        self.hidden_states = hidden_states
        self.hidden_states_scale = hidden_states_scale
        self.topk_ids = topk_ids
        self.topk_weights = topk_weights
        return SimpleNamespace(
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )


def _make_dispatcher(*, dispatch_type, max_num_tokens, quant_config):
    dispatcher = object.__new__(FlashinferDispatcher)
    dispatcher.runner_backend = None
    dispatcher.dispatch_type = dispatch_type
    dispatcher.hidden_size = 128
    dispatcher.max_num_tokens = max_num_tokens
    dispatcher.ep_size = 1
    dispatcher.invalid_token_expert_id = 8
    dispatcher.payload_in_workspace = False
    dispatcher.trtllm_output_in_workspace = False
    dispatcher.trtllm_output_in_workspace_logged = False
    dispatcher.quant_config = quant_config
    dispatcher.moe_a2a = _RecordingCommunication()
    dispatcher.decode_a2a_namespace = _WorkspaceNamespace.DECODE
    dispatcher.sp_prefill_moe_a2a = None
    return dispatcher


def _dispatch_without_dp(*, dispatcher, hidden_states, topk_output):
    with (
        patch(
            "sglang.srt.layers.moe.token_dispatcher.flashinfer.get_dp_global_num_tokens",
            return_value=None,
        ),
        patch(
            "sglang.srt.layers.moe.token_dispatcher.flashinfer.is_dp_attention_enabled",
            return_value=False,
        ),
    ):
        return dispatcher.dispatch(hidden_states=hidden_states, topk_output=topk_output)


class TestFlashinferDispatcher(CustomTestCase):
    def test_empty_mxfp8_dispatch_uses_same_payload_dtype_as_nonempty_rank(self):
        dispatcher = _make_dispatcher(
            dispatch_type=FlashinferA2ADispatchType.MXFP8,
            max_num_tokens=0,
            quant_config={"use_mxfp8": True},
        )
        topk_output = StandardTopKOutput(
            topk_weights=torch.empty((0, 1), dtype=torch.float32),
            topk_ids=torch.empty((0, 1), dtype=torch.int32),
            router_logits=None,
        )

        output = _dispatch_without_dp(
            dispatcher=dispatcher,
            hidden_states=torch.empty((0, 128), dtype=torch.bfloat16),
            topk_output=topk_output,
        )

        moe_a2a = dispatcher.moe_a2a
        self.assertEqual(
            [
                moe_a2a.hidden_states.dtype,
                moe_a2a.hidden_states_scale.dtype,
                moe_a2a.topk_ids.dtype,
                moe_a2a.topk_weights.dtype,
            ],
            [torch.float8_e4m3fn, torch.uint8, torch.int32, torch.float32],
        )
        self.assertEqual(output.hidden_states.dtype, torch.float8_e4m3fn)
        self.assertEqual(output.hidden_states_scale.dtype, torch.uint8)

    def test_int64_expert_ids_reach_moe_a2a_as_int32(self):
        """Expert IDs reach the A2A as int32 even when routing emits int64,
        as static EP dispatch does; FlashInfer's kernel rejects any other dtype."""
        dispatcher = _make_dispatcher(
            dispatch_type=FlashinferA2ADispatchType.BF16,
            max_num_tokens=4,
            quant_config={},
        )
        topk_ids = torch.tensor([[0, 5], [7, 2], [3, 3]], dtype=torch.int64)
        topk_output = StandardTopKOutput(
            topk_weights=torch.full((3, 2), 0.5, dtype=torch.float32),
            topk_ids=topk_ids,
            router_logits=None,
        )

        _dispatch_without_dp(
            dispatcher=dispatcher,
            hidden_states=torch.zeros((3, 128), dtype=torch.bfloat16),
            topk_output=topk_output,
        )

        moe_a2a = dispatcher.moe_a2a
        self.assertEqual(moe_a2a.topk_ids.dtype, torch.int32)
        self.assertTrue(torch.equal(moe_a2a.topk_ids, topk_ids.to(torch.int32)))

    def test_bypassed_topk_is_routed_before_moe_a2a(self):
        """Runners that fuse routing hand over bypassed top-k; the A2A still
        needs explicit expert IDs and weights, so routing must run first."""
        dispatcher = _make_dispatcher(
            dispatch_type=FlashinferA2ADispatchType.BF16,
            max_num_tokens=4,
            quant_config={},
        )
        hidden_states = torch.zeros((3, 128), dtype=torch.bfloat16)
        router_logits = torch.tensor(
            [[0.1, 3.0, 0.2, 2.0], [5.0, 0.0, 4.0, 0.1], [0.0, 0.1, 0.2, 9.0]]
        )
        logical_to_physical = torch.tensor([2, 0, 3, 1])
        topk_output = BypassedTopKOutput(
            hidden_states=hidden_states,
            raw_router_logits=router_logits,
            topk_config=TopKConfig(top_k=2, torch_native=True),
            expert_location_dispatch_info=ExpertLocationDispatchInfo(
                ep_dispatch_algorithm="static",
                partial_logical_to_rank_dispatch_physical_map=logical_to_physical,
                partial_logical_to_all_physical_map=logical_to_physical[:, None],
                partial_logical_to_all_physical_map_num_valid=torch.ones(
                    4, dtype=torch.int64
                ),
                num_physical_experts=4,
            ),
        )

        _dispatch_without_dp(
            dispatcher=dispatcher,
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

        moe_a2a = dispatcher.moe_a2a
        expert_ids = moe_a2a.topk_ids
        self.assertEqual(expert_ids.dtype, torch.int32)
        self.assertEqual(
            expert_ids.sort(dim=-1).values.tolist(), [[0, 1], [2, 3], [1, 3]]
        )
        self.assertEqual(moe_a2a.topk_weights.shape, (3, 2))


def _fake_fp4_quantize(x, global_scale, is_sf_swizzled_layout):
    rows = x.shape[0]
    return (
        torch.zeros((rows, x.shape[1] // 2), dtype=torch.uint8),
        torch.zeros((rows, x.shape[1] // 16), dtype=torch.uint8),
    )


def _nvfp4_trtllm_dispatcher(*, output_in_workspace):
    dispatcher = _make_dispatcher(
        dispatch_type=FlashinferA2ADispatchType.NVFP4,
        max_num_tokens=4,
        quant_config={"input_global_scale": torch.ones(())},
    )
    dispatcher.runner_backend = MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED
    dispatcher.trtllm_output_in_workspace = output_in_workspace
    return dispatcher


def _routed_topk(num_tokens):
    return StandardTopKOutput(
        topk_weights=torch.full((num_tokens, 2), 0.5),
        topk_ids=torch.zeros((num_tokens, 2), dtype=torch.int32),
        router_logits=None,
    )


def _run_trtllm_fp4(dispatch_output, *, rows):
    """The FlashInfer A2A + TRT-LLM Gen FP4 runner with only its kernel faked; the
    kernel writes ``rows`` into the output it is given."""
    from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm as runner
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

    def routed_moe(**kwargs):
        kwargs["output"].copy_(rows)
        return [kwargs["output"]]

    flashinfer = ModuleType("flashinfer")
    flashinfer.__path__ = []
    fused_moe = ModuleType("flashinfer.fused_moe")
    fused_moe.trtllm_fp4_block_scale_moe = None
    fused_moe.trtllm_fp4_block_scale_routed_moe = routed_moe
    weight = torch.empty(0, dtype=torch.uint8)
    quant_info = runner.FlashInferTrtllmFp4MoeQuantInfo(
        w13_weight=weight,
        w2_weight=weight,
        w13_weight_scale=weight,
        w2_weight_scale=weight,
        g1_scale_c=weight,
        g1_alphas=weight,
        g2_alphas=weight,
        w13_input_scale_quant=weight,
        global_num_experts=8,
        local_expert_offset=0,
        local_num_experts=8,
        intermediate_size_per_partition=64,
        routing_method_type=0,
    )
    with (
        patch.dict(
            "sys.modules",
            {"flashinfer": flashinfer, "flashinfer.fused_moe": fused_moe},
        ),
        patch.object(runner, "get_activation_type", return_value=0),
        patch.object(runner, "trtllm_moe_enable_pdl", return_value=False),
        patch.object(
            runner,
            "_get_fp4_moe_output_buffer",
            side_effect=lambda num_tokens, hidden_size, dtype, **_: torch.empty(
                (num_tokens, hidden_size), dtype=dtype
            ),
        ),
    ):
        return runner.fused_experts_flashinfer_to_flashinfer_trtllm(
            dispatch_output, quant_info, MoeRunnerConfig()
        )


class TestMoeA2AOutputInWorkspace(CustomTestCase):
    """SGLANG_OPT_MOE_A2A_OUTPUT_IN_WORKSPACE: TRT-LLM Gen finalizes NVFP4 decode
    rows into the A2A combine input buffer, which the combine reads in place."""

    def test_switch_needs_a_trtllm_gen_runner(self):
        trtllm = MoeRunnerBackend.FLASHINFER_TRTLLM
        for backend, child, eligible in (
            (trtllm, True, True),
            (MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED, True, True),
            (MoeRunnerBackend.FLASHINFER_CUTLASS, True, False),
            (None, True, False),
            (trtllm, False, False),
        ):
            with (
                self.subTest(backend=backend, child=child),
                envs.SGLANG_OPT_MOE_A2A_OUTPUT_IN_WORKSPACE.override(child),
            ):
                self.assertEqual(
                    _trtllm_output_in_workspace_eligible(
                        backend, _WorkspaceNamespace.DECODE
                    ),
                    eligible,
                )

    def test_nvfp4_decode_rows_reach_the_combine_in_its_input_buffer(self):
        dispatcher = _nvfp4_trtllm_dispatcher(output_in_workspace=True)
        hidden = torch.zeros((3, 128), dtype=torch.bfloat16)
        with patch(
            "sglang.srt.layers.moe.token_dispatcher.flashinfer.fp4_quantize",
            _fake_fp4_quantize,
            create=True,
        ):
            dispatch_output = _dispatch_without_dp(
                dispatcher=dispatcher, hidden_states=hidden, topk_output=_routed_topk(3)
            )
        moe_a2a = dispatcher.moe_a2a
        self.assertIsNotNone(dispatch_output.moe_output)
        self.assertIs(dispatch_output.moe_output, moe_a2a.combine_input_buffer)
        self.assertEqual(dispatch_output.moe_output.dtype, torch.bfloat16)

        rows = torch.arange(3 * 128, dtype=torch.bfloat16).view(3, 128)
        combine_input = _run_trtllm_fp4(dispatch_output, rows=rows)
        self.assertIs(combine_input.hidden_states, moe_a2a.combine_input_buffer)
        combined = dispatcher.combine(combine_input)
        self.assertTrue(moe_a2a.combined_in_place)
        self.assertTrue(torch.equal(combined, rows))

    def test_other_dispatches_and_outputs_keep_the_staging_copy(self):
        """No buffer is handed out with the switch off, for a BF16 dispatch or an SP
        prefill dispatch; the runner finalizes elsewhere unless the buffer fits."""
        nvfp4 = {"input_global_scale": torch.ones(())}
        dispatches = {
            "switch off": (False, nvfp4, False),
            "bf16 dispatch": (True, {}, False),
            "sp prefill": (True, nvfp4, True),
        }
        module = "sglang.srt.layers.moe.token_dispatcher.flashinfer"
        for case, (on, quant_config, sp_prefill) in dispatches.items():
            with self.subTest(case=case):
                dispatcher = _nvfp4_trtllm_dispatcher(output_in_workspace=on)
                dispatcher.quant_config = quant_config
                dispatcher.sp_prefill_moe_a2a = _RecordingCommunication()
                with (
                    patch(f"{module}.fp4_quantize", _fake_fp4_quantize, create=True),
                    patch(f"{module}.get_is_extend_in_batch", return_value=sp_prefill),
                    patch(
                        f"{module}.get_forward",
                        return_value=SimpleNamespace(sp_active=sp_prefill),
                    ),
                ):
                    dispatch_output = _dispatch_without_dp(
                        dispatcher=dispatcher,
                        hidden_states=torch.zeros((3, 128), dtype=torch.bfloat16),
                        topk_output=_routed_topk(3),
                    )
                self.assertIsNone(dispatch_output.moe_output)

        from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
            FlashinferDispatchOutput,
        )
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )

        hidden = torch.zeros((3, 64), dtype=torch.uint8)
        scale = torch.zeros((3, 8), dtype=torch.uint8)
        rows = torch.ones((3, 128), dtype=torch.bfloat16)
        buffers = {
            "other shape": torch.zeros((4, 128), dtype=torch.bfloat16),
            "other dtype": torch.zeros((3, 128), dtype=torch.float16),
            "prefill all-gather": None,
        }
        for case, buffer in buffers.items():
            with self.subTest(case=case):
                if buffer is None:
                    output = StandardDispatchOutput(hidden, scale, _routed_topk(3))
                else:
                    output = FlashinferDispatchOutput(
                        hidden, scale, _routed_topk(3), buffer
                    )
                combine_input = _run_trtllm_fp4(output, rows=rows)
                self.assertIsNot(combine_input.hidden_states, buffer)
                self.assertTrue(torch.equal(combine_input.hidden_states, rows))


class TestFlashinferRunnerDispatchFormats(CustomTestCase):
    def test_cutlass_preserves_prefill_and_decode_combine_formats(self):
        """AG prefill cannot use the decode workspace or A2A kernel layout."""
        from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass as runner
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
            FlashinferCombineInput,
            FlashinferDispatchOutput,
        )
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
            StandardDispatchOutput,
        )
        from sglang.srt.runtime_context import get_context, get_parallel

        layer = SimpleNamespace(
            moe_tp_size=1, moe_tp_rank=0, moe_ep_size=4, moe_ep_rank=0
        )
        config = MoeRunnerConfig(layer=layer)
        quant_info = runner.FlashInferCutlassMoeQuantInfo(
            quant_type="bf16", w13_weight=torch.empty(0), w2_weight=torch.empty(0)
        )
        hidden = torch.zeros((2, 128), dtype=torch.bfloat16)
        topk = StandardTopKOutput(
            torch.ones((2, 1)), torch.zeros((2, 1), dtype=torch.int32), None
        )

        for prefill in (True, False):
            with self.subTest(prefill=prefill):
                workspace = torch.empty_like(hidden)
                dispatch = (
                    StandardDispatchOutput(hidden, None, topk)
                    if prefill
                    else FlashinferDispatchOutput(hidden, None, topk, workspace)
                )

                def kernel(**kwargs):
                    # This flag changes FlashInfer's distributed input layout.
                    self.assertEqual(kwargs["enable_alltoall"], not prefill)
                    kwargs["output"].copy_(kwargs["input"] + 1)
                    return (kwargs["output"],)

                with (
                    get_context().override_server_args(model_path="dummy"),
                    get_parallel().override(tp_group=None),
                    patch.object(
                        runner,
                        "_flashinfer_cutlass_fused_moe",
                        return_value=(kernel, None),
                    ),
                    patch.object(runner, "_activation_type", return_value=None),
                ):
                    result = runner.fused_experts_flashinfer_to_flashinfer_cutlass(
                        dispatch, quant_info, config
                    )
                self.assertIsInstance(
                    result,
                    StandardCombineInput if prefill else FlashinferCombineInput,
                )
                torch.testing.assert_close(
                    result.hidden_states, torch.ones_like(hidden)
                )
                if not prefill:
                    self.assertIs(result.hidden_states, workspace)

    def test_cutedsl_preserves_prefill_and_decode_combine_formats(self):
        """AG prefill (including idle ranks) must reach RS, not A2A combine."""
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
            CuteDslFp4MoeQuantInfo,
            fused_experts_flashinfer_to_flashinfer_cutedsl_fp4,
        )
        from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
            FlashinferCombineInput,
            FlashinferDispatchOutput,
        )
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
            StandardDispatchOutput,
        )

        # Replace only the GPU kernel boundary; routing and combine selection
        # execute the real runner on CPU tensors.
        wrapper = SimpleNamespace(run=lambda **kwargs: kwargs["x"] + 1)
        weight = torch.empty(0)
        quant_info = CuteDslFp4MoeQuantInfo(
            w13_weight=weight,
            w2_weight=weight,
            w13_weight_sf=weight,
            w2_weight_sf=weight,
            w1_alpha=weight,
            w2_alpha=weight,
            a1_scale=weight,
            a2_scale=weight,
            wrapper=wrapper,
            quant_mode="w4a16",
        )
        for num_tokens in (0, 2):
            hidden = torch.zeros((num_tokens, 128), dtype=torch.bfloat16)
            topk = StandardTopKOutput(
                torch.ones((num_tokens, 1)),
                torch.zeros((num_tokens, 1), dtype=torch.int32),
                None,
            )
            for prefill in (True, False):
                with self.subTest(num_tokens=num_tokens, prefill=prefill):
                    workspace = torch.empty_like(hidden)
                    dispatch = (
                        StandardDispatchOutput(hidden, None, topk)
                        if prefill
                        else FlashinferDispatchOutput(hidden, None, topk, workspace)
                    )
                    result = fused_experts_flashinfer_to_flashinfer_cutedsl_fp4(
                        dispatch, quant_info, MoeRunnerConfig()
                    )
                    self.assertIsInstance(
                        result,
                        StandardCombineInput if prefill else FlashinferCombineInput,
                    )
                    torch.testing.assert_close(
                        result.hidden_states, torch.ones_like(hidden)
                    )
                    if not prefill:
                        self.assertIs(result.hidden_states, workspace)


if __name__ == "__main__":
    unittest.main()
