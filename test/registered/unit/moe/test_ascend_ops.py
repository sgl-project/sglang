"""CPU unit tests for the Ascend MoE dispatcher contracts.

These tests deliberately stop at the Python dispatcher boundary.  The
``torch_npu`` kernels themselves require an Ascend runner and are covered by
hardware integration tests when an NPU job is available.
"""

import sys
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.moe.routing import (
    MXFP8_QUANT_MODE,
    RoutedTokens,
    npu_moe_token_unpermute,
)
from sglang.srt.layers.moe.token_dispatcher.ascend_ops import (
    AscendOpsCombineInput,
    AscendOpsDispatcher,
    _AllGatherState,
    _AllToAllVState,
    _MC2State,
)
from sglang.srt.layers.moe.utils import (
    AscendOpsConfig,
    DispatcherOutputDtype,
    MoeA2ABackend,
)
from sglang.srt.models.deepseek_common.moe_transport import MoeRowTransport
from sglang.srt.runtime_context import get_flags, get_forward, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAscendOpsConfig(CustomTestCase):
    # Protects the configuration boundary from accepting an advertised but
    # unimplemented decode implementation.
    def test_fused_op_is_rejected_during_config_validation(self):
        with self.assertRaisesRegex(ValueError, "fused_op.*not implemented"):
            AscendOpsConfig.from_json_str('{"decode":"fused_op"}')

    # Protects the default path used when no Ascend Ops config is supplied.
    def test_low_latency_is_the_default_decode_mode(self):
        config = AscendOpsConfig.from_json_str(None)
        self.assertEqual(config.decode, "low_latency")


class TestAscendOpsDispatcherContracts(CustomTestCase):
    def _dispatcher_for_quant_mode(self, quant_config=None):
        dispatcher = AscendOpsDispatcher.__new__(AscendOpsDispatcher)
        dispatcher.quant_config = quant_config or {}
        return dispatcher

    # Protects mode-specific quantization settings from being ignored in favor
    # of the shared dispatcher dtype.
    def test_mode_specific_dtype_overrides_shared_dtype(self):
        dispatcher = self._dispatcher_for_quant_mode(
            {
                "dispatcher_output_dtype": "bf16",
                "low_latency_dispatcher_output_dtype": "mxfp8",
            }
        )
        self.assertEqual(
            dispatcher._quant_mode("mc2", "low_latency"),
            4,
        )
        self.assertEqual(dispatcher._quant_mode("routing"), -1)

    # Protects MC2 from receiving a routing-only dispatcher dtype it cannot
    # encode in its own quantization enum.
    def test_unsupported_mc2_dtype_is_rejected(self):
        dispatcher = self._dispatcher_for_quant_mode(
            {"dispatcher_output_dtype": DispatcherOutputDtype.FP8.value}
        )
        with self.assertRaisesRegex(NotImplementedError, "mc2.*fp8"):
            dispatcher._quant_mode("mc2")

    def test_alltoallv_communicates_mode_specific_payload_and_scale(self):
        """Ascend Ops routing honors the low-latency wire dtype before fallback."""
        module = "sglang.srt.layers.moe.token_dispatcher.ascend_ops"
        hidden = torch.arange(128).reshape(2, 64).to(torch.bfloat16)
        ids = torch.tensor([[0], [1]], dtype=torch.int32)
        weights = torch.ones((2, 1))
        scales = torch.tensor([[[127, 128]], [[129, 130]]], dtype=torch.uint8)

        def route(x, topk_ids, num_experts, quant_mode):
            # Model only the external NPU boundary; dtype resolution, counts,
            # scale transport and dispatch state use the production code.
            quantized = quant_mode == MXFP8_QUANT_MODE
            return RoutedTokens(
                x.to(torch.float8_e4m3fn) if quantized else x,
                torch.arange(2, dtype=torch.int32),
                torch.ones(2, dtype=torch.int64),
                scales if quantized else None,
            )

        cases = (
            (
                {
                    "normal_dispatcher_output_dtype": "bf16",
                    "low_latency_dispatcher_output_dtype": "mxfp8",
                },
                True,
            ),
            (
                {
                    "dispatcher_output_dtype": "mxfp8",
                    "low_latency_dispatcher_output_dtype": "bf16",
                },
                False,
            ),
            ({"dispatcher_output_dtype": "mxfp8"}, True),
            ({}, False),
        )
        for config, quantized in cases:
            with self.subTest(config=config):
                dispatcher = self._dispatcher_for_quant_mode(config)
                dispatcher.num_experts = dispatcher.num_local_experts = 2
                wire_dtypes = []

                def exchange(output, payload, **kwargs):
                    wire_dtypes.append(payload.dtype)
                    self.assertEqual(kwargs["input_split_sizes"], [2])
                    self.assertEqual(kwargs["output_split_sizes"], [2])
                    output.copy_(payload)

                with (
                    patch(f"{module}.dist.get_world_size", return_value=1),
                    patch(f"{module}.dist.get_rank", return_value=0),
                    patch(f"{module}.dist.all_to_all_single", side_effect=exchange),
                    patch(f"{module}.npu_moe_init_routing", side_effect=route),
                ):
                    result = dispatcher._dispatch_alltoallv(hidden, weights, ids, None)

                expected_dtype = torch.float8_e4m3fn if quantized else torch.bfloat16
                self.assertEqual(
                    wire_dtypes,
                    [torch.uint8, expected_dtype] if quantized else [expected_dtype],
                )
                self.assertEqual(result.hidden_states.dtype, expected_dtype)
                torch.testing.assert_close(
                    result.hidden_states.float(), hidden.to(expected_dtype).float()
                )
                if quantized:
                    torch.testing.assert_close(result.hidden_states_scale, scales)
                else:
                    self.assertIsNone(result.hidden_states_scale)

    def test_allgather_communicates_prefill_mxfp8_payload_and_scale(self):
        """AllGather must quantize before communication when prefill requests it."""
        module = "sglang.srt.layers.moe.token_dispatcher.ascend_ops"
        dispatcher = self._dispatcher_for_quant_mode(
            {"low_latency_dispatcher_output_dtype": "mxfp8"}
        )
        hidden = torch.ones((1, 64), dtype=torch.bfloat16)
        payload = hidden.to(torch.float8_e4m3fn)
        scales = torch.tensor([[[127, 128]]], dtype=torch.uint8)
        wire_dtypes = []

        def gather(output, local, group):
            wire_dtypes.append(local.dtype)
            output.copy_(local)

        with (
            get_forward().scoped(ascend_allgather_row_sizes=[1]),
            patch(f"{module}.dist.get_world_size", return_value=1),
            patch(f"{module}.dist.get_rank", return_value=0),
            patch(f"{module}.dist.all_gather_into_tensor", side_effect=gather),
            patch(
                f"{module}.torch_npu",
                SimpleNamespace(
                    npu_dynamic_mx_quant=lambda *args, **kwargs: (payload, scales)
                ),
                create=True,
            ),
        ):
            result = dispatcher._dispatch_allgather(
                hidden, torch.ones((1, 1)), torch.zeros((1, 1), dtype=torch.int32), None
            )

        self.assertEqual(
            wire_dtypes, [torch.float8_e4m3fn, torch.uint8, torch.float32, torch.int32]
        )
        torch.testing.assert_close(result.hidden_states_scale, scales)

    # Protects AllGather row metadata when attention CP/DP creates uneven shards.
    def test_allgather_row_sizes_use_actual_rank_local_rows(self):
        def fake_all_gather(outputs, local_rows, group):
            outputs[0].copy_(local_rows)
            outputs[1].fill_(3)

        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_world_size",
                return_value=2,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.all_gather",
                side_effect=fake_all_gather,
            ),
        ):
            row_sizes = AscendOpsDispatcher._allgather_row_sizes(
                2, group=None, device=torch.device("cpu")
            )

        self.assertEqual(row_sizes, [2, 3])

    def test_allgather_uses_only_a_complete_tp_layout(self):
        """CP subgroups cannot supply the row counts for a larger TP group."""
        module = "sglang.srt.layers.moe.token_dispatcher.ascend_ops"
        dispatcher = self._dispatcher_for_quant_mode()
        transport = MoeRowTransport.__new__(MoeRowTransport)
        transport.batch = SimpleNamespace(
            attn_cp_metadata=SimpleNamespace(per_rank_actual_token=[2, 2]),
            global_num_tokens_padded_cpu=[5, 3],
        )
        transport.mlp_reduce_scatter = False
        transport.gathered_rows = nullcontext()
        device = torch.device("cpu")

        def gather(outputs, local_rows, group):
            outputs[0].copy_(local_rows)
            for output, rows in zip(outputs[1:], (2, 3, 3)):
                output.fill_(rows)

        with (
            get_forward().scoped(is_extend_in_batch=True),
            get_flags().moe.override(
                a2a_backend=MoeA2ABackend.ASCEND_OPS,
                moe_ascend_ops_config=AscendOpsConfig(prefill="allgather"),
            ),
            patch(f"{module}.dist.get_world_size", return_value=4),
            patch(f"{module}.dist.get_rank", return_value=0),
            patch(f"{module}.dist.all_gather", side_effect=gather) as exchange,
        ):
            transport._use_cp = False
            transport._use_tp_attn_a2a_scatter = True
            with (
                get_parallel().override(
                    tp_size=4, attn_cp_size=1, attn_dp_size=2, attn_tp_size=2
                ),
                transport.scope(),
            ):
                self.assertEqual(
                    dispatcher._resolve_allgather_row_sizes(3, None, device),
                    [3, 2, 2, 1],
                )
                exchange.assert_not_called()

            transport._use_cp = True
            with (
                get_parallel().override(
                    tp_size=4, attn_cp_size=2, attn_dp_size=2, attn_tp_size=1
                ),
                transport.scope(),
            ):
                self.assertEqual(
                    dispatcher._resolve_allgather_row_sizes(2, None, device),
                    [2, 2, 3, 3],
                )

            transport.batch.attn_cp_metadata.per_rank_actual_token = [4, 4, 4, 4]
            exchange.reset_mock()
            with (
                get_parallel().override(
                    tp_size=4, attn_cp_size=4, attn_dp_size=1, attn_tp_size=1
                ),
                transport.scope(),
            ):
                self.assertEqual(
                    dispatcher._resolve_allgather_row_sizes(4, None, device),
                    [4, 4, 4, 4],
                )
                exchange.assert_not_called()

        self.assertIsNone(get_forward().ascend_allgather_row_sizes)

    def test_allgather_unpermute_passes_gmm_output_directly(self):
        """The vendor op must receive the original rows and dtype."""
        hidden = torch.tensor([[2.0], [4.0]], dtype=torch.bfloat16)
        indices = torch.tensor([1, 0], dtype=torch.int32)
        weights = torch.tensor([[0.25, 0.75]])

        def unpermute(*, permuted_tokens, sorted_indices, probs):
            self.assertIs(permuted_tokens, hidden)
            rows = permuted_tokens.index_select(0, sorted_indices)
            return (
                (rows.float() * probs.reshape(-1, 1))
                .sum(0, keepdim=True)
                .to(permuted_tokens.dtype)
            )

        torch_npu = SimpleNamespace(npu_moe_token_unpermute=unpermute)
        with patch.dict(sys.modules, {"torch_npu": torch_npu}):
            result = npu_moe_token_unpermute(hidden, indices, weights)

        torch.testing.assert_close(result, torch.tensor([[2.5]], dtype=torch.bfloat16))

    # Protects the forward-state handoff: combine must work without a cached
    # dispatch result stored on the dispatcher instance.
    def test_combine_uses_forward_state_instead_of_dispatcher_state(self):
        dispatcher = self._dispatcher_for_quant_mode()
        state = _MC2State(
            num_tokens=1,
            assist_info_for_combine=torch.empty(0),
            ep_send_counts=torch.empty(0),
            x_active_mask=None,
            expand_scales=None,
            topk_ids=torch.zeros((1, 1), dtype=torch.int32),
            topk_weights=torch.ones((1, 1)),
        )
        combine_input = AscendOpsCombineInput(
            hidden_states=torch.zeros((1, 4)),
            forward_state=state,
        )
        with patch.object(
            dispatcher, "_combine_mc2", return_value="combined"
        ) as combine:
            result = dispatcher.combine(combine_input)

        self.assertEqual(result, "combined")
        combine.assert_called_once_with(combine_input.hidden_states, state)

    # Protects the explicit state-to-combine mapping for both prefill paths.
    def test_combine_dispatches_all_prefill_state_types(self):
        dispatcher = self._dispatcher_for_quant_mode()
        hidden_states = torch.zeros((1, 4))
        states = (
            (_AllGatherState(None, [1], torch.bfloat16), "allgather"),
            (
                _AllToAllVState(
                    None,
                    [1],
                    [1],
                    torch.ones((1, 1)),
                    torch.zeros((1, 1), dtype=torch.int32),
                    torch.zeros(1, dtype=torch.int32),
                    torch.Size((1, 4)),
                ),
                "alltoallv",
            ),
        )
        for state, mode in states:
            with self.subTest(mode=mode):
                combine_name = f"_combine_{mode}"
                with patch.object(
                    dispatcher, combine_name, return_value=mode
                ) as combine:
                    result = dispatcher.combine(
                        AscendOpsCombineInput(hidden_states, state)
                    )
                self.assertEqual(result, mode)
                combine.assert_called_once_with(hidden_states, state)


if __name__ == "__main__":
    unittest.main()
