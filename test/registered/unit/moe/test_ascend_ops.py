"""CPU unit tests for the Ascend MoE dispatcher contracts.

These tests deliberately stop at the Python dispatcher boundary.  The
``torch_npu`` kernels themselves require an Ascend runner and are covered by
hardware integration tests when an NPU job is available.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.moe.prefill_graph import (
    MoePrefillGraphState,
    get_moe_prefill_graph_state,
    install_capture_state,
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


class TestAscendOpsPrefillGraph(CustomTestCase):
    def _topk_output(self, num_tokens):
        return (
            torch.ones((num_tokens, 2)),
            torch.zeros((num_tokens, 2), dtype=torch.int32),
            None,
        )

    # Guards the capture path: prefill graph capture forces
    # is_extend_in_batch False, so the bound state is the only thing that keeps
    # dispatch on allgather instead of falling into the decode MC2 path.
    def test_capture_state_selects_capacity_allgather(self):
        dispatcher = AscendOpsDispatcher.__new__(AscendOpsDispatcher)
        dispatcher.tp_group = None
        hidden_states = torch.zeros((8, 4))
        topk_weights, topk_ids, _ = self._topk_output(8)

        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.get_moe_prefill_graph_state",
                return_value=MoePrefillGraphState(),
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.get_is_extend_in_batch",
                return_value=False,
            ),
            patch.object(
                dispatcher, "_dispatch_allgather", return_value="allgather"
            ) as allgather,
            patch.object(dispatcher, "_dispatch_low_latency") as low_latency,
        ):
            result = dispatcher.dispatch(hidden_states, (topk_weights, topk_ids, None))

        self.assertEqual(result, "allgather")
        allgather.assert_called_once_with(
            hidden_states, topk_weights, topk_ids, None, capacity=hidden_states.shape[0]
        )
        low_latency.assert_not_called()

    # Guards the capture contract: capacity mode must not run the row-count
    # exchange, which syncs the host and cannot execute inside a captured
    # graph.  Every rank contributes the bucket width instead.
    def test_capacity_allgather_uses_bucket_rows_without_exchange(self):
        dispatcher = AscendOpsDispatcher.__new__(AscendOpsDispatcher)
        dispatcher.quant_config = {}
        capacity, world = 4, 2

        def fake_all_gather(output, local_rows, group=None):
            rows = local_rows.shape[0]
            for rank in range(world):
                output[rank * rows : (rank + 1) * rows].copy_(local_rows)

        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_world_size",
                return_value=world,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.all_gather_into_tensor",
                side_effect=fake_all_gather,
            ),
            patch.object(
                AscendOpsDispatcher,
                "_allgather_row_sizes",
                side_effect=AssertionError("row exchange must not run"),
            ),
        ):
            output = dispatcher._dispatch_allgather(
                torch.zeros((capacity, 4)),
                torch.ones((capacity, 2)),
                torch.zeros((capacity, 2), dtype=torch.int32),
                group=None,
                capacity=capacity,
            )

        self.assertEqual(output.forward_state.row_sizes, [capacity] * world)
        self.assertEqual(output.hidden_states.shape[0], world * capacity)

    # Guards the backend gate: only ascend_ops + allgather binds the state, so
    # every other configuration keeps its existing runtime dispatch.
    def test_capture_state_is_bound_only_for_ascend_ops_allgather(self):
        batch = SimpleNamespace(moe_prefill_graph_state=None)
        cases = (
            (MoeA2ABackend.DEEPEP, "allgather", False),
            (MoeA2ABackend.ASCEND_OPS, "alltoallv", False),
            (MoeA2ABackend.ASCEND_OPS, "allgather", True),
        )

        for a2a_backend, prefill, expected in cases:
            with self.subTest(a2a=a2a_backend.value, prefill=prefill):
                batch.moe_prefill_graph_state = None
                with (
                    patch(
                        "sglang.srt.hardware_backend.npu.moe.prefill_graph.get_moe_a2a_backend",
                        return_value=a2a_backend,
                    ),
                    patch(
                        "sglang.srt.hardware_backend.npu.moe.prefill_graph.get_moe_ascend_ops_config",
                        return_value=AscendOpsConfig(prefill=prefill),
                    ),
                ):
                    install_capture_state(batch)
                self.assertEqual(batch.moe_prefill_graph_state is not None, expected)

    # Guards the eager path: outside a runner-owned forward no batch is
    # published, and the dispatcher must see "no capture" rather than raise.
    def test_state_lookup_is_none_without_a_forward_batch(self):
        with patch(
            "sglang.srt.hardware_backend.npu.moe.prefill_graph.try_get_forward_batch",
            return_value=None,
        ):
            self.assertIsNone(get_moe_prefill_graph_state())


if __name__ == "__main__":
    unittest.main()
