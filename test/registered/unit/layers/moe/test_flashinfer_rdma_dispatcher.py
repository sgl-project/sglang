import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.flashinfer import FlashinferCombineInput
from sglang.srt.layers.moe.token_dispatcher.flashinfer_rdma import (
    FlashinferRdmaDispatcher,
    _validate_parallel_topology,
    close_flashinfer_rdma_buffers,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MODULE = "sglang.srt.layers.moe.token_dispatcher.flashinfer_rdma"


def _parallel(ep, rank):
    return SimpleNamespace(
        tp_size=ep,
        attn_dp_size=ep,
        attn_tp_size=1,
        attn_cp_size=1,
        attn_dcp_size=1,
        moe_ep_size=ep,
        moe_tp_size=1,
        moe_dp_size=1,
        tp_rank=rank,
        attn_dp_rank=rank,
        moe_ep_rank=rank,
    )


class TestFlashinferRdmaDispatcher(CustomTestCase):
    def test_native_mxfp4_wire_contract_before_buffer_allocation(self):
        group = Mock()
        group.size.return_value, group.rank.return_value = 4, 0
        parallel = _parallel(4, 0)
        parallel.tp_group = parallel.moe_ep_group = SimpleNamespace(device_group=group)
        config = SimpleNamespace(
            runner_backend=MoeRunnerBackend.FLASHINFER_MXFP4,
            apply_router_weight_on_input=False,
        )
        tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            kwargs["device"] = "cpu"
            return tensor(*args, **kwargs)

        with (
            patch(f"{MODULE}.get_parallel", return_value=parallel),
            patch(
                f"{MODULE}.dist.get_process_group_ranks", return_value=list(range(4))
            ),
            patch(f"{MODULE}.dist.all_reduce") as reduce,
            patch(f"{MODULE}.torch.tensor", side_effect=cpu_tensor),
            patch(
                f"{MODULE}._shared_state",
                return_value=SimpleNamespace(packed=torch.empty(2, 5120)),
            ) as shared,
        ):
            for wire in ("bf16", "nvfp4"):
                with (
                    self.subTest(wire=wire),
                    envs.SGLANG_FLASHINFER_RDMA_WIRE_DTYPE.override(wire),
                    envs.SGLANG_FLASHINFER_RDMA_NVFP4_GLOBAL_SCALE.override(0.1),
                ):
                    dispatcher = FlashinferRdmaDispatcher(
                        group, 6, 384, 96, 5120, config, 54
                    )
                    self.assertEqual(shared.call_args.kwargs["wire_dtype"], wire)
                    dispatcher.set_quant_config({})
                    if wire == "nvfp4":
                        expected = tensor(0.1, dtype=torch.float32).item()
                        self.assertEqual(dispatcher.wire_global_scale, expected)
                        self.assertTrue(
                            torch.all(dispatcher.packed_global_scales == expected)
                        )
                    else:
                        self.assertIsNone(dispatcher.wire_global_scale)
                        self.assertIsNone(dispatcher.packed_global_scales)
            self.assertEqual(reduce.call_count, 4)
            for scale in (0.0, -1.0, float("nan"), float("inf"), 1e50, 1e-50):
                with (
                    self.subTest(scale=scale),
                    envs.SGLANG_FLASHINFER_RDMA_WIRE_DTYPE.override("nvfp4"),
                    envs.SGLANG_FLASHINFER_RDMA_NVFP4_GLOBAL_SCALE.override(scale),
                    self.assertRaises(ValueError),
                ):
                    FlashinferRdmaDispatcher(group, 6, 384, 96, 5120, config, 54)
            self.assertEqual(shared.call_count, 2)

    def test_topology_requires_matching_token_and_expert_owners(self):
        for ep in (4, 8, 16, 32):
            ranks = list(range(ep, 2 * ep))
            arguments = dict(
                group_ranks=ranks, group_rank=1, tp_ranks=ranks, ep_ranks=ranks
            )
            _validate_parallel_topology(_parallel(ep, 1), **arguments)
            for name, value in (
                ("attn_dp_size", 1),
                ("moe_tp_size", 2),
                ("moe_ep_rank", 0),
            ):
                parallel = _parallel(ep, 1)
                setattr(parallel, name, value)
                with (
                    self.subTest(ep=ep, name=name),
                    self.assertRaisesRegex(ValueError, name),
                ):
                    _validate_parallel_topology(parallel, **arguments)
            with self.assertRaisesRegex(ValueError, "ordered membership"):
                _validate_parallel_topology(
                    _parallel(ep, 1), **{**arguments, "ep_ranks": ranks[::-1]}
                )

    def test_nvfp4_scale_must_be_shared_finite_and_positive(self):
        dispatcher = object.__new__(FlashinferRdmaDispatcher)
        dispatcher.native_mxfp4 = False
        dispatcher.group = object()
        for scale in (None, torch.ones(2), torch.ones(1, dtype=torch.bfloat16)):
            with (
                self.subTest(scale=scale),
                self.assertRaisesRegex(ValueError, "static FP32"),
            ):
                dispatcher.set_quant_config({"input_global_scale": scale})
        for scale in (0.0, -1.0, float("inf"), float("nan")):
            with (
                self.subTest(scale=scale),
                patch(f"{MODULE}.dist.all_reduce"),
                self.assertRaisesRegex(ValueError, "scales disagree"),
            ):
                dispatcher.set_quant_config({"input_global_scale": torch.tensor(scale)})
        with patch(f"{MODULE}.dist.all_reduce") as reduce:
            dispatcher.set_quant_config({"input_global_scale": torch.tensor(2.0)})
            self.assertEqual(reduce.call_count, 2)
        with patch(f"{MODULE}.dist.all_reduce", side_effect=[None, None]) as reduce:
            reduce.side_effect = lambda tensor, **kwargs: tensor.fill_(
                1.0 if kwargs["op"] == torch.distributed.ReduceOp.MIN else 2.0
            )
            with self.assertRaisesRegex(ValueError, "scales disagree"):
                dispatcher.set_quant_config({"input_global_scale": torch.tensor(1.0)})

    def test_prefill_preserves_routes_and_zeros_partial_output(self):
        dispatcher = object.__new__(FlashinferRdmaDispatcher)
        dispatcher.group = Mock()
        dispatcher.group.size.return_value = 4
        dispatcher.ep_rank, dispatcher.hidden = 1, 16
        dispatcher.state = SimpleNamespace(owner=None)
        dispatcher._pending = None
        hidden = torch.ones((2, 16), dtype=torch.bfloat16)
        ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
        global_hidden = hidden.repeat(4, 1)
        global_ids, global_weights = ids.repeat(4, 1), weights.repeat(4, 1)
        dispatcher.tp_group = Mock()
        dispatcher.tp_group.all_gatherv.return_value = (
            global_hidden,
            global_ids,
            global_weights,
        )
        dispatcher.tp_group.reduce_scatterv.return_value = hidden
        with (
            patch(
                f"{MODULE}.torch.cuda.is_current_stream_capturing", return_value=False
            ),
            patch(f"{MODULE}.get_dp_global_num_tokens", return_value=[2] * 4),
        ):
            output = dispatcher._prefill_dispatch(hidden, ids, weights)
        self.assertIs(output.topk_output.topk_weights, global_weights)
        self.assertIs(output.topk_output.topk_ids, global_ids)
        self.assertEqual(torch.count_nonzero(output.moe_output).item(), 0)
        result = dispatcher.combine(FlashinferCombineInput(output.moe_output))
        self.assertIs(result, hidden)
        dispatcher.tp_group.reduce_scatterv.assert_called_once_with(
            output.moe_output, sizes=[2] * 4
        )
        self.assertIsNone(dispatcher.state.owner)

    def test_shared_workspace_cannot_interleave_or_close_in_flight(self):
        dispatcher = object.__new__(FlashinferRdmaDispatcher)
        dispatcher._pending = None
        dispatcher.state = SimpleNamespace(owner=object(), buffer=Mock())
        with self.assertRaisesRegex(RuntimeError, "must not interleave"):
            dispatcher.dispatch(None, None)
        with self.assertRaisesRegex(RuntimeError, "preceding dispatch"):
            dispatcher.combine(None)
        resources = SimpleNamespace(
            buffers={("flashinfer_rdma_alltoall", 1): dispatcher.state}
        )
        with (
            patch(f"{MODULE}.get_resources", return_value=resources),
            patch(f"{MODULE}.torch.cuda.synchronize"),
        ):
            with self.assertRaisesRegex(RuntimeError, "during dispatch/combine"):
                close_flashinfer_rdma_buffers()
            dispatcher.state.buffer.close.assert_not_called()
            dispatcher.state.owner = None
            close_flashinfer_rdma_buffers()
            dispatcher.state.buffer.close.assert_called_once()
            self.assertEqual(resources.buffers, {})


if __name__ == "__main__":
    unittest.main()
