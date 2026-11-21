import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.flashinfer import FlashinferCombineInput
from sglang.srt.layers.moe.token_dispatcher.flashinfer_rdma import (
    FlashinferRdmaDispatcher,
    _shared_state,
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
            envs.SGLANG_FLASHINFER_RDMA_DISPATCH_DTYPE.override(None),
            patch(f"{MODULE}.get_parallel", return_value=parallel),
            patch(
                f"{MODULE}.get_spec",
                return_value=SimpleNamespace(speculative_adaptive=False),
            ),
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
                    self.assertEqual(shared.call_args.kwargs["dispatch_dtype"], wire)
                    self.assertEqual(shared.call_args.kwargs["combine_dtype"], wire)
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

    def test_dispatch_layout_and_combine_codec_are_independent(self):
        module = "flashinfer.comm.rdma_moe_alltoall"
        buffer = Mock()
        fake = SimpleNamespace(
            RdmaMoeAlltoAll=buffer,
            RdmaMoeAlltoAllConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            RdmaDispatchLayout=lambda data, scales: SimpleNamespace(
                data_bytes=data, scale_bytes=scales
            ),
        )
        resources = SimpleNamespace(buffers={})
        group = Mock()
        group.size.return_value = 4
        cpu = torch.device("cpu")
        with (
            patch.dict("sys.modules", {module: fake}),
            patch(f"{MODULE}.get_resources", return_value=resources),
            patch(f"{MODULE}.torch.cuda.current_device", return_value=0),
            patch(f"{MODULE}.torch.device", return_value=cpu),
        ):
            states = []
            for dispatch in ("bf16", "nvfp4", "mxfp8"):
                for combine in ("bf16", "nvfp4"):
                    state = _shared_state(
                        group,
                        hidden=64,
                        num_experts=8,
                        topk=2,
                        capacity=8,
                        dispatch_dtype=dispatch,
                        combine_dtype=combine,
                    )
                    states.append(state)
                    config = buffer.call_args.kwargs["config"]
                    self.assertEqual(config.communication_dtype, combine)
                    if dispatch == "mxfp8":
                        self.assertEqual(config.dispatch_layout.data_bytes, 64)
                        self.assertEqual(config.dispatch_layout.scale_bytes, 2)
                        self.assertEqual(state.packed.shape, (64, 64))
                        self.assertEqual(state.packed_sf.shape, (64, 2))
                        self.assertEqual(state.packed.dtype, torch.uint8)
                    elif dispatch != combine:
                        self.assertEqual(config.dispatch_dtype, dispatch)
            self.assertEqual(len({id(s) for s in states}), 6)
            self.assertIs(
                _shared_state(
                    group,
                    hidden=64,
                    num_experts=8,
                    topk=2,
                    capacity=8,
                    dispatch_dtype="mxfp8",
                    combine_dtype="nvfp4",
                ),
                states[-1],
            )
            self.assertEqual(buffer.call_count, 6)

    def test_live_rows_bound_and_metadata_fallback(self):
        dispatcher = object.__new__(FlashinferRdmaDispatcher)
        dispatcher.group = Mock()
        dispatcher.topk, dispatcher.ep_rank = 6, 0
        for ep in (4, 8, 16, 32):
            dispatcher.group.size.return_value = ep
            full = ep * 54 * dispatcher.topk
            dispatcher.state = SimpleNamespace(packed=torch.empty(full, 1))
            dispatcher.rows_follow_batch = True
            cases = [
                ([6] * ep, 6, ep * 6 * dispatcher.topk),
                ([0, 3] + [1] * (ep - 2), 0, (ep + 1) * dispatcher.topk),
                ([0] * ep, 0, dispatcher.topk),
                ([54] * ep, 54, full),
                (None, 6, full),
                ([6] * (ep - 1), 6, full),
                ([6] * ep, 12, full),
            ]
            for sizes, tokens, expected in cases:
                with (
                    self.subTest(ep=ep, sizes=sizes, tokens=tokens),
                    patch(f"{MODULE}.get_dp_global_num_tokens", return_value=sizes),
                ):
                    self.assertEqual(dispatcher._live_rows(tokens), expected)
            dispatcher.rows_follow_batch = False
            with patch(f"{MODULE}.get_dp_global_num_tokens", return_value=[6] * ep):
                self.assertEqual(dispatcher._live_rows(6), full)

    def test_live_rows_reach_all_codecs_and_combine(self):
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        hidden, tokens, rows, full = 64, 2, 12, 64
        for native, codec in (
            (False, "nvfp4"),
            (True, "bf16"),
            (True, "nvfp4"),
            (True, "mxfp8"),
        ):
            with self.subTest(native=native, codec=codec):
                width = hidden // 2 if codec == "nvfp4" else hidden
                sf_width = {"bf16": 0, "nvfp4": 4, "mxfp8": 2}[codec]
                dispatcher = object.__new__(FlashinferRdmaDispatcher)
                dispatcher.hidden, dispatcher.topk = hidden, 2
                dispatcher.capacity, dispatcher.ep_rank, dispatcher.num_experts = (
                    8,
                    0,
                    8,
                )
                dispatcher.native_mxfp4, dispatcher.dispatch_dtype = native, codec
                dispatcher.combine_dtype = "nvfp4"
                dispatcher.wire_global_scale = 1.0
                dispatcher.quant_config = {"input_global_scale": torch.ones(1)}
                dispatcher.rows_follow_batch = True
                dispatcher.group = Mock()
                dispatcher.group.size.return_value = 4
                dispatcher._pending = None
                counts = torch.tensor([1, 0], dtype=torch.int32)
                state = SimpleNamespace(
                    owner=None,
                    buffer=Mock(),
                    outputs=(),
                    packed=torch.zeros(
                        full,
                        width,
                        dtype=torch.bfloat16 if codec == "bf16" else torch.uint8,
                    ),
                    packed_sf=torch.zeros(full, sf_width, dtype=torch.uint8),
                    ids=torch.zeros(full, 1, dtype=torch.int32),
                    weights=torch.ones(full, 1),
                    slot_map=torch.zeros(full, dtype=torch.int32),
                    compact_output=torch.zeros(full, hidden, dtype=torch.bfloat16),
                    expert_output=torch.zeros(2, 32, hidden, dtype=torch.bfloat16),
                    combine_scales=torch.ones(2, 32, 1),
                )
                dispatcher.state = state
                dispatcher.packed_global_scales = torch.ones(full, 1)
                state.buffer.dispatch.return_value = (
                    None,
                    None,
                    counts,
                    object(),
                    None,
                )
                state.buffer.dequantize_nvfp4_to_bf16.side_effect = (
                    lambda x, scales, sf: torch.zeros(
                        x.shape[0], hidden, dtype=torch.bfloat16
                    )
                )
                state.buffer.combine.return_value = (torch.zeros(tokens, hidden), None)
                quantize = Mock(
                    return_value=(
                        torch.zeros(tokens, width, dtype=torch.uint8),
                        torch.zeros(tokens, sf_width, dtype=torch.uint8),
                    )
                )
                interleave = Mock(side_effect=lambda x: x)
                kernels = SimpleNamespace(
                    pack_expert_rows=Mock(), unpack_expert_rows=Mock()
                )
                topk = StandardTopKOutput(
                    torch.ones(tokens, 2),
                    torch.zeros(tokens, 2, dtype=torch.int32),
                    None,
                )
                with (
                    patch(f"{MODULE}.get_is_extend_in_batch", return_value=False),
                    patch(
                        f"{MODULE}.get_dp_global_num_tokens", return_value=[2, 0, 1, 3]
                    ),
                    patch.dict(
                        "sys.modules",
                        {
                            "flashinfer": SimpleNamespace(
                                mxfp8_quantize=quantize,
                                nvfp4_block_scale_interleave=interleave,
                            ),
                            "sglang.srt.layers.quantization.fp4_utils": SimpleNamespace(
                                fp4_quantize=quantize
                            ),
                            "sglang.srt.layers.moe.token_dispatcher.flashinfer_rdma_kernels": kernels,
                        },
                    ),
                ):
                    output = dispatcher.dispatch(
                        torch.zeros(tokens, hidden, dtype=torch.bfloat16), topk
                    )
                    self.assertEqual(output.hidden_states.shape[0], rows)
                    self.assertEqual(output.topk_output.topk_ids.shape[0], rows)
                    self.assertEqual(output.moe_output.shape[0], rows)
                    for tensor in kernels.pack_expert_rows.call_args.args[3:]:
                        self.assertEqual(tensor.shape[0], rows)
                    if native and codec == "nvfp4":
                        for (
                            tensor
                        ) in state.buffer.dequantize_nvfp4_to_bf16.call_args.args:
                            self.assertEqual(tensor.shape[0], rows)
                    if interleave.called:
                        self.assertEqual(interleave.call_args.args[0].shape[0], rows)
                    dispatcher.combine(FlashinferCombineInput(output.moe_output))
                    self.assertIs(kernels.unpack_expert_rows.call_args.args[2], counts)
                    self.assertIsNone(state.owner)

    def test_mxfp8_runner_uses_transmitted_scales_without_requantization(self):
        from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass as runner
        from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
            FlashinferDispatchOutput,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        hidden = torch.zeros((3, 64), dtype=torch.float8_e4m3fn)
        scales = torch.ones(128 * 4, dtype=torch.uint8)
        weights = torch.zeros((1, 8, 32), dtype=torch.uint8)
        weight_scales = torch.ones((1, 8, 4), dtype=torch.uint8)
        quant = runner.FlashInferCutlassMxfp4MoeQuantInfo(
            weights,
            weights,
            weight_scales,
            weight_scales,
            mxfp4_weight_global_scale=torch.ones(1),
        )
        topk = StandardTopKOutput(
            torch.ones(3, 1), torch.zeros((3, 1), dtype=torch.int32), None
        )
        dispatch = FlashinferDispatchOutput(hidden, scales, topk, None, torch.bfloat16)
        config = SimpleNamespace(
            moe_tp_size=1, moe_tp_rank=0, moe_ep_size=4, moe_ep_rank=0
        )
        quantize = Mock(return_value=(hidden, scales))
        compute = Mock()
        with (
            patch.dict(
                "sys.modules", {"flashinfer": SimpleNamespace(mxfp8_quantize=quantize)}
            ),
            patch.object(
                runner,
                "_flashinfer_cutlass_fused_moe",
                return_value=(compute, SimpleNamespace(Swiglu=0, SwigluStep=1)),
            ),
            patch.object(runner, "use_symmetric_memory", return_value=nullcontext()),
            patch.object(
                runner, "get_parallel", return_value=SimpleNamespace(tp_group=None)
            ),
            patch.object(runner, "is_allocation_symmetric", return_value=False),
            patch.object(runner, "_cutlass_moe_tune_max_num_tokens", return_value=4096),
        ):
            result = runner.fused_experts_rdma_to_flashinfer_mxfp4(
                dispatch, quant, config
            )
            self.assertEqual(result.hidden_states.dtype, torch.bfloat16)
            self.assertIs(compute.call_args.kwargs["input"], hidden)
            self.assertIs(compute.call_args.kwargs["input_sf"], scales)
            quantize.assert_not_called()
            for bad_scales in (None, scales[:128], scales.float()):
                with self.assertRaisesRegex(ValueError, "Prequantized MXFP8"):
                    runner.fused_experts_rdma_to_flashinfer_mxfp4(
                        dispatch._replace(hidden_states_scale=bad_scales), quant, config
                    )
            runner.fused_experts_rdma_to_flashinfer_mxfp4(
                dispatch._replace(
                    hidden_states=hidden.to(torch.bfloat16), hidden_states_scale=None
                ),
                quant,
                config,
            )
            quantize.assert_called_once()
            empty = runner.fused_experts_rdma_to_flashinfer_mxfp4(
                dispatch._replace(hidden_states=hidden[:0]), quant, config
            )
            self.assertEqual(empty.hidden_states.shape, (0, 64))
            self.assertEqual(empty.hidden_states.dtype, torch.bfloat16)

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
