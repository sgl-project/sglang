import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.communicator import ops as comm_ops
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.models.bailing_moe import BailingMoEModel
from sglang.srt.models.bailing_moe_v3 import BailingMoELinearModel
from sglang.srt.models.glm4_moe import Glm4MoeModel
from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteModel
from sglang.srt.models.glm5_next import Glm5NextModel
from sglang.srt.models.gpt_oss import GptOssModel
from sglang.srt.models.laguna import LagunaModel
from sglang.srt.models.qwen3_vl_moe import Qwen3MoeLLMModel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

NUM_LAYERS = 4
MODELS = (
    BailingMoEModel,
    BailingMoELinearModel,
    Glm4MoeModel,
    Glm4MoeLiteModel,
    Glm5NextModel,
    GptOssModel,
    LagunaModel,
    Qwen3MoeLLMModel,
)


def all_reduce(hidden_states):
    return hidden_states * 2


# The group a deferred FFN output owes its sum over.
GROUP = SimpleNamespace(all_reduce=all_reduce)


class DeferringLayer(nn.Module):
    def __init__(self, defer, return_topk=False):
        super().__init__()
        self.return_topk = return_topk
        self.layer_communicator = comm.LayerCommunicator.__new__(comm.LayerCommunicator)
        self.layer_communicator._steps = comm.BoundarySteps(
            attention=comm.StageEntry(
                prepare=partial(
                    comm_ops._consumer_step,
                    step=partial(
                        comm_ops._read_input,
                        layer_input=None,
                        enters_stack=False,
                        read=comm.NORM_QUANT_READ,
                        update=comm.ADD,
                    ),
                    carried_fusions=(),
                ),
                input_rows=comm.Layout(frozenset()),
                input_move=comm.CommunicateSimpleFn._trivial,
                handoff=comm_ops._hand_qkv_hook_its_input,
            ),
            ffn=comm.StageEntry(
                prepare=partial(
                    comm_ops._read_input,
                    layer_input=None,
                    enters_stack=False,
                    read=comm.NORM_READ,
                    update=comm.ADD,
                ),
                input_rows=comm.Layout(frozenset()),
            ),
            ffn_output=comm.StageOutput(
                comm.Layout(frozenset()),
                group=comm.SumGroup.TP,
                leaves_for_next_layer=True,
            ),
            ffn_output_move=comm.CommunicateSummableTensorPairFn._trivial,
            ffn_sum_is_movable=True,
        )
        self.layer_communicator._ffn_sum_moves_to_next_layer = lambda batch, **_: defer
        self.layer_communicator.is_last_layer = False
        self.layer_communicator._sp_steps = None
        self.layer_communicator._input_scattered_steps = None
        self.layer_communicator._cp_steps = None
        self.layer_communicator.ffn_reduction_group = lambda forward_batch: GROUP
        self.layer_communicator._ffn_leaves_sum_to_reduce_scatter = (
            lambda batch, dp_step: False
        )
        self.layer_communicator._complete_ffn_output_now = (
            lambda hidden, residual, **_: (all_reduce(hidden), residual)
        )

    def forward(
        self,
        positions=None,
        hidden_states=None,
        forward_batch=None,
        residual=None,
        *args,
        **kwargs,
    ):
        stream = forward_batch.residual_stream
        assert residual is stream
        hidden_states, residual = stream.input(hidden_states)
        hidden_states = comm.reduce_output(hidden_states)
        if residual is None:
            residual = hidden_states.clone()
        else:
            # later norms mutate the residual; captured snapshots must stay intact
            residual.add_(hidden_states)
        stream.write(residual)
        capture = kwargs.get("capture_output")
        if capture is not None:
            capture(residual.clone())
        with self.layer_communicator.ffn_exit(forward_batch) as ffn_exit:
            partial = torch.full_like(hidden_states, 0.5)
        hidden_states, residual = ffn_exit.finish(partial, stream)
        if self.return_topk:
            return hidden_states, residual, None
        return hidden_states, residual


class SumNorm(nn.Module):
    def forward(self, hidden_states, residual=None, **kwargs):
        if residual is None:
            return hidden_states
        return hidden_states + residual, hidden_states + residual


def build_model(model_cls, *, defer, capture):
    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    model.start_layer = 0
    model.end_layer = NUM_LAYERS
    model.first_k_dense_replace = 0
    model.layers_to_capture = [1, 2, 3] if capture else []
    model.capture_aux_hidden_states = capture
    model.use_hf_deepstack_order = False
    model.dflash_capture = capture
    model.enable_a2a_moe = False
    model.config = SimpleNamespace(mhc=False)
    if model_cls is BailingMoELinearModel:
        # Bailing-v3 captures after the layer; other models use boundary indices
        model.layers_to_capture = [0, 1, 2] if capture else []
    model.layers = nn.ModuleList(
        DeferringLayer(
            defer and i < NUM_LAYERS - 1, return_topk=model_cls is Glm5NextModel
        )
        for i in range(NUM_LAYERS)
    )
    model.norm = SumNorm()
    return model


class TestAuxCaptureDeferredAllreduce(CustomTestCase):
    def test_skipping_an_empty_capture_does_not_complete_its_output(self):
        boundary = comm.LayerCommunicator.__new__(comm.LayerCommunicator)
        partial = torch.empty(0, 4)
        owed = comm.UnreducedOutput(partial, group=GROUP)
        with patch.object(GROUP, "all_reduce") as reduce:
            hidden, captured = boundary.capture_output(owed, partial, skip_empty=True)
        self.assertIs(hidden, owed)
        self.assertIsNone(captured)
        reduce.assert_not_called()

    def test_snapshot_without_a_residual_does_not_alias_the_main_output(self):
        boundary = comm.LayerCommunicator.__new__(comm.LayerCommunicator)
        hidden = torch.ones(2, 4)
        captured = boundary.snapshot(hidden, None)
        hidden.zero_()
        torch.testing.assert_close(captured, torch.ones_like(hidden))

    def test_capture_matches_eager_reduction(self):
        inputs = torch.tensor([[0.25, -0.5, 0.75, 1.0], [1.5, 2.0, -1.0, 0.0]])
        batch = SimpleNamespace(
            can_run_tbo=False,
            forward_mode=ForwardMode.DECODE,
            capture_hidden_mode=SimpleNamespace(need_capture=lambda: True),
        )
        for model_cls in MODELS:
            for defer in (False, True):
                for capture in (False, True):
                    with (
                        self.subTest(
                            model=model_cls.__name__, defer=defer, capture=capture
                        ),
                        patch.object(
                            GROUP, "all_reduce", side_effect=all_reduce
                        ) as reduce,
                    ):
                        model = build_model(model_cls, defer=defer, capture=capture)
                        result = model(
                            input_ids=None,
                            positions=None,
                            forward_batch=batch,
                            input_embeds=inputs.clone(),
                        )
                        output, snapshots = result if capture else (result, [])
                        torch.testing.assert_close(output, inputs + NUM_LAYERS)
                        self.assertEqual(len(snapshots), 3 if capture else 0)
                        for boundary, snapshot in enumerate(snapshots, 1):
                            torch.testing.assert_close(snapshot, inputs + boundary)
                        self.assertEqual(
                            reduce.call_count, NUM_LAYERS - 1 if defer else 0
                        )

    def test_terminal_capture_uses_the_final_norm_boundary(self):
        inputs = torch.ones(2, 4)
        batch = SimpleNamespace(
            can_run_tbo=False,
            forward_mode=ForwardMode.DECODE,
            capture_hidden_mode=SimpleNamespace(need_capture=lambda: True),
        )
        for cls in (GptOssModel, LagunaModel, BailingMoELinearModel):
            with self.subTest(model=cls.__name__):
                model = build_model(cls, defer=True, capture=True)
                model.layers_to_capture = [
                    NUM_LAYERS - 1 if cls is BailingMoELinearModel else NUM_LAYERS
                ]
                result, snapshots = model(
                    input_ids=None,
                    positions=None,
                    forward_batch=batch,
                    input_embeds=inputs.clone(),
                )
                self.assertEqual(len(snapshots), 1)
                torch.testing.assert_close(snapshots[0], inputs + NUM_LAYERS)
                torch.testing.assert_close(result, inputs + NUM_LAYERS)


class TestPipelineResidualReception(CustomTestCase):
    def test_models_keep_the_received_residual_contribution(self):
        inputs = torch.zeros(2, 4)
        initial_residual = torch.full_like(inputs, 0.25)
        batch = SimpleNamespace(
            can_run_tbo=False,
            forward_mode=ForwardMode.DECODE,
            capture_hidden_mode=SimpleNamespace(need_capture=lambda: False),
        )
        for model_cls in MODELS:
            with self.subTest(model=model_cls.__name__):
                model = build_model(model_cls, defer=True, capture=False)
                model.pp_group.is_first_rank = False
                result = model(
                    input_ids=None,
                    positions=None,
                    forward_batch=batch,
                    pp_proxy_tensors=PPProxyTensors(
                        {
                            "hidden_states": inputs.clone(),
                            "residual": initial_residual.clone(),
                        }
                    ),
                )
                torch.testing.assert_close(result, initial_residual + NUM_LAYERS)

    def test_bailing_capture_does_not_include_a_previous_pipeline_ranks_layer(self):
        model = build_model(BailingMoELinearModel, defer=True, capture=True)
        model.pp_group.is_first_rank = False
        model.start_layer = 2
        model.layers_to_capture = [model.start_layer - 1]
        inputs = torch.zeros(2, 4)
        residual = torch.full_like(inputs, 0.25)
        batch = SimpleNamespace(
            can_run_tbo=False,
            forward_mode=ForwardMode.DECODE,
            capture_hidden_mode=SimpleNamespace(need_capture=lambda: True),
        )
        result = model(
            None,
            None,
            batch,
            pp_proxy_tensors=PPProxyTensors(
                {"hidden_states": inputs, "residual": residual}
            ),
        )
        self.assertIsInstance(result, torch.Tensor)
        torch.testing.assert_close(result, torch.full_like(inputs, 2.25))

    def test_written_streams_do_not_read_a_separate_residual(self):
        comm_instance = comm.LayerCommunicator.__new__(comm.LayerCommunicator)
        comm_instance._batch_steps = lambda fb: SimpleNamespace(
            attention=SimpleNamespace(input_sum=None), ffn=None
        )
        comm_instance._residual = SimpleNamespace(
            ffn_update=SimpleNamespace(at_producer=True)
        )
        streams = torch.randn(2, 4, 3)
        hidden, residual = comm_instance.from_pp(
            PPProxyTensors({"hidden_states": streams}),
            SimpleNamespace(residual_stream=None),
        )
        self.assertIs(hidden, streams)
        self.assertIsNone(residual.pending)
        self.assertIs(residual.residual, hidden)

    def test_optional_residual_and_declared_partial_keep_the_wire_values(self):
        comm_instance = comm.LayerCommunicator.__new__(comm.LayerCommunicator)
        comm_instance._batch_steps = lambda fb: SimpleNamespace(
            attention=SimpleNamespace(input_sum=None), ffn=None
        )
        partial = torch.randn(2, 4)
        prior = torch.randn_like(partial)
        hidden, residual = comm_instance.from_pp(
            PPProxyTensors({"hidden_states": partial, "residual": prior}),
            SimpleNamespace(residual_stream=None),
        )
        self.assertIs(hidden, partial)
        self.assertIs(residual.residual, prior)
        self.assertIs(residual.pending.value, partial)
        missing = PPProxyTensors({"hidden_states": partial})
        with self.assertRaises(KeyError):
            comm_instance.from_pp(missing, SimpleNamespace(residual_stream=None))
        hidden, residual = comm_instance.from_pp(
            missing, SimpleNamespace(residual_stream=None), allow_missing_residual=True
        )
        self.assertIs(hidden, partial)
        self.assertIsNone(residual.pending)
        self.assertIs(residual.residual, hidden)


if __name__ == "__main__":
    unittest.main()
