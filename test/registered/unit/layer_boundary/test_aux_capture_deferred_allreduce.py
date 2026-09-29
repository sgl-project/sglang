import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers import layer_boundary as comm
from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
from sglang.srt.layers.layer_boundary import StageKind
from sglang.srt.layers.layer_boundary import prepare as comm_ops
from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.layers.layer_boundary.ops import identity_output
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.models.bailing_moe import BailingMoEModel
from sglang.srt.models.bailing_moe_v3 import BailingMoELinearModel
from sglang.srt.models.glm4_moe import Glm4MoeModel
from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteModel
from sglang.srt.models.glm5_next import Glm5NextModel
from sglang.srt.models.gpt_oss import GptOssModel
from sglang.srt.models.laguna import LagunaModel
from sglang.srt.models.llama4 import Llama4Model
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.models.qwen3_vl import Qwen3LLMModel
from sglang.srt.models.qwen3_vl_moe import Qwen3MoeLLMModel
from sglang.test.boundary_fixtures import (
    finish_exit,
    identity_input,
    stub_plan,
    stub_stage,
)
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
    Qwen3Model,
    Qwen3LLMModel,
    Llama4Model,
)


def all_reduce(hidden_states):
    return hidden_states * 2


# The group a deferred FFN output owes its sum over.
GROUP = SimpleNamespace(all_reduce=all_reduce)


class DeferringLayer(nn.Module):
    def __init__(self, defer, return_topk=False, stage_api=False):
        super().__init__()
        self.return_topk = return_topk
        self.stage_api = stage_api
        self.layer_communicator = stub_plan()
        self.layer_communicator.norm = None
        self.attn_boundary = stub_stage(self.layer_communicator, StageKind.ATTENTION)
        self.layer_communicator._paths[BatchVariant.ORDINARY] = comm.StageSteps(
            entry=comm.StageEntry(
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
                input_move=identity_input,
                handoff=comm_ops._hand_qkv_hook_its_input,
            ),
            output=comm.StageOutput(
                comm.Layout(frozenset()),
                group=comm.SumGroup.TP,
                leaves_for_next_layer=True,
            ),
            output_move=identity_output,
        )
        self.layer_communicator.output._ffn_sum_moves_to_next_layer = (
            lambda batch, steps, **_: defer
        )
        self.layer_communicator.terminal = False
        self.layer_communicator._paths[BatchVariant.SEQUENCE_PARALLEL] = None
        self.layer_communicator._paths[BatchVariant.INPUT_SCATTERED] = None
        self.layer_communicator._paths[BatchVariant.CONTEXT_PARALLEL] = None
        self.layer_communicator.output.ffn_reduction_group = lambda forward_batch: GROUP
        self.layer_communicator.output._ffn_leaves_sum_to_reduce_scatter = (
            lambda batch, dp_step: False
        )
        self.layer_communicator.output._complete_ffn_output_now = (
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
        stream = forward_batch.residual_stream if self.stage_api else None
        if stream is not None:
            residual = stream
        if isinstance(residual, ResidualStream):
            hidden_states, residual = residual.input(hidden_states)
        hidden_states = comm.reduce_output(hidden_states)
        if residual is None:
            residual = hidden_states.clone()
        else:
            # later norms mutate the residual; captured snapshots must stay intact
            residual.add_(hidden_states)
        capture = kwargs.get("capture_output")
        if capture is not None:
            capture(residual.clone())
        with self.layer_communicator.output.ffn_exit(
            forward_batch, stream=ResidualStream()
        ) as ffn_exit:
            partial = torch.full_like(hidden_states, 0.5)
        if stream is not None:
            stream.write(residual)
            residual = stream
        hidden_states, residual = finish_exit(ffn_exit, partial, residual)
        if self.stage_api:
            return (hidden_states, None) if self.return_topk else hidden_states
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
    model.config = SimpleNamespace(mhc=False, num_hidden_layers=NUM_LAYERS)
    if model_cls is BailingMoELinearModel:
        # Bailing-v3 captures after the layer; other models use boundary indices
        model.layers_to_capture = [0, 1, 2] if capture else []
    model.layers = nn.ModuleList(
        DeferringLayer(
            defer and i < NUM_LAYERS - 1,
            return_topk=model_cls is Glm5NextModel,
            stage_api=model_cls
            in (
                BailingMoEModel,
                BailingMoELinearModel,
                GptOssModel,
                Glm4MoeModel,
                Glm4MoeLiteModel,
                LagunaModel,
                Qwen3MoeLLMModel,
                Qwen3Model,
                Qwen3LLMModel,
                Llama4Model,
                Glm5NextModel,
            ),
        )
        for i in range(NUM_LAYERS)
    )
    model.norm = SumNorm()
    return model


class TestAuxCaptureDeferredAllreduce(CustomTestCase):
    def test_skipping_an_empty_capture_does_not_complete_its_output(self):
        boundary = stub_plan()
        boundary.norm = None
        partial = torch.empty(0, 4)
        batch = SimpleNamespace(residual_stream=ResidualStream(partial))
        owed = batch.residual_stream.leave(
            comm.UnreducedOutput(partial, group=GROUP), comm.ADD
        )
        with patch.object(GROUP, "all_reduce") as reduce:
            hidden, captured = stub_stage(boundary, StageKind.ATTENTION).capture_output(
                owed, batch, skip_empty=True
            )
        self.assertIs(hidden, owed)
        self.assertIsNone(captured)
        reduce.assert_not_called()

    def test_snapshot_without_a_residual_does_not_alias_the_main_output(self):
        hidden = torch.ones(2, 4)
        captured = ResidualStream().snapshot(hidden)
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

    def test_glm_capture_owns_contracted_and_multi_rank_gathered_outputs(self):
        inputs = torch.ones(2, 4)
        original_capture = AuxHiddenStateList.capture
        for mhc in (False, True):
            for size in (1, 2):
                with self.subTest(mhc=mhc, size=size):
                    ownership = []

                    def capture(collector, value, *, owned=False):
                        ownership.append(owned)
                        original_capture(collector, value, owned=owned)

                    group = SimpleNamespace(
                        world_size=size,
                        all_gather=lambda value, dim: (
                            value if size == 1 else torch.cat([value] * size, dim=dim)
                        ),
                    )
                    model = build_model(Glm5NextModel, defer=True, capture=True)
                    model.enable_a2a_moe = True
                    model.config = SimpleNamespace(mhc=mhc, hc_mult=2)
                    batch = SimpleNamespace(
                        can_run_tbo=False,
                        forward_mode=ForwardMode.DECODE,
                        capture_hidden_mode=SimpleNamespace(need_capture=lambda: True),
                    )
                    with (
                        patch.object(AuxHiddenStateList, "capture", capture),
                        patch(
                            "sglang.srt.models.glm5_next.get_parallel",
                            return_value=SimpleNamespace(attn_tp_group=group),
                        ),
                    ):
                        _, snapshots = model(
                            None, None, batch, input_embeds=inputs.clone()
                        )
                    self.assertEqual(ownership, [mhc or size > 1] * 3)
                    for boundary, value in enumerate(snapshots, 1):
                        torch.testing.assert_close(
                            value,
                            torch.full((2 * size, 2 if mhc else 4), 1.0 + boundary),
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

    def test_llama4_split_prefill_preserves_the_stream_between_segments(self):
        from sglang.srt.models.llama4 import Llama4ForCausalLM, Llama4Model

        model = build_model(Llama4Model, defer=True, capture=False)
        lm = SimpleNamespace(
            model=model,
            lm_head=None,
            logits_processor=lambda ids, hidden, head, fb: hidden,
        )
        fb = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        inputs = torch.ones(2, 4)
        result = Llama4ForCausalLM.forward_split_prefill(
            lm, None, None, fb, (0, 2), inputs.clone()
        )
        self.assertIsNone(result)
        self.assertIsNotNone(fb.residual_stream)
        result = Llama4ForCausalLM.forward_split_prefill(
            lm, None, None, fb, (2, NUM_LAYERS)
        )
        torch.testing.assert_close(result, inputs + NUM_LAYERS)
        self.assertIsNone(fb.residual_stream)


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
            if model_cls is Llama4Model:
                continue  # Llama4 has no pipeline model entry.
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
        comm_instance = stub_plan()
        comm_instance.norm = None
        comm_instance._batch_steps = lambda batch: SimpleNamespace(
            entry=SimpleNamespace(input_sum=None)
        )
        batch = SimpleNamespace(residual_stream=None)
        streams = torch.randn(2, 4, 3)
        from dataclasses import replace

        from sglang.srt.layers.layer_boundary import declare_ffn

        stage = stub_stage(comm_instance, StageKind.ATTENTION)
        stage.declaration = replace(
            stage.declaration,
            previous=declare_ffn(update=SimpleNamespace(at_producer=True)),
        )
        hidden = stage.from_pp(PPProxyTensors({"hidden_states": streams}), batch)
        residual = batch.residual_stream
        self.assertIs(hidden, streams)
        self.assertIsNone(residual.pending)
        self.assertIs(residual.residual, hidden)

    def test_optional_residual_and_declared_partial_keep_the_wire_values(self):
        comm_instance = stub_plan()
        comm_instance.norm = None
        comm_instance._batch_steps = lambda batch: SimpleNamespace(
            entry=SimpleNamespace(input_sum=None)
        )
        batch = SimpleNamespace(residual_stream=None)
        partial = torch.randn(2, 4)
        prior = torch.randn_like(partial)
        hidden = stub_stage(comm_instance, StageKind.ATTENTION).from_pp(
            PPProxyTensors({"hidden_states": partial, "residual": prior}), batch
        )
        residual = batch.residual_stream
        self.assertIs(hidden, partial)
        self.assertIs(residual.residual, prior)
        self.assertIs(residual.pending.value, partial)
        missing = PPProxyTensors({"hidden_states": partial})
        with self.assertRaises(KeyError):
            stub_stage(comm_instance, StageKind.ATTENTION).from_pp(missing, batch)
        hidden = stub_stage(comm_instance, StageKind.ATTENTION).from_pp(
            missing, batch, allow_missing_residual=True
        )
        residual = batch.residual_stream
        self.assertIs(hidden, partial)
        self.assertIsNone(residual.pending)
        self.assertIs(residual.residual, hidden)


if __name__ == "__main__":
    unittest.main()
