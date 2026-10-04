"""Numerical and interface coverage for boundary/graph integration regressions."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import test_declared_decoder_boundary as fixture
import torch

from sglang.srt.layers.layer_boundary import (
    PLAIN_ADD,
    declare_attn,
    declare_ffn,
)
from sglang.srt.layers.layer_boundary import exit as exits
from sglang.srt.layers.layer_boundary import (
    make_stages,
)
from sglang.srt.layers.layer_boundary import stage as stages
from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.layers.layer_boundary.fusions.cutedsl import CuteDSLFusion
from sglang.srt.layers.layer_boundary.layout import SumGroup
from sglang.srt.layers.layer_boundary.prepare import _dispatch_by_update
from sglang.srt.layers.layer_boundary.residual import batch
from sglang.srt.layers.layer_boundary.residual.add_norm import REPLACE_AT_EXIT
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import patch_communicator

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestBoundaryIntegrations(unittest.TestCase):
    def test_dp_mixer_keeps_attention_group_reduction(self):
        parallel = fixture.parallel_of(attn_dp=2, attn_tp=2)
        rows = fixture.comm.Layout(frozenset())
        produced = fixture.comm.OutputContract(
            rows, group=SumGroup.ATTN_TP, may_defer_to_next=True, update=PLAIN_ADD
        )
        plan = SimpleNamespace(path_for=lambda _: SimpleNamespace(output=produced))
        boundary = exits.ExitPolicy(plan)
        boundary._sum_deferral_allowed = lambda _: True
        summed = Mock(side_effect=lambda value, *args, **kwargs: value * 2)
        with (
            fixture.planning(parallel),
            patch.object(exits, "is_dp_attention_enabled", return_value=True),
            patch.object(exits, "sum_output", summed),
        ):
            stream = ResidualStream(torch.zeros(3, 4))
            result = exits.MixerExit(boundary, None, stream=stream)
            value = torch.ones(3, 4)
            torch.testing.assert_close(result.finish(value), value * 2)
            self.assertEqual(summed.call_args.args[1], SumGroup.ATTN_TP)
            self.assertIsNone(stream.pending.owed)

    def test_pipeline_preserves_declared_sum_for_one_receiver_completion(self):
        parallel = fixture.parallel_of(
            attn_dp=1, attn_tp=2, enable_attn_tp_input_scattered=True
        )
        with fixture.planning(parallel):
            attn, _ = make_stages(
                (declare_attn(), fixture.Norm()),
                (declare_ffn(), fixture.Norm()),
                previous=declare_ffn(),
            )
        attn.plan.path_for = lambda _: attn.plan.paths[BatchVariant.INPUT_SCATTERED]
        fb = SimpleNamespace(residual_stream=ResidualStream(torch.full((4, 4), 3.0)))
        hidden = fb.residual_stream.record(
            torch.ones(4, 4), PLAIN_ADD, declared_sum=SumGroup.TP
        )
        wire = batch.to_pp(hidden, fb)
        self.assertIsNone(fb.residual_stream)
        torch.testing.assert_close(wire["hidden_states"], torch.ones(4, 4))
        hidden = attn.from_pp(wire, fb)
        reductions = []

        def reduce_scatter(value, residual):
            reductions.append("RS")
            return value.chunk(2)[0] * 2, residual.chunk(2)[0]

        with (
            fixture.planning(parallel),
            patch_communicator("tp_reduce_scatter", reduce_scatter),
        ):
            # Rebind the path with the numerical collective stand-in.
            attn, _ = make_stages(
                (declare_attn(), fixture.Norm()),
                (declare_ffn(), fixture.Norm()),
                previous=declare_ffn(),
            )
            attn.plan.path_for = lambda _: attn.plan.paths[BatchVariant.INPUT_SCATTERED]
            attn.plan.qkv_latent_func = None
            entry = attn.entry(fb)
            hidden, residual = fb.residual_stream.input(hidden)
            output, _ = entry.prepare(
                hidden,
                residual,
                fb,
                fixture.Norm(),
                pending=fb.residual_stream.pending,
                update=PLAIN_ADD,
            )
        self.assertEqual(reductions, ["RS"])
        torch.testing.assert_close(output, torch.full((2, 4), 10.0))

    def test_terminal_finalize_requires_explicit_final_consumer(self):
        parallel = fixture.parallel_of(attn_dp=1, attn_tp=2)
        fb = SimpleNamespace(
            input_ids=torch.zeros(2),
            forward_mode=ForwardMode.DECODE,
            dp_padding_mode=SimpleNamespace(is_max_len=lambda: False),
        )
        for accepted in (False, True):
            fusion = CuteDSLFusion()
            fusion.install(
                SimpleNamespace(), defers_finalize=True, terminal_finalize=accepted
            )
            with (
                fixture.planning(parallel),
                patch.object(fusion, "_finalize_eligible", return_value=True),
                patch(
                    "sglang.srt.layers.layer_boundary.fusions.cutedsl.get_parallel",
                    return_value=parallel,
                ),
            ):
                _, ffn = make_stages(
                    (declare_attn(), fixture.Norm()),
                    (declare_ffn(sparse=True), fixture.Norm(), {"fusions": fusion}),
                    terminal=True,
                )
                fb.residual_stream = ResidualStream(torch.ones(2, 4))
                with patch.object(
                    ffn.plan.output, "_dp_reduce_scatter_step", return_value=None
                ):
                    decision = ffn.exit(fb)
                self.assertEqual(decision.defer_moe_finalize, accepted)

    def test_absent_residual_uses_non_partial_update_path(self):
        plain = Mock(side_effect=AssertionError("cannot add a missing residual"))
        generic = Mock(return_value=("input", None))
        self.assertEqual(
            _dispatch_by_update(
                None, None, None, None, paths={True: plain, False: generic}
            ),
            ("input", None),
        )
        plain.assert_not_called()

    def test_lora_or_shared_tp1_defers_only_when_fusion_is_eligible(self):
        group = object()
        residual = torch.ones(2, 4)
        fb = SimpleNamespace(
            input_ids=torch.zeros(2), residual_stream=ResidualStream(residual)
        )
        for lora, shared in ((True, False), (False, True), (False, False)):
            for enabled in (False, True):
                with (
                    patch.object(
                        exits.envs.SGLANG_SHARED_EXPERT_TP1, "get", return_value=shared
                    ),
                    patch.object(exits, "_ffn_has_tokens", return_value=True),
                    patch.object(
                        exits, "post_experts_sum_is_one_all_reduce", return_value=False
                    ),
                    patch.object(
                        exits,
                        "get_lora",
                        return_value=SimpleNamespace(enable_lora=lora),
                    ),
                    patch.object(
                        exits,
                        "get_moe_a2a_backend",
                        return_value=SimpleNamespace(is_none=lambda: True),
                    ),
                    patch.object(
                        exits,
                        "get_exec",
                        return_value=SimpleNamespace(
                            comm=SimpleNamespace(enable_quant_communications=False)
                        ),
                    ),
                    patch.object(
                        exits,
                        "get_parallel",
                        return_value=SimpleNamespace(tp_group=group),
                    ),
                    patch.object(
                        exits, "post_experts_reduction_group", return_value=group
                    ),
                    patch.object(
                        exits, "flashinfer_ar_fusion_applies", return_value=enabled
                    ),
                    patch.object(exits, "aiter_ar_fusion_applies", return_value=False),
                ):
                    self.assertEqual(
                        exits._batch_allows_deferred_sum(fb),
                        enabled and (lora or shared),
                    )

    def test_ffn_writing_the_next_stream_hands_on_a_complete_output(self):
        # It has each part's sum completed (sum_part) before building the
        # stream from the parts, so its exit owes no sum and never defers one.
        for attn_dp in (1, 2):
            with (
                self.subTest(attn_dp=attn_dp),
                fixture.planning(fixture.parallel_of(attn_dp=attn_dp, attn_tp=2)),
            ):
                for sparse in (False, True):
                    _, ffn = make_stages(
                        (declare_attn(), fixture.Norm()),
                        (
                            declare_ffn(sparse=sparse, update=REPLACE_AT_EXIT),
                            fixture.Norm(),
                        ),
                        previous=declare_ffn(sparse=sparse, update=REPLACE_AT_EXIT),
                    )
                    for variant, path in ffn.plan.paths.items():
                        self.assertIsNone(path.output.group, (sparse, variant))
                        self.assertFalse(path.output.may_defer_to_next)

    def test_only_an_ffn_writing_the_next_stream_sums_its_parts(self):
        summed = Mock(side_effect=lambda value, *args, **kwargs: value * 2)
        fb = SimpleNamespace()
        with (
            fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=2)),
            patch.object(stages, "sum_output", summed),
        ):
            for update in (PLAIN_ADD, REPLACE_AT_EXIT):
                _, ffn = make_stages(
                    (declare_attn(), fixture.Norm()),
                    (declare_ffn(update=update), fixture.Norm()),
                )
                if update is PLAIN_ADD:
                    with self.assertRaises(RuntimeError):
                        ffn.sum_part(torch.ones(2, 4), fb, SumGroup.TP)
                    continue
                part = ffn.sum_part(torch.ones(2, 4), fb, SumGroup.MOE_OUTPUT)
        torch.testing.assert_close(part, torch.full((2, 4), 2.0))
        summed.assert_called_once()
        self.assertIs(summed.call_args.args[1], SumGroup.MOE_OUTPUT)
        self.assertFalse(summed.call_args.kwargs["may_quantize"])

    def test_dense_decoder_forwards_capture_callback(self):
        from sglang.srt.models.llama4 import Llama4DecoderLayer
        from sglang.srt.models.qwen3 import Qwen3DecoderLayer

        for cls in (Qwen3DecoderLayer, Llama4DecoderLayer):
            captures = []

            def prepare(hidden, fb, capture=None, **kwargs):
                if capture is not None:
                    capture(hidden)
                return hidden

            layer = SimpleNamespace(
                attn_boundary=SimpleNamespace(
                    prepare=prepare, finish=lambda hidden, fb: hidden
                ),
                ffn_boundary=SimpleNamespace(
                    prepare=lambda hidden, fb, **kw: hidden,
                    finish=lambda hidden, fb: hidden,
                ),
                self_attn=lambda **kw: kw["hidden_states"],
                mlp=lambda hidden, **kw: hidden,
                feed_forward=lambda hidden, fb: hidden,
            )
            value = torch.ones(2, 4)
            cls.forward(layer, None, value, None, capture_output=captures.append)
            self.assertEqual(len(captures), 1)
            self.assertIs(captures[0], value)


if __name__ == "__main__":
    unittest.main()
