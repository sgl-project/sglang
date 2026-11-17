"""Numerical and interface coverage for boundary/graph integration regressions."""

import unittest
from functools import partial
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
from sglang.srt.layers.layer_boundary import stage as stages
from sglang.srt.layers.layer_boundary.contracts import (
    BatchVariant,
    ExitFacts,
    StageKind,
)
from sglang.srt.layers.layer_boundary.fusions.cutedsl import CuteDSLFusion
from sglang.srt.layers.layer_boundary.layout import SumGroup
from sglang.srt.layers.layer_boundary.residual import batch
from sglang.srt.layers.layer_boundary.residual.add_norm import REPLACE_AT_EXIT
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.boundary_fixtures import build_stages
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import patch_communicator

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestBoundaryIntegrations(unittest.TestCase):
    def test_dp_mixer_keeps_attention_group_reduction(self):
        parallel = fixture.parallel_of(attn_dp=2, attn_tp=2)
        rows = fixture.comm.Layout(frozenset())
        produced = fixture.comm.OutputContract(
            rows, group=SumGroup.ATTN_TP, may_defer_to_next=True, update=PLAIN_ADD
        )
        path = SimpleNamespace(output=produced)
        plan = SimpleNamespace(finishes_directly=False, path_for=lambda _: path)
        boundary = exits.ExitPolicy(plan)
        summed = Mock(side_effect=lambda value, *args, **kwargs: value * 2)
        with (
            fixture.planning(parallel),
            patch.object(exits, "is_dp_attention_enabled", return_value=True),
            patch.object(exits, "sum_output", summed),
        ):
            # Under attention DP a mixer completes its sum even where it could
            # otherwise carry it to the next attention.
            with patch.object(exits, "_sum_deferral_allowed", return_value=True):
                path.exit = exits.exit_facts(
                    StageKind.ATTENTION, plan, BatchVariant.ORDINARY, path
                )
            stream = ResidualStream(torch.zeros(3, 4))
            result = exits.MixerExit(boundary, None, stream=stream)
            value = torch.ones(3, 4)
            torch.testing.assert_close(result.finish(value), value * 2)
            self.assertEqual(summed.call_args.args[1], SumGroup.ATTN_TP)
            self.assertIsNone(stream.pending.owed)

    def test_a_pipeline_handoff_carries_its_sum_completed(self):
        # The pipeline sends each tensor as one slice per attention-TP rank and
        # gathers the slices back, so the sender completes a declared sum; the
        # next rank's entry takes the value as complete and only slices it.
        calls = []

        def all_reduce(value):
            calls.append("AR")
            return value * 2

        def reduce_scatter(value, residual):
            calls.append("RS")
            return value.chunk(2)[0] * 2, residual.chunk(2)[0]

        def slice_rows(value, residual):
            calls.append("slice")
            return value.chunk(2)[0], residual.chunk(2)[0]

        parallel = fixture.parallel_of(
            attn_dp=1,
            attn_tp=2,
            enable_attn_tp_input_scattered=True,
            tp_group=SimpleNamespace(name="tp", ranks=[0, 1], all_reduce=all_reduce),
        )
        with (
            fixture.planning(parallel),
            patch_communicator("tp_reduce_scatter", reduce_scatter),
            patch_communicator("tp_slice", slice_rows),
        ):
            attn, _ = build_stages(
                (declare_attn(), fixture.Norm()),
                (declare_ffn(), fixture.Norm()),
                previous=declare_ffn(),
            )
            attn.plan.path_for = lambda _: attn.plan.paths[BatchVariant.INPUT_SCATTERED]
            attn.plan.qkv_latent_func = None
            fb = SimpleNamespace(
                residual_stream=ResidualStream(torch.full((4, 4), 3.0))
            )
            hidden = fb.residual_stream.record(
                torch.ones(4, 4), PLAIN_ADD, declared_sum=SumGroup.TP
            )
            wire = batch.to_pp(hidden, fb)
            self.assertIsNone(fb.residual_stream)
            torch.testing.assert_close(wire["hidden_states"], torch.full((4, 4), 2.0))
            hidden = attn.from_pp(wire, fb)
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
        self.assertEqual(calls, ["AR", "slice"])
        # Norm: 2 * (h + r) on the slice, with h the completed sum.
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
                _, ffn = build_stages(
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
                    facts = ExitFacts(
                        single_sum=False, fused_consumer_sum=exits._fused_consumer_sum()
                    )
                    self.assertEqual(
                        exits._batch_allows_deferred_sum(fb, facts),
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
                    _, ffn = build_stages(
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

    def test_an_ffn_that_completes_its_own_sum_owes_none(self):
        # Its compute completes the sum, so the exit neither sums nor defers it,
        # under DP and on an input-scattered batch too.
        for attn_dp, scattered in ((1, False), (2, False), (1, True)):
            parallel = fixture.parallel_of(
                attn_dp=attn_dp,
                attn_tp=2,
                enable_attn_tp_input_scattered=scattered,
            )
            with (
                self.subTest(attn_dp=attn_dp, input_scattered=scattered),
                fixture.planning(parallel),
            ):
                for sparse in (False, True):
                    stages = {
                        complete: build_stages(
                            (declare_attn(), fixture.Norm()),
                            (
                                declare_ffn(sparse=sparse, output_complete=complete),
                                fixture.Norm(),
                            ),
                            previous=declare_ffn(sparse=sparse),
                        )[1]
                        for complete in (False, True)
                    }
                    owed = stages[False].plan.paths[BatchVariant.ORDINARY].output
                    self.assertIsNotNone(owed.group, sparse)
                    if scattered:
                        self.assertIn(
                            BatchVariant.INPUT_SCATTERED, stages[True].plan.paths
                        )
                    for variant, path in stages[True].plan.paths.items():
                        self.assertIsNone(path.output.group, (sparse, variant))
                        self.assertFalse(path.output.may_defer_to_next)
                        self.assertFalse(path.output.may_reduce_scatter)

    def test_only_an_ffn_writing_the_next_stream_sums_its_parts(self):
        summed = Mock(side_effect=lambda value, *args, **kwargs: value * 2)
        fb = SimpleNamespace()
        with (
            fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=2)),
            patch.object(stages, "sum_output", summed),
        ):
            for update in (PLAIN_ADD, REPLACE_AT_EXIT):
                _, ffn = build_stages(
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
                _quant_formats=lambda fb, positions: ("", ""),
            )
            value = torch.ones(2, 4)
            cls.forward(layer, None, value, None, capture_output=captures.append)
            self.assertEqual(len(captures), 1)
            self.assertIs(captures[0], value)

    def test_qwen3_decoder_calls_the_ffn_entry_bound_under_attention_dp(self):
        """Under attention DP the FFN entry reads before its DP gather, so it
        cannot hand on a quantized input: the decoder must call that entry as
        bound, in prefill and in a decode that would otherwise fuse the fp8
        quant, and asking the entry to quantize fails clearly."""
        from sglang.srt.layers.layer_boundary import prepare as prepare_steps
        from sglang.srt.layers.layer_boundary.residual.add_norm import (
            NORM_QUANT_READOUT,
            PLAIN_RESIDUAL_OPS,
        )
        from sglang.srt.layers.quantization import fp8_activation
        from sglang.srt.models import qwen3
        from sglang.test.boundary_fixtures import Norm, make_test_stages

        class Module:
            def __init__(self, fn, **attrs):
                self.fn = fn
                self.__dict__.update(attrs)

            def __call__(self, *args, **kwargs):
                return self.fn(*args, **kwargs)

        exec_ = SimpleNamespace(
            deterministic=SimpleNamespace(
                rl_on_policy_target=None, enable_deterministic_inference=False
            )
        )
        with (
            fixture.planning(fixture.parallel_of(attn_dp=2, attn_tp=1)),
            patch.object(prepare_steps, "dp_gather", lambda hidden, *args: hidden),
            patch.object(prepare_steps, "is_allocation_symmetric", lambda: False),
            patch.object(qwen3, "get_exec", lambda: exec_),
            patch.object(fp8_activation, "get_exec", lambda: exec_),
            patch.object(
                fp8_activation, "get_lora", lambda: SimpleNamespace(enable_lora=False)
            ),
            patch.object(qwen3, "_use_aiter", True),
            patch.object(qwen3, "_is_gfx95", True),
            patch.object(qwen3, "aiter_takes_fp8_group_input", lambda linear: True),
        ):
            ffn = make_test_stages(
                first=True,
                attention_norm=Norm(),
                ffn_norm=Norm(),
                residual=PLAIN_RESIDUAL_OPS._replace(ffn_readout=NORM_QUANT_READOUT),
            ).ffn
            self.assertIs(
                ffn.plan.paths.get(BatchVariant.ORDINARY)
                .entry.prepare.keywords["step"]
                .func,
                prepare_steps._reduce_update_read_dp_gather,
            )
            bf16_linear = SimpleNamespace(
                params_dtype=torch.bfloat16, weight=torch.empty(256, 256), bias=None
            )
            for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
                with self.subTest(mode=mode):
                    attn_calls = []
                    layer = SimpleNamespace(
                        attn_boundary=SimpleNamespace(
                            prepare=lambda hidden, fb, **kw: (
                                attn_calls.append(kw) or hidden
                            ),
                            # The attention's output enters the stream, owed
                            # to the FFN's read as the real finish leaves it.
                            finish=lambda hidden, fb: fb.residual_stream.record(
                                hidden,
                                PLAIN_ADD,
                                declared_sum=ffn.entry(fb).declared_sum,
                            ),
                            reads_quantized=lambda fb: True,
                        ),
                        ffn_boundary=SimpleNamespace(
                            prepare=ffn.prepare,
                            finish=lambda hidden, fb: hidden,
                            reads_quantized=ffn.reads_quantized,
                        ),
                        self_attn=Module(
                            lambda **kw: kw["hidden_states"], qkv_proj=bf16_linear
                        ),
                        mlp=Module(
                            lambda hidden, **kw: hidden, gate_up_proj=bf16_linear
                        ),
                        _decode_quant_formats=None,
                    )
                    layer._quant_formats = partial(
                        qwen3.Qwen3DecoderLayer._quant_formats, layer
                    )
                    fb = SimpleNamespace(
                        forward_mode=mode,
                        attn_cp_metadata=None,
                        residual_stream=ResidualStream(torch.zeros(2, 4)),
                    )
                    out = qwen3.Qwen3DecoderLayer.forward(
                        layer, torch.arange(2), torch.ones(2, 4), fb
                    )
                    # Norm stand-in: 2 * (input + residual), unquantized.
                    torch.testing.assert_close(out, torch.full((2, 4), 2.0))
                    # The attention, whose entry reads on its own rows, still
                    # fuses the quant in decode.
                    self.assertEqual(
                        attn_calls[0].get("quant_format"),
                        "fp8" if mode is ForwardMode.DECODE else None,
                    )
            fb = SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                residual_stream=ResidualStream(torch.zeros(2, 4)),
            )
            owed = fb.residual_stream.record(
                torch.ones(2, 4), PLAIN_ADD, declared_sum=ffn.entry(fb).declared_sum
            )
            with self.assertRaisesRegex(
                NotImplementedError, "_reduce_update_read_dp_gather"
            ):
                ffn.prepare(owed, fb, quant_format="fp8")

    def test_qwen3_decoder_takes_the_previous_layers_owed_output(self):
        """Under attention TP the FFN hands its unfinished sum to the next
        layer's attention read as an opaque OwedOutput, so a Qwen3 layer must
        not read its input before the boundary does: a two-layer stack runs in
        prefill and in a decode that checks the fp8 quant fusion."""
        from sglang.srt.layers.layer_boundary import prepare as prepare_steps
        from sglang.srt.layers.layer_boundary.residual.stream import OwedOutput
        from sglang.srt.layers.quantization import fp8_activation
        from sglang.srt.models import qwen3
        from sglang.test.boundary_fixtures import Norm, make_test_stages

        class Module:
            def __init__(self, **attrs):
                self.__dict__.update(attrs)

            def __call__(self, *args, **kwargs):
                return kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]

        exec_ = SimpleNamespace(
            deterministic=SimpleNamespace(
                rl_on_policy_target=None, enable_deterministic_inference=False
            )
        )
        rows_seen = []
        # One rank of two stands for both: every sum leaves its input as is.
        parallel = fixture.parallel_of(
            attn_dp=1,
            attn_tp=2,
            tp_group=SimpleNamespace(name="tp", ranks=[0, 1], all_reduce=lambda x: x),
            attn_tp_group=SimpleNamespace(
                name="attn_tp", ranks=[0, 1], all_reduce=lambda x: x
            ),
        )
        with (
            fixture.planning(parallel),
            patch.object(
                prepare_steps, "attn_tp_all_reduce", lambda hidden, *a, **kw: hidden
            ),
            patch.object(
                prepare_steps, "tensor_model_parallel_all_reduce", lambda hidden: hidden
            ),
            patch.object(qwen3, "get_exec", lambda: exec_),
            patch.object(fp8_activation, "get_exec", lambda: exec_),
            patch.object(
                fp8_activation, "get_lora", lambda: SimpleNamespace(enable_lora=False)
            ),
            patch.object(qwen3, "_use_aiter", True),
            patch.object(qwen3, "_is_gfx95", True),
            patch.object(qwen3, "aiter_takes_fp8_group_input", lambda linear: True),
            # Keep the read in bf16 so the CPU stand-in norm can serve it.
            patch.object(
                qwen3,
                "aiter_prefers_bf16_input",
                lambda linear, num_tokens: rows_seen.append(num_tokens) or True,
            ),
        ):
            stages = [
                make_test_stages(first=i == 0, attention_norm=Norm(), ffn_norm=Norm())
                for i in range(2)
            ]
            linear = SimpleNamespace(params_dtype=torch.bfloat16)
            layers = []
            for stage in stages:
                layer = SimpleNamespace(
                    attn_boundary=stage.attn,
                    ffn_boundary=stage.ffn,
                    self_attn=Module(qkv_proj=linear),
                    mlp=Module(gate_up_proj=linear),
                    _decode_quant_formats=None,
                )
                layer._quant_formats = partial(
                    qwen3.Qwen3DecoderLayer._quant_formats, layer
                )
                layers.append(layer)
            for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
                with self.subTest(mode=mode):
                    rows_seen.clear()
                    fb = SimpleNamespace(
                        forward_mode=mode,
                        attn_cp_metadata=None,
                        residual_stream=ResidualStream(),
                        # The FFN exit defers its sum only for a batch with tokens.
                        input_ids=torch.zeros(3),
                    )
                    hidden = torch.ones(3, 4)
                    for layer in layers:
                        hidden = qwen3.Qwen3DecoderLayer.forward(
                            layer, torch.arange(3), hidden, fb
                        )
                        self.assertIsInstance(hidden, OwedOutput)
                    # The fusion check sees this forward's token rows.
                    self.assertEqual(
                        rows_seen, [3] * 4 if mode is ForwardMode.DECODE else []
                    )

    @staticmethod
    def _qwen3_decode_quant_formats(
        qkv_proj, gate_up_proj, num_tokens, compiling=False
    ):
        from sglang.srt import batch_invariant_ops
        from sglang.srt.layers.quantization import fp8_activation
        from sglang.srt.models import qwen3

        exec_ = SimpleNamespace(
            deterministic=SimpleNamespace(
                rl_on_policy_target=None, enable_deterministic_inference=False
            )
        )
        boundary = SimpleNamespace(reads_quantized=lambda fb: True)
        layer = SimpleNamespace(
            attn_boundary=boundary,
            ffn_boundary=boundary,
            self_attn=SimpleNamespace(qkv_proj=qkv_proj),
            mlp=SimpleNamespace(gate_up_proj=gate_up_proj),
            _decode_quant_formats=None,
        )
        with (
            patch.object(qwen3, "_use_aiter", True),
            patch.object(qwen3, "_is_gfx95", True),
            patch.object(qwen3, "aiter_takes_fp8_group_input", lambda linear: True),
            patch.object(qwen3, "get_exec", lambda: exec_),
            patch.object(fp8_activation, "get_exec", lambda: exec_),
            patch.object(
                fp8_activation, "get_lora", lambda: SimpleNamespace(enable_lora=False)
            ),
            patch.object(
                batch_invariant_ops, "is_batch_invariant_mode_enabled", lambda: False
            ),
            patch("torch.compiler.is_compiling", lambda: compiling),
        ):
            return qwen3.Qwen3DecoderLayer._quant_formats(
                layer,
                SimpleNamespace(forward_mode=ForwardMode.DECODE),
                torch.arange(num_tokens),
            )

    def test_qwen3_fused_norm_quant_needs_bf16_and_no_compile(self):
        """The fused fp8 norm hands the GEMM a pre-quantized input, for which it
        returns bf16, so an fp16 model keeps the unfused norm; it also stays
        unfused while torch.compile traces, as the fused SiLU-and-mul does."""
        for dtype, compiling, expected in (
            (torch.bfloat16, False, ("fp8", "fp8")),
            (torch.float16, False, ("", "")),
            (torch.bfloat16, True, ("", "")),
        ):
            linear = SimpleNamespace(
                params_dtype=dtype, weight=torch.empty(256, 256), bias=None
            )
            with self.subTest(dtype=dtype, compiling=compiling):
                self.assertEqual(
                    self._qwen3_decode_quant_formats(
                        linear, linear, num_tokens=8, compiling=compiling
                    ),
                    expected,
                )

    def test_qwen3_fused_norm_quant_yields_to_the_small_m_gemm(self):
        """Where the gfx950 small-M GEMM, which quantizes its bf16 input itself,
        beats the pre-quantized GEMM (Qwen3-0.6B qkv_proj up to 16 rows), the
        norm before it stays unfused; the other linear, larger batches and
        SGLANG_ROCM_SMALLM_FP8_BS=0 keep the fusion."""
        qkv_proj, gate_up_proj = (
            SimpleNamespace(
                params_dtype=torch.bfloat16,
                weight=torch.empty(n, 1024, device="meta"),
                bias=None,
            )
            for n in (4096, 6144)
        )
        for num_tokens, smallm_on, expected in (
            (16, "1", ("", "fp8")),
            (17, "1", ("fp8", "fp8")),
            (16, "0", ("fp8", "fp8")),
        ):
            with (
                self.subTest(num_tokens=num_tokens, smallm_on=smallm_on),
                patch.dict("os.environ", {"SGLANG_ROCM_SMALLM_FP8_BS": smallm_on}),
            ):
                self.assertEqual(
                    self._qwen3_decode_quant_formats(
                        qkv_proj, gate_up_proj, num_tokens=num_tokens
                    ),
                    expected,
                )

    def test_quantized_reads_stay_where_the_input_stays_on_its_rows(self):
        """A decode read that is the entry's last step on the stage's rows keeps
        the fused quant, the attention's with or without attention DP; the FFN
        entry under attention DP, which gathers after its read, does not."""
        from sglang.srt.layers.layer_boundary.residual.add_norm import (
            NORM_QUANT_READOUT,
            PLAIN_RESIDUAL_OPS,
        )
        from sglang.test.boundary_fixtures import Norm, make_test_stages

        fb = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        for attn_dp, attn_tp in ((1, 1), (1, 2), (2, 1), (2, 2)):
            with (
                self.subTest(attn_dp=attn_dp, attn_tp=attn_tp),
                fixture.planning(fixture.parallel_of(attn_dp=attn_dp, attn_tp=attn_tp)),
            ):
                stages = make_test_stages(
                    attention_norm=Norm(),
                    ffn_norm=Norm(),
                    residual=PLAIN_RESIDUAL_OPS._replace(
                        ffn_readout=NORM_QUANT_READOUT
                    ),
                )
                self.assertTrue(stages.attn.reads_quantized(fb))
                self.assertEqual(stages.ffn.reads_quantized(fb), attn_dp == 1)


if __name__ == "__main__":
    unittest.main()
