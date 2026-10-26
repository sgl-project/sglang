"""A custom output sum, channelwise convolution, and terminal read share one owner."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_declared_decoder_boundary as fixture
import torch

from sglang.srt.layers.layer_boundary import (
    ProducerReduction,
    ReadoutFusion,
    append_stages,
    declare_attn,
    declare_ffn,
    layer_stack,
    prepare,
)
from sglang.srt.layers.layer_boundary.layout import SumGroup
from sglang.srt.layers.layer_boundary.output import SumImplementation, UnreducedOutput
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual import bind_update
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
from sglang.srt.layers.layer_boundary.residual.short_conv import (
    ShortConvRead,
    ShortConvUpdate,
)
from sglang.srt.layers.layer_boundary.residual.stream import (
    OwedOutput,
    ResidualStream,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import patch_communicator

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestShortConvBoundary(unittest.TestCase):
    def test_sum_conv_gather_add_norm_and_tail(self):
        for defer in (False, True):
            for scattered in (False, True):
                for lora in (False, True):
                    with self.subTest(defer=defer, scattered=scattered, lora=lora):
                        self.run_chain(defer, scattered, lora=lora)

    def run_chain(self, defer, scattered, lora=False):
        calls = []
        batch = SimpleNamespace(
            positions=torch.tensor([3, 8]),
            input_ids=torch.tensor([1, 2]),
            forward_mode=ForwardMode.DECODE,
            residual_stream=None,
        )

        def complete(value, forward_batch):
            calls.append("sum")
            value = value * 2 + 0.7
            return value[:, :2] if scattered else value

        def conv(value, positions, forward_batch):
            self.assertIs(forward_batch, batch)
            calls.append("conv")
            return value.square() * 0.1 + positions[:, None]

        def gather(value):
            calls.append("hidden_gather")
            return torch.cat((value, value + 5), dim=-1)

        norm = fixture.Norm()
        read = ShortConvRead()
        update = ShortConvUpdate(conv, None, SumGroup.TP, gather if scattered else None)
        custom = SumImplementation(complete, lambda fb: defer)
        parallel = fixture.parallel_of(attn_dp=1, attn_tp=2, pp_size=1)
        with (
            fixture.planning(parallel, boundary_reduction="ar"),
            patch_communicator("get_lora", lambda: SimpleNamespace(enable_lora=lora)),
            layer_stack(final_read=read),
        ):
            stages = [
                stage
                for _ in range(2)
                for stage in append_stages(
                    (declare_attn(read=read, update=update, sum_impl=custom), norm),
                    (declare_ffn(read=read, update=update, sum_impl=custom), norm),
                )
            ]
        initial = torch.tensor([[0.2, -0.4, 0.6, 0.8], [-0.7, 0.9, 0.1, 0.3]])
        hidden, residual = initial, initial
        residual_batch.start(batch)
        with (
            fixture.planning(parallel, boundary_reduction="ar"),
            patch_communicator("get_lora", lambda: SimpleNamespace(enable_lora=lora)),
        ):
            for index, stage in enumerate(stages):
                hidden = stage.prepare(hidden, batch)
                torch.testing.assert_close(hidden, 2 * residual)
                output = hidden * (0.2 + 0.03 * index)
                hidden = stage.finish(output, batch)
                expected = output * 2 + 0.7
                if scattered:
                    expected = expected[:, :2]
                expected = expected.square() * 0.1 + batch.positions[:, None]
                if scattered:
                    expected = torch.cat((expected, expected + 5), dim=-1)
                residual = residual + expected
            if defer:
                self.assertIsInstance(hidden, OwedOutput)
            result, tail = residual_batch.read_output(hidden, batch, read, norm)
        torch.testing.assert_close(result, 2 * residual)
        torch.testing.assert_close(tail, residual)
        self.assertEqual(calls.count("sum"), 4)
        self.assertEqual(calls.count("conv"), 4)
        self.assertEqual(calls.count("hidden_gather"), 4 if scattered else 0)
        self.assertIsNone(batch.residual_stream)
        if isinstance(hidden, OwedOutput):
            self.assertIsNone(hidden.contribution.value)
            self.assertIsNone(hidden.contribution.owed)

    def test_fusion_receives_the_actual_update_and_rejects_before_mutation(self):
        update = ShortConvUpdate(
            lambda *args: self.fail("unfused conv"), None, SumGroup.TP
        )
        actual = bind_update(update, SimpleNamespace(positions=None))
        events = []
        expected = (torch.ones(2, 4), torch.zeros(2, 4))

        def run(*args, update):
            events.append(update)
            return expected

        rejected = ReadoutFusion(
            SumGroup.TP,
            lambda *args, **kwargs: self.fail("rejected fusion ran"),
            accepts_update=lambda u: False,
        )
        accepted = ReadoutFusion(
            SumGroup.TP, run, reads=True, accepts_update=lambda u: u is actual
        )
        result = prepare._run_read_fusions(
            (rejected, accepted),
            torch.ones(2, 4),
            torch.zeros(2, 4),
            None,
            None,
            read=ShortConvRead(),
            update=actual,
        )
        self.assertIs(result, expected)
        self.assertEqual(events, [actual])

    def test_explicit_positions_belong_to_the_producer_invocation(self):
        batch = SimpleNamespace(positions=torch.tensor([2, 4]))
        positions = torch.tensor([17, 23])
        update = bind_update(
            ShortConvUpdate(lambda h, p, fb: h + p[:, None], None, SumGroup.TP),
            batch,
            positions,
        )
        value = torch.zeros(2, 4)
        batch.positions = torch.tensor([101, 103])
        torch.testing.assert_close(
            update.contribution(value), positions[:, None].expand_as(value).to(value)
        )

    def test_final_read_requires_explicit_acceptance_and_releases_on_failure(self):
        update = bind_update(
            ShortConvUpdate(lambda *args: self.fail("conv"), None, SumGroup.TP),
            SimpleNamespace(positions=None),
        )
        batch = SimpleNamespace(residual_stream=ResidualStream(torch.ones(2, 4)))
        output = batch.residual_stream.record(
            torch.zeros(2, 4), update, declared_sum=SumGroup.TP
        )
        with self.assertRaises(NotImplementedError):
            residual_batch.read_output(output, batch, SimpleNamespace(), None)
        self.assertIsNone(batch.residual_stream)
        self.assertIsNone(output.contribution.value)

    def test_terminal_nonplain_update_requires_final_read(self):
        update = ShortConvUpdate(lambda *args: None, None, SumGroup.TP)
        with fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=2, pp_size=1)):
            with self.assertRaises(NotImplementedError), layer_stack():
                append_stages(
                    (declare_ffn(read=ShortConvRead(), update=update), fixture.Norm())
                )

    def test_already_reduced_output_does_not_run_custom_sum(self):
        calls = []
        custom = SumImplementation(
            lambda value, batch: calls.append("sum") or value * 2
        )
        parallel = fixture.parallel_of(attn_dp=1, attn_tp=2)
        batch = SimpleNamespace(
            input_ids=torch.ones(2, dtype=torch.long),
            forward_mode=ForwardMode.DECODE,
            residual_stream=None,
        )
        value = torch.randn(2, 4)
        with fixture.planning(parallel), layer_stack():
            (stage,) = append_stages((declare_ffn(sum_impl=custom), fixture.Norm()))
        with fixture.planning(parallel):
            residual_batch.start(batch)
            stage.prepare(value, batch)
            stage.plan.output.complete_now(
                value, batch.residual_stream, batch, already_reduced=True
            )
        self.assertEqual(calls, [])
        torch.testing.assert_close(batch.residual_stream.pending.value, value)

    def test_inkling_keeps_unpadded_batches_on_declared_dense_and_moe_rows(self):
        from sglang.srt.arg_groups.parallel_hook import _boundary_parallelism_overrides
        from sglang.srt.layers.layer_boundary import BatchVariant
        from sglang.srt.layers.layer_boundary.residual.add_norm import (
            PLAIN_RESIDUAL_OPS,
        )
        from sglang.srt.models import inkling

        for architecture in (
            "InklingForConditionalGeneration",
            "InklingForConditionalGenerationMTP",
        ):
            parallel = fixture.parallel_of(
                attn_dp=1,
                attn_tp=2,
                moe_dense_tp_size=1,
                disable_attn_tp_gather=True,
                enable_attn_tp_input_scattered=True,
            )
            overrides = _boundary_parallelism_overrides(
                parallel, "inkling", architecture
            )
            self.assertEqual(
                overrides,
                {"enable_attn_tp_input_scattered": False},
            )
            parallel = SimpleNamespace(**(vars(parallel) | overrides))
            self.assertTrue(parallel.disable_attn_tp_gather)
            self.assertEqual(parallel.moe_dense_tp_size, 1)
            config = SimpleNamespace(dense_mlp_idx=1, use_sconv=False)
            custom = SumImplementation(lambda value, batch: value)
            with (
                fixture.planning(parallel),
                patch.object(inkling, "get_parallel", return_value=parallel),
                layer_stack(),
            ):
                stages = []
                for layer_id in (0, 1):
                    declarations = inkling.inkling_stages(
                        config, layer_id, ops=PLAIN_RESIDUAL_OPS, sums=(custom, custom)
                    )
                    stages.extend(
                        append_stages(
                            *(
                                (declaration, fixture.Norm())
                                for declaration in declarations
                            )
                        )
                    )
            batch = SimpleNamespace(
                input_ids=torch.ones(3, dtype=torch.long),
                forward_mode=ForwardMode.EXTEND,
                attn_cp_metadata=None,
            )
            with fixture.planning(parallel):
                for stage in stages:
                    self.assertIs(stage.plan.variant_for(batch), BatchVariant.ORDINARY)

    def test_custom_sums_reject_unimplemented_rows_and_mixer_exits(self):
        custom = SumImplementation(lambda value, batch: value)
        cases = (
            (fixture.parallel_of(attn_dp=2, attn_tp=2), declare_ffn, False),
            (
                fixture.parallel_of(
                    attn_dp=1, attn_tp=2, enable_attn_tp_input_scattered=True
                ),
                declare_ffn,
                False,
            ),
            (fixture.parallel_of(attn_dp=1, attn_tp=2), declare_ffn, True),
            (
                fixture.parallel_of(attn_dp=1, attn_tp=2),
                lambda **kw: declare_attn(
                    reduction=ProducerReduction.EXIT_SCOPED, **kw
                ),
                False,
            ),
        )
        from sglang.srt.layers import layernorm_sp

        for parallel, declare, sp in cases:
            with self.subTest(
                dp=parallel.attn_dp_size,
                scattered=parallel.enable_attn_tp_input_scattered,
                sp=sp,
                declare=declare,
            ):
                with (
                    fixture.planning(parallel),
                    patch.object(layernorm_sp, "layernorm_sp_enabled", lambda: sp),
                ):
                    with self.assertRaisesRegex(
                        NotImplementedError, "custom output sums"
                    ):
                        with layer_stack():
                            append_stages((declare(sum_impl=custom), fixture.Norm()))

    def test_replay_handoff_rejects_tensor_with_declared_debt(self):
        batch = SimpleNamespace(residual_stream=ResidualStream(torch.ones(2, 4)))
        hidden = batch.residual_stream.record(
            torch.ones(2, 4), PLAIN_ADD, declared_sum=SumGroup.TP
        )
        with self.assertRaisesRegex(RuntimeError, "complete the contribution"):
            residual_batch.export_replay_handoff(hidden, batch)
        self.assertIsNotNone(batch.residual_stream.pending.owed)

    def test_replay_handoff_rebinds_to_live_positions_and_releases_capture_state(self):
        batch = SimpleNamespace(
            positions=torch.tensor([2, 4]),
            residual_stream=ResidualStream(torch.ones(2, 4)),
        )
        spec = ShortConvUpdate(lambda h, p, fb: h + p[:, None], None, SumGroup.TP)
        actual = bind_update(spec, batch)
        hidden = batch.residual_stream.record(torch.zeros(2, 4), actual)
        pending = batch.residual_stream.pending
        handoff = residual_batch.export_replay_handoff(hidden, batch)
        self.assertIsNone(batch.residual_stream)
        self.assertIsNone(pending.update)
        self.assertIs(handoff.update, spec)
        live = SimpleNamespace(positions=torch.tensor([7, 11]), residual_stream=None)
        restored = residual_batch.import_replay_handoff(
            handoff, live, update_context=live.positions
        )
        result, _ = residual_batch.read_output(
            restored, live, ShortConvRead(), fixture.Norm()
        )
        torch.testing.assert_close(result, torch.tensor([[16.0] * 4, [24.0] * 4]))
        self.assertIsNone(live.residual_stream)

    def test_replay_handoff_preserves_written_state(self):
        value = torch.ones(2, 4)
        batch = SimpleNamespace(residual_stream=ResidualStream(value))
        handoff = residual_batch.export_replay_handoff(value, batch)
        self.assertTrue(handoff.written)
        restored = residual_batch.import_replay_handoff(handoff, batch)
        self.assertIs(restored, value)
        self.assertIs(batch.residual_stream.residual, value)
        self.assertIsNone(batch.residual_stream.pending)

    def test_normalization_handoff_completes_producer_before_captured_add_norm(self):
        for convolve in (False, True):
            with self.subTest(convolve=convolve):
                events = []
                value = torch.full((2, 4), 0.1, dtype=torch.bfloat16)
                residual = torch.full((2, 4), 0.0003, dtype=torch.bfloat16)
                batch = SimpleNamespace(
                    positions=torch.tensor([2, 4]),
                    residual_stream=ResidualStream(residual),
                )

                def conv(value, positions, live):
                    self.assertIs(live, batch)
                    events.append("conv")
                    return value * 3

                update = (
                    bind_update(ShortConvUpdate(conv, None, SumGroup.TP), batch)
                    if convolve
                    else PLAIN_ADD
                )
                output = batch.residual_stream.record(
                    UnreducedOutput(
                        value,
                        complete_sum=lambda v: events.append("sum") or v * 2,
                    ),
                    update,
                )
                pending = batch.residual_stream.pending
                read = ShortConvRead()
                handoff = read.export_normalization_handoff(output, batch)
                self.assertEqual(events, ["sum", "conv"] if convolve else ["sum"])
                self.assertIsNone(batch.residual_stream)
                self.assertIsNone(pending.update)
                self.assertIs(handoff.residual, residual)
                expected = value * (6 if convolve else 2)
                torch.testing.assert_close(handoff.value, expected)

                def norm(v, r, **kw):
                    self.assertIs(v, handoff.value)
                    self.assertIs(r, residual)
                    events.append("norm")
                    merged = v.float() + r.float()
                    return merged, merged.to(v.dtype)

                normalized, written = read.read_normalization_handoff(handoff, norm)
                torch.testing.assert_close(
                    normalized, expected.float() + residual.float()
                )
                self.assertEqual(events[-1], "norm")
                self.assertFalse(torch.equal(normalized, (expected + residual).float()))
                torch.testing.assert_close(
                    written, (expected.float() + residual.float()).to(value)
                )

    def test_tp1_moe_merges_shared_without_running_custom_sum(self):
        from sglang.srt.layers.layer_boundary.residual import LayerResidualOps
        from sglang.srt.models import inkling
        from sglang.srt.models.inkling_common.kernels.comm import (
            stash_ar_shared,
            take_ar_shared,
        )

        parallel = fixture.parallel_of(attn_dp=1, attn_tp=1, pp_size=1)
        read = ShortConvRead()
        config = SimpleNamespace(dense_mlp_idx=0, use_sconv=False)
        custom = SumImplementation(lambda *a: self.fail("single-rank custom sum"))
        with (
            fixture.planning(parallel),
            patch.object(inkling, "get_parallel", return_value=parallel),
            layer_stack(final_read=read),
        ):
            stages = append_stages(
                *[
                    (decl, fixture.Norm())
                    for decl in inkling.inkling_stages(
                        config,
                        0,
                        ops=LayerResidualOps(read, PLAIN_ADD, read, PLAIN_ADD),
                        sums=(custom, custom),
                    )
                ]
            )
        batch = SimpleNamespace(
            input_ids=torch.ones(2, dtype=torch.long),
            forward_mode=ForwardMode.DECODE,
            residual_stream=None,
        )
        value = torch.ones(2, 4)
        try:
            with fixture.planning(parallel):
                residual_batch.start(batch)
                hidden = stages[0].prepare(value, batch)
                hidden = stages[0].finish(hidden * 0.1, batch)
                operand = stages[1].prepare(hidden, batch)
                routed, shared = operand * 0.2, operand * 0.3
                stash_ar_shared(shared)
                output = stages[1].finish(routed, batch)
                torch.testing.assert_close(
                    batch.residual_stream.pending.value, routed + shared
                )
                self.assertIsNone(take_ar_shared(value.shape[0]))
                residual_batch.read_output(output, batch, read, fixture.Norm())
                self.assertIsNone(batch.residual_stream)
        finally:
            take_ar_shared(value.shape[0])


class TestMtpInvocationCleanup(unittest.TestCase):
    def test_exception_clears_residual_and_shared_stash(self):
        from sglang.srt.models import inkling

        model = inkling.InklingMTPLayer.__new__(inkling.InklingMTPLayer)
        torch.nn.Module.__init__(model)
        model.embed_tokens = torch.nn.Embedding(4, 2)
        model.main_model_embed_norm = None
        model.hidden_norm = model.embed_norm = torch.nn.Identity()
        model.input_proj = torch.nn.Linear(4, 2)
        model.log_scaling_n_floor = None
        streams = []

        def fail(value, positions, batch, **kwargs):
            stream = residual_batch.stream_of(batch)
            stream.residual = value
            streams.append(stream)
            raise RuntimeError("draft failed")

        model.transformer_block = fail
        ids = torch.tensor([1])
        batch = SimpleNamespace(
            input_ids=ids, spec_info=SimpleNamespace(hidden_states=torch.ones(1, 2))
        )
        with (
            patch.object(inkling, "take_ar_shared") as take,
            self.assertRaisesRegex(RuntimeError, "draft failed"),
        ):
            model(ids, ids, batch)
        self.assertIsNone(batch.residual_stream)
        self.assertIsNone(streams[0].residual)
        take.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
