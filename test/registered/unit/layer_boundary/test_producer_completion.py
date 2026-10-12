"""Producer updates survive binding, carried completion and terminal handoffs."""

import unittest
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import test_declared_decoder_boundary as fixture
import torch

from sglang.srt.layers.layer_boundary import (
    ReadoutFusion,
    append_stages,
    declare_attn,
    declare_ffn,
    layer_stack,
)
from sglang.srt.layers.layer_boundary.facts import facts_of
from sglang.srt.layers.layer_boundary.layout import SumGroup
from sglang.srt.layers.layer_boundary.output import DeferredFinalize, UnreducedOutput
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    NORM_READOUT,
    PLAIN_ADD,
    NormReadout,
    PlainAdd,
)
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Nonplain(PlainAdd):
    is_plain_add = False
    quantized_sum = False

    @staticmethod
    def update(value, residual):
        return value.square() + residual


class TestProducerCompletion(unittest.TestCase):
    def test_pipeline_cut_rejects_deferred_nonplain_but_local_stage_allows_it(self):
        parallel = fixture.parallel_of(attn_dp=1, attn_tp=2, pp_size=2)
        with fixture.planning(parallel):
            with layer_stack():
                append_stages(
                    (declare_ffn(update=Nonplain()), fixture.Norm()),
                    (declare_ffn(), fixture.Norm()),
                )
            for declare in (declare_attn, declare_ffn):
                with self.subTest(stage=declare.__name__):
                    with self.assertRaisesRegex(
                        NotImplementedError, "pipeline boundaries"
                    ):
                        with layer_stack(next_layers=[lambda: (declare_attn(),)]):
                            append_stages((declare(update=Nonplain()), fixture.Norm()))

    def test_pair_export_rejects_nonplain_before_starting_completion(self):
        for exporter in (residual_batch.to_pp, residual_batch.final_norm):
            with self.subTest(export=exporter.__name__):
                group = SimpleNamespace(
                    all_reduce=Mock(side_effect=AssertionError("sum"))
                )
                value = torch.ones(2, 4)
                batch = SimpleNamespace(residual_stream=ResidualStream(value))
                output = batch.residual_stream.record(
                    UnreducedOutput(value, group), Nonplain()
                )
                with self.assertRaisesRegex(
                    NotImplementedError, "plain residual update"
                ):
                    if exporter is residual_batch.final_norm:
                        exporter(output, batch, fixture.Norm())
                    else:
                        exporter(output, batch)
                group.all_reduce.assert_not_called()
                self.assertIsNotNone(batch.residual_stream.pending)

    def test_carried_finalize_uses_only_candidates_for_its_completion_group(self):
        events = []

        class Completion(DeferredFinalize, frozen=True):
            def complete(self):
                events.append("complete")
                return torch.full((2, 4), 3.0)

        class Read(NormReadout):
            completing_fusions = (
                ReadoutFusion(
                    SumGroup.ATTN_TP,
                    lambda *a, **kw: self.fail("wrong completion group"),
                    carried=True,
                ),
            )

        parallel = fixture.parallel_of(attn_dp=1, attn_tp=2, pp_size=1)
        with fixture.planning(parallel), layer_stack():
            (stage,) = append_stages((declare_ffn(read=Read()), fixture.Norm()))
        batch = SimpleNamespace(
            input_ids=torch.ones(2, dtype=torch.long),
            forward_mode=ForwardMode.DECODE,
            residual_stream=ResidualStream(torch.ones(2, 4)),
        )
        hidden = batch.residual_stream.record(Completion(), PLAIN_ADD)
        with fixture.planning(parallel):
            result = stage.prepare(hidden, batch)
        torch.testing.assert_close(result, torch.full((2, 4), 8.0))
        self.assertEqual(events, ["complete"])
        self.assertIsNone(batch.residual_stream.pending)

    def test_binding_is_required_and_callable_when_present(self):
        for protocol, declare, keyword in (
            (PLAIN_ADD, declare_ffn, "update"),
            (NORM_READOUT, declare_attn, "read"),
        ):
            for binding in ("missing", 3):
                with self.subTest(protocol=keyword, binding=binding):
                    names = (
                        (
                            "is_plain_add",
                            "applied_at_exit",
                            "outlives_layer",
                            "writes_stream",
                            "quantized_sum",
                            "replay_update",
                        )
                        if keyword == "update"
                        else (
                            "is_plain_norm",
                            "reads_before_dp_gather",
                            "reads_after_attn_tp_gather",
                            "completing_fusions",
                            "gathering_reads",
                        )
                    )
                    fake = SimpleNamespace(
                        **{name: getattr(protocol, name, None) for name in names}
                    )
                    if binding != "missing":
                        fake.bind = binding
                    with self.assertRaisesRegex(TypeError, "bind"):
                        declare(**{keyword: fake})

    def test_neighbour_facts_retain_no_execution_callbacks(self):
        def walk(value):
            self.assertFalse(callable(value))
            if isinstance(value, msgspec.Struct):
                for name in value.__struct_fields__:
                    walk(getattr(value, name))
            elif is_dataclass(value):
                for field in fields(value):
                    walk(getattr(value, field.name))
            elif isinstance(value, Mapping):
                for key, item in value.items():
                    walk(key)
                    walk(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    walk(item)

        walk(facts_of(declare_ffn(update=Nonplain())))


if __name__ == "__main__":
    unittest.main()
