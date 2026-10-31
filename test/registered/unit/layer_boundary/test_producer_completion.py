"""Producer updates survive binding, carried completion and terminal handoffs."""

import unittest
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial
from types import FunctionType, MethodType, SimpleNamespace
from unittest.mock import Mock, patch

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
from sglang.srt.layers.layer_boundary.contracts import SumImplementation
from sglang.srt.layers.layer_boundary.facts import (
    ReadFacts,
    TransformFacts,
    UpdateFacts,
    facts_of,
)
from sglang.srt.layers.layer_boundary.layout import SumGroup
from sglang.srt.layers.layer_boundary.output import (
    DeferredFinalize,
    OutputTransform,
    UnreducedOutput,
)
from sglang.srt.layers.layer_boundary.residual import PendingUpdateRead
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    PLAIN_ADD,
    NormReadout,
    PlainAdd,
)
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class Nonplain(PlainAdd):
    is_plain_add = False
    quantized_sum = False

    @staticmethod
    def update(value, residual):
        return value.square() + residual


class TestProducerCompletion(CustomTestCase):
    def test_update_context_is_bound_once_for_each_completion_route(self):
        class Read(PendingUpdateRead):
            bind = None
            completing_fusions = ()
            accepts_pending_update = staticmethod(lambda update: True)
            init_residual = staticmethod(lambda value: value)
            read = staticmethod(lambda value, norm: value)
            update_and_read = staticmethod(
                lambda update, value, residual, norm: update.update(value, residual)
            )

        for route in ("finish", "exit", "complete_now", "mixer"):
            for at_exit in (False, True):
                with self.subTest(route=route, at_exit=at_exit):
                    bindings = []

                    class Update(Nonplain):
                        applied_at_exit = at_exit

                        def bind(self, batch, context=None):
                            bindings.append(context)
                            bound = Nonplain()
                            bound.applied_at_exit = at_exit
                            bound.update = lambda value, residual: (
                                value * context + residual
                            )
                            return bound

                    parallel = fixture.parallel_of(attn_dp=1, attn_tp=1)
                    declare = declare_attn if route == "mixer" else declare_ffn
                    with fixture.planning(parallel), layer_stack(final_read=Read()):
                        (stage,) = append_stages(
                            (declare(update=Update()), fixture.Norm())
                        )
                    for context in (3, 7):
                        batch = SimpleNamespace(
                            input_ids=torch.ones(2, dtype=torch.long),
                            forward_mode=ForwardMode.DECODE,
                            residual_stream=ResidualStream(torch.ones(2, 4)),
                        )
                        with fixture.planning(parallel):
                            value = torch.full((2, 4), 2.0)
                            if route == "exit":
                                result = stage.exit(
                                    batch, update_context=context
                                ).finish(value)
                            elif route == "complete_now":
                                result = stage.complete_now(
                                    value, batch, update_context=context
                                )
                            else:
                                result = stage.finish(
                                    value, batch, update_context=context
                                )

                            result = residual_batch.read_output(
                                result, batch, Read(), None
                            )
                        torch.testing.assert_close(
                            result, torch.full((2, 4), 2 * context + 1.0)
                        )
                    self.assertEqual(bindings, [3, 7])

    def test_terminal_pair_export_releases_retained_contribution(self):
        """An old owed handle must not retain the update after terminal export."""
        for route in ("pp", "norm", "failing_norm"):
            with self.subTest(route=route):
                value = torch.ones(2, 4)
                update = PlainAdd()
                update.context = object()
                stream = ResidualStream(value)
                batch = SimpleNamespace(residual_stream=stream)
                group = SimpleNamespace(all_reduce=lambda value: value * 3)
                handle = stream.record(UnreducedOutput(value, group), update)
                if route == "pp":
                    result = residual_batch.to_pp(handle, batch)
                    torch.testing.assert_close(result["hidden_states"], value * 3)
                    torch.testing.assert_close(result["residual"], value)
                elif route == "norm":
                    result = residual_batch.final_norm(handle, batch, fixture.Norm())
                    torch.testing.assert_close(result, value * 8)
                else:
                    with self.assertRaisesRegex(RuntimeError, "norm failed"):
                        residual_batch.final_norm(
                            handle, batch, Mock(side_effect=RuntimeError("norm failed"))
                        )
                self.assertIsNone(batch.residual_stream)
                self.assertIsNone(handle.contribution.update)
                self.assertIsNone(handle.contribution.value)
                self.assertIsNone(handle.contribution.owed)
                self.assertIsNone(stream.pending)
                self.assertIsNone(stream.residual)

    def test_terminal_read_binds_each_context_before_read_or_completion(self):
        """Terminal reads must not run declaration callbacks with missing state."""
        events = []

        class Read(NormReadout, PendingUpdateRead):
            def __init__(self, route, context=None):
                self.route, self.context = route, context
                self.completing_fusions = (
                    (ReadoutFusion(SumGroup.ATTN_TP, self.complete, reads=True),)
                    if route == "fused"
                    else ()
                )

            def bind(self, batch, context):
                events.append(("bind", context))
                return Read(self.route, batch.context if context is None else context)

            def accepts_pending_update(self, update):
                return update.is_plain_add

            def read(self, value, norm, **kwargs):
                assert self.context is not None, "unbound terminal read"
                events.append("read")
                return value + self.context, value

            def update_and_read(self, update, value, residual, norm, **kwargs):
                return self.read(update.update(value, residual), norm)

            def complete(self, value, residual, batch, norm):
                assert self.context is not None, "unbound terminal fusion"
                events.append("complete")
                return self.read(value * 3 + residual, norm)

        for route in ("written", "pending", "fused"):
            for context in (17, 29):
                for explicit_context in (False, True):
                    with self.subTest(
                        route=route, context=context, explicit=explicit_context
                    ):
                        events.clear()
                        value = torch.ones(2, 4)
                        stream = ResidualStream(value)
                        batch = SimpleNamespace(residual_stream=stream, context=context)
                        hidden = value
                        if route != "written":
                            hidden = stream.record(
                                value,
                                PLAIN_ADD,
                                declared_sum=SumGroup.ATTN_TP
                                if route == "fused"
                                else None,
                            )
                        kwargs = {"read_context": context} if explicit_context else {}
                        result, residual = residual_batch.read_output(
                            hidden, batch, Read(route), None, **kwargs
                        )
                        expected = {"written": 1, "pending": 2, "fused": 4}[route]
                        torch.testing.assert_close(
                            result, torch.full_like(value, expected + context)
                        )
                        torch.testing.assert_close(
                            residual, torch.full_like(value, expected)
                        )
                        self.assertEqual(
                            events,
                            [("bind", context if explicit_context else None)]
                            + (["complete"] if route == "fused" else [])
                            + ["read"],
                        )
                        self.assertIsNone(batch.residual_stream)
                        self.assertIsNone(stream.pending)

    def test_terminal_read_rejects_changed_binding_capabilities_and_cleans_up(self):
        """Binding cannot change routing facts or callbacks before terminal execution."""
        for change in ("rows", "callbacks"):
            with self.subTest(change=change):

                class Read(NormReadout, PendingUpdateRead):
                    accepts_pending_update = staticmethod(lambda update: True)

                    def bind(self, batch, context):
                        class Bound(NormReadout):
                            pass

                        bound = Bound(reads_before_dp_gather=change == "rows")
                        if change == "callbacks":
                            bound.gathering_reads = (lambda *args: None,)
                        return bound

                value = torch.ones(2, 4)
                stream = ResidualStream(value)
                batch = SimpleNamespace(residual_stream=stream)
                hidden = stream.record(value, PLAIN_ADD)
                contribution = stream.pending
                with self.assertRaisesRegex(ValueError, "invocation read changed"):
                    residual_batch.read_output(hidden, batch, Read(), None)
                self.assertIsNone(contribution.update)
                self.assertIsNone(batch.residual_stream)

    def test_update_binding_rejects_bindable_results_before_any_exit_route(self):
        """Immediate writes and deferred records must reject an unbound result alike."""
        parallel = fixture.parallel_of(attn_dp=1, attn_tp=1)
        for at_exit in (False, True):
            with self.subTest(at_exit=at_exit):

                class Update(Nonplain):
                    applied_at_exit = at_exit
                    update = Mock(side_effect=AssertionError("unbound update ran"))

                    def bind(self, batch, context):
                        return self

                with fixture.planning(parallel), layer_stack():
                    stage, _ = append_stages(
                        (declare_ffn(update=Update()), fixture.Norm()),
                        (declare_ffn(), fixture.Norm()),
                    )
                value = torch.ones(2, 4)
                batch = SimpleNamespace(
                    residual_stream=ResidualStream(value),
                    input_ids=torch.ones(2, dtype=torch.long),
                    forward_mode=ForwardMode.DECODE,
                )
                with (
                    fixture.planning(parallel),
                    self.assertRaisesRegex(
                        ValueError, "invocation update must have bind = None"
                    ),
                ):
                    stage.finish(value, batch)
                self.assertIsNone(batch.residual_stream.pending)

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

        for a2a, dense in ((True, None), (False, 1)):
            parallel = fixture.parallel_of(attn_dp=2, attn_tp=1, pp_size=2)
            with (
                self.subTest(a2a=a2a, dense=dense),
                fixture.planning(parallel, a2a=a2a),
            ):
                with self.assertRaisesRegex(NotImplementedError, "pipeline boundaries"):
                    with layer_stack(next_layers=[lambda: (declare_attn(),)]):
                        append_stages(
                            (
                                declare_ffn(
                                    update=Nonplain(), sparse=a2a, dense_tp_size=dense
                                ),
                                fixture.Norm(),
                            )
                        )

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

    def test_neighbour_facts_do_not_retain_layer_state(self):
        import test_kimi_k3_stage_kernels as k3_fixture

        from sglang.srt.layers.layer_boundary.row_transport import RowTransport
        from sglang.srt.models import kimi_k3

        seen = set()

        def walk(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            self.assertNotIsInstance(value, (torch.nn.Module, torch.Tensor))
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
            elif isinstance(value, (tuple, list, set, frozenset)):
                for item in value:
                    walk(item)
            elif isinstance(value, partial):
                walk(value.func)
                walk(value.args)
                walk(value.keywords)
            elif isinstance(value, MethodType):
                walk(value.__self__)
                walk(value.__func__)
            elif isinstance(value, FunctionType):
                walk(value.__defaults__)
                walk(value.__kwdefaults__)
                for cell in value.__closure__ or ():
                    walk(cell.cell_contents)
            elif not isinstance(
                value, (type(None), bool, int, float, str, bytes, Enum)
            ):
                self.fail(f"unexpected execution object: {type(value)}")

        update = Nonplain()
        update.layer = torch.nn.Linear(4, 4)
        declaration = declare_ffn(
            update=update,
            output_transform=OutputTransform(update.layer.forward),
            sum_impl=SumImplementation(update.layer.forward),
            row_transport=RowTransport({}, bind=update.layer.forward),
        )
        facts = facts_of(declaration)
        self.assertIsInstance(facts.read, ReadFacts)
        self.assertIsInstance(facts.update, UpdateFacts)
        self.assertIsInstance(facts.output_transform, TransformFacts)
        walk(facts)
        # The checker must reject an opaque update even when it is not callable.
        with self.assertRaisesRegex(AssertionError, "unexpected execution object"):
            walk(update)

        config = SimpleNamespace(**vars(k3_fixture.CONFIG), attn_res_block_size=4)
        with (
            fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=2), a2a=True),
            patch.object(kimi_k3, "_shards_moe_rows", return_value=True),
        ):
            declarations = kimi_k3._stage_facts(config, 2, sp_collective=True)
        for declaration in declarations:
            walk(facts_of(declaration))

    def test_pipeline_sender_uses_the_remote_k3_gather(self):
        import test_kimi_k3_stage_kernels as k3_fixture

        from sglang.srt.layers.communication import k3_sp_collective
        from sglang.srt.layers.layer_boundary import BatchVariant, ops
        from sglang.srt.models import kimi_k3

        config = SimpleNamespace(**vars(k3_fixture.CONFIG), attn_res_block_size=4)
        gather = Mock(side_effect=lambda value: torch.cat((value, value + 10)))
        with (
            fixture.planning(
                fixture.parallel_of(attn_dp=1, attn_tp=2, pp_size=2), a2a=True
            ),
            patch.object(kimi_k3, "_shards_moe_rows", return_value=True),
            patch.object(k3_sp_collective, "all_gather", gather),
            layer_stack(
                next_layers=[
                    lambda: kimi_k3._stage_facts(config, 2, sp_collective=True)
                ]
            ),
        ):
            (stage,) = append_stages((declare_ffn(sparse=True), fixture.Norm()))
        move = stage.plan.paths[BatchVariant.ORDINARY].output_move
        self.assertIs(move.keywords["gather"], gather)
        value = torch.ones(2, 4)
        with patch.object(
            ops, "attn_tp_gather", side_effect=AssertionError("default gather")
        ):
            output, _ = move(
                hidden_states=value,
                residual=torch.zeros_like(value),
                forward_batch=None,
            )
        gather.assert_called_once()
        torch.testing.assert_close(output, torch.cat((value, value + 10)))

    def test_invocation_callbacks_complete_selected_sum_and_gather_paths(self):
        from sglang.srt.layers.layer_boundary import prepare
        from sglang.srt.layers.layer_boundary.boundary import bind_entry
        from sglang.srt.layers.layer_boundary.contracts import (
            EdgeContract,
            InputContract,
            OutputContract,
        )
        from sglang.srt.layers.layer_boundary.layout import Layout, TokenAxis
        from sglang.srt.layers.layer_boundary.ops import GatheredInput
        from sglang.srt.layers.layer_boundary.residual.stream import DeclaredSum

        events = []

        class Read(NormReadout):
            is_plain_norm = False

            def __init__(self, context=None):
                self.context = context
                self.completing_fusions = tuple(
                    ReadoutFusion(
                        group,
                        partial(self.complete, group, scatters),
                        scatters=scatters,
                        reads=True,
                        carried=not scatters,
                    )
                    for group, scatters in (
                        (SumGroup.TP, False),
                        (SumGroup.ATTN_TP, True),
                        (SumGroup.ATTN_TP, False),
                    )
                )
                self.gathering_reads = (self.gather,)

            def bind(self, batch, context):
                events.append(("bind", context))
                return Read(context)

            def complete(self, group, scatters, value, residual, batch, norm):
                assert self.context is not None, "unbound completion"
                events.append(("complete", group, scatters, self.context))
                return torch.full_like(residual, self.context), residual

            def gather(self, value, residual, norm):
                assert self.context is not None, "unbound gathering read"
                events.append(("gather", self.context))
                return torch.full_like(residual, self.context), residual

        full = Layout(frozenset())
        shard = Layout(frozenset({TokenAxis.ATTN_TP}))
        group = object()
        with patch.object(
            prepare,
            "_sum_group",
            side_effect=lambda selected: (
                group if selected is SumGroup.ATTN_TP else None
            ),
        ):
            for route in ("sum", "scatter", "gather", "carried"):
                source = shard if route == "gather" else full
                target = shard if route == "scatter" else full
                rows = shard if route in ("gather", "scatter") else full
                declared = route in ("sum", "scatter")
                edge = EdgeContract(
                    OutputContract(
                        source,
                        group=SumGroup.ATTN_TP if declared else None,
                        always_partial=declared,
                    ),
                    InputContract(target, read=Read()),
                    rows,
                    rows,
                )
                entry = bind_entry(edge)
                for context in (17, 29):
                    with self.subTest(route=route, context=context):
                        events.clear()
                        value = torch.ones(2, 4)
                        hidden = (
                            UnreducedOutput(value, group)
                            if route == "carried"
                            else value
                        )
                        result, _ = entry.prepare(
                            hidden,
                            value,
                            SimpleNamespace(),
                            None,
                            pending=(
                                SimpleNamespace(owed=DeclaredSum(SumGroup.ATTN_TP))
                                if declared
                                else None
                            ),
                            read_context=context,
                        )
                        if isinstance(result, GatheredInput):
                            result = result.value
                        torch.testing.assert_close(
                            result, torch.full_like(value, context)
                        )
                        callback = (
                            ("gather", context)
                            if route == "gather"
                            else (
                                "complete",
                                SumGroup.ATTN_TP,
                                route == "scatter",
                                context,
                            )
                        )
                        self.assertEqual(events, [("bind", context), callback])

    def test_output_transform_preserves_invocation_binding_and_read_order(self):
        from sglang.srt.layers.layer_boundary import prepare
        from sglang.srt.layers.layer_boundary.boundary import bind_entry
        from sglang.srt.layers.layer_boundary.contracts import (
            EdgeContract,
            InputContract,
            OutputContract,
        )
        from sglang.srt.layers.layer_boundary.layout import Layout
        from sglang.srt.layers.layer_boundary.residual.stream import DeclaredSum

        events = []
        batch = SimpleNamespace()
        forbidden = Mock(side_effect=AssertionError("fusion bypassed transform"))

        class Read(NormReadout):
            completing_fusions = (
                ReadoutFusion(SumGroup.ATTN_TP, forbidden, reads=True, carried=True),
            )
            gathering_reads = (forbidden,)

            def __init__(self, context=None):
                self.context = context

            def bind(self, forward_batch, context):
                self_test.assertIs(forward_batch, batch)
                events.append(("bind", context))
                return Read(context)

            def update_and_read(self, update, value, residual, norm, **kwargs):
                assert self.context is not None, "unbound fallback read"
                events.append(("read", self.context))
                written = update.update(value, residual)
                return written + self.context, written

        def reduce(value, forward_batch, **kwargs):
            self.assertIs(forward_batch, batch)
            events.append("sum")
            return value + 3

        def transform(value):
            events.append("transform")
            return value.square()

        self_test = self
        full = Layout(frozenset())
        with patch.object(prepare, "attn_tp_all_reduce", side_effect=reduce):
            for bindable in (True, False):
                reader = Read() if bindable else Read(0)
                if not bindable:
                    reader.bind = None
                edge = EdgeContract(
                    OutputContract(
                        full,
                        group=SumGroup.ATTN_TP,
                        always_partial=True,
                        transform=OutputTransform(transform),
                    ),
                    InputContract(full, read=reader),
                    full,
                    full,
                )
                entry = bind_entry(edge)
                for context in (17, 29):
                    with self.subTest(bindable=bindable, context=context):
                        events.clear()
                        value = torch.ones(2, 4)
                        result, residual = entry.prepare(
                            value,
                            value,
                            batch,
                            None,
                            pending=SimpleNamespace(owed=DeclaredSum(SumGroup.ATTN_TP)),
                            read_context=context,
                        )
                        effective = context if bindable else 0
                        torch.testing.assert_close(result, value * (17 + effective))
                        torch.testing.assert_close(residual, value * 17)
                        self.assertEqual(
                            events,
                            ([("bind", context)] if bindable else [])
                            + ["sum", "transform", ("read", effective)],
                        )
                forbidden.assert_not_called()

    def test_invocation_read_reaches_normal_and_cp_consumers(self):
        from sglang.srt.layers.layer_boundary import prepare
        from sglang.srt.layers.layer_boundary.boundary import CpMoves, bind_entry
        from sglang.srt.layers.layer_boundary.contracts import (
            EdgeContract,
            InputContract,
            OutputContract,
        )
        from sglang.srt.layers.layer_boundary.layout import Layout, TokenAxis

        class Read(NormReadout):
            def bind(self, batch, context):
                bound = Read()
                bound.bind = None
                bound.context = context
                return bound

            def update_and_read(self, update, value, residual, norm, **kw):
                return value + residual + self.context, residual

        full = Layout(frozenset())
        local = Layout(frozenset({TokenAxis.ATTN_CP}))
        moves = CpMoves(
            gather=prepare._then_moe_cp_gather, take_back=lambda h, r, b: (h[:2], r)
        )
        batch = SimpleNamespace(
            attn_cp_metadata=SimpleNamespace(per_rank_actual_token=[2, 2])
        )
        for shape in ("local", "gather", "move"):
            with self.subTest(shape=shape):
                source = full if shape in ("local", "move") else local
                residual_rows = local if shape in ("gather", "move") else full
                target = full if shape == "gather" else residual_rows
                edge = EdgeContract(
                    OutputContract(source),
                    InputContract(target, read=Read()),
                    residual_rows,
                    residual_rows,
                )
                entry = bind_entry(edge, cp_moves=moves)
                with (
                    patch.object(
                        prepare,
                        "moe_cp_gather",
                        side_effect=lambda h, *a: torch.cat([h, h]),
                    ),
                    patch.object(prepare, "get_moe_cp_size", return_value=2),
                ):
                    for context in (3, 7):
                        value, _ = entry.prepare(
                            torch.ones(4 if shape == "move" else 2, 4),
                            torch.ones(2, 4),
                            batch,
                            None,
                            read_context=context,
                        )
                        torch.testing.assert_close(
                            value, torch.full_like(value, context + 2)
                        )


if __name__ == "__main__":
    unittest.main()
