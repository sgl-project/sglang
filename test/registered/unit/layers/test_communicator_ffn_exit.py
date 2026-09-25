import ast
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

import sglang
from sglang.srt.layers import communicator as comm
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    UnreducedOutput,
    reduce_output,
)
from sglang.srt.runtime_context import get_forward
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_group(scale=3):
    """The group a deferred sum is owed over, with a stubbed all-reduce."""
    return types.SimpleNamespace(all_reduce=MagicMock(side_effect=lambda h: h * scale))


def make_communicator(
    *,
    fuse,
    reduce_scatter,
    reduce_scatter_step=None,
    scatters_to_local_tokens=False,
    group=None,
    cls=LayerCommunicator,
):
    """A communicator whose decisions and postprocess are stubbed, built
    without the process-wide parallel state."""
    communicator = cls.__new__(cls)
    if cls is LayerCommunicator:
        communicator.should_defer_ffn_reduction = MagicMock(return_value=fuse)
    communicator.should_use_reduce_scatter = MagicMock(return_value=reduce_scatter)
    communicator.is_last_layer = False
    communicator._sp_variant = None
    communicator._postprocess_scatters_to_local_tokens = scatters_to_local_tokens
    communicator._reduce_scatter_step = MagicMock(return_value=reduce_scatter_step)
    communicator.ffn_reduction_group = MagicMock(return_value=group or make_group())
    communicator.postprocess_layer = MagicMock(
        side_effect=lambda hidden_states, residual, forward_batch: (
            hidden_states + 1,
            residual,
        )
    )
    return communicator


def published_flags():
    forward = get_forward()
    return forward.fuse_mlp_allreduce, forward.mlp_reduce_scatter


class TestFfnExit(CustomTestCase):
    def setUp(self):
        self.hidden_states = torch.ones(3, 4)
        self.residual = torch.zeros(3, 4)
        self.forward_batch = object()

    def run_exit(self, communicator):
        with communicator.ffn_exit(self.forward_batch) as ffn_exit:
            seen = published_flags()
            hidden_states = self.hidden_states * 2
        return seen, ffn_exit.finish(hidden_states, self.residual)

    def test_reduction_left_to_next_layer_is_unreduced(self):
        communicator = make_communicator(fuse=True, reduce_scatter=False)
        seen, (hidden_states, residual) = self.run_exit(communicator)
        self.assertEqual(seen, (True, False))
        self.assertIsInstance(hidden_states, UnreducedOutput)
        self.assertIs(residual, self.residual)
        communicator.postprocess_layer.assert_not_called()

    def test_reduction_left_to_next_layer_declares_its_group(self):
        group = make_group()
        communicator = make_communicator(fuse=True, reduce_scatter=False, group=group)
        _, (hidden_states, _) = self.run_exit(communicator)
        self.assertIs(hidden_states.group, group)
        self.assertIsNone(hidden_states.reduce_and_redistribute)

    def test_a_reduce_scatter_is_left_to_the_next_layer(self):
        """Under attention DP the reduce-scatter postprocess would run goes to the
        next layer with the partial sum."""
        step = MagicMock()
        communicator = make_communicator(
            fuse=False,
            reduce_scatter=True,
            reduce_scatter_step=step,
            scatters_to_local_tokens=True,
        )
        seen, (hidden_states, residual) = self.run_exit(communicator)
        self.assertEqual(seen, (False, True))
        self.assertIsInstance(hidden_states, UnreducedOutput)
        bound = hidden_states.reduce_and_redistribute
        self.assertIs(bound.func, comm._to_local_tokens)
        self.assertEqual(bound.args, (step, self.forward_batch))
        self.assertIs(residual, self.residual)
        communicator.postprocess_layer.assert_not_called()
        step.assert_not_called()

    def test_a_deferred_sum_carries_the_scatter_back_under_attention_dp(self):
        group = make_group()
        communicator = make_communicator(
            fuse=True, reduce_scatter=False, scatters_to_local_tokens=True, group=group
        )
        _, (hidden_states, _) = self.run_exit(communicator)
        self.assertIsInstance(hidden_states, UnreducedOutput)
        bound = hidden_states.reduce_and_redistribute
        self.assertIs(bound.func, comm._all_reduce_then_to_local_tokens)
        self.assertEqual(bound.args, (group, self.forward_batch))
        communicator._reduce_scatter_step.assert_not_called()
        communicator.postprocess_layer.assert_not_called()
        group.all_reduce.assert_not_called()

    def test_postprocess_completes_other_exits(self):
        for reduce_scatter in (False, True):
            with self.subTest(reduce_scatter=reduce_scatter):
                communicator = make_communicator(
                    fuse=False, reduce_scatter=reduce_scatter
                )
                seen, (hidden_states, residual) = self.run_exit(communicator)
                self.assertEqual(seen, (False, reduce_scatter))
                self.assertNotIsInstance(hidden_states, UnreducedOutput)
                communicator.postprocess_layer.assert_called_once()
                torch.testing.assert_close(hidden_states, self.hidden_states * 2 + 1)
                self.assertIs(residual, self.residual)

    def test_finish_applies_the_decision_the_ffn_saw(self):
        """finish() applies the decision published to the FFN, even when the
        decision methods would answer differently by then."""
        for fuse in (True, False):
            with self.subTest(fuse=fuse):
                communicator = make_communicator(fuse=fuse, reduce_scatter=False)
                with communicator.ffn_exit(self.forward_batch) as ffn_exit:
                    seen = published_flags()
                    decide = communicator.should_defer_ffn_reduction
                    decide.return_value = not fuse
                    hidden_states = self.hidden_states * 2
                hidden_states, _ = ffn_exit.finish(hidden_states, self.residual)
                self.assertEqual(seen, (fuse, False))
                self.assertEqual(isinstance(hidden_states, UnreducedOutput), fuse)
                self.assertEqual(communicator.postprocess_layer.called, not fuse)

    def test_flags_are_restored_after_the_ffn(self):
        before = published_flags()
        self.run_exit(make_communicator(fuse=True, reduce_scatter=True))
        self.assertEqual(published_flags(), before)

    def test_subclass_decisions_are_used(self):
        class NeverDefers(LayerCommunicator):
            def should_defer_ffn_reduction(self, forward_batch):
                return False

        communicator = make_communicator(
            fuse=True, reduce_scatter=False, cls=NeverDefers
        )
        seen, _ = self.run_exit(communicator)
        self.assertEqual(seen, (False, False))
        communicator.postprocess_layer.assert_called_once()

    def test_deferral_implies_fusion_and_passes_the_handoff_through(self):
        """A deferring communicator publishes both flags, and a non-tensor
        handoff leaves finish() untouched for the next layer's input norm."""

        class Defers(LayerCommunicator):
            def should_defer_moe_finalize(self, forward_batch, m=None):
                return True

            def should_fuse_mlp_allreduce_with_next_layer(self, forward_batch):
                return False

        communicator = make_communicator(fuse=False, reduce_scatter=False, cls=Defers)
        handoff = object()
        with communicator.ffn_exit(self.forward_batch) as ffn_exit:
            seen = published_flags() + (get_forward().defer_moe_finalize,)
        self.assertEqual(seen, (True, False, True))
        self.assertEqual(
            ffn_exit.finish(handoff, self.residual), (handoff, self.residual)
        )
        communicator.postprocess_layer.assert_not_called()
        self.assertFalse(get_forward().defer_moe_finalize)

        # The MoE may decline per forward; a tensor then takes the fused exit.
        hidden_states, _ = ffn_exit.finish(self.hidden_states * 2, self.residual)
        self.assertIsInstance(hidden_states, UnreducedOutput)


class TestReduceOutput(CustomTestCase):
    def setUp(self):
        self.group = make_group()
        self.all_reduce = self.group.all_reduce

    def communicator(self, *, fuse):
        return make_communicator(fuse=fuse, reduce_scatter=False, group=self.group)

    def test_reduction_left_by_the_last_layer_runs_once(self):
        communicator = self.communicator(fuse=True)
        with communicator.ffn_exit(object()) as ffn_exit:
            hidden_states = torch.ones(3, 4)
        hidden_states, _ = ffn_exit.finish(hidden_states, torch.zeros(3, 4))

        hidden_states = reduce_output(hidden_states)
        self.all_reduce.assert_called_once()
        torch.testing.assert_close(hidden_states, torch.full((3, 4), 3.0))
        self.assertNotIsInstance(hidden_states, UnreducedOutput)

        reduce_output(hidden_states)
        self.all_reduce.assert_called_once()

    def test_a_reduce_scatter_left_by_the_last_layer_runs_once(self):
        local = torch.full((1, 4), 7.0)
        step = MagicMock(return_value=local)
        partial = torch.ones(3, 4)
        hidden_states = reduce_output(
            UnreducedOutput(partial, reduce_and_redistribute=step)
        )
        step.assert_called_once_with(partial)
        self.assertIs(hidden_states, local)
        self.all_reduce.assert_not_called()

    def test_complete_hidden_states_pass_through(self):
        communicator = self.communicator(fuse=False)
        with communicator.ffn_exit(object()) as ffn_exit:
            hidden_states = torch.ones(3, 4)
        hidden_states, _ = ffn_exit.finish(hidden_states, torch.zeros(3, 4))

        self.assertIs(reduce_output(hidden_states), hidden_states)
        self.assertIsNone(reduce_output(None))
        self.all_reduce.assert_not_called()

    def test_finish_layer_stack_completes_the_last_layer(self):
        for fuse in (True, False):
            with self.subTest(fuse=fuse):
                self.all_reduce.reset_mock()
                communicator = self.communicator(fuse=fuse)
                residual = torch.zeros(3, 4)
                with communicator.ffn_exit(object()) as ffn_exit:
                    hidden_states = torch.ones(3, 4)
                hidden_states, _ = ffn_exit.finish(hidden_states, residual)

                hidden_states, residual_out = communicator.finish_layer_stack(
                    hidden_states, residual, object()
                )
                self.assertEqual(self.all_reduce.call_count, int(fuse))
                expected = 3.0 if fuse else 2.0  # all-reduce stub / postprocess stub
                torch.testing.assert_close(hidden_states, torch.full((3, 4), expected))
                self.assertIs(residual_out, residual)


class TestSelectFfnCompletion(CustomTestCase):
    """What the next layer's input runs in place of postprocess, chosen before
    the FFN runs."""

    def communicator(
        self, *, fuse=False, is_last_layer=False, scatters=True, sp_variant=None
    ):
        communicator = LayerCommunicator.__new__(LayerCommunicator)
        communicator.is_last_layer = is_last_layer
        communicator._sp_variant = sp_variant
        communicator._postprocess_scatters_to_local_tokens = scatters
        communicator.allow_reduce_scatter = True
        communicator.layer_scatter_modes = types.SimpleNamespace(is_layer_sparse=True)
        communicator.should_defer_ffn_reduction = lambda forward_batch: fuse
        communicator.should_use_reduce_scatter = lambda forward_batch: not fuse
        self.group = make_group()
        communicator.ffn_reduction_group = lambda: self.group
        return communicator

    def left(self, communicator, step, forward_batch=None):
        with patch.object(
            comm, "_reduce_and_redistribute_output_step", return_value=step
        ):
            completion = communicator._select_ffn_completion(forward_batch)
        if completion.leave is None:
            return None
        return completion.leave(torch.ones(3, 4))

    def test_a_reduce_scatter_is_bound_for_the_next_layer(self):
        forward_batch = object()
        for step in (
            comm._reduce_and_redistribute_output_varlen,
            comm._reduce_and_redistribute_output_max_len,
        ):
            with self.subTest(step=step.__name__):
                left = self.left(self.communicator(), step, forward_batch)
                bound = left.reduce_and_redistribute
                self.assertIs(bound.func, comm._to_local_tokens)
                self.assertEqual(bound.args, (step, forward_batch))

    def test_postprocess_keeps_everything_else(self):
        reduce_scatter = comm._reduce_and_redistribute_output_varlen
        for name, communicator, step in (
            ("scatter only", self.communicator(), None),
            ("last layer", self.communicator(is_last_layer=True), reduce_scatter),
            ("other postprocess", self.communicator(scatters=False), reduce_scatter),
        ):
            with self.subTest(name):
                self.assertIsNone(self.left(communicator, step))

    def test_postprocess_keeps_an_active_layernorm_sp_region(self):
        communicator = self.communicator(sp_variant=object())
        with get_forward().scoped(sp_active=True):
            self.assertIsNone(
                self.left(communicator, comm._reduce_and_redistribute_output_varlen)
            )

    def test_a_deferred_sum_keeps_its_layout_or_scatters_back(self):
        forward_batch = object()
        kept = self.left(self.communicator(fuse=True, scatters=False), None)
        self.assertIs(kept.group, self.group)
        self.assertIsNone(kept.reduce_and_redistribute)
        moved = self.left(self.communicator(fuse=True), None, forward_batch)
        self.assertIsNone(moved.group)
        bound = moved.reduce_and_redistribute
        self.assertIs(bound.func, comm._all_reduce_then_to_local_tokens)
        self.assertEqual(bound.args, (self.group, forward_batch))

    def test_the_all_reduce_runs_before_the_scatter(self):
        calls = []
        partial = torch.ones(3, 4)
        local = torch.empty(1, 4)
        group = types.SimpleNamespace(
            all_reduce=lambda x: calls.append(("all_reduce", x)) or x * 2
        )
        with (
            patch.object(comm, "_dp_scatter_group", return_value="group"),
            patch.object(
                comm,
                "get_local_dp_buffer",
                side_effect=lambda group: calls.append(("buffer", group)) or local,
            ),
            patch.object(
                comm,
                "dp_scatter",
                side_effect=lambda out, full, fb: calls.append(("scatter", out)),
            ),
        ):
            result = comm._all_reduce_then_to_local_tokens(group, object(), partial)
        self.assertEqual([c[0] for c in calls], ["all_reduce", "buffer", "scatter"])
        self.assertIs(calls[0][1], partial)
        self.assertIs(result, local)


SRT_DIR = Path(sglang.__file__).resolve().parent / "srt"
# Attributes a decoder layer holds its FFN, or parts of it, in.
FFN_ATTRS = {"mlp", "moe", "shared_expert", "shared_experts", "share_expert"}


def can_be_true(value, params):
    """Whether a use_dp_attention_reduce value can turn it on: anything but
    False or a function forwarding its own parameter of that name."""
    if isinstance(value, ast.Constant) and value.value is False:
        return False
    return not (isinstance(value, ast.Name) and value.id in params)


class SourceIndex:
    def __init__(self):
        self.paths = {
            "sglang.srt." + str(p.relative_to(SRT_DIR))[:-3].replace("/", "."): p
            for p in SRT_DIR.rglob("*.py")
        }
        self._trees = {}

    def tree(self, module):
        if module not in self._trees:
            self._trees[module] = ast.parse(self.paths[module].read_text())
        return self._trees[module]

    def resolve(self, module, name):
        """The (module, ClassDef) that `name` refers to in `module`, if any."""
        for node in self.tree(module).body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                return module, node
        for node in ast.walk(self.tree(module)):
            if isinstance(node, ast.ImportFrom) and node.module in self.paths:
                for alias in node.names:
                    if (alias.asname or alias.name) == name:
                        return self.resolve(node.module, alias.name)
        return None


def submodule_constructions(cls):
    """(attribute, class name, call) for each `self.attr = Name(...)`."""
    for node in ast.walk(cls):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            for target in node.targets:
                if (
                    isinstance(func, ast.Name)
                    and isinstance(target, ast.Attribute)
                    and getattr(target.value, "id", None) == "self"
                ):
                    yield target.attr, func.id, node.value


def attention_tp_reductions(cls):
    """Lines in a class that can turn on use_dp_attention_reduce."""
    found = []
    for fn in ast.walk(cls):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        for node in ast.walk(fn):
            if isinstance(node, ast.keyword) and node.arg == "use_dp_attention_reduce":
                if can_be_true(node.value, params):
                    found.append(node.value.lineno)
            elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Attribute) and t.attr == "use_dp_attention_reduce"
                for t in node.targets
            ):
                if can_be_true(node.value, params):
                    found.append(node.lineno)
    return found


class TestFfnExitUsersOweATpSum(CustomTestCase):
    """Under attention DP the next layer's input completes a deferred FFN sum
    with an all-reduce over TP. An FFN whose linear layers reduce over the
    attention-TP group only (use_dp_attention_reduce) owes a different sum, so
    no FFN of a layer that uses ffn_exit may turn it on."""

    def test_no_ffn_behind_ffn_exit_reduces_over_attention_tp(self):
        index = SourceIndex()
        layers, ffn_classes, offenders = [], set(), []
        for module, path in sorted(index.paths.items()):
            source = path.read_text()
            if (
                not module.startswith("sglang.srt.models.")
                or ".ffn_exit(" not in source
            ):
                continue
            for cls in index.tree(module).body:
                if not isinstance(cls, ast.ClassDef):
                    continue
                if ".ffn_exit(" not in ast.get_source_segment(source, cls):
                    continue
                layers.append(f"{module}.{cls.name}")
                todo = []
                for attr, name, call in submodule_constructions(cls):
                    if attr not in FFN_ATTRS:
                        continue
                    todo.append((module, name))
                    offenders += [
                        f"{module}.{cls.name}:{kw.value.lineno}"
                        for kw in call.keywords
                        if kw.arg == "use_dp_attention_reduce"
                        and can_be_true(kw.value, set())
                    ]
                # Everything an FFN class builds is part of the FFN.
                while todo:
                    resolved = index.resolve(*todo.pop())
                    if resolved is None:
                        continue
                    where, ffn = resolved
                    if (where, ffn.name) in ffn_classes:
                        continue
                    ffn_classes.add((where, ffn.name))
                    offenders += [
                        f"{where}.{ffn.name}:{line}"
                        for line in attention_tp_reductions(ffn)
                    ]
                    todo += [(where, n) for _, n, _ in submodule_constructions(ffn)]
        self.assertTrue(layers and ffn_classes, "no FFN behind ffn_exit found")
        self.assertEqual(sorted(set(offenders)), [])


if __name__ == "__main__":
    unittest.main()
