"""Stage producers leave their output sums to the stage boundary."""

import ast
import unittest
from pathlib import Path

import torch

import sglang.srt.models
from sglang.srt.layers.layer_boundary.stage import (
    StageBoundary,
    check_stage_producers,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Calls that complete a TP / EP sum of a stage output in compute. A fused
# kernel that carries the sum inside it counts as well: every MoE finalize
# all-reduce variant does.
_SUMS = {
    "reduce_moe_output",
    "post_experts_all_reduce",
    "tensor_model_parallel_all_reduce",
    "moe_expert_parallel_all_reduce",
    "moe_tensor_model_parallel_all_reduce",
}
_SUM_PREFIXES = ("moe_finalize_all_reduce",)


def _is_sum(name) -> bool:
    return name in _SUMS or (name is not None and name.startswith(_SUM_PREFIXES))


_BUILDS_STAGES = (
    "append_stages(",
    "declare_attn(",
    "declare_ffn(",
    "_build_stages(",
    "_init_stage_boundary(",
)


# A vocabulary-parallel lookup leaves each rank holding only its own shard's
# rows, so the class completes that sum itself. An embedding is never a decoder
# stage, so no sum inside one is a stage output and the whole class is exempt.
# Only a class that names one of these bases directly is exempt.
_VOCAB_PARALLEL_BASES = ("VocabParallelEmbedding", "ParallelLMHead")


def _is_vocab_parallel(node) -> bool:
    return isinstance(node, ast.ClassDef) and any(
        base in _VOCAB_PARALLEL_BASES
        for base in (ast.unparse(parent) for parent in node.bases)
    )


def _call_name(node: ast.Call):
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)


def _requires(expr, names) -> bool:
    """Whether ``expr`` holds only when one of ``names`` is true: the flag
    itself, or an ``and`` with it as a conjunct. A negation, a comparison or an
    ``or`` does not require it, so it guards nothing."""
    if isinstance(expr, ast.Attribute):
        return expr.attr in names
    if isinstance(expr, ast.Name):
        return expr.id in names
    if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.And):
        return any(_requires(value, names) for value in expr.values)
    return False


def _reduce_results_names(tree) -> set:
    """``reduce_results`` and the names bound to a condition that requires it.
    A branch on such a name is guarded like a branch on the condition."""
    names = {"reduce_results"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _requires(node.value, names):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _unguarded_sums(source: str):
    """Sum calls not under ``if ... self.reduce_results ...``: such a branch is
    a shared class built without stage boundaries by another model. Sums inside
    a vocabulary-parallel embedding complete its lookup, not a stage output.
    Nor under ``if ... enable_cp_tp_group_sharing ...``: there linear attention
    partitions its heads over the TP group while attention TP is 1, so the
    boundary owes no sum and the layer completes the partition's own."""
    found = []
    guards = _reduce_results_names(ast.parse(source)) | {"enable_cp_tp_group_sharing"}

    def visit(node, guarded):
        if _is_vocab_parallel(node):
            return
        if isinstance(node, ast.If) and _requires(node.test, guards):
            for child in node.body:
                visit(child, True)
            for child in node.orelse:
                visit(child, guarded)
            return
        if isinstance(node, ast.Call) and not guarded and _is_sum(_call_name(node)):
            found.append(f"{node.lineno}: {ast.unparse(node)}")
        for child in ast.iter_child_nodes(node):
            visit(child, guarded)

    visit(ast.parse(source), False)
    return found


class TestStageProducersLeaveSums(CustomTestCase):
    def test_stage_models_do_not_sum_stage_outputs_in_compute(self):
        (models,) = map(Path, sglang.srt.models.__path__)
        checked, offenders = 0, {}
        for path in sorted(models.rglob("*.py")):
            source = path.read_text()
            if not any(marker in source for marker in _BUILDS_STAGES):
                continue
            checked += 1
            if sums := _unguarded_sums(source):
                offenders[str(path.relative_to(models))] = sums
        self.assertGreater(checked, 30)
        self.assertEqual(offenders, {})

    def test_fused_sums_count_and_only_a_required_flag_guards(self):
        source = (
            "class Moe:\n"
            "    def forward(self, x):\n"
            "        fused = self.reduce_results and x.is_cuda\n"
            "        if fused:\n"
            "            moe_finalize_all_reduce_mhc_quant(x)\n"
            "        skip = not self.reduce_results\n"
            "        if skip:\n"
            "            moe_finalize_all_reduce_mhc_combine(x)\n"
            "        if not self.reduce_results:\n"
            "            tensor_model_parallel_all_reduce(x)\n"
            "        if self.reduce_results is False:\n"
            "            moe_finalize_all_reduce(x)\n"
            "        if x.is_cuda or self.reduce_results:\n"
            "            tensor_model_parallel_all_reduce(x)\n"
        )
        self.assertEqual(
            _unguarded_sums(source),
            [
                "8: moe_finalize_all_reduce_mhc_combine(x)",
                "10: tensor_model_parallel_all_reduce(x)",
                "12: moe_finalize_all_reduce(x)",
                "14: tensor_model_parallel_all_reduce(x)",
            ],
        )

    def test_guard_rejects_a_producer_that_sums_its_own_output(self):
        class Producer(torch.nn.Module):
            def __init__(self, reduce_results):
                super().__init__()
                self.reduce_results = reduce_results

        class Layer(torch.nn.Module):
            def __init__(self, reduce_results, *, staged=True):
                super().__init__()
                self.mlp = Producer(reduce_results)
                if staged:
                    self.ffn_boundary = object.__new__(StageBoundary)

        class Model(torch.nn.Module):
            def __init__(self, *layers):
                super().__init__()
                self.layers = torch.nn.ModuleList(layers)

        check_stage_producers(Model(Layer(False), Layer(True, staged=False)))
        with self.assertRaisesRegex(ValueError, r"layers\.1\.mlp"):
            check_stage_producers(Model(Layer(False), Layer(True)))


if __name__ == "__main__":
    unittest.main()
