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

# Calls that complete a TP / EP sum of a stage output in compute.
_SUMS = {
    "reduce_moe_output",
    "post_experts_all_reduce",
    "tensor_model_parallel_all_reduce",
    "moe_expert_parallel_all_reduce",
    "moe_tensor_model_parallel_all_reduce",
}
_BUILDS_STAGES = (
    "make_stages(",
    "declare_attn(",
    "declare_ffn(",
    "_build_stages(",
    "_init_stage_boundary(",
)


# A vocabulary-parallel lookup leaves each rank holding only its own shard's
# rows, so the class completes that sum itself. An embedding is never a decoder
# stage, so its sum is never a stage output.
_VOCAB_PARALLEL_BASES = ("VocabParallelEmbedding", "ParallelLMHead")


def _is_vocab_parallel(node) -> bool:
    return isinstance(node, ast.ClassDef) and any(
        base in _VOCAB_PARALLEL_BASES
        for base in (ast.unparse(parent) for parent in node.bases)
    )


def _call_name(node: ast.Call):
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)


def _unguarded_sums(source: str):
    """Sum calls not under ``if ... self.reduce_results ...``: such a branch is
    a shared class built without stage boundaries by another model. Sums inside
    a vocabulary-parallel embedding complete its lookup, not a stage output."""
    found = []

    def visit(node, guarded):
        if _is_vocab_parallel(node):
            return
        if isinstance(node, ast.If) and "reduce_results" in ast.unparse(node.test):
            for child in node.body:
                visit(child, True)
            for child in node.orelse:
                visit(child, guarded)
            return
        if isinstance(node, ast.Call) and not guarded and _call_name(node) in _SUMS:
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
