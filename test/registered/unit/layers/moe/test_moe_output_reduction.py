"""When a MoE block all-reduces its own output, and why it does not."""

import ast
import contextlib
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import sglang
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.utils import (
    post_experts_output_is_complete,
    reduce_moe_output,
)
from sglang.srt.runtime_context import get_forward
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MODELS_DIR = Path(sglang.__file__).resolve().parent / "srt" / "models"


def a2a(name=None):
    names = ("flashinfer", "pplx", "flashinfer_megamoe")
    return types.SimpleNamespace(
        **{f"is_{n}": (lambda n=n: n == name) for n in names},
        is_none=lambda: name is None,
    )


@contextlib.contextmanager
def moe_config(
    *, tp_size=2, dwdp_size=1, backend=None, fp4_allgather=False, reduce_scatterv=False
):
    with (
        patch.object(
            moe_utils,
            "get_parallel",
            return_value=types.SimpleNamespace(tp_size=tp_size, dwdp_size=dwdp_size),
        ),
        patch.object(moe_utils, "get_moe_a2a_backend", return_value=a2a(backend)),
        patch.object(
            moe_utils,
            "should_use_flashinfer_cutlass_moe_fp4_allgather",
            return_value=fp4_allgather,
        ),
        patch.object(
            moe_utils, "should_use_dp_reduce_scatterv", return_value=reduce_scatterv
        ),
    ):
        yield


class TestReduceMoeOutput(CustomTestCase):
    def all_reduces(self, **flags):
        calls = []
        with (
            patch(
                "sglang.srt.distributed.communication_op.tensor_model_parallel_all_reduce",
                side_effect=lambda x: calls.append(x) or x * 2,
            ),
            get_forward().scoped(**flags),
        ):
            output = reduce_moe_output(torch.ones(2, 3))
        return len(calls), output

    def test_partial_output_is_reduced_once(self):
        with moe_config():
            count, output = self.all_reduces()
        self.assertEqual(count, 1)
        torch.testing.assert_close(output, torch.full((2, 3), 2.0))

    def test_single_rank_has_nothing_to_sum(self):
        with moe_config(tp_size=1):
            self.assertEqual(self.all_reduces()[0], 0)

    def test_a_later_step_owns_the_sum(self):
        for flags, config in (
            ({"fuse_mlp_allreduce": True}, {}),
            ({"mlp_reduce_scatter": True}, {}),
            ({}, {"reduce_scatterv": True}),
        ):
            with self.subTest(flags=flags, config=config), moe_config(**config):
                self.assertEqual(self.all_reduces(**flags)[0], 0)
                # Who runs the sum does not change whether one is owed.
                self.assertFalse(post_experts_output_is_complete(is_tp_path=True))

    def test_output_is_already_complete(self):
        for config in (
            {"backend": "flashinfer"},
            {"backend": "pplx"},
            {"backend": "flashinfer_megamoe"},
            {"dwdp_size": 2},
            {"fp4_allgather": True},
        ):
            with self.subTest(**config), moe_config(**config):
                self.assertEqual(self.all_reduces()[0], 0)
                self.assertTrue(post_experts_output_is_complete(is_tp_path=True))

    def test_fp4_allgather_only_completes_the_tp_sum(self):
        with moe_config(fp4_allgather=True):
            self.assertFalse(post_experts_output_is_complete(is_tp_path=False))


def open_coded_reductions(tree):
    """``if`` statements that test should_skip_post_experts_all_reduce for the TP
    path and call tensor_model_parallel_all_reduce themselves."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        tests_skip = any(
            isinstance(sub, ast.Call)
            and getattr(sub.func, "id", None) == "should_skip_post_experts_all_reduce"
            and any(
                kw.arg == "is_tp_path" and getattr(kw.value, "value", None) is True
                for kw in sub.keywords
            )
            for sub in ast.walk(node.test)
        )
        reduces = any(
            isinstance(sub, ast.Call)
            and getattr(sub.func, "id", None) == "tensor_model_parallel_all_reduce"
            for stmt in node.body
            for sub in ast.walk(stmt)
        )
        if tests_skip and reduces:
            yield node.lineno


class TestModelsUseTheSharedReduction(CustomTestCase):
    def test_no_model_open_codes_the_tp_reduction(self):
        found = [
            f"{path.relative_to(MODELS_DIR)}:{line}"
            for path in sorted(MODELS_DIR.rglob("*.py"))
            for line in open_coded_reductions(ast.parse(path.read_text()))
        ]
        self.assertEqual(found, [], "use reduce_moe_output")


if __name__ == "__main__":
    unittest.main()
