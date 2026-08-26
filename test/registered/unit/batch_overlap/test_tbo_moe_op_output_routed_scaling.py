"""Regression: TBO op_output must not re-apply an already-fused routed_scaling_factor.

DeepseekV2MoE.op_output is the epilogue of the two-batch-overlap (TBO)
DeepEP MoE pipeline. It multiplied the routed output by
self.routed_scaling_factor in every non-aiter case (even when the factor was
already folded into topk_weights), while the equivalent non-TBO
path forward_deepep guards that multiply with
experts.should_fuse_routed_scaling_factor_in_topk or _use_aiter.

When the guard flag is true the factor is already applied upstream: topk.py
folds it into topk_weights for quant methods that set
should_fuse_routed_scaling_factor_in_topk (e.g. ModelOpt NVFP4), and
aiter folds it into aiter_biased_grouped_topk. The TBO epilogue therefore
applied it a second time and every TBO MoE layer silently returned output
scaled by routed_scaling_factor**2 (2.5**2 for DeepSeek-V3).

The verbatim op_output source is AST-extracted from the model file and
executed against a stub state, so this test needs no GPU, weights, or server
and runs on CPU in milliseconds.

Usage:
    python -m pytest test/registered/unit/batch_overlap/test_tbo_moe_op_output_routed_scaling.py -v
"""

import ast
import importlib.util
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

# CPU-only unit test; no CUDA/distributed dependencies.
register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ROUTED_SCALING_FACTOR = 2.5  # DeepSeek-V3-family value


def _model_file() -> Path:
    """Locate deepseek_v2.py in this checkout (fall back to installed sglang)."""
    repo_file = (
        Path(__file__).resolve().parents[4]
        / "python"
        / "sglang"
        / "srt"
        / "models"
        / "deepseek_v2.py"
    )
    if repo_file.exists():
        return repo_file
    spec = importlib.util.find_spec("sglang")
    assert spec is not None and spec.origin is not None, "sglang not importable"
    return Path(spec.origin).resolve().parent / "srt" / "models" / "deepseek_v2.py"


def _extract_op_output_source() -> str:
    path = _model_file()
    source = path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekV2MoE":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "op_output":
                    return textwrap.dedent(ast.get_source_segment(source, item))
    raise AssertionError(f"DeepseekV2MoE.op_output not found in {path}")


class _StubA2ABackend:
    """op_output's mori fast path is a slice-only op; keep it disabled."""

    def is_mori(self):
        return False


def _load_op_output(use_aiter: bool):
    """Bind the verbatim op_output source to a stub module namespace."""
    namespace = {
        "torch": torch,
        "_use_aiter": use_aiter,
        "get_moe_a2a_backend": lambda: _StubA2ABackend(),
    }
    exec(compile(_extract_op_output_source(), str(_model_file()), "exec"), namespace)
    return namespace["op_output"]


class _StubState:
    """The _StateDict surface op_output touches: attribute get/set + pop."""

    def __init__(self, **fields):
        object.__setattr__(self, "_fields", fields)

    def pop(self, name):
        return self._fields.pop(name)

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, "_fields")[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self._fields[name] = value


class TestTboOpOutputRoutedScalingFactor(unittest.TestCase):
    def _run_op_output(self, should_fuse, use_aiter, has_shared):
        """Return (op_output result, routed combine output, shared output)."""
        torch.manual_seed(0)
        combine = torch.randn(4, 6)
        shared = torch.randn(4, 6) if has_shared else None
        state = _StubState(
            hidden_states_after_combine=combine.clone(),
            shared_output=shared.clone() if shared is not None else None,
        )
        moe = SimpleNamespace(
            routed_scaling_factor=ROUTED_SCALING_FACTOR,
            experts=SimpleNamespace(
                should_fuse_routed_scaling_factor_in_topk=should_fuse
            ),
        )
        _load_op_output(use_aiter)(moe, state)
        return state.hidden_states_mlp_output, combine, shared

    def test_routed_scaling_factor_applied_exactly_once(self):
        # All 2**3 combos of (folded-in-topk, aiter, shared experts): the
        # routed output must carry exactly one factor of
        # routed_scaling_factor, whether it was folded upstream or applied
        # here.
        for should_fuse in (False, True):
            for use_aiter in (False, True):
                for has_shared in (False, True):
                    with self.subTest(
                        should_fuse=should_fuse,
                        use_aiter=use_aiter,
                        has_shared=has_shared,
                    ):
                        out, combine, shared = self._run_op_output(
                            should_fuse, use_aiter, has_shared
                        )
                        folded_upstream = should_fuse or use_aiter
                        scale = 1.0 if folded_upstream else ROUTED_SCALING_FACTOR
                        expected = combine * scale
                        if shared is not None:
                            expected = expected + shared
                        torch.testing.assert_close(out, expected)

    def test_fold_flag_changes_the_result(self):
        # Guard against the parametrized test passing vacuously: with the
        # fold flag on, the pre-fix epilogue returned
        # combine * routed_scaling_factor (the factor squared in total),
        # which must differ from the correct output.
        out, combine, _ = self._run_op_output(
            should_fuse=True, use_aiter=False, has_shared=False
        )
        self.assertFalse(
            torch.allclose(out, combine * ROUTED_SCALING_FACTOR),
            "op_output re-applied the already-fused routed_scaling_factor",
        )


if __name__ == "__main__":
    unittest.main()
