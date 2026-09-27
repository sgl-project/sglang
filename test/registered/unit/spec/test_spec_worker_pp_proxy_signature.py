"""Spec-v2 workers must accept the scheduler's pp_proxy_tensors kwarg (#40155).

The non-overlap scheduler path always calls
``model_worker.forward_batch_generation(..., pp_proxy_tensors=...)``.
Workers that omit the parameter die on the first step with TypeError.
Pin the parameter on every concrete override under speculative/.
"""

import ast
import unittest
from pathlib import Path

from sglang.srt.speculative import base_spec_worker as base_spec_worker_mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SPEC_ROOT = Path(base_spec_worker_mod.__file__).resolve().parent

# Classes that define forward_batch_generation and must accept pp_proxy_tensors.
# BaseSpecWorker is the ABC; StandaloneWorkerV2 inherits EAGLEWorkerV2's method.
_REQUIRED_PP_PROXY_WORKERS = frozenset(
    {
        "EAGLEWorkerV2",
        "FrozenKVMTPWorkerV2",
        "MultiLayerEagleWorkerV2",
        "DFlashWorkerV2",
        "DSparkWorkerV2",
        "NGRAMWorker",
        "UnoWorkerV2",
    }
)


def _forward_batch_generation_defs(root: Path) -> dict[str, ast.FunctionDef]:
    found: dict[str, ast.FunctionDef] = {}
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == "forward_batch_generation"
                ):
                    found[node.name] = item
    return found


def _arg_names(func: ast.FunctionDef) -> set[str]:
    args = func.args
    names = {arg.arg for arg in args.args}
    names.update(arg.arg for arg in args.kwonlyargs)
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)
    return names


class TestSpecWorkerPpProxySignature(CustomTestCase):
    def test_every_concrete_worker_accepts_pp_proxy_tensors(self):
        defs = _forward_batch_generation_defs(_SPEC_ROOT)
        missing_classes = sorted(_REQUIRED_PP_PROXY_WORKERS - defs.keys())
        self.assertEqual(
            missing_classes,
            [],
            "expected worker classes missing forward_batch_generation: "
            + ", ".join(missing_classes),
        )
        missing_param = sorted(
            name
            for name in _REQUIRED_PP_PROXY_WORKERS
            if "pp_proxy_tensors" not in _arg_names(defs[name])
            and "kwargs" not in _arg_names(defs[name])
        )
        self.assertEqual(
            missing_param,
            [],
            "scheduler always passes pp_proxy_tensors; missing on: "
            + ", ".join(missing_param),
        )

    def test_base_declares_abstract_forward_batch_generation(self):
        defs = _forward_batch_generation_defs(_SPEC_ROOT)
        self.assertIn("BaseSpecWorker", defs)
        base_def = defs["BaseSpecWorker"]
        self.assertTrue(
            any(
                isinstance(dec, ast.Name) and dec.id == "abstractmethod"
                for dec in base_def.decorator_list
            )
            or any(
                isinstance(dec, ast.Attribute) and dec.attr == "abstractmethod"
                for dec in base_def.decorator_list
            ),
            "BaseSpecWorker.forward_batch_generation must be abstract",
        )
        self.assertIn("pp_proxy_tensors", _arg_names(base_def))

    def test_required_list_covers_all_concrete_overrides(self):
        """Fail when a new worker overrides forward_batch_generation without
        being added to _REQUIRED_PP_PROXY_WORKERS."""
        defs = _forward_batch_generation_defs(_SPEC_ROOT)
        extras = set(defs) - _REQUIRED_PP_PROXY_WORKERS - {"BaseSpecWorker"}
        self.assertEqual(
            extras,
            set(),
            "new worker overrides forward_batch_generation; add it to "
            "_REQUIRED_PP_PROXY_WORKERS: " + ", ".join(sorted(extras)),
        )


if __name__ == "__main__":
    unittest.main()
