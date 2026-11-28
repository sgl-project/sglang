"""Spec workers must accept the pp_proxy_tensors the scheduler always passes.

The non-overlap speculative path calls
``model_worker.forward_batch_generation(batch, pp_proxy_tensors=...)``.
Workers that omit the keyword TypeError on the first request (#40155).
This pins the interface on BaseSpecWorker and on every spec worker that
implements the method, so a new worker without the kwarg fails here
instead of in serving.
"""

import ast
import inspect
import unittest
from pathlib import Path

from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SPEC_ROOT = Path(inspect.getfile(BaseSpecWorker)).resolve().parent

# Concrete workers that define forward_batch_generation. Inheritors that
# do not override (StandaloneWorkerV2) are covered by the parent class.
_REQUIRED_CLASSES = frozenset(
    {
        "BaseSpecWorker",
        "EAGLEWorkerV2",
        "DFlashWorkerV2",
        "NGRAMWorker",
        "DSparkWorkerV2",
        "MultiLayerEagleWorkerV2",
        "FrozenKVMTPWorkerV2",
        "UnoWorkerV2",
    }
)


def _param_names(func: ast.FunctionDef) -> set[str]:
    names = {arg.arg for arg in func.args.posonlyargs}
    names.update(arg.arg for arg in func.args.args)
    names.update(arg.arg for arg in func.args.kwonlyargs)
    return names


def _forward_batch_generation_classes(root: Path) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name == "forward_batch_generation"
                ):
                    found[node.name] = _param_names(item)
                    break
    return found


class TestSpecWorkerPpProxyTensors(CustomTestCase):
    def test_base_spec_worker_declares_pp_proxy_tensors(self):
        params = inspect.signature(BaseSpecWorker.forward_batch_generation).parameters
        self.assertIn("pp_proxy_tensors", params)

    def test_every_spec_worker_accepts_pp_proxy_tensors(self):
        found = _forward_batch_generation_classes(_SPEC_ROOT)
        missing_classes = sorted(_REQUIRED_CLASSES - found.keys())
        self.assertEqual(
            missing_classes,
            [],
            "expected spec workers were not found; the scan missed a file",
        )
        missing_kwarg = sorted(
            name for name, params in found.items() if "pp_proxy_tensors" not in params
        )
        self.assertEqual(
            missing_kwarg,
            [],
            "forward_batch_generation must accept pp_proxy_tensors "
            "(the non-overlap scheduler always passes it): " + ", ".join(missing_kwarg),
        )

    def test_subclass_missing_kwarg_fails_at_class_definition(self):
        with self.assertRaisesRegex(TypeError, "pp_proxy_tensors"):

            class _MissingKwargWorker(BaseSpecWorker):
                def forward_batch_generation(self, batch, on_publish=None):
                    raise NotImplementedError


if __name__ == "__main__":
    unittest.main()
