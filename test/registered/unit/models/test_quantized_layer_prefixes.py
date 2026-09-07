"""Keep per-module quantization decisions tied to qualified layer names."""

import ast
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]
MODELS_ROOT = REPO_ROOT / "python/sglang/srt/models"

# These constructors forward ``prefix`` to QuantizationConfig.get_quant_method.
QUANTIZATION_AWARE_LAYERS = {
    "ColumnParallelLinear",
    "FusedMoE",
    "MergedColumnParallelLinear",
    "ParallelLMHead",
    "QKVParallelLinear",
    "RadixAttention",
    "ReplicatedLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
}


def _call_name(node: ast.Call) -> str | None:
    return getattr(node.func, "id", None) or getattr(node.func, "attr", None)


def _is_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


class TestQuantizedLayerPrefixes(CustomTestCase):
    def test_active_quantization_configs_have_explicit_prefixes(self):
        self.assertTrue(MODELS_ROOT.is_dir())
        checked = 0
        missing = []
        for source_path in sorted(MODELS_ROOT.rglob("*.py")):
            source = source_path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source, filename=str(source_path))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                layer_name = _call_name(node)
                if layer_name not in QUANTIZATION_AWARE_LAYERS:
                    continue
                keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
                quant_config = keywords.get("quant_config")
                if quant_config is None or _is_none(quant_config):
                    continue
                checked += 1
                if "prefix" not in keywords:
                    relative_path = source_path.relative_to(REPO_ROOT)
                    missing.append(f"{relative_path}:{node.lineno} ({layer_name})")

        self.assertGreater(checked, 0)
        self.assertEqual(
            [],
            missing,
            "quantization-aware layers with an active quant_config need a "
            "qualified prefix so per-module schemes and exclusion lists can "
            "match them:\n  " + "\n  ".join(missing),
        )


if __name__ == "__main__":
    unittest.main()
