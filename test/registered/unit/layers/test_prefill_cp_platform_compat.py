import ast
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


_REPO_ROOT = Path(__file__).resolve().parents[4]
_LEGACY_CP_UTILS = "sglang.srt.layers.utils.cp_utils"


def _imports_from(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            imported.update(alias.name for alias in node.names)
    return imported


class TestPrefillCPPlatformCompatibility(CustomTestCase):
    def test_generic_modeling_does_not_import_legacy_prefill_cp(self):
        model_files = (
            "deepseek_v2.py",
            "deepseek_nextn.py",
            "deepseek_v4_nextn.py",
            "qwen2_moe.py",
            "qwen3_moe.py",
            "mellum.py",
        )

        violations = {}
        for name in model_files:
            path = _REPO_ROOT / "python/sglang/srt/models" / name
            imported = _imports_from(path, _LEGACY_CP_UTILS)
            if imported:
                violations[name] = sorted(imported)

        self.assertEqual(violations, {})

    def test_legacy_cp_utils_imports_are_platform_compatibility_only(self):
        allowed_consumers = {
            "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py",
            # Shared modules retain only the branches required by HIP/NPU.
            "python/sglang/srt/layers/attention/dsa/dsa_indexer.py",
            "python/sglang/srt/layers/attention/dsa/dsa_npu_indexer.py",
            "python/sglang/srt/layers/attention/dsa_backend.py",
            "python/sglang/srt/layers/attention/dsv4/compressor.py",
            "python/sglang/srt/layers/cp/utils.py",
            "python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py",
            # DeepSeek-V4 keeps the HIP-only CP+TBO implementation in the shared
            # model file until the ROCm owners migrate it independently.
            "python/sglang/srt/models/deepseek_v4.py",
        }

        actual_consumers = set()
        source_root = _REPO_ROOT / "python/sglang/srt"
        for path in source_root.rglob("*.py"):
            if path == source_root / "layers/utils/cp_utils.py":
                continue
            if _imports_from(path, _LEGACY_CP_UTILS):
                actual_consumers.add(path.relative_to(_REPO_ROOT).as_posix())

        self.assertEqual(actual_consumers, allowed_consumers)


if __name__ == "__main__":
    unittest.main()
