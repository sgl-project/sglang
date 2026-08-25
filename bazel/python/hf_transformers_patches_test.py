import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


def install_namespace_packages() -> None:
    root = (
        Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"] / "python/sglang"
    )
    for name, path in (
        ("sglang", root),
        ("sglang.srt", root / "srt"),
        ("sglang.srt.utils", root / "srt" / "utils"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


sys.modules.pop("torch", None)
sys.modules.pop("transformers", None)
install_namespace_packages()

from sglang.srt.utils import hf_transformers_patches  # noqa: E402


class HfTransformersPatchesTest(unittest.TestCase):
    def test_no_transformers_install_is_a_torch_free_noop(self) -> None:
        hf_transformers_patches._applied = False
        hf_transformers_patches.apply_all()
        self.assertTrue(hf_transformers_patches._applied)
        self.assertNotIn("transformers", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def test_rope_compat_recurses_without_transformers(self) -> None:
        text_config = SimpleNamespace(rope_scaling={"rope_type": "yarn", "factor": 4.0})
        config = SimpleNamespace(
            rope_scaling={"rope_type": "llama3", "factor": 8.0},
            text_config=text_config,
        )

        hf_transformers_patches.normalize_rope_scaling_compat(config)

        self.assertEqual(config.rope_scaling["type"], "llama3")
        self.assertEqual(text_config.rope_scaling["type"], "yarn")


if __name__ == "__main__":
    unittest.main()
