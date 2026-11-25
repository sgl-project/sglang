"""Every srt model file the diffusion runtime imports must trigger diffusion CI."""

import ast
import fnmatch
import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
FILTERS = ("multimodal_gen", "multimodal_gen_server")


def _srt_model_imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("sglang.srt.models."):
                found.add(node.module)
            elif node.module == "sglang.srt.models":
                found.update(f"sglang.srt.models.{a.name}" for a in node.names)
    return found


def _model_files() -> set[str]:
    """srt model modules multimodal_gen imports, with the srt models they import."""
    pending = set()
    for path in (ROOT / "python/sglang/multimodal_gen").rglob("*.py"):
        if "test" not in path.relative_to(ROOT / "python/sglang/multimodal_gen").parts:
            pending |= _srt_model_imports(path)
    files = set()
    while pending:
        module = pending.pop()
        path = ROOT / "python" / (module.replace(".", "/") + ".py")
        rel = str(path.relative_to(ROOT))
        if path.exists() and rel not in files:
            files.add(rel)
            pending |= _srt_model_imports(path)
    return files


def _expand_braces(pattern: str) -> list[str]:
    match = re.search(r"\{([^{}]*)\}", pattern)
    if match is None:
        return [pattern]
    head, tail = pattern[: match.start()], pattern[match.end() :]
    return [
        expanded
        for option in match.group(1).split(",")
        for expanded in _expand_braces(head + option + tail)
    ]


class TestDiffusionSrtPaths(unittest.TestCase):
    def test_imported_srt_models_trigger_diffusion_ci(self):
        workflow = ROOT / ".github/workflows/_pr-test-detect-changes.yml"
        steps = yaml.safe_load(workflow.read_text())["jobs"]["detect"]["steps"]
        filters = next(
            yaml.safe_load(step["with"]["filters"])
            for step in steps
            if "dorny/paths-filter" in step.get("uses", "")
        )
        files = _model_files()
        self.assertIn("python/sglang/srt/models/qwen3_vl.py", files)
        for name in FILTERS:
            patterns = [p for raw in filters[name] for p in _expand_braces(raw)]
            for file in sorted(files):
                with self.subTest(filter=name, file=file):
                    self.assertTrue(
                        any(fnmatch.fnmatch(file, p) for p in patterns),
                        f"add {file} to the {name} filter in {workflow.name}",
                    )


if __name__ == "__main__":
    unittest.main()
