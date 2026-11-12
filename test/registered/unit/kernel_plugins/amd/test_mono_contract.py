"""A rename in SGLang that the monokernel framework reads fails here, not on a GPU."""

import ast
import unittest
from functools import lru_cache
from pathlib import Path

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, stage="base-a", runner_config="cpu")

ROOT = Path(sglang.__file__).resolve().parent  # python/sglang

# Names the framework reads that its CPU tests reach only through fakes: GPU runner,
# batch and process-group attributes, string protocols, and server arguments read
# only at launch. Imports and the rest fail the framework tests directly.
READS = {
    "srt/arg_groups/fields/*.py": (
        "cuda_graph_config",
        "speculative_algorithm",
        "speculative_num_draft_tokens",
    ),
    "srt/configs/model_config.py": ("dtype", "hf_config"),
    "srt/distributed/parallel_state.py": ("world_size", "cpu_group"),
    "srt/layers/attention/base_attn_backend.py": ("decode_attention_backend_str",),
    "srt/model_executor/forward_batch_info.py": (
        "forward_mode",
        "batch_size",
        "input_ids",
    ),
    "srt/model_executor/model_runner_components/weight_updater.py": (
        '"_derived_weight_cache_error"',
    ),
    "srt/model_executor/runner/decode_cuda_graph_runner.py": ("capture_bs",),
    "srt/runtime_context.py": ('"tp_group"',),
}


@lru_cache(maxsize=None)
def _provided(path: Path) -> frozenset:
    """Names a file provides: definitions, attributes, imports and string literals."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for t in ast.walk(target):
                    if isinstance(t, ast.Name):
                        names.add(t.id)
                    elif isinstance(t, ast.Attribute):
                        names.add(t.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(f'"{node.value}"')
    return frozenset(names)


class TestSglangReads(CustomTestCase):
    def test_sglang_provides_what_the_framework_reads(self):
        for pattern, names in READS.items():
            with self.subTest(pattern=pattern):
                files = sorted(ROOT.glob(pattern))
                self.assertTrue(files, f"no file matches {pattern}")
                provided = frozenset().union(*map(_provided, files))
                missing = [name for name in names if name not in provided]
                self.assertFalse(
                    missing,
                    f"SGLang no longer provides {missing} in {pattern}; repair the "
                    "monokernel framework and READS",
                )


if __name__ == "__main__":
    unittest.main()
