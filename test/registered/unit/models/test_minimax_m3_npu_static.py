import ast
import re
import unittest
from pathlib import Path

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ModuleNotFoundError:
    register_cpu_ci = None

if register_cpu_ci is not None:
    register_cpu_ci(est_time=2, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text()


class TestMiniMaxM3NPUStaticContracts(unittest.TestCase):
    def test_model_has_explicit_npu_prepare_path(self):
        source = _read("python/sglang/srt/models/minimax_m3.py")
        tree = ast.parse(source)
        function_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }

        self.assertIn("_is_npu = is_npu()", source)
        self.assertIn("split_qkv_tp_rmsnorm_rope", source)
        self.assertIn("get_attention_tp_group", source)
        self.assertIn("forward_prepare_npu", function_names)
        self.assertRegex(source, r"if\s+_is_npu:\s*\n\s+s = self\.forward_prepare_npu")
        self.assertNotRegex(
            source,
            r"_fuse_qkv_index_enabled\s*=.*_is_npu",
            "NPU must not enter CUDA/ROCm qkv+index fused GEMM path.",
        )

    def test_npu_memory_pool_exposes_minimax_sparse_wrapper(self):
        source = _read("python/sglang/srt/hardware_backend/npu/memory_pool_npu.py")
        tree = ast.parse(source)
        class_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        }

        self.assertIn("NPUMHATokenToKOnlyPool", class_names)
        self.assertIn("NPUMiniMaxSparseKVPool", class_names)
        self.assertIn("torch_npu.npu_scatter_nd_update_", source)

    def test_ascend_pool_selection_prefers_minimax_sparse_pool(self):
        source = _read("python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py")
        ascend_branch = source[
            source.index('self.server_args.attention_backend == "ascend"') :
        ]

        minimax_match = re.search(
            r"is_minimax_sparse\(self\.model_config\.hf_config\)", ascend_branch
        )
        generic_mha_match = re.search(r"NPUMHATokenToKVPool", ascend_branch)

        self.assertIsNotNone(minimax_match)
        self.assertIsNotNone(generic_mha_match)
        self.assertLess(minimax_match.start(), generic_mha_match.start())
        self.assertIn("NPUMiniMaxSparseKVPool", ascend_branch)

    def test_minimax_sparse_backend_has_npu_guardrails(self):
        source = _read("python/sglang/srt/layers/attention/minimax_sparse_backend.py")

        self.assertIn("is_npu", source)
        self.assertIn("_raise_npu_sparse_not_ready", source)
        self.assertRegex(source, r"self\.is_npu\s*=\s*is_npu\(\)")

    def test_swigluoai_has_npu_eager_path(self):
        source = _read("python/sglang/srt/models/minimax_m3.py")
        tree = ast.parse(source)
        function_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }

        self.assertIn("_swigluoai_torch", function_names)
        self.assertRegex(
            source,
            r"(?s)elif hidden_act == \"swigluoai\".*?if _is_npu:",
            "NPU must avoid the torch.compile/Triton swigluoai helper.",
        )


if __name__ == "__main__":
    unittest.main()
