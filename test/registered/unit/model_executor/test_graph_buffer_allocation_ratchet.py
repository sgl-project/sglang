"""Hand-made ``self.<attr> = torch.<alloc>(...)`` buffers in graph-owning modules
may only go down; graph-read buffers register through the registry instead."""

import ast
import pathlib
import unittest

import sglang
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SCANNED_DIRS = (
    "srt/layers/attention",
    "srt/speculative",
    "srt/model_executor/runner",
)

ALLOCATORS = frozenset(
    {
        "zeros",
        "ones",
        "empty",
        "full",
        "arange",
        "zeros_like",
        "ones_like",
        "empty_like",
        "full_like",
    }
)

FROZEN_COUNTS = {
    "srt/layers/attention/aiter_backend.py": 28,
    "srt/layers/attention/deepseek_v4_backend.py": 7,
    "srt/layers/attention/deepseek_v4_backend_hip_radix.py": 3,
    "srt/layers/attention/dsa/dsa_indexer_kpool.py": 2,
    "srt/layers/attention/dsa/gfx950_fused_indexer.py": 10,
    "srt/layers/attention/dsa_backend.py": 11,
    "srt/layers/attention/dsv4/compressor.py": 1,
    "srt/layers/attention/dsv4/dsv41_sparse.py": 1,
    "srt/layers/attention/dsv4/metadata.py": 1,
    "srt/layers/attention/dsv4/sparse_prefill_utils.py": 1,
    "srt/layers/attention/flashattention_backend.py": 19,
    "srt/layers/attention/flashinfer_backend.py": 15,
    "srt/layers/attention/flashinfer_mla_backend.py": 13,
    "srt/layers/attention/flashmla_backend.py": 5,
    "srt/layers/attention/hpc_ops_backend.py": 3,
    "srt/layers/attention/hybrid_linear_attn_backend.py": 4,
    "srt/layers/attention/linear/inkling_sconv_backend.py": 8,
    "srt/layers/attention/linear/kda_backend.py": 1,
    "srt/layers/attention/linear/kda_flashinfer_prefill_graph.py": 4,
    "srt/layers/attention/linear/short_conv_backend.py": 1,
    "srt/layers/attention/mamba/mamba.py": 4,
    "srt/layers/attention/mamba/mamba1.py": 2,
    "srt/layers/attention/mamba/mixer2_rms_norm_gated.py": 1,
    "srt/layers/attention/minicpm/attention_adapter.py": 4,
    "srt/layers/attention/minicpm/cache.py": 1,
    "srt/layers/attention/minimax_sparse_backend.py": 1,
    "srt/layers/attention/minimax_sparse_nvfp4_backend.py": 2,
    "srt/layers/attention/minimax_sparse_ops/msa_plan.py": 9,
    "srt/layers/attention/qwen_sparse_attn_backend.py": 22,
    "srt/layers/attention/triton_backend.py": 26,
    "srt/layers/attention/trtllm_mha_backend.py": 12,
    "srt/layers/attention/trtllm_mla_backend.py": 6,
    "srt/layers/attention/vision.py": 1,
    "srt/layers/attention/wave_backend.py": 8,
    "srt/layers/attention/xpu_backend.py": 8,
    "srt/speculative/base_spec_worker.py": 2,
    "srt/speculative/dflash_info_v2.py": 5,
    "srt/speculative/dflash_worker_v2.py": 32,
    "srt/speculative/dspark_components/dspark_deferred_commit.py": 4,
    "srt/speculative/dspark_components/dspark_draft.py": 2,
    "srt/speculative/dspark_components/dspark_draft_sampler.py": 7,
    "srt/speculative/dspark_components/dspark_observability.py": 6,
    "srt/speculative/dspark_components/dspark_planner.py": 2,
    "srt/speculative/dspark_components/dspark_worker_v2.py": 1,
    "srt/speculative/eagle_draft_cuda_graph_runner.py": 2,
    "srt/speculative/eagle_draft_extend_cuda_graph_runner.py": 1,
    "srt/speculative/eagle_info.py": 1,
    "srt/speculative/eagle_post_verify_graphs.py": 1,
    "srt/speculative/eagle_worker_v2.py": 2,
    "srt/speculative/frozen_kv_mtp_worker_v2.py": 2,
    "srt/speculative/lilicorr_utils.py": 8,
    "srt/speculative/multi_layer_eagle_worker_v2.py": 5,
    "srt/speculative/ngram_info.py": 1,
    "srt/speculative/ngram_worker.py": 6,
    "srt/speculative/standalone_worker_v2.py": 2,
    "srt/speculative/uno_worker_v2.py": 1,
    "srt/model_executor/runner/prefill_cuda_graph_runner.py": 3,
}


def _allocates(node: ast.AST) -> bool:
    return any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in ALLOCATORS
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "torch"
        for call in ast.walk(node)
    )


def _is_self_attribute(target: ast.AST) -> bool:
    if isinstance(target, ast.Subscript):
        target = target.value
    return (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    )


def _count_self_allocations(source: str) -> int:
    count = 0
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if any(_is_self_attribute(t) for t in targets) and _allocates(value):
            count += 1
    return count


def scan_counts(package_root: pathlib.Path) -> dict:
    counts = {}
    for directory in SCANNED_DIRS:
        for path in sorted((package_root / directory).rglob("*.py")):
            count = _count_self_allocations(path.read_text())
            if count:
                counts[path.relative_to(package_root).as_posix()] = count
    return counts


class TestGraphBufferAllocationRatchet(unittest.TestCase):
    def test_hand_made_allocations_only_go_down(self):
        counts = scan_counts(pathlib.Path(sglang.__file__).parent)
        grown = {
            path: (FROZEN_COUNTS.get(path, 0), count)
            for path, count in counts.items()
            if count > FROZEN_COUNTS.get(path, 0)
        }
        self.assertFalse(
            grown,
            "New hand-made persistent allocations (frozen, now): "
            f"{grown}. Register graph-read buffers through "
            "CudaGraphBufferRegistry.register_buffer; raise the frozen count "
            "only for a buffer no graph reads.",
        )
        shrunk = {
            path: (frozen, counts.get(path, 0))
            for path, frozen in FROZEN_COUNTS.items()
            if counts.get(path, 0) < frozen
        }
        self.assertFalse(
            shrunk,
            f"Allocations went down (frozen, now): {shrunk}. Lower FROZEN_COUNTS.",
        )

    def test_counter_sees_attribute_and_subscript_targets(self):
        source = (
            "class A:\n"
            "    def __init__(self):\n"
            "        self.a = torch.zeros(4)\n"
            "        self.b: object = torch.empty(4) if x else None\n"
            "        self.c[k] = torch.full((2,), 1)\n"
            "        local = torch.zeros(4)\n"
            "        self.d = local.view(2, 2)\n"
        )
        self.assertEqual(_count_self_allocations(source), 3)


if __name__ == "__main__":
    unittest.main()
