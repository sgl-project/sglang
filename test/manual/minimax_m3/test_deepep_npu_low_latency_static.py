import ast
import importlib.util
import os
import sys
import types
import unittest
from enum import Enum
from pathlib import Path
from unittest import mock

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEEPEP_PATH = "python/sglang/srt/layers/moe/token_dispatcher/deepep.py"


def _read_deepep_source() -> str:
    return (REPO_ROOT / DEEPEP_PATH).read_text()


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name!r} not found")


def _install_fake_modules():
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.distributed",
        "sglang.srt.distributed.parallel_state",
        "sglang.srt.environ",
        "sglang.srt.eplb",
        "sglang.srt.eplb.expert_distribution",
        "sglang.srt.layers",
        "sglang.srt.layers.dp_attention",
        "sglang.srt.layers.moe",
        "sglang.srt.layers.moe.token_dispatcher",
        "sglang.srt.layers.moe.token_dispatcher.base",
        "sglang.srt.layers.moe.topk",
        "sglang.srt.layers.moe.utils",
        "sglang.srt.utils",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))

    parallel_state = sys.modules["sglang.srt.distributed.parallel_state"]
    parallel_state.get_tp_group = lambda: types.SimpleNamespace(barrier=lambda: None)

    class FakeEnv:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    envs = types.SimpleNamespace(
        SGLANG_ZBAL_LOCAL_MEM_SIZE=FakeEnv(0),
        SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE=FakeEnv(False),
        SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=FakeEnv(16),
    )
    sys.modules["sglang.srt.environ"].envs = envs

    recorder = types.SimpleNamespace(on_deepep_dispatch_low_latency=lambda _: None)
    sys.modules[
        "sglang.srt.eplb.expert_distribution"
    ].get_global_expert_distribution_recorder = lambda: recorder

    layers = sys.modules["sglang.srt.layers"]
    layers.deep_gemm_wrapper = types.SimpleNamespace(
        ENABLE_JIT_DEEPGEMM=False,
        DEEPGEMM_BLACKWELL=False,
    )
    sys.modules["sglang.srt.layers.dp_attention"].get_is_extend_in_batch = lambda: False

    base = sys.modules["sglang.srt.layers.moe.token_dispatcher.base"]

    class DispatchOutputFormat(Enum):
        DEEPEP_NORMAL = "deepep_normal"
        DEEPEP_LL = "deepep_ll"

    class CombineInputFormat(Enum):
        DEEPEP_NORMAL = "deepep_normal"
        DEEPEP_LL = "deepep_ll"

    base.BaseDispatcher = object
    base.BaseDispatcherConfig = object
    base.CombineInput = object
    base.CombineInputFormat = CombineInputFormat
    base.DispatcherBaseHooks = object
    base.DispatchOutput = object
    base.DispatchOutputFormat = DispatchOutputFormat
    sys.modules["sglang.srt.layers.moe.topk"].TopKOutput = object

    moe_utils = sys.modules["sglang.srt.layers.moe.utils"]

    class DeepEPMode(Enum):
        NORMAL = "normal"
        LOW_LATENCY = "low_latency"
        AUTO = "auto"

        def enable_normal(self):
            return self in (DeepEPMode.NORMAL, DeepEPMode.AUTO)

        def enable_low_latency(self):
            return self in (DeepEPMode.LOW_LATENCY, DeepEPMode.AUTO)

        def is_normal(self):
            return self == DeepEPMode.NORMAL

        def is_low_latency(self):
            return self == DeepEPMode.LOW_LATENCY

    class DeepEPOutputDtype(Enum):
        BF16 = "bf16"
        FP8 = "fp8"
        INT8 = "int8"
        NVFP4 = "nvfp4"

    moe_utils.DeepEPMode = DeepEPMode
    moe_utils.DeepEPOutputDtype = DeepEPOutputDtype
    moe_utils.get_deepep_config = lambda: None
    moe_utils.get_deepep_output_dtype = lambda _: DeepEPOutputDtype.BF16
    moe_utils.is_tbo_enabled = lambda: False

    utils = sys.modules["sglang.srt.utils"]
    utils.get_bool_env_var = lambda _: False
    utils.get_cuda_version = lambda: (12, 0)
    utils.is_blackwell = lambda: False
    utils.is_flashinfer_available = lambda: False
    utils.is_hip = lambda: False
    utils.is_npu = lambda: False
    utils.load_json_config = lambda _: {}


def _load_deepep_module():
    _install_fake_modules()
    module_path = REPO_ROOT / DEEPEP_PATH
    spec = importlib.util.spec_from_file_location("_deepep_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestDeepEPNPULowLatencyStatic(unittest.TestCase):
    def test_npu_low_latency_preserves_negative_topk_and_enables_kernel_mask(self):
        module = _load_deepep_module()
        module._is_npu = True
        hidden_states = torch.ones((1, 4), dtype=torch.bfloat16)
        topk_ids = torch.tensor([[-1, 34, -1, 109]], dtype=torch.int64)
        topk_weights = torch.tensor([[0.5, 1.0, 0.25, 0.75]], dtype=torch.float32)

        with mock.patch.dict(os.environ, {}, clear=True):
            (
                _,
                prepared_ids,
                prepared_weights,
            ) = module._prepare_low_latency_dispatch_inputs(
                hidden_states,
                topk_ids,
                topk_weights,
                num_max_dispatch_tokens_per_rank=16,
            )
            self.assertEqual(os.environ.get("MOE_ENABLE_TOPK_NEG_ONE"), "1")

        self.assertEqual(prepared_ids.dtype, torch.int32)
        self.assertTrue(torch.equal(prepared_ids, topk_ids.to(torch.int32)))
        self.assertTrue(torch.equal(prepared_weights, topk_weights))

    def test_low_latency_dispatch_prepares_npu_runtime_inputs(self):
        source = _read_deepep_source()
        tree = ast.parse(source)
        helper = _find_function(tree, "_prepare_low_latency_dispatch_inputs")
        helper_source = ast.get_source_segment(source, helper)

        self.assertIn("torch.int32 if _is_npu else torch.int64", helper_source)
        self.assertIn("MOE_ENABLE_TOPK_NEG_ONE", helper_source)
        self.assertIn("num_max_dispatch_tokens_per_rank", helper_source)
        self.assertIn("raise RuntimeError", helper_source)
        self.assertIn(".contiguous()", helper_source)
        self.assertNotIn("SGLANG_DEEPEP_DEBUG_DISPATCH", helper_source)
        self.assertNotIn("logger.warning", helper_source)
        self.assertNotIn(".item()", helper_source)

    def test_low_latency_dispatch_uses_prepared_inputs_before_runtime_call(self):
        source = _read_deepep_source()
        tree = ast.parse(source)
        low_latency_class = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and node.name == "_DeepEPDispatcherImplLowLatency"
        )
        dispatch_a = next(
            node
            for node in low_latency_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "dispatch_a"
        )
        dispatch_a_source = ast.get_source_segment(source, dispatch_a)

        prepare_pos = dispatch_a_source.index("_prepare_low_latency_dispatch_inputs")
        dispatch_core_pos = dispatch_a_source.index("self._dispatch_core")
        self.assertLess(prepare_pos, dispatch_core_pos)


if __name__ == "__main__":
    unittest.main()
