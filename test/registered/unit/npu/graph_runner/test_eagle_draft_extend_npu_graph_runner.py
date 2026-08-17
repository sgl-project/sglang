"""
Unit tests for sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_extend_npu_graph_runner.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

# ---------------------------------------------------------------------------
# Mock the entire sglang package tree so we never trigger the heavy
# sglang.__init__ import chain (triton, IPython, aiohttp, transformers
# patches, ...).  Only the specific submodules the target file needs are
# replaced with real lightweight stubs.
# ---------------------------------------------------------------------------
for _pkg in (
    "sglang",
    "sglang.test",
    "sglang.test.ci",
    "sglang.srt",
    "sglang.srt.configs",
    "sglang.srt.speculative",
    "sglang.srt.hardware_backend",
    "sglang.srt.hardware_backend.npu",
    "sglang.srt.hardware_backend.npu.graph_runner",
):
    sys.modules.setdefault(_pkg, MagicMock())

# register_npu_ci is a runtime no-op (CI parses it via AST).
_ci_mod = types.ModuleType("sglang.test.ci.ci_register")
_ci_mod.register_npu_ci = lambda *a, **kw: None
sys.modules.setdefault("sglang.test.ci.ci_register", _ci_mod)

from sglang.test.ci.ci_register import register_npu_ci  # noqa: E402

register_npu_ci(est_time=5, suite="base-a-test-1-npu-a2")

# ---------------------------------------------------------------------------
# Stub sglang.srt.configs.model_config with faithful copies of
# is_deepseek_dsa / is_deepseek_v4 (keep in sync with source).
# ---------------------------------------------------------------------------
_cfg_mod = types.ModuleType("sglang.srt.configs.model_config")

_DSA_ARCHS = (
    "DeepseekV3ForCausalLM",
    "DeepseekV32ForCausalLM",
    "DeepseekV3ForCausalLMNextN",
    "MistralLarge3ForCausalLM",
    "PixtralForConditionalGeneration",
    "GlmMoeDsaForCausalLM",
    "GlmMoeDsaForCausalLMNextN",
    "LongcatFlashForCausalLM",
    "LongcatFlashForCausalLMNextN",
)
_V4_ARCHS = (
    "DeepseekV4ForCausalLM",
    "DeepseekV4ForCausalLMNextN",
    "DeepseekV4ForCausalLMDSpark",
)


def _hf_arch(config):
    archs = getattr(config, "architectures", None)
    return archs[0] if archs else None


def _hf_attr(config, name):
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def is_deepseek_dsa(config):
    return _hf_arch(config) in _DSA_ARCHS and _hf_attr(config, "index_topk") is not None


def is_deepseek_v4(config):
    return _hf_arch(config) in _V4_ARCHS


_cfg_mod.is_deepseek_dsa = is_deepseek_dsa
_cfg_mod.is_deepseek_v4 = is_deepseek_v4
sys.modules["sglang.srt.configs.model_config"] = _cfg_mod

# ---------------------------------------------------------------------------
# Stub the parent class (real one pulls in the full speculative stack).
# ---------------------------------------------------------------------------
_parent_mod = types.ModuleType(
    "sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner"
)


class _FakeParent:
    def __init__(self, eagle_worker=None):
        pass

    def _cache_loc_dtype(self):
        return torch.int64

    def _replay_graph(self, shape_key, forward_batch):
        return self.backend.replay(shape_key, forward_batch)


_parent_mod.EAGLEDraftExtendCudaGraphRunner = _FakeParent
sys.modules["sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner"] = _parent_mod

# ---------------------------------------------------------------------------
# Load the target module directly from file path (bypasses sglang import chain).
# ---------------------------------------------------------------------------
_TARGET_FILE = (
    Path(__file__).resolve().parents[5]
    / "python"
    / "sglang"
    / "srt"
    / "hardware_backend"
    / "npu"
    / "graph_runner"
    / "eagle_draft_extend_npu_graph_runner.py"
)

_spec = importlib.util.spec_from_file_location(
    "sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_extend_npu_graph_runner",
    str(_TARGET_FILE),
)
_target_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _target_mod
_spec.loader.exec_module(_target_mod)

EAGLEDraftExtendNpuGraphRunner = _target_mod.EAGLEDraftExtendNpuGraphRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner(hf_config, bs=4, raw_bs=2):
    r = object.__new__(EAGLEDraftExtendNpuGraphRunner)
    r.model_runner = MagicMock()
    r.model_runner.model_config.hf_config = hf_config
    r.backend = MagicMock()
    r.bs = bs
    r.raw_bs = raw_bs
    return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEAGLEDraftExtendNpuGraphRunner(unittest.TestCase):
    def test_init_delegates_to_super(self):
        parent_cls = EAGLEDraftExtendNpuGraphRunner.__bases__[0]
        calls = []
        orig = parent_cls.__init__

        def spy(self, *a, **kw):
            calls.append((a, kw))

        parent_cls.__init__ = spy
        try:
            inst = object.__new__(EAGLEDraftExtendNpuGraphRunner)
            ew = MagicMock()
            EAGLEDraftExtendNpuGraphRunner.__init__(inst, ew)
        finally:
            parent_cls.__init__ = orig
        self.assertEqual(calls, [((ew,), {})])

    def test_cache_loc_dtype(self):
        r = object.__new__(EAGLEDraftExtendNpuGraphRunner)
        self.assertEqual(r._cache_loc_dtype(), torch.int32)

    def test_replay_graph_non_dsa(self):
        """Non-DSA/V4 -> replay_with_input_update with padded seq_lens."""
        cfg = SimpleNamespace(architectures=["LlamaForCausalLM"])
        r = _make_runner(cfg, bs=4, raw_bs=2)
        fb = MagicMock()
        fb.seq_lens_cpu = torch.tensor([10, 20])
        r._replay_graph("key", fb)
        r.backend.replay_with_input_update.assert_called_once_with(
            "key",
            seq_lens=[10, 20, 0, 0],
            attr_name="actual_seq_lengths_kv",
            attr_type=[],
        )
        r.backend.replay.assert_not_called()

    def test_replay_graph_dsa(self):
        """DSA -> backend.replay(shape_key, forward_batch)."""
        cfg = SimpleNamespace(architectures=["DeepseekV3ForCausalLM"], index_topk=8)
        r = _make_runner(cfg)
        fb = MagicMock()
        r._replay_graph("key", fb)
        r.backend.replay.assert_called_once_with("key", fb)
        r.backend.replay_with_input_update.assert_not_called()

    def test_replay_graph_v4(self):
        """V4 -> backend.replay(shape_key, forward_batch)."""
        cfg = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
        r = _make_runner(cfg)
        fb = MagicMock()
        r._replay_graph("key", fb)
        r.backend.replay.assert_called_once_with("key", fb)

    def test_replay_graph_dsa_without_index_topk_falls_back(self):
        """DSA arch but no index_topk -> non-DSA branch."""
        cfg = SimpleNamespace(architectures=["DeepseekV3ForCausalLM"])
        r = _make_runner(cfg, bs=2, raw_bs=1)
        fb = MagicMock()
        fb.seq_lens_cpu = torch.tensor([5])
        r._replay_graph("key", fb)
        r.backend.replay_with_input_update.assert_called_once()
        r.backend.replay.assert_not_called()


if __name__ == "__main__":
    unittest.main()
