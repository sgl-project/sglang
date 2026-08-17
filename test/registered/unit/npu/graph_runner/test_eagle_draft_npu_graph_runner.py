"""
Unit tests for sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner.
"""

import importlib.util
import sys
import types
import unittest
from enum import IntEnum, auto
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.distributed  # noqa: F401 鈥?ensure available for patching

# ---------------------------------------------------------------------------
# Mock the entire sglang package tree (same pattern as the extend-runner UT).
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

_ci_mod = types.ModuleType("sglang.test.ci.ci_register")
_ci_mod.register_npu_ci = lambda *a, **kw: None
sys.modules.setdefault("sglang.test.ci.ci_register", _ci_mod)

from sglang.test.ci.ci_register import register_npu_ci  # noqa: E402

register_npu_ci(est_time=3, suite="base-a-test-1-npu-a2")

# ---------------------------------------------------------------------------
# Stub sglang.srt.configs.model_config (keep in sync with source).
# ---------------------------------------------------------------------------
_cfg_mod = types.ModuleType("sglang.srt.configs.model_config")


class AttentionArch(IntEnum):
    MLA = auto()
    MHA = auto()


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


_cfg_mod.AttentionArch = AttentionArch
_cfg_mod.is_deepseek_dsa = is_deepseek_dsa
_cfg_mod.is_deepseek_v4 = is_deepseek_v4
sys.modules["sglang.srt.configs.model_config"] = _cfg_mod

# ---------------------------------------------------------------------------
# Stub parent class.
# ---------------------------------------------------------------------------
_parent_mod = types.ModuleType("sglang.srt.speculative.eagle_draft_cuda_graph_runner")


class _FakeParent:
    def __init__(self, eagle_worker=None):
        pass

    def can_run_graph(self, forward_batch):
        return True

    def _replay_graph(self, shape_key, forward_batch):
        return self.backend.replay(shape_key, forward_batch)


_parent_mod.EAGLEDraftCudaGraphRunner = _FakeParent
sys.modules["sglang.srt.speculative.eagle_draft_cuda_graph_runner"] = _parent_mod

# ---------------------------------------------------------------------------
# Load the target module directly from file.
# ---------------------------------------------------------------------------
_TARGET_FILE = (
    Path(__file__).resolve().parents[5]
    / "python"
    / "sglang"
    / "srt"
    / "hardware_backend"
    / "npu"
    / "graph_runner"
    / "eagle_draft_npu_graph_runner.py"
)

_spec = importlib.util.spec_from_file_location(
    "sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner",
    str(_TARGET_FILE),
)
_target_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _target_mod
_spec.loader.exec_module(_target_mod)

EAGLEDraftNpuGraphRunner = _target_mod.EAGLEDraftNpuGraphRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner(
    hf_config=None, bs=4, raw_bs=2, speculative_num_steps=4, seed_dsa=False, dp_size=1
):
    r = object.__new__(EAGLEDraftNpuGraphRunner)
    r._init_arch_map()
    r.model_runner = MagicMock()
    if hf_config is not None:
        r.model_runner.model_config.hf_config = hf_config
    r.backend = MagicMock()
    r.bs = bs
    r.raw_bs = raw_bs
    r.speculative_num_steps = speculative_num_steps
    r.eagle_worker = MagicMock()
    r.eagle_worker.seed_dsa_topk_from_draft_extend = seed_dsa
    r.attn_dp_size = dp_size
    r.device = "cpu"
    return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEAGLEDraftNpuGraphRunner(unittest.TestCase):
    def test_init_delegates_to_super(self):
        """__init__ calls _init_arch_map() then super().__init__()."""
        parent_cls = EAGLEDraftNpuGraphRunner.__bases__[0]
        order = []
        orig_init = parent_cls.__init__
        orig_arch = EAGLEDraftNpuGraphRunner._init_arch_map

        def spy_super(self, *a, **kw):
            order.append("super")

        def spy_arch(self):
            order.append("arch_map")

        parent_cls.__init__ = spy_super
        EAGLEDraftNpuGraphRunner._init_arch_map = spy_arch
        try:
            inst = object.__new__(EAGLEDraftNpuGraphRunner)
            ew = MagicMock()
            EAGLEDraftNpuGraphRunner.__init__(inst, ew)
        finally:
            parent_cls.__init__ = orig_init
            EAGLEDraftNpuGraphRunner._init_arch_map = orig_arch
        self.assertEqual(order, ["arch_map", "super"])

    def test_init_arch_map(self):
        r = object.__new__(EAGLEDraftNpuGraphRunner)
        r._init_arch_map()
        self.assertEqual(r.attr_name[AttentionArch.MLA], "actual_seq_lengths_kv")
        self.assertEqual(r.attr_name[AttentionArch.MHA], "context_lens")
        self.assertEqual(r.attr_type[AttentionArch.MLA], [])
        self.assertIsInstance(r.attr_type[AttentionArch.MHA], torch.Tensor)

    def test_cache_loc_dtype(self):
        r = object.__new__(EAGLEDraftNpuGraphRunner)
        self.assertEqual(r._cache_loc_dtype(), torch.int32)

    def test_get_update_attr_name_and_type(self):
        r = object.__new__(EAGLEDraftNpuGraphRunner)
        r._init_arch_map()
        self.assertEqual(r._get_update_attr_name(), "actual_seq_lengths_kv")
        self.assertEqual(r._get_update_attr_type(), [])

    def test_can_run_graph_early_return(self):
        """Early return (no all_reduce) when seed disabled or dp_size <= 1."""
        fb = MagicMock()
        # seed disabled
        r = _make_runner(seed_dsa=False, dp_size=4)
        with patch("torch.distributed.all_reduce") as mock_ar:
            result = r.can_run_graph(fb)
        self.assertTrue(result)
        mock_ar.assert_not_called()
        # dp_size <= 1
        r = _make_runner(seed_dsa=True, dp_size=1)
        with patch("torch.distributed.all_reduce") as mock_ar:
            result = r.can_run_graph(fb)
        self.assertTrue(result)
        mock_ar.assert_not_called()

    def test_can_run_graph_idle(self):
        """Full path, idle mode -> seed_ready=True -> can_run=True."""
        r = _make_runner(seed_dsa=True, dp_size=4)
        fb = MagicMock()
        fb.forward_mode.is_idle.return_value = True
        fb.spec_info = None
        with patch("torch.distributed.all_reduce") as mock_ar:
            result = r.can_run_graph(fb)
        self.assertTrue(result)
        mock_ar.assert_called_once()

    def test_can_run_graph_seed_not_ready(self):
        """Full path, not idle and spec_info=None -> seed_ready=False -> False."""
        r = _make_runner(seed_dsa=True, dp_size=4)
        fb = MagicMock()
        fb.forward_mode.is_idle.return_value = False
        fb.spec_info = None
        with patch("torch.distributed.all_reduce") as mock_ar:
            result = r.can_run_graph(fb)
        self.assertFalse(result)
        mock_ar.assert_called_once()

    def test_can_run_graph_spec_info_no_indices(self):
        """Full path, not idle, spec_info exists but dsa_topk_indices=None -> False."""
        r = _make_runner(seed_dsa=True, dp_size=4)
        fb = MagicMock()
        fb.forward_mode.is_idle.return_value = False
        fb.spec_info = MagicMock()
        fb.spec_info.dsa_topk_indices = None
        with patch("torch.distributed.all_reduce") as mock_ar:
            result = r.can_run_graph(fb)
        self.assertFalse(result)
        mock_ar.assert_called_once()

    def test_can_run_graph_seed_ready(self):
        """Full path, not idle but dsa_topk_indices set -> seed_ready=True -> True."""
        r = _make_runner(seed_dsa=True, dp_size=4)
        fb = MagicMock()
        fb.forward_mode.is_idle.return_value = False
        fb.spec_info = MagicMock()
        fb.spec_info.dsa_topk_indices = torch.tensor([1, 2])
        with patch("torch.distributed.all_reduce") as mock_ar:
            result = r.can_run_graph(fb)
        self.assertTrue(result)
        mock_ar.assert_called_once()

    def test_replay_graph_non_dsa(self):
        """Non-DSA/V4: multi-step seq_lens + cpu_update_input list."""
        cfg = SimpleNamespace(architectures=["LlamaForCausalLM"])
        r = _make_runner(hf_config=cfg, bs=4, raw_bs=2, speculative_num_steps=4)
        fb = MagicMock()
        fb.seq_lens_cpu = torch.tensor([10, 20, 30, 40])
        r._replay_graph("key", fb)
        expected = [
            {"actual_seq_lengths_kv": [11, 21, 0, 0]},
            {"actual_seq_lengths_kv": [12, 22, 0, 0]},
            {"actual_seq_lengths_kv": [13, 23, 0, 0]},
        ]
        r.backend.replay_with_input_update.assert_called_once_with(
            "key", seq_lens=None, cpu_update_input=expected
        )
        r.backend.replay.assert_not_called()

    def test_replay_graph_dsa_and_v4(self):
        """DSA and V4 both go to backend.replay."""
        fb = MagicMock()
        for arch, extra in [
            ("DeepseekV3ForCausalLM", {"index_topk": 8}),
            ("DeepseekV4ForCausalLM", {}),
        ]:
            with self.subTest(arch=arch):
                cfg = SimpleNamespace(architectures=[arch], **extra)
                r = _make_runner(hf_config=cfg)
                r._replay_graph("key", fb)
                r.backend.replay.assert_called_once_with("key", fb)
                r.backend.replay_with_input_update.assert_not_called()
                r.backend.reset_mock()

    def test_replay_graph_dsa_without_index_topk_falls_back(self):
        """DSA arch but no index_topk -> non-DSA branch."""
        cfg = SimpleNamespace(architectures=["DeepseekV3ForCausalLM"])
        r = _make_runner(hf_config=cfg, bs=2, raw_bs=1, speculative_num_steps=2)
        fb = MagicMock()
        fb.seq_lens_cpu = torch.tensor([5, 10])
        r._replay_graph("key", fb)
        r.backend.replay_with_input_update.assert_called_once()
        r.backend.replay.assert_not_called()


if __name__ == "__main__":
    unittest.main()
