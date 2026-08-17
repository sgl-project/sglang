"""
Unit tests for sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner.
"""

import importlib.util
import sys
import types
import unittest
from enum import IntEnum, auto
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

# Ensure torch.npu is mockable on CPU hosts.
if not hasattr(torch, "npu"):
    torch.npu = MagicMock()

# ---------------------------------------------------------------------------
# Mock the entire sglang package tree.
# ---------------------------------------------------------------------------
for _pkg in (
    "sglang",
    "sglang.test",
    "sglang.test.ci",
    "sglang.srt",
    "sglang.srt.configs",
    "sglang.srt.distributed",
    "sglang.srt.distributed.parallel_state",
    "sglang.srt.compilation",
    "sglang.srt.layers",
    "sglang.srt.model_executor",
    "sglang.srt.hardware_backend",
    "sglang.srt.hardware_backend.npu",
    "sglang.srt.hardware_backend.npu.graph_runner",
):
    sys.modules.setdefault(_pkg, MagicMock())

_ci_mod = types.ModuleType("sglang.test.ci.ci_register")
_ci_mod.register_npu_ci = lambda *a, **kw: None
sys.modules.setdefault("sglang.test.ci.ci_register", _ci_mod)

from sglang.test.ci.ci_register import register_npu_ci  # noqa: E402

register_npu_ci(est_time=5, suite="base-a-test-1-npu-a2")

# ---------------------------------------------------------------------------
# Stub sglang.srt.configs.model_config
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
# Stub sglang.srt.distributed.parallel_state
# ---------------------------------------------------------------------------
_ps_mod = types.ModuleType("sglang.srt.distributed.parallel_state")


class GroupCoordinator:
    pass


_ps_mod.GroupCoordinator = GroupCoordinator
sys.modules["sglang.srt.distributed.parallel_state"] = _ps_mod

# ---------------------------------------------------------------------------
# Stub sglang.srt.environ
# ---------------------------------------------------------------------------
_environ_mod = types.ModuleType("sglang.srt.environ")
envs = MagicMock()
envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get.return_value = False
_environ_mod.envs = envs
sys.modules["sglang.srt.environ"] = _environ_mod

# ---------------------------------------------------------------------------
# Stub sglang.srt.utils
# ---------------------------------------------------------------------------
from contextlib import nullcontext  # noqa: E402

_utils_mod = types.ModuleType("sglang.srt.utils")


def empty_context():
    return nullcontext()


def get_bool_env_var(name, default="False"):
    return False


def get_compiler_backend(name):
    return "inductor"


def is_npu():
    return False


_utils_mod.empty_context = empty_context
_utils_mod.get_bool_env_var = get_bool_env_var
_utils_mod.get_compiler_backend = get_compiler_backend
_utils_mod.is_npu = is_npu
sys.modules["sglang.srt.utils"] = _utils_mod

# ---------------------------------------------------------------------------
# Stub sglang.srt.compilation.torch_compile_decoration
# ---------------------------------------------------------------------------
_tc_mod = types.ModuleType("sglang.srt.compilation.torch_compile_decoration")
_tc_mod.patch_model = None
sys.modules["sglang.srt.compilation.torch_compile_decoration"] = _tc_mod
# Connect parent package so `from sglang.srt.compilation import
# torch_compile_decoration` resolves to our stub, not a MagicMock child.
_compilation_mod = types.ModuleType("sglang.srt.compilation")
_compilation_mod.torch_compile_decoration = _tc_mod
sys.modules["sglang.srt.compilation"] = _compilation_mod

# ---------------------------------------------------------------------------
# Stub sglang.srt.layers.logits_processor
# ---------------------------------------------------------------------------
_lp_mod = types.ModuleType("sglang.srt.layers.logits_processor")


class LogitsProcessorOutput:
    def __init__(self, next_token_logits=None, full_logits=None, hidden_states=None):
        self.next_token_logits = next_token_logits
        self.full_logits = full_logits
        self.hidden_states = hidden_states


_lp_mod.LogitsProcessorOutput = LogitsProcessorOutput
sys.modules["sglang.srt.layers.logits_processor"] = _lp_mod

# ---------------------------------------------------------------------------
# Stub sglang.srt.model_executor.forward_batch_info
# ---------------------------------------------------------------------------
_fb_mod = types.ModuleType("sglang.srt.model_executor.forward_batch_info")


class ForwardBatch:
    pass


class PPProxyTensors:
    def __init__(self, tensors):
        self.tensors = tensors


_fb_mod.ForwardBatch = ForwardBatch
_fb_mod.PPProxyTensors = PPProxyTensors
sys.modules["sglang.srt.model_executor.forward_batch_info"] = _fb_mod

# ---------------------------------------------------------------------------
# Stub parent class DecodeCudaGraphRunner
# ---------------------------------------------------------------------------
_runner_mod = types.ModuleType("sglang.srt.model_executor.runner")


class _FakeParent:
    def __init__(
        self,
        model_runner,
        *,
        attn_backend=None,
        speculative_num_steps=None,
        speculative_num_draft_tokens=None,
    ):
        self.is_dllm = False
        self.enable_torch_compile = False

    def _make_graph_key(self, bs):
        return f"key_{bs}"

    def load_batch(self, forward_batch, pp_proxy_tensors=None):
        pass


_runner_mod.DecodeCudaGraphRunner = _FakeParent
sys.modules["sglang.srt.model_executor.runner"] = _runner_mod

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
    / "npu_graph_runner.py"
)

_spec = importlib.util.spec_from_file_location(
    "sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner",
    str(_TARGET_FILE),
)
_target_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _target_mod
_spec.loader.exec_module(_target_mod)

NPUGraphRunner = _target_mod.NPUGraphRunner
patch_model_npu = _target_mod.patch_model_npu

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner(
    hf_config=None,
    bs=4,
    raw_bs=2,
    raw_num_token=4,
    captured_req_width=1,
    is_dllm=False,
    if_use_v2=False,
):
    r = object.__new__(NPUGraphRunner)
    r.is_dllm = is_dllm
    r.if_use_v2 = if_use_v2
    r._init_arch_map()
    r.update_attr_name = None
    r.update_attr_type = None
    r.model_runner = MagicMock()
    r.model_runner.is_draft_worker = False
    r.model_runner.spec_algorithm.is_dflash.return_value = False
    if hf_config is not None:
        r.model_runner.model_config.hf_config = hf_config
    r.backend = MagicMock()
    r.bs = bs
    r.raw_bs = raw_bs
    r.raw_num_token = raw_num_token
    r.captured_req_width = captured_req_width
    r.buffers = MagicMock()
    r.graphs = {}
    r.load_batch = MagicMock()
    r._make_graph_key = lambda bs: f"key_{bs}"
    return r


def _make_forward_batch(seq_lens, is_target_verify=False, needs_init=True):
    fb = MagicMock()
    fb.needs_forward_metadata_init.return_value = needs_init
    fb.forward_mode.is_target_verify.return_value = is_target_verify
    fb.seq_lens = torch.tensor(seq_lens)
    fb.input_ids = MagicMock()
    fb.positions = MagicMock()
    fb.mrope_positions = None
    fb.input_embeds = None
    return fb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNPUGraphRunner(unittest.TestCase):
    def test_init_arch_map_non_dllm(self):
        r = _make_runner(is_dllm=False)
        self.assertEqual(r.attr_name[AttentionArch.MLA], "actual_seq_lengths_kv")
        self.assertEqual(r.attr_name[AttentionArch.MHA], "context_lens")
        self.assertEqual(r.attr_name["TARGET_VERIFY"], "actual_seq_kvlen")
        self.assertEqual(r.attr_type[AttentionArch.MLA], [])
        self.assertIsInstance(r.attr_type[AttentionArch.MHA], torch.Tensor)
        self.assertEqual(r.attr_type["TARGET_VERIFY"], [])

    def test_init_arch_map_dllm(self):
        """DLLM: MHA also uses actual_seq_lengths_kv (same as MLA)."""
        r = _make_runner(is_dllm=True)
        r._init_arch_map()
        self.assertEqual(r.attr_name[AttentionArch.MHA], "actual_seq_lengths_kv")
        self.assertEqual(r.attr_name[AttentionArch.MLA], "actual_seq_lengths_kv")

    def test_cache_loc_dtype(self):
        r = object.__new__(NPUGraphRunner)
        self.assertEqual(r._cache_loc_dtype(), torch.int32)

    def test_get_update_attr_name_and_type(self):
        """Non-V2 -> MLA key; V2 -> TARGET_VERIFY key."""
        # Non-V2
        r = _make_runner(if_use_v2=False)
        self.assertEqual(r._get_update_attr_name(), "actual_seq_lengths_kv")
        self.assertEqual(r._get_update_attr_type(), [])
        # V2
        r = _make_runner(if_use_v2=True)
        self.assertEqual(r._get_update_attr_name(), "actual_seq_kvlen")
        self.assertEqual(r._get_update_attr_type(), [])

    def test_update_inputs_tensor_conversion(self):
        """update_attr_type is Tensor -> seq_lens converted to int32."""
        r = _make_runner()
        r.update_attr_name = "ctx_lens"
        r.update_attr_type = torch.Tensor()
        r.bs = 4
        r.graphs = {4: MagicMock()}
        r._update_inputs([10, 20, 0, 0])
        update_kwargs = r.graphs[4].update.call_args.kwargs
        converted = update_kwargs["cpu_update_input"][0]["ctx_lens"]
        self.assertIsInstance(converted, torch.Tensor)
        self.assertEqual(converted.dtype, torch.int32)

    def test_update_inputs_list_type(self):
        """update_attr_type is list -> seq_lens passed as-is."""
        r = _make_runner()
        r.update_attr_name = "ctx_lens"
        r.update_attr_type = []
        r.bs = 4
        r.graphs = {4: MagicMock()}
        r._update_inputs([10, 20, 0, 0])
        update_kwargs = r.graphs[4].update.call_args.kwargs
        self.assertEqual(
            update_kwargs["cpu_update_input"],
            [{"ctx_lens": [10, 20, 0, 0]}],
        )

    def test_execute_buffer_copy_path(self):
        """needs_forward_metadata_init=False -> buffer copy path (else branch)."""
        cfg = SimpleNamespace(architectures=["LlamaForCausalLM"])
        r = _make_runner(hf_config=cfg, bs=4, raw_bs=2, raw_num_token=2)
        r.backend.replay_with_input_update.return_value = LogitsProcessorOutput(
            next_token_logits=torch.randn(10, 100),
            hidden_states=torch.randn(10, 64),
        )
        fb = _make_forward_batch([10, 20], needs_init=False)
        r.execute(fb)
        # Verify input_ids and positions were copied to buffers
        r.buffers.input_ids.__getitem__.assert_called()
        r.buffers.positions.__getitem__.assert_called()

    def test_execute_non_dsa_decode(self):
        cfg = SimpleNamespace(architectures=["LlamaForCausalLM"])
        r = _make_runner(hf_config=cfg, bs=4, raw_bs=2)
        r.backend.replay_with_input_update.return_value = LogitsProcessorOutput(
            next_token_logits=torch.randn(10, 100),
            hidden_states=torch.randn(10, 64),
        )
        fb = _make_forward_batch([10, 20], is_target_verify=False)
        result = r.execute(fb)
        kwargs = r.backend.replay_with_input_update.call_args.kwargs
        self.assertEqual(kwargs["seq_lens"], [10, 20, 0, 0])
        self.assertIsInstance(result, LogitsProcessorOutput)

    def test_execute_non_dsa_target_verify(self):
        """Non-DSA target_verify: (seq_lens + captured_req_width) + zeros."""
        cfg = SimpleNamespace(architectures=["LlamaForCausalLM"])
        r = _make_runner(hf_config=cfg, bs=4, raw_bs=2, captured_req_width=1)
        r.backend.replay_with_input_update.return_value = LogitsProcessorOutput(
            next_token_logits=torch.randn(10, 100),
            hidden_states=torch.randn(10, 64),
        )
        fb = _make_forward_batch([10, 20], is_target_verify=True)
        r.execute(fb)
        kwargs = r.backend.replay_with_input_update.call_args.kwargs
        # [10, 20] + 1 = [11, 21], then + [0, 0] padding
        self.assertEqual(kwargs["seq_lens"], [11, 21, 0, 0])

    def test_execute_dsa(self):
        """DSA -> backend.replay, not replay_with_input_update."""
        cfg = SimpleNamespace(architectures=["DeepseekV3ForCausalLM"], index_topk=8)
        r = _make_runner(hf_config=cfg)
        r.backend.replay.return_value = LogitsProcessorOutput(
            next_token_logits=torch.randn(10, 100),
        )
        fb = _make_forward_batch([10, 20])
        r.execute(fb)
        r.backend.replay.assert_called_once()
        r.backend.replay_with_input_update.assert_not_called()

    def test_execute_output_non_dllm(self):
        """Non-DLLM output: slices next_token_logits, drops full_logits."""
        cfg = SimpleNamespace(architectures=["LlamaForCausalLM"])
        r = _make_runner(hf_config=cfg, is_dllm=False, raw_num_token=2)
        r.backend.replay_with_input_update.return_value = LogitsProcessorOutput(
            next_token_logits=torch.randn(10, 100),
            full_logits=torch.randn(10, 100),
            hidden_states=torch.randn(10, 64),
        )
        fb = _make_forward_batch([10, 20])
        result = r.execute(fb)
        self.assertIsNotNone(result.next_token_logits)
        self.assertEqual(
            result.next_token_logits.shape[0], 2
        )  # sliced to raw_num_token
        self.assertIsNone(result.full_logits)
        self.assertEqual(result.hidden_states.shape[0], 2)

    def test_execute_output_dllm(self):
        """DLLM output: slices full_logits, drops next_token_logits."""
        cfg = SimpleNamespace(architectures=["LlamaForCausalLM"])
        r = _make_runner(hf_config=cfg, is_dllm=True, raw_num_token=2)
        r.backend.replay_with_input_update.return_value = LogitsProcessorOutput(
            next_token_logits=torch.randn(10, 100),
            full_logits=torch.randn(10, 100),
            hidden_states=torch.randn(10, 64),
        )
        fb = _make_forward_batch([10, 20])
        result = r.execute(fb)
        self.assertIsNone(result.next_token_logits)
        self.assertIsNotNone(result.full_logits)
        self.assertEqual(result.full_logits.shape[0], 2)
        self.assertEqual(result.hidden_states.shape[0], 2)

    def test_execute_output_pp_proxy(self):
        """PP proxy tensors: each tensor sliced to [:bs] (not raw_bs)."""
        cfg = SimpleNamespace(architectures=["LlamaForCausalLM"])
        r = _make_runner(hf_config=cfg, bs=4, raw_bs=2)
        r.backend.replay_with_input_update.return_value = PPProxyTensors(
            {
                "hidden": torch.randn(8, 64),
                "logits": torch.randn(8, 100),
            }
        )
        fb = _make_forward_batch([10, 20])
        result = r.execute(fb)
        self.assertIsInstance(result, PPProxyTensors)
        # bs=4 (not raw_bs=2): PP keeps padding for downstream stage
        self.assertEqual(result.tensors["hidden"].shape, (4, 64))
        self.assertEqual(result.tensors["logits"].shape, (4, 100))

    def test_init_sets_if_use_v2(self):
        """__init__ detects V2 architectures."""
        model_runner = MagicMock()
        model_runner.model_config.hf_config.architectures = ["MiMoV2ForCausalLM"]
        r = NPUGraphRunner(model_runner)
        self.assertTrue(r.if_use_v2)
        self.assertIsNone(r.update_attr_name)
        self.assertFalse(r.use_fia)
        # patch_model was monkey-patched
        self.assertIs(_tc_mod.patch_model, patch_model_npu)


if __name__ == "__main__":
    unittest.main()
