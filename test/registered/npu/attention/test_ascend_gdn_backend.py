"""Unit tests for AscendGDNAttnBackend — shape logic, branches, metadata."""

import sys
import unittest
from enum import IntEnum, auto
from unittest.mock import MagicMock, patch

# ---- Mock modules unavailable on CPU/Windows before any sglang import ----
_mock = MagicMock()
for _mod in (
    "torch_npu",
    "triton",
    "triton.language",
    "sgl_kernel_npu",
    "sgl_kernel_npu.attention",
    "sgl_kernel_npu.attention.sinks_attention",
    "sgl_kernel_npu.fla",
    "sgl_kernel_npu.fla.fused_gdn_gating",
    "sgl_kernel_npu.mamba",
    "sgl_kernel_npu.mamba.causal_conv1d",
    "sglang.srt.utils.hf_transformers_patches",
    "sglang.global_config",
    "sglang.lang.api",
    "sglang.lang.backend.runtime_endpoint",
    "sglang.lang.choices",
    "sglang.utils",
    "sglang.srt.configs.model_config",
    "sglang.srt.dllm.config",
    "sglang.srt.hardware_backend.npu.attention.ascend_torch_native_backend",
    "sglang.srt.hardware_backend.npu.attention.mla_preprocess",
    "sglang.srt.layers.attention.base_attn_backend",
    "sglang.srt.layers.attention.dsa.utils",
    "sglang.srt.layers.radix_attention",
    "sglang.srt.layers.utils.cp_utils",
    "sglang.srt.mem_cache.swa_memory_pool",
    "sglang.srt.runtime_context",
    "sglang.srt.speculative.spec_info",
    "sglang.srt.utils",
    "aiohttp",
    "sglang.test.ci.ci_register",
    "sglang.test.test_utils",
):
    sys.modules[_mod] = _mock

# --- Base class stubs (must be real classes, not MagicMock instances) ---


class _ForwardMetadataStub:
    """Plain object — no MagicMock auto-attribute, so hasattr works correctly."""

    pass


class _MambaAttnBackendBaseStub:
    """Stub for MambaAttnBackendBase → provides self.forward_metadata."""

    def __init__(self, model_runner):
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.forward_metadata = _ForwardMetadataStub()
        self.state_indices_list_gdn = []

    def init_forward_metadata(self, forward_batch):
        pass

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=True):
        pass

    def _track_mamba_state_decode(
        self, forward_batch, conv_states, ssm_states, cache_indices
    ):
        pass

    def _track_mamba_state_extend(self, forward_batch, h, ssm_states, forward_metadata):
        pass


# MambaAttnBackendBase lives in hybrid_linear_attn_backend
_mamba_mod = type(sys)("sglang.srt.layers.attention.hybrid_linear_attn_backend")
_mamba_mod.MambaAttnBackendBase = _MambaAttnBackendBaseStub
sys.modules["sglang.srt.layers.attention.hybrid_linear_attn_backend"] = _mamba_mod

# AscendMambaAttnBackendBase lives in ascend_hybrid_linear_attn_backend
_ascend_mamba_mod = type(sys)(
    "sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend"
)
_ascend_mamba_mod.AscendMambaAttnBackendBase = _MambaAttnBackendBaseStub
sys.modules[
    "sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend"
] = _ascend_mamba_mod


# GDNKernelDispatcher — must be a real class stub, not MagicMock
class _GDNKernelDispatcherStub:
    def __init__(self, decode_backend, prefill_backend, **kwargs):
        self.decode_backend = decode_backend
        self.prefill_backend = prefill_backend

    def decode(self, **kwargs):
        """Stub — overridden per-test via mock."""
        pass

    def extend(self, **kwargs):
        """Stub — overridden per-test via mock."""
        pass


_gdn_backend_mod = type(sys)("sglang.srt.layers.attention.linear.gdn_backend")
_gdn_backend_mod.GDNKernelDispatcher = _GDNKernelDispatcherStub
sys.modules["sglang.srt.layers.attention.linear.gdn_backend"] = _gdn_backend_mod

_linear_utils_mod = type(sys)("sglang.srt.layers.attention.linear.utils")
_linear_utils_mod.get_linear_attn_decode_backend = lambda: "decode_backend_stub"
_linear_utils_mod.get_linear_attn_prefill_backend = lambda: "prefill_backend_stub"
sys.modules["sglang.srt.layers.attention.linear.utils"] = _linear_utils_mod


# RadixLinearAttention
class _RadixLinearAttentionStub:
    pass


_radix_linear_mod = type(sys)("sglang.srt.layers.radix_linear_attention")
_radix_linear_mod.RadixLinearAttention = _RadixLinearAttentionStub
sys.modules["sglang.srt.layers.radix_linear_attention"] = _radix_linear_mod


# MambaPool
class _MambaPoolStub:
    class SpeculativeState:
        """Real class so isinstance(..., SpeculativeState) passes in TARGET_VERIFY."""

        pass


_mem_pool_mod = type(sys)("sglang.srt.mem_cache.memory_pool")
_mem_pool_mod.MambaPool = _MambaPoolStub
sys.modules["sglang.srt.mem_cache.memory_pool"] = _mem_pool_mod


# ForwardBatch, ForwardMode — we need a real ForwardMode enum for branch testing
class _ForwardModeStub(IntEnum):
    EXTEND = auto()
    DECODE = auto()
    MIXED = auto()
    IDLE = auto()
    TARGET_VERIFY = auto()
    DRAFT_EXTEND_V2 = auto()
    PREBUILT = auto()
    SPLIT_PREFILL = auto()
    DLLM_EXTEND = auto()

    def is_extend(self, include_draft_extend_v2=False):
        return self in (
            _ForwardModeStub.EXTEND,
            _ForwardModeStub.MIXED,
            _ForwardModeStub.TARGET_VERIFY,
            _ForwardModeStub.SPLIT_PREFILL,
            _ForwardModeStub.DLLM_EXTEND,
        ) or (include_draft_extend_v2 and self == _ForwardModeStub.DRAFT_EXTEND_V2)

    def is_prefill(self, include_draft_extend_v2=False):
        return self.is_extend(include_draft_extend_v2=include_draft_extend_v2)

    def is_decode(self):
        return self == _ForwardModeStub.DECODE

    def is_target_verify(self):
        return self == _ForwardModeStub.TARGET_VERIFY

    def is_draft_extend_v2(self):
        return self == _ForwardModeStub.DRAFT_EXTEND_V2

    def is_idle(self):
        return self == _ForwardModeStub.IDLE


_fb_info_mod = type(sys)("sglang.srt.model_executor.forward_batch_info")
_fb_info_mod.ForwardBatch = MagicMock
_fb_info_mod.ForwardMode = _ForwardModeStub
sys.modules["sglang.srt.model_executor.forward_batch_info"] = _fb_info_mod

# ModelRunner
_model_runner_mod = type(sys)("sglang.srt.model_executor.model_runner")
_model_runner_mod.ModelRunner = MagicMock
sys.modules["sglang.srt.model_executor.model_runner"] = _model_runner_mod

# Eagle info
_eagle_mod = type(sys)("sglang.srt.speculative.eagle_info")
_eagle_mod.EagleDraftInput = MagicMock
_eagle_mod.EagleVerifyInput = MagicMock
sys.modules["sglang.srt.speculative.eagle_info"] = _eagle_mod

# sglang.version
_ver = type(sys)("sglang.version")
_ver.__version__ = "0.0.0.dev0"
sys.modules["sglang.version"] = _ver

import torch

from sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend import (
    AscendGDNAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def register_npu_ci(est_time, suite=None, nightly=False, disabled=None):
    def decorator(cls):
        return cls

    return decorator


class CustomTestCase(unittest.TestCase):
    pass


register_npu_ci(est_time=3, suite="stage-a-unit-test-npu")


def _make_model_runner():
    """Create a minimal ModelRunner mock with nested mamba_cache structure."""
    mr = MagicMock()
    # simulate: model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0]
    # conv[0] is a 4D tensor: (num_layers, batch, d_inner, d_state) or similar
    conv_tensor = torch.zeros(2, 4, 64, 128)
    mamba_cache = MagicMock()
    mamba_cache.conv = [conv_tensor]
    mamba_pool = MagicMock()
    mamba_pool.mamba_cache = mamba_cache
    req_pool = MagicMock()
    req_pool.mamba_pool = mamba_pool
    mr.req_to_token_pool = req_pool
    mr.dtype = torch.float16
    return mr


def _make_backend(
    mamba_cache_indices=None,
    has_mamba_track_mask=False,
):
    """Create an AscendGDNAttnBackend with optional forward_metadata overrides."""
    model_runner = _make_model_runner()
    backend = AscendGDNAttnBackend(model_runner)
    if mamba_cache_indices is not None:
        backend.forward_metadata.mamba_cache_indices = mamba_cache_indices
    backend.forward_metadata.has_mamba_track_mask = has_mamba_track_mask
    return backend


# ---------------------------------------------------------------------------
#  Helpers for forward_decode / forward_extend / fused_recurrent_xxx tests
# ---------------------------------------------------------------------------


def _make_layer(**overrides):
    """Mock RadixLinearAttention with typical GDN dimensions."""
    layer = MagicMock()
    layer.layer_id = 0
    layer.q_dim = 64
    layer.k_dim = 64
    layer.v_dim = 128
    layer.num_q_heads = 8
    layer.num_k_heads = 8
    layer.num_v_heads = 8
    layer.head_q_dim = 8
    layer.head_k_dim = 8
    layer.head_v_dim = 16
    layer.conv_weights = torch.randn(8, 4)  # (d_inner, kernel_size=4)
    layer.bias = torch.randn(8)
    layer.activation = "silu"
    layer.A_log = torch.randn(8)
    layer.dt_bias = torch.randn(8)
    for k, v in overrides.items():
        setattr(layer, k, v)
    return layer


def _make_fb(forward_mode=ForwardMode.DECODE, batch_size=2, **overrides):
    """Mock ForwardBatch."""
    fb = MagicMock()
    fb.forward_mode = forward_mode
    fb.batch_size = batch_size
    fb.spec_info = MagicMock()
    fb.spec_info.draft_token_num = 3
    fb.extend_prefix_lens = torch.tensor([0, 0])
    fb.extend_seq_lens_cpu = torch.tensor([1, 1])
    fb.num_token_non_padded_cpu = batch_size
    fb.input_ids = torch.zeros(batch_size, dtype=torch.int64)
    for k, v in overrides.items():
        setattr(fb, k, v)
    return fb


def _setup_forward_metadata(backend, batch_size=2):
    """Set up forward_metadata fields needed by forward methods."""
    m = backend.forward_metadata
    m.query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    m.mamba_cache_indices = torch.arange(batch_size, dtype=torch.int32)
    m.retrieve_next_token = None
    m.retrieve_next_sibling = None
    m.retrieve_parent_token = None
    m.has_mamba_track_mask = False
    m.mamba_cache_indices_gdn = torch.arange(batch_size, dtype=torch.int32)


def _setup_layer_cache(
    backend,
    batch_size=2,
    d_inner=256,
    d_state=128,
    ssm_heads=4,
    ssm_head_dim=64,
    is_target_verify=False,
):
    """Set up mamba2_layer_cache return value."""
    if is_target_verify:
        from sglang.srt.mem_cache.memory_pool import MambaPool

        cache = MambaPool.SpeculativeState()
    else:
        cache = MagicMock()
    cache.conv = [torch.zeros(batch_size, d_inner, d_state)]
    cache.temporal = torch.zeros(batch_size, ssm_heads, ssm_head_dim, ssm_head_dim)
    if is_target_verify:
        cache.intermediate_ssm = torch.zeros(
            batch_size, ssm_heads, ssm_head_dim, ssm_head_dim
        )
        cache.intermediate_conv_window = [torch.zeros(batch_size, d_inner, d_state)]
    backend.req_to_token_pool.mamba2_layer_cache.return_value = cache
    return cache


class TestInit(CustomTestCase):
    """Tests for AscendGDNAttnBackend.__init__."""

    def test_conv_states_shape(self):
        """conv_states_shape = (..., d_inner, d_state) after swapping last 2 dims."""
        backend = _make_backend()
        expected_shape = torch.Size((2, 4, 128, 64))
        self.assertEqual(backend.conv_states_shape, expected_shape)

    def test_kernel_dispatcher_created(self):
        """kernel_dispatcher is created with correct backend args."""
        backend = _make_backend()
        self.assertIsNotNone(backend.kernel_dispatcher)
        self.assertEqual(
            backend.kernel_dispatcher.decode_backend, "decode_backend_stub"
        )
        self.assertEqual(
            backend.kernel_dispatcher.prefill_backend, "prefill_backend_stub"
        )

    def test_dtype_stored(self):
        """model_runner.dtype is accessible."""
        mr = _make_model_runner()
        backend = AscendGDNAttnBackend(mr)
        self.assertEqual(backend.req_to_token_pool, mr.req_to_token_pool)


class TestPrepareMambaTrackMetadata(CustomTestCase):
    """Tests for AscendGDNAttnBackend._prepare_mamba_track_metadata."""

    def test_no_track_mask(self):
        """has_mamba_track_mask=False → nothing happens."""
        backend = _make_backend(has_mamba_track_mask=False)
        fb = MagicMock()
        fb.mamba_track_mask = None
        fb.mamba_track_indices = None
        backend._prepare_mamba_track_metadata(fb)
        # Should not crash, and conv_states_mask_indices should not be set
        self.assertFalse(hasattr(backend.forward_metadata, "conv_states_mask_indices"))

    def test_with_track_mask(self):
        """has_mamba_track_mask=True → sets conv_states_mask_indices."""
        backend = _make_backend(has_mamba_track_mask=True)
        fb = MagicMock()
        fb.mamba_track_mask = torch.tensor([False, True, False, True])
        fb.mamba_track_indices = torch.tensor([10, 20, 30, 40])
        backend._prepare_mamba_track_metadata(fb)
        # mamba_track_mask.nonzero() → indices [1, 3]
        # mamba_track_indices[[1, 3]] → [20, 40]
        expected = torch.tensor([20, 40])
        self.assertTrue(
            torch.equal(backend.forward_metadata.conv_states_mask_indices, expected)
        )

    def test_all_false_track_mask(self):
        """All entries False → empty conv_states_mask_indices."""
        backend = _make_backend(has_mamba_track_mask=True)
        fb = MagicMock()
        fb.mamba_track_mask = torch.tensor([False, False, False])
        fb.mamba_track_indices = torch.tensor([10, 20, 30])
        backend._prepare_mamba_track_metadata(fb)
        self.assertEqual(backend.forward_metadata.conv_states_mask_indices.numel(), 0)

    def test_all_true_track_mask(self):
        """All entries True → all indices selected."""
        backend = _make_backend(has_mamba_track_mask=True)
        fb = MagicMock()
        fb.mamba_track_mask = torch.tensor([True, True, True])
        fb.mamba_track_indices = torch.tensor([10, 20, 30])
        backend._prepare_mamba_track_metadata(fb)
        self.assertTrue(
            torch.equal(
                backend.forward_metadata.conv_states_mask_indices,
                torch.tensor([10, 20, 30]),
            )
        )


class TestPrepareGdnInputs(CustomTestCase):
    """Tests for AscendGDNAttnBackend.prepare_gdn_inputs."""

    def test_decode_mode(self):
        """DECODE mode: ssm_state_indices = cache_indices, seq_lengths = 1."""
        backend = _make_backend(
            mamba_cache_indices=torch.tensor([0, 1, 2]),
        )
        bs = 3
        backend.prepare_gdn_inputs(
            bs=bs,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
        )
        # num_accept_tokens / actual_seq_lengths: shape (bs,), all ones
        self.assertEqual(backend.num_accept_tokens.shape, (bs,))
        self.assertEqual(backend.actual_seq_lengths.shape, (bs,))
        self.assertTrue((backend.num_accept_tokens == 1).all().item())
        self.assertTrue((backend.actual_seq_lengths == 1).all().item())
        # ssm_state_indices == cache_indices (identity)
        expected = torch.tensor([0, 1, 2], dtype=torch.int32)
        self.assertTrue(torch.equal(backend.ssm_state_indices, expected))

    def test_target_verify_mode(self):
        """TARGET_VERIFY mode: ssm_state_indices = arange(bs * draft_token_num)."""
        backend = _make_backend(
            mamba_cache_indices=torch.tensor([0, 1, 2]),
        )
        bs = 3
        spec_info = MagicMock()
        spec_info.draft_token_num = 5
        backend.prepare_gdn_inputs(
            bs=bs,
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=spec_info,
        )
        # actual_seq_lengths = ones * draft_token_num = (5,5,5)
        self.assertTrue((backend.actual_seq_lengths == 5).all().item())
        # ssm_state_indices: arange(bs * draft_token_num) = arange(15)
        expected = torch.arange(15, dtype=torch.int32)
        self.assertTrue(torch.equal(backend.ssm_state_indices, expected))

    def test_tensor_device(self):
        """All created tensors are on the same device as cache_indices."""
        cache_indices = torch.tensor([0, 1], device="cpu")
        backend = _make_backend(mamba_cache_indices=cache_indices)
        backend.prepare_gdn_inputs(
            bs=2,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
        )
        self.assertEqual(backend.num_accept_tokens.device, cache_indices.device)
        self.assertEqual(backend.actual_seq_lengths.device, cache_indices.device)
        self.assertEqual(backend.ssm_state_indices.device, cache_indices.device)

    def test_tensor_dtype(self):
        """num_accept_tokens and actual_seq_lengths are int32."""
        backend = _make_backend(
            mamba_cache_indices=torch.tensor([0, 1]),
        )
        backend.prepare_gdn_inputs(
            bs=2,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
        )
        self.assertEqual(backend.num_accept_tokens.dtype, torch.int32)
        self.assertEqual(backend.actual_seq_lengths.dtype, torch.int32)

    def test_bs_zero(self):
        """bs=0: all tensors are empty (shape (0,))."""
        backend = _make_backend(
            mamba_cache_indices=torch.tensor([], dtype=torch.int32),
        )
        backend.prepare_gdn_inputs(
            bs=0,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
        )
        self.assertEqual(backend.num_accept_tokens.shape, (0,))
        self.assertEqual(backend.actual_seq_lengths.shape, (0,))
        self.assertEqual(backend.ssm_state_indices.shape, (0,))


class TestInitForwardMetadata(CustomTestCase):
    """Tests for init_forward_metadata / init_forward_metadata_out_graph."""

    def test_draft_extend_v2_skips(self):
        """DRAFT_EXTEND_V2 → returns early, no prepare_gdn_inputs call."""
        backend = _make_backend(
            mamba_cache_indices=torch.tensor([0, 1]),
        )
        fb = MagicMock()
        fb.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        fb.batch_size = 2
        fb.spec_info = None
        # Should not raise — early return
        backend.init_forward_metadata(fb)
        # num_accept_tokens should NOT be set (early return before prepare_gdn_inputs)
        self.assertFalse(hasattr(backend, "num_accept_tokens"))

    def test_decode_calls_prepare_gdn_inputs(self):
        """DECODE mode → prepare_gdn_inputs runs, sets num_accept_tokens."""
        backend = _make_backend(
            mamba_cache_indices=torch.tensor([0, 1]),
        )
        fb = MagicMock()
        fb.forward_mode = ForwardMode.DECODE
        fb.batch_size = 2
        fb.spec_info = None
        backend.init_forward_metadata(fb)
        self.assertTrue(hasattr(backend, "num_accept_tokens"))
        self.assertEqual(backend.num_accept_tokens.shape, (2,))
        self.assertEqual(backend.graph_mode, False)

    def test_graph_mode_set(self):
        """init_forward_metadata_out_graph sets graph_mode=True."""
        backend = _make_backend(
            mamba_cache_indices=torch.tensor([0, 1]),
        )
        fb = MagicMock()
        fb.forward_mode = ForwardMode.DECODE
        fb.batch_size = 2
        fb.spec_info = None
        fb.mamba_track_mask = None
        fb.mamba_track_indices = None
        backend.init_forward_metadata_out_graph(fb)
        self.assertEqual(backend.graph_mode, True)

    def test_graph_mode_draft_extend_v2_skips(self):
        """init_forward_metadata_out_graph with DRAFT_EXTEND_V2 → early return."""
        backend = _make_backend(
            mamba_cache_indices=torch.tensor([0, 1]),
        )
        fb = MagicMock()
        fb.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        fb.batch_size = 2
        fb.spec_info = None
        backend.init_forward_metadata_out_graph(fb)
        self.assertFalse(hasattr(backend, "num_accept_tokens"))
        self.assertFalse(hasattr(backend, "graph_mode"))

    def test_graph_mode_target_verify(self):
        """init_forward_metadata_out_graph with TARGET_VERIFY → full init, graph_mode=True."""
        backend = _make_backend(
            mamba_cache_indices=torch.tensor([0, 1]),
        )
        fb = MagicMock()
        fb.forward_mode = ForwardMode.TARGET_VERIFY
        fb.batch_size = 2
        fb.spec_info = MagicMock()
        fb.spec_info.draft_token_num = 3
        fb.mamba_track_mask = None
        fb.mamba_track_indices = None
        backend.init_forward_metadata_out_graph(fb)
        self.assertTrue(hasattr(backend, "num_accept_tokens"))
        self.assertEqual(backend.graph_mode, True)
        self.assertTrue((backend.actual_seq_lengths == 3).all().item())
        expected_indices = torch.arange(6, dtype=torch.int32)
        self.assertTrue(torch.equal(backend.ssm_state_indices, expected_indices))

    def test_extend_mode(self):
        """init_forward_metadata with EXTEND → prepare_gdn_inputs runs."""
        backend = _make_backend(
            mamba_cache_indices=torch.tensor([0, 1]),
        )
        fb = MagicMock()
        fb.forward_mode = ForwardMode.EXTEND
        fb.batch_size = 2
        fb.spec_info = None
        fb.mamba_track_mask = None
        fb.mamba_track_indices = None
        backend.init_forward_metadata(fb)
        self.assertTrue(hasattr(backend, "num_accept_tokens"))
        self.assertEqual(backend.graph_mode, False)
        self.assertTrue(
            torch.equal(
                backend.ssm_state_indices, torch.tensor([0, 1], dtype=torch.int32)
            )
        )

    def test_target_verify_mode(self):
        """init_forward_metadata with TARGET_VERIFY → ssm_state_indices = arange(bs * draft_token_num)."""
        backend = _make_backend(
            mamba_cache_indices=torch.tensor([0, 1]),
        )
        fb = MagicMock()
        fb.forward_mode = ForwardMode.TARGET_VERIFY
        fb.batch_size = 2
        fb.spec_info = MagicMock()
        fb.spec_info.draft_token_num = 3
        fb.mamba_track_mask = None
        fb.mamba_track_indices = None
        backend.init_forward_metadata(fb)
        self.assertTrue(hasattr(backend, "num_accept_tokens"))
        self.assertEqual(backend.graph_mode, False)
        self.assertTrue((backend.actual_seq_lengths == 3).all().item())
        expected_indices = torch.arange(6, dtype=torch.int32)
        self.assertTrue(torch.equal(backend.ssm_state_indices, expected_indices))


# ---------------------------------------------------------------------------
#  forward_decode
# ---------------------------------------------------------------------------

_MOD = "sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend"


class TestForwardDecode(CustomTestCase):
    """Tests for AscendGDNAttnBackend.forward_decode."""

    def setUp(self):
        self.backend = _make_backend(
            mamba_cache_indices=torch.tensor([0, 1], dtype=torch.int32),
        )
        _setup_forward_metadata(self.backend)
        _setup_layer_cache(self.backend)
        self.layer = _make_layer()
        self.fb = _make_fb(ForwardMode.DECODE, batch_size=2)
        self.x = torch.randn(2, 256)  # (bs, q_dim + k_dim + v_dim = 64+64+128)
        self.a = torch.randn(2, 8)
        self.b = torch.randn(2, 8)

        # Replace kernel_dispatcher methods with mocks
        self.backend.kernel_dispatcher.decode = MagicMock(
            return_value=torch.randn(2, 8, 16)
        )
        self.backend._track_mamba_state_decode = MagicMock()

    @patch(_MOD + ".causal_conv1d_update")
    def test_returns_decode_output(self, mock_conv):
        """forward_decode returns kernel_dispatcher.decode output."""
        mock_conv.return_value = self.x.clone()
        result = self.backend.forward_decode(
            self.layer, self.fb, self.x, self.a, self.b
        )
        self.assertTrue(
            torch.equal(result, self.backend.kernel_dispatcher.decode.return_value)
        )

    @patch(_MOD + ".causal_conv1d_update")
    def test_split_shapes(self, mock_conv):
        """torch.split produces Q, K, V with correct dims."""
        mock_conv.return_value = self.x.clone()
        # Capture split arguments by wrapping kernel_dispatcher.decode
        decode_args = {}

        def _capture(**kw):
            decode_args.update(kw)
            return torch.randn(2, 8, 16)

        self.backend.kernel_dispatcher.decode = MagicMock(side_effect=_capture)
        self.backend.forward_decode(self.layer, self.fb, self.x, self.a, self.b)
        q, k, v = decode_args["q"], decode_args["k"], decode_args["v"]
        self.assertEqual(q.shape, (1, 2, 8, 8))  # (1, bs, n_q_heads, head_q_dim)
        self.assertEqual(k.shape, (1, 2, 8, 8))  # (1, bs, n_k_heads, head_k_dim)
        self.assertEqual(v.shape, (1, 2, 8, 16))  # (1, bs, n_v_heads, head_v_dim)

    @patch(_MOD + ".causal_conv1d_update")
    def test_track_called(self, mock_conv):
        """_track_mamba_state_decode is called after decode."""
        mock_conv.return_value = self.x.clone()
        self.backend.forward_decode(self.layer, self.fb, self.x, self.a, self.b)
        self.backend._track_mamba_state_decode.assert_called_once_with(
            self.fb,
            self.backend.req_to_token_pool.mamba2_layer_cache.return_value.conv[0],
            self.backend.req_to_token_pool.mamba2_layer_cache.return_value.temporal,
            self.backend.forward_metadata.mamba_cache_indices,
        )


# ---------------------------------------------------------------------------
#  forward_extend  (non-TARGET_VERIFY)
# ---------------------------------------------------------------------------


class TestForwardExtend(CustomTestCase):
    """Tests for AscendGDNAttnBackend.forward_extend — non-target-verify path."""

    def setUp(self):
        self.backend = _make_backend(
            mamba_cache_indices=torch.tensor([0, 1], dtype=torch.int32),
        )
        _setup_forward_metadata(self.backend)
        _setup_layer_cache(self.backend)
        self.layer = _make_layer()
        self.fb = _make_fb(ForwardMode.EXTEND, batch_size=2)
        self.fb.extend_prefix_lens = torch.tensor([0, 0])
        self.seq_len = 2
        self.x = torch.randn(self.seq_len, 256)  # (seq_len, q_dim+k_dim+v_dim)
        self.a = torch.randn(self.seq_len, 8)
        self.b = torch.randn(self.seq_len, 8)

        self.backend._track_mamba_state_extend = MagicMock()
        self.backend.kernel_dispatcher.extend = MagicMock(
            return_value=(
                torch.randn(self.seq_len, 8, 16),  # core_attn_out
                torch.randn(2, 4, 64, 64),  # last_recurrent_state
                torch.randn(2, 4, 64, 64),  # h
            )
        )

    @patch(_MOD + ".causal_conv1d_fn")
    @patch(_MOD + ".fused_gdn_gating")
    def test_has_initial_states_false(self, mock_gate, mock_conv):
        """extend_prefix_lens = [0,0] → has_initial_states = [False, False]."""
        mock_conv.return_value = torch.randn(256, self.seq_len)
        mock_gate.return_value = (
            torch.randn(self.seq_len, 8),  # g
            torch.randn(self.seq_len, 8),  # beta
        )
        self.backend.forward_extend(self.layer, self.fb, self.x, self.a, self.b)
        _, kwargs = mock_conv.call_args
        self.assertTrue((kwargs["has_initial_state"] == False).all().item())

    @patch(_MOD + ".causal_conv1d_fn")
    @patch(_MOD + ".fused_gdn_gating")
    def test_has_initial_states_true(self, mock_gate, mock_conv):
        """extend_prefix_lens = [5,3] → has_initial_states = [True, True]."""
        self.fb.extend_prefix_lens = torch.tensor([5, 3])
        mock_conv.return_value = torch.randn(256, self.seq_len)
        mock_gate.return_value = (
            torch.randn(self.seq_len, 8),
            torch.randn(self.seq_len, 8),
        )
        self.backend.forward_extend(self.layer, self.fb, self.x, self.a, self.b)
        _, kwargs = mock_conv.call_args
        self.assertTrue((kwargs["has_initial_state"] == True).all().item())

    @patch(_MOD + ".causal_conv1d_fn")
    @patch(_MOD + ".fused_gdn_gating")
    def test_has_initial_states_mixed(self, mock_gate, mock_conv):
        """extend_prefix_lens = [5,0] → has_initial_states = [True, False]."""
        self.fb.extend_prefix_lens = torch.tensor([5, 0])
        mock_conv.return_value = torch.randn(256, self.seq_len)
        mock_gate.return_value = (
            torch.randn(self.seq_len, 8),
            torch.randn(self.seq_len, 8),
        )
        self.backend.forward_extend(self.layer, self.fb, self.x, self.a, self.b)
        _, kwargs = mock_conv.call_args
        expected = torch.tensor([True, False])
        self.assertTrue(torch.equal(kwargs["has_initial_state"], expected))

    @patch(_MOD + ".causal_conv1d_fn")
    @patch(_MOD + ".fused_gdn_gating")
    def test_mamba_track_mask_updates_conv(self, mock_gate, mock_conv):
        """has_mamba_track_mask=True → conv_states written with track data."""
        self.backend.forward_metadata.has_mamba_track_mask = True
        self.backend.forward_metadata.track_conv_indices = torch.tensor([0])
        self.backend.forward_metadata.conv_states_mask_indices = torch.tensor([0])
        mock_conv.return_value = torch.randn(256, self.seq_len)
        mock_gate.return_value = (
            torch.randn(self.seq_len, 8),
            torch.randn(self.seq_len, 8),
        )
        conv_states = (
            self.backend.req_to_token_pool.mamba2_layer_cache.return_value.conv[0]
        )
        original = conv_states.clone()
        self.backend.forward_extend(self.layer, self.fb, self.x, self.a, self.b)
        # conv_states should be modified (track mask wrote data into it)
        self.assertFalse(torch.equal(conv_states, original))
        # The position at conv_states_mask_indices=[0] should be non-zero
        self.assertFalse((conv_states.transpose(1, 2)[0] == 0).all().item())

    @patch(_MOD + ".causal_conv1d_fn")
    @patch(_MOD + ".fused_gdn_gating")
    def test_split_and_view_shapes(self, mock_gate, mock_conv):
        """torch.split + view produce correct shapes in non-target-verify."""
        mock_conv.return_value = torch.randn(256, self.seq_len)
        mock_gate.return_value = (
            torch.randn(self.seq_len, 8),
            torch.randn(self.seq_len, 8),
        )

        extend_args = {}

        def _capture(**kw):
            extend_args.update(kw)
            return (
                torch.randn(self.seq_len, 8, 16),  # core_attn_out
                torch.randn(2, 4, 64, 64),  # last_recurrent_state
                None,  # h
            )

        self.backend.kernel_dispatcher.extend = MagicMock(side_effect=_capture)
        self.backend.forward_extend(self.layer, self.fb, self.x, self.a, self.b)
        q = extend_args["q"]
        self.assertEqual(q.shape, (1, self.seq_len, 8, 8))

    @patch(_MOD + ".causal_conv1d_fn")
    @patch(_MOD + ".fused_gdn_gating")
    def test_ssm_states_update(self, mock_gate, mock_conv):
        """last_recurrent_state is not None → ssm_states[cache_indices] updated."""
        mock_conv.return_value = torch.randn(256, self.seq_len)
        mock_gate.return_value = (
            torch.randn(self.seq_len, 8),
            torch.randn(self.seq_len, 8),
        )
        ssm_states = (
            self.backend.req_to_token_pool.mamba2_layer_cache.return_value.temporal
        )
        cache_indices = self.backend.forward_metadata.mamba_cache_indices
        original = ssm_states.clone()
        self.backend.forward_extend(self.layer, self.fb, self.x, self.a, self.b)
        # ssm_states[cache_indices] should be updated with last_recurrent_state
        self.assertFalse(
            torch.equal(ssm_states[cache_indices], original[cache_indices])
        )
        # Verify the update matches what kernel_dispatcher.extend returned
        last_recurrent_state = self.backend.kernel_dispatcher.extend.return_value[1]
        expected = last_recurrent_state.to(ssm_states.dtype, copy=False)
        self.assertTrue(torch.equal(ssm_states[cache_indices], expected))

    @patch(_MOD + ".causal_conv1d_fn")
    @patch(_MOD + ".fused_gdn_gating")
    def test_track_mamba_state_extend_called(self, mock_gate, mock_conv):
        """h is not None → _track_mamba_state_extend is called."""
        mock_conv.return_value = torch.randn(256, self.seq_len)
        mock_gate.return_value = (
            torch.randn(self.seq_len, 8),
            torch.randn(self.seq_len, 8),
        )
        self.backend.forward_extend(self.layer, self.fb, self.x, self.a, self.b)
        self.backend._track_mamba_state_extend.assert_called_once()

    @patch(_MOD + ".causal_conv1d_fn")
    @patch(_MOD + ".fused_gdn_gating")
    def test_conv_for_prefill_slice(self, mock_gate, mock_conv):
        """conv_states_for_prefill is passed as keyword argument conv_states."""
        mock_conv.return_value = torch.randn(256, self.seq_len)
        mock_gate.return_value = (
            torch.randn(self.seq_len, 8),
            torch.randn(self.seq_len, 8),
        )
        self.backend.forward_extend(self.layer, self.fb, self.x, self.a, self.b)
        _, kwargs = mock_conv.call_args
        self.assertIn("conv_states", kwargs)
        self.assertEqual(kwargs["conv_states"].shape[0], 2)  # batch_size

    @patch(_MOD + ".causal_conv1d_fn")
    @patch(_MOD + ".fused_gdn_gating")
    def test_h_none_skips_track(self, mock_gate, mock_conv):
        """h=None → _track_mamba_state_extend is NOT called."""
        mock_conv.return_value = torch.randn(256, self.seq_len)
        mock_gate.return_value = (
            torch.randn(self.seq_len, 8),
            torch.randn(self.seq_len, 8),
        )
        self.backend.kernel_dispatcher.extend = MagicMock(
            return_value=(
                torch.randn(self.seq_len, 8, 16),  # core_attn_out
                torch.randn(2, 4, 64, 64),  # last_recurrent_state
                None,  # h is None
            )
        )
        self.backend.forward_extend(self.layer, self.fb, self.x, self.a, self.b)
        self.backend._track_mamba_state_extend.assert_not_called()


# ---------------------------------------------------------------------------
#  forward_extend  (TARGET_VERIFY)
# ---------------------------------------------------------------------------


class TestForwardExtendTargetVerify(CustomTestCase):
    """Tests for forward_extend — is_target_verify = True branch."""

    def setUp(self):
        draft_token_num = 3
        self.backend = _make_backend(
            mamba_cache_indices=torch.tensor([0, 1], dtype=torch.int32),
        )
        _setup_forward_metadata(self.backend)
        _setup_layer_cache(self.backend, is_target_verify=True)
        self.backend.graph_mode = False
        self.layer = _make_layer()
        self.fb = _make_fb(ForwardMode.TARGET_VERIFY, batch_size=2)
        self.fb.num_token_non_padded_cpu = 2 * draft_token_num  # no padding
        self.seq_len = 2 * draft_token_num  # 6
        self.x = torch.randn(self.seq_len, 256)
        self.a = torch.randn(self.seq_len, 8)
        self.b = torch.randn(self.seq_len, 8)

        self.backend.fused_recurrent_gated_delta_rule_update = MagicMock(
            return_value=torch.randn(self.seq_len, 8, 16)
        )

    @patch(_MOD + ".causal_conv1d_update_v2")
    @patch(_MOD + ".fused_gdn_gating_kernel_without_sigmoid")
    def test_has_initial_states(self, mock_gate, mock_conv):
        """TARGET_VERIFY: fused_recurrent called with correctly reshaped beta/g."""
        mock_conv.return_value = self.x.view(2, 3, -1).contiguous().view(6, -1)
        mock_gate.return_value = (
            torch.randn(2 * 3, 8),  # g
            torch.randn(2 * 3, 8),  # beta
        )
        self.backend.forward_extend(self.layer, self.fb, self.x, self.a, self.b)
        # fused_recurrent_gated_delta_rule_update should be called
        self.backend.fused_recurrent_gated_delta_rule_update.assert_called_once()
        # beta and g should be reshaped to (batch_size, draft_token_num, num_v_heads)
        _, kwargs = self.backend.fused_recurrent_gated_delta_rule_update.call_args
        self.assertEqual(kwargs["beta"].shape, (2, 3, 8))
        self.assertEqual(kwargs["g"].shape, (2, 3, 8))

    @patch(_MOD + ".causal_conv1d_update_v2")
    @patch(_MOD + ".fused_gdn_gating_kernel_without_sigmoid")
    def test_output_padding(self, mock_gate, mock_conv):
        """core_attn_out shorter than num_token_padding → zero-padded up."""
        mock_conv.return_value = self.x.clone()
        mock_gate.return_value = (
            torch.randn(2 * 3, 8),
            torch.randn(2 * 3, 8),
        )
        # fused_recurrent returns only 3 rows → after view(-1, 8, 16) → (3, 8, 16)
        # 3 < num_token_padding=6 → triggers zero-padding cat at L287-297
        self.backend.fused_recurrent_gated_delta_rule_update.return_value = torch.randn(
            1, 3, 8, 16
        )
        result = self.backend.forward_extend(
            self.layer, self.fb, self.x, self.a, self.b
        )
        # Should be padded from 3 to 6 (num_token_padding)
        self.assertEqual(result.shape[0], 6)
        # Last 3 rows should be zeros (padding)
        self.assertTrue((result[3:] == 0).all().item())

    @patch(_MOD + ".causal_conv1d_update_v2")
    @patch(_MOD + ".fused_gdn_gating_kernel_without_sigmoid")
    def test_graph_mode_skips_padding(self, mock_gate, mock_conv):
        """graph_mode=True → padding slice and zero-padding cat are both skipped."""
        self.backend.graph_mode = True
        self.fb.num_token_non_padded_cpu = 4  # less than padding
        self.seq_len = 2 * 3
        self.x = torch.randn(self.seq_len, 256)
        mock_conv.return_value = self.x.view(2, 3, -1).contiguous().view(6, -1)
        mock_gate.return_value = (
            torch.randn(2 * 3, 8),
            torch.randn(2 * 3, 8),
        )
        self.backend.fused_recurrent_gated_delta_rule_update.return_value = torch.randn(
            6, 8, 16
        )
        result = self.backend.forward_extend(
            self.layer, self.fb, self.x, self.a, self.b
        )
        # graph_mode=True → no padding slice on mixed_qkv (full input passed to conv)
        _, kwargs = mock_conv.call_args
        self.assertEqual(kwargs["x"].shape[1], 3)  # draft_token_num, not sliced
        # No zero-padding appended to result
        self.assertEqual(result.shape[0], 6)
        # fused_recurrent was called
        self.backend.fused_recurrent_gated_delta_rule_update.assert_called_once()


# ---------------------------------------------------------------------------
#  fused_recurrent_gated_delta_rule_update
# ---------------------------------------------------------------------------


class TestFusedRecurrentGatedDeltaRuleUpdate(CustomTestCase):
    """Tests for AscendGDNAttnBackend.fused_recurrent_gated_delta_rule_update."""

    def setUp(self):
        self.backend = _make_backend(
            mamba_cache_indices=torch.tensor([0, 1], dtype=torch.int32),
        )
        _setup_forward_metadata(self.backend)
        self.backend.num_accept_tokens = torch.ones(2, dtype=torch.int32)
        self.backend.actual_seq_lengths = torch.ones(2, dtype=torch.int32) * 3
        self.backend.ssm_state_indices = torch.arange(
            2 * 3, dtype=torch.int32
        )  # batch * seq_len
        self.backend.graph_mode = False

        self.batch_size = 2
        self.seq_len = 3
        self.nh = 8
        self.nvh = 8
        self.hk = 64
        self.hv = 64
        self.mix_qkv = torch.randn(self.batch_size, self.seq_len, 256)
        self.recurrent_state = torch.randn(self.batch_size, self.nvh, self.hk, self.hv)
        self.beta = torch.randn(self.batch_size, self.seq_len, self.nvh)
        self.g = torch.randn(self.batch_size, self.seq_len, self.nvh)
        self.cache_indices = torch.tensor([0, 1], dtype=torch.int32)

    @patch("torch.ops.npu.recurrent_gated_delta_rule", create=True)
    def test_graph_mode_true(self, mock_kernel):
        """graph_mode=True → uses mamba_cache_indices_gdn."""
        mock_kernel.return_value = torch.randn(2, 3, 8, 64)
        self.backend.graph_mode = True
        self.backend.forward_metadata.mamba_cache_indices_gdn = torch.tensor(
            [100, 101, 102, 103, 104, 105], dtype=torch.int32
        )
        self.backend.fused_recurrent_gated_delta_rule_update(
            self.mix_qkv,
            self.nh,
            self.nvh,
            self.hk,
            self.hv,
            self.recurrent_state,
            self.beta,
            self.g,
            self.cache_indices,
        )
        _, kwargs = mock_kernel.call_args
        self.assertTrue(
            torch.equal(
                kwargs["ssm_state_indices"],
                torch.tensor([[100, 101, 102], [103, 104, 105]], dtype=torch.int32),
            )
        )

    @patch("torch.ops.npu.recurrent_gated_delta_rule", create=True)
    def test_graph_mode_false(self, mock_kernel):
        """graph_mode=False → uses self.ssm_state_indices."""
        mock_kernel.return_value = torch.randn(2, 3, 8, 64)
        self.backend.graph_mode = False
        self.backend.ssm_state_indices = torch.tensor(
            [5, 6, 7, 8, 9, 10], dtype=torch.int32
        )
        self.backend.fused_recurrent_gated_delta_rule_update(
            self.mix_qkv,
            self.nh,
            self.nvh,
            self.hk,
            self.hv,
            self.recurrent_state,
            self.beta,
            self.g,
            self.cache_indices,
        )
        _, kwargs = mock_kernel.call_args
        self.assertTrue(
            torch.equal(
                kwargs["ssm_state_indices"],
                torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.int32),
            )
        )

    @patch("torch.ops.npu.recurrent_gated_delta_rule", create=True)
    def test_num_accept_tokens_graph_mode(self, mock_kernel):
        """graph_mode=True → num_accept_tokens = ones(batch_size)."""
        mock_kernel.return_value = torch.randn(2, 3, 8, 64)
        self.backend.graph_mode = True
        self.backend.forward_metadata.mamba_cache_indices_gdn = torch.arange(
            self.batch_size * self.seq_len, dtype=torch.int32
        )
        self.backend.fused_recurrent_gated_delta_rule_update(
            self.mix_qkv,
            self.nh,
            self.nvh,
            self.hk,
            self.hv,
            self.recurrent_state,
            self.beta,
            self.g,
            self.cache_indices,
        )
        _, kwargs = mock_kernel.call_args
        self.assertTrue((kwargs["num_accepted_tokens"] == 1).all().item())

    @patch("torch.ops.npu.recurrent_gated_delta_rule", create=True)
    def test_num_accept_tokens_no_graph(self, mock_kernel):
        """graph_mode=False → num_accept_tokens = self.num_accept_tokens."""
        mock_kernel.return_value = torch.randn(2, 3, 8, 64)
        self.backend.graph_mode = False
        self.backend.num_accept_tokens = torch.tensor([7, 7], dtype=torch.int32)
        self.backend.fused_recurrent_gated_delta_rule_update(
            self.mix_qkv,
            self.nh,
            self.nvh,
            self.hk,
            self.hv,
            self.recurrent_state,
            self.beta,
            self.g,
            self.cache_indices,
        )
        _, kwargs = mock_kernel.call_args
        self.assertTrue(
            torch.equal(
                kwargs["num_accepted_tokens"],
                torch.tensor([7, 7], dtype=torch.int32),
            )
        )

    @patch("torch.ops.npu.recurrent_gated_delta_rule", create=True)
    def test_scale_computation(self, mock_kernel):
        """scale = 1 / sqrt(head_k_dim)."""
        mock_kernel.return_value = torch.randn(2, 3, 8, 64)
        self.backend.fused_recurrent_gated_delta_rule_update(
            self.mix_qkv,
            self.nh,
            self.nvh,
            self.hk,
            self.hv,
            self.recurrent_state,
            self.beta,
            self.g,
            self.cache_indices,
        )
        _, kwargs = mock_kernel.call_args
        expected_scale = 1.0 / (self.hk**0.5)
        self.assertAlmostEqual(kwargs["scale"], expected_scale, places=5)

    @patch("torch.ops.npu.recurrent_gated_delta_rule", create=True)
    def test_beta_cast_to_bf16(self, mock_kernel):
        """beta is cast to bfloat16 before kernel call."""
        mock_kernel.return_value = torch.randn(2, 3, 8, 64)
        self.backend.fused_recurrent_gated_delta_rule_update(
            self.mix_qkv,
            self.nh,
            self.nvh,
            self.hk,
            self.hv,
            self.recurrent_state,
            self.beta,
            self.g,
            self.cache_indices,
        )
        _, kwargs = mock_kernel.call_args
        self.assertEqual(kwargs["beta"].dtype, torch.bfloat16)

    @patch("torch.ops.npu.recurrent_gated_delta_rule", create=True)
    def test_g_cast_to_f32(self, mock_kernel):
        """g is cast to float32 before kernel call."""
        mock_kernel.return_value = torch.randn(2, 3, 8, 64)
        self.backend.fused_recurrent_gated_delta_rule_update(
            self.mix_qkv,
            self.nh,
            self.nvh,
            self.hk,
            self.hv,
            self.recurrent_state,
            self.beta,
            self.g,
            self.cache_indices,
        )
        _, kwargs = mock_kernel.call_args
        self.assertEqual(kwargs["g"].dtype, torch.float32)

    @patch("torch.ops.npu.recurrent_gated_delta_rule", create=True)
    def test_intermediate_state_view(self, mock_kernel):
        """intermediate_state is not None → reshaped to (bs*seq, nv, hk, hv) before kernel."""
        mock_kernel.return_value = torch.randn(2, 3, 8, 64)
        inter = torch.randn(2 * 3, 8, 64, 64)
        result = self.backend.fused_recurrent_gated_delta_rule_update(
            self.mix_qkv,
            self.nh,
            self.nvh,
            self.hk,
            self.hv,
            self.recurrent_state,
            self.beta,
            self.g,
            self.cache_indices,
            intermediate_state=inter,
        )
        # intermediate_state passed to kernel should be reshaped to
        # (-1, num_value_heads, head_k_dim, head_v_dim)
        _, kwargs = mock_kernel.call_args
        self.assertEqual(
            kwargs["intermediate_state"].shape,
            (self.batch_size * self.seq_len, self.nvh, self.hk, self.hv),
        )
        self.assertIsNotNone(result)

    @patch("torch.ops.npu.recurrent_gated_delta_rule", create=True)
    def test_intermediate_state_none(self, mock_kernel):
        """intermediate_state=None → no reshape, passed as None to kernel."""
        mock_kernel.return_value = torch.randn(2, 3, 8, 64)
        self.backend.fused_recurrent_gated_delta_rule_update(
            self.mix_qkv,
            self.nh,
            self.nvh,
            self.hk,
            self.hv,
            self.recurrent_state,
            self.beta,
            self.g,
            self.cache_indices,
            # intermediate_state not passed → defaults to None
        )
        _, kwargs = mock_kernel.call_args
        self.assertIsNone(kwargs["intermediate_state"])


if __name__ == "__main__":
    unittest.main()
