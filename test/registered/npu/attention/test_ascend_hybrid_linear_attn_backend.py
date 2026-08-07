"""Unit tests for AscendMambaAttnBackendBase and AscendHybridLinearAttnBackend."""

import sys
import unittest
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Optional
from unittest.mock import MagicMock, patch

# ---- Mock modules unavailable on CPU/Windows before any sglang import ----
_mock = MagicMock()
for _mod in (
    "torch_npu",
    "triton",
    "triton.language",
    "sgl_kernel_npu",
    "sgl_kernel_npu.mamba",
    "sgl_kernel_npu.mamba.mamba_state_update_triton",
    "sglang.srt.utils.hf_transformers_patches",
    "sglang.global_config",
    "sglang.lang.api",
    "sglang.lang.backend.runtime_endpoint",
    "sglang.lang.choices",
    "sglang.utils",
    "sglang.srt.configs.model_config",
    "sglang.srt.dllm.config",
    "sglang.srt.layers.attention.base_attn_backend",
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

# --- ForwardMetadata stub (matches real dataclass shape) ---


@dataclass(kw_only=True)
class ForwardMetadata:
    query_start_loc: "object"
    mamba_cache_indices: "object"
    mamba_cache_indices_gdn: Optional[object] = None
    mamba_track_indices: Optional[object] = None
    retrieve_next_token: Optional[object] = None
    retrieve_next_sibling: Optional[object] = None
    retrieve_parent_token: Optional[object] = None


_mamba2_meta_mod = type(sys)("sglang.srt.layers.attention.mamba.mamba2_metadata")
_mamba2_meta_mod.ForwardMetadata = ForwardMetadata
sys.modules["sglang.srt.layers.attention.mamba.mamba2_metadata"] = _mamba2_meta_mod

# --- ForwardMode stub ---


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

    def is_idle(self):
        return self == _ForwardModeStub.IDLE

    def is_decode_or_idle(self):
        return self in (_ForwardModeStub.DECODE, _ForwardModeStub.IDLE)

    def is_target_verify(self):
        return self == _ForwardModeStub.TARGET_VERIFY

    def is_draft_extend_v2(self):
        return self == _ForwardModeStub.DRAFT_EXTEND_V2


_fb_info_mod = type(sys)("sglang.srt.model_executor.forward_batch_info")
_fb_info_mod.ForwardBatch = MagicMock
_fb_info_mod.ForwardMode = _ForwardModeStub
sys.modules["sglang.srt.model_executor.forward_batch_info"] = _fb_info_mod

_model_runner_mod = type(sys)("sglang.srt.model_executor.model_runner")
_model_runner_mod.ModelRunner = MagicMock
sys.modules["sglang.srt.model_executor.model_runner"] = _model_runner_mod

_eagle_mod = type(sys)("sglang.srt.speculative.eagle_info")
_eagle_mod.EagleDraftInput = MagicMock
_eagle_mod.EagleVerifyInput = MagicMock
sys.modules["sglang.srt.speculative.eagle_info"] = _eagle_mod

# --- MambaAttnBackendBase stub ---


class _MambaAttnBackendBaseStub:
    """Stub for MambaAttnBackendBase — provides required attributes."""

    def __init__(self, model_runner):
        self.pad_slot_id = -1
        self.device = "cpu"
        self.topk = 0
        self.is_draft_worker = False
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = getattr(model_runner, "token_to_kv_pool", None)
        self.forward_metadata = None
        self.state_indices_list = []
        self.mamba_track_indices_buf = None
        self.query_start_loc_list = []
        self.retrieve_next_token_list = []
        self.retrieve_next_sibling_list = []
        self.retrieve_parent_token_list = []
        self.cached_cuda_graph_decode_query_start_loc = None
        self.cached_cuda_graph_verify_query_start_loc = None
        self.conv_states_shape = None

    def get_cuda_graph_seq_len_fill_value(self):
        return 0


class _HybridLinearAttnBackendStub:
    """Stub for HybridLinearAttnBackend — accepts 3 init args."""

    def __init__(self, full_attn_backend, linear_attn_backend, full_attn_layers):
        self.full_attn_backend = full_attn_backend
        self.linear_attn_backend = linear_attn_backend
        self.full_attn_layers = full_attn_layers


_hybrid_mod = type(sys)("sglang.srt.layers.attention.hybrid_linear_attn_backend")
_hybrid_mod.MambaAttnBackendBase = _MambaAttnBackendBaseStub
_hybrid_mod.HybridLinearAttnBackend = _HybridLinearAttnBackendStub
sys.modules["sglang.srt.layers.attention.hybrid_linear_attn_backend"] = _hybrid_mod

# sglang.version
_ver = type(sys)("sglang.version")
_ver.__version__ = "0.0.0.dev0"
sys.modules["sglang.version"] = _ver

import torch

from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
    AscendHybridLinearAttnBackend,
    AscendMamba2AttnBackend,
    AscendMambaAttnBackendBase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

_MOD = "sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend"


def register_npu_ci(est_time, suite=None, nightly=False, disabled=None):
    def decorator(cls):
        return cls

    return decorator


class CustomTestCase(unittest.TestCase):
    pass


register_npu_ci(est_time=3, suite="stage-a-unit-test-npu")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _make_model_runner():
    """Create a minimal ModelRunner mock."""
    mr = MagicMock()
    mr.device = "cpu"
    mr.is_draft_worker = False
    mr.server_args = MagicMock()
    mr.server_args.speculative_eagle_topk = 0

    req_pool = MagicMock()
    req_pool.get_mamba_indices = MagicMock(
        side_effect=lambda indices: indices.to(torch.int32)
    )
    mr.req_to_token_pool = req_pool
    return mr


def _make_backend():
    """Create an AscendMambaAttnBackendBase instance."""
    mr = _make_model_runner()
    backend = AscendMambaAttnBackendBase(mr)
    return backend


def _init_graph(backend, max_bs=2, max_num_tokens=6):
    """Call init_cuda_graph_state so lists are populated."""
    backend.init_cuda_graph_state(max_bs, max_num_tokens)
    return backend


def _make_spec_info(draft_token_num=3, topk=1):
    """Create a mock spec_info."""
    spec = MagicMock()
    spec.draft_token_num = draft_token_num
    spec.topk = topk
    spec.retrive_next_token = torch.zeros(2, draft_token_num, dtype=torch.int32)
    spec.retrive_next_sibling = torch.zeros(2, draft_token_num, dtype=torch.int32)
    return spec


# ---------------------------------------------------------------------------
#  TestInit
# ---------------------------------------------------------------------------


class TestInit(CustomTestCase):
    """Tests for AscendMambaAttnBackendBase.__init__."""

    def test_state_indices_list_gdn_created(self):
        """__init__ creates state_indices_list_gdn as empty list."""
        backend = _make_backend()
        self.assertEqual(backend.state_indices_list_gdn, [])

    def test_inherits_parent_attributes(self):
        """Parent class attributes are accessible."""
        backend = _make_backend()
        self.assertEqual(backend.state_indices_list, [])
        self.assertEqual(backend.query_start_loc_list, [])
        self.assertEqual(backend.pad_slot_id, -1)
        self.assertEqual(backend.device, "cpu")

    def test_ascend_mamba2_attn_backend_inherits(self):
        """AscendMamba2AttnBackend is a subclass with no overrides."""
        mr = _make_model_runner()
        backend = AscendMamba2AttnBackend(mr)
        self.assertEqual(backend.state_indices_list_gdn, [])
        self.assertIsInstance(backend, AscendMambaAttnBackendBase)


# ---------------------------------------------------------------------------
#  init_cuda_graph_state
# ---------------------------------------------------------------------------


class TestInitCudaGraphState(CustomTestCase):
    """Tests for AscendMambaAttnBackendBase.init_cuda_graph_state."""

    def test_divisibility_assertion(self):
        """max_num_tokens not divisible by max_bs raises AssertionError."""
        backend = _make_backend()
        with self.assertRaises(AssertionError):
            backend.init_cuda_graph_state(max_bs=2, max_num_tokens=5)

    def test_state_indices_list_shapes(self):
        """state_indices_list has max_bs entries with increasing sizes."""
        backend = _init_graph(_make_backend(), max_bs=3, max_num_tokens=9)
        self.assertEqual(len(backend.state_indices_list), 3)
        # Entry i has shape (i+1,)
        for i in range(3):
            self.assertEqual(backend.state_indices_list[i].shape, (i + 1,))

    def test_state_indices_list_gdn_shapes(self):
        """state_indices_list_gdn has max_bs entries with increasing sizes * draft_token_num."""
        backend = _init_graph(_make_backend(), max_bs=3, max_num_tokens=9)
        draft_token_num = 9 // 3  # = 3
        self.assertEqual(len(backend.state_indices_list_gdn), 3)
        for i in range(3):
            self.assertEqual(
                backend.state_indices_list_gdn[i].shape, ((i + 1) * draft_token_num,)
            )

    def test_query_start_loc_list_shapes(self):
        """query_start_loc_list has max_bs entries with shape (i+2,)."""
        backend = _init_graph(_make_backend(), max_bs=3, max_num_tokens=9)
        self.assertEqual(len(backend.query_start_loc_list), 3)
        for i in range(3):
            self.assertEqual(backend.query_start_loc_list[i].shape, (i + 2,))

    def test_retrieve_lists_shapes(self):
        """retrieve_next_token/sibling/parent lists have correct shapes."""
        backend = _init_graph(_make_backend(), max_bs=2, max_num_tokens=6)
        draft_token_num = 3
        for name in (
            "retrieve_next_token_list",
            "retrieve_next_sibling_list",
            "retrieve_parent_token_list",
        ):
            lst = getattr(backend, name)
            self.assertEqual(len(lst), 2)
            for i in range(2):
                self.assertEqual(lst[i].shape, (i + 1, draft_token_num))

    def test_cached_decode_query_start_loc(self):
        """cached_cuda_graph_decode_query_start_loc = arange(0, max_bs+1)."""
        backend = _init_graph(_make_backend(), max_bs=3, max_num_tokens=9)
        expected = torch.arange(0, 4, dtype=torch.int32)
        self.assertTrue(
            torch.equal(backend.cached_cuda_graph_decode_query_start_loc, expected)
        )

    def test_cached_verify_query_start_loc(self):
        """cached_cuda_graph_verify_query_start_loc = arange(0, max_bs*draft+1, step=draft)."""
        backend = _init_graph(_make_backend(), max_bs=2, max_num_tokens=6)
        # draft_token_num = 3, so arange(0, 2*3+1, step=3) = [0, 3, 6]
        expected = torch.arange(0, 7, step=3, dtype=torch.int32)
        self.assertTrue(
            torch.equal(backend.cached_cuda_graph_verify_query_start_loc, expected)
        )

    def test_all_buffers_filled_with_pad_slot_id(self):
        """All pre-allocated buffers are filled with pad_slot_id."""
        backend = _init_graph(_make_backend(), max_bs=2, max_num_tokens=6)
        for tensor in backend.state_indices_list:
            self.assertTrue((tensor == -1).all().item())
        for tensor in backend.state_indices_list_gdn:
            self.assertTrue((tensor == -1).all().item())

    def test_single_batch(self):
        """max_bs=1 edge case."""
        backend = _init_graph(_make_backend(), max_bs=1, max_num_tokens=4)
        self.assertEqual(len(backend.state_indices_list), 1)
        self.assertEqual(backend.state_indices_list[0].shape, (1,))
        self.assertEqual(backend.state_indices_list_gdn[0].shape, (4,))


# ---------------------------------------------------------------------------
#  _capture_metadata
# ---------------------------------------------------------------------------


class TestCaptureMetadata(CustomTestCase):
    """Tests for AscendMambaAttnBackendBase._capture_metadata."""

    def setUp(self):
        self.backend = _init_graph(_make_backend(), max_bs=2, max_num_tokens=6)
        self.req_pool_indices = torch.tensor([10, 20], dtype=torch.int32)

    def test_decode_mode(self):
        """DECODE mode → query_start_loc from cached_decode, returns ForwardMetadata."""
        meta = self.backend._capture_metadata(
            bs=2,
            req_pool_indices=self.req_pool_indices,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
        )
        self.assertIsInstance(meta, ForwardMetadata)
        # mamba_cache_indices should be copied from mamba_indices
        expected = self.req_pool_indices
        self.assertTrue(torch.equal(meta.mamba_cache_indices[:2], expected))
        # query_start_loc should be [0, 1, 2]
        self.assertTrue(
            torch.equal(
                meta.query_start_loc, torch.tensor([0, 1, 2], dtype=torch.int32)
            )
        )

    def test_idle_mode(self):
        """IDLE mode → same as decode (is_decode_or_idle branch)."""
        meta = self.backend._capture_metadata(
            bs=2,
            req_pool_indices=self.req_pool_indices,
            forward_mode=ForwardMode.IDLE,
            spec_info=None,
        )
        self.assertIsInstance(meta, ForwardMetadata)
        self.assertTrue(
            torch.equal(
                meta.query_start_loc, torch.tensor([0, 1, 2], dtype=torch.int32)
            )
        )

    def test_target_verify_mode(self):
        """TARGET_VERIFY → ssm_state_indices copied to state_indices_list_gdn."""
        spec_info = _make_spec_info(draft_token_num=3, topk=1)
        meta = self.backend._capture_metadata(
            bs=2,
            req_pool_indices=self.req_pool_indices,
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=spec_info,
        )
        self.assertIsInstance(meta, ForwardMetadata)
        # mamba_cache_indices_gdn should be set
        self.assertIsNotNone(meta.mamba_cache_indices_gdn)
        # query_start_loc should be [0, 3, 6]
        self.assertTrue(
            torch.equal(
                meta.query_start_loc, torch.tensor([0, 3, 6], dtype=torch.int32)
            )
        )

    def test_target_verify_topk_gt_1(self):
        """TARGET_VERIFY with topk>1 → returns retrieve_next_token/sibling."""
        spec_info = _make_spec_info(draft_token_num=3, topk=2)
        meta = self.backend._capture_metadata(
            bs=2,
            req_pool_indices=self.req_pool_indices,
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=spec_info,
        )
        self.assertIsInstance(meta, ForwardMetadata)
        self.assertIsNotNone(meta.retrieve_next_token)
        self.assertIsNotNone(meta.retrieve_next_sibling)
        self.assertIsNotNone(meta.retrieve_parent_token)
        # mamba_cache_indices_gdn should NOT be set for topk>1
        self.assertIsNone(meta.mamba_cache_indices_gdn)

    def test_invalid_mode_raises(self):
        """EXTEND mode → raises ValueError."""
        with self.assertRaises(ValueError):
            self.backend._capture_metadata(
                bs=2,
                req_pool_indices=self.req_pool_indices,
                forward_mode=ForwardMode.EXTEND,
                spec_info=None,
            )

    def test_state_indices_copied(self):
        """mamba_indices are copied into state_indices_list[bs-1]."""
        self.backend._capture_metadata(
            bs=2,
            req_pool_indices=self.req_pool_indices,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
        )
        self.assertTrue(
            torch.equal(self.backend.state_indices_list[1][:2], self.req_pool_indices)
        )


# ---------------------------------------------------------------------------
#  _replay_metadata
# ---------------------------------------------------------------------------


class TestReplayMetadata(CustomTestCase):
    """Tests for AscendMambaAttnBackendBase._replay_metadata."""

    def setUp(self):
        self.backend = _init_graph(_make_backend(), max_bs=2, max_num_tokens=6)
        self.req_pool_indices = torch.tensor([10, 20], dtype=torch.int32)

    def test_seq_lens_none_sets_num_padding_zero(self):
        """seq_lens_cpu=None → num_padding=0."""
        meta = self.backend._replay_metadata(
            bs=2,
            req_pool_indices=self.req_pool_indices.clone(),
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            seq_lens_cpu=None,
        )
        self.assertIsInstance(meta, ForwardMetadata)
        self.assertTrue(
            torch.equal(
                meta.query_start_loc, torch.tensor([0, 1, 2], dtype=torch.int32)
            )
        )

    def test_decode_no_padding(self):
        """DECODE with num_padding=0 → query_start_loc = [0,1,2]."""
        seq_lens = torch.tensor([5, 10], dtype=torch.int32)
        meta = self.backend._replay_metadata(
            bs=2,
            req_pool_indices=self.req_pool_indices.clone(),
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            seq_lens_cpu=seq_lens,
        )
        self.assertTrue(
            torch.equal(
                meta.query_start_loc, torch.tensor([0, 1, 2], dtype=torch.int32)
            )
        )

    def test_decode_with_padding(self):
        """DECODE with padding → query_start_loc padded region filled with bs-num_padding."""
        # seq_lens contains a 0 (fill value) → num_padding=1
        seq_lens = torch.tensor([5, 0], dtype=torch.int32)
        meta = self.backend._replay_metadata(
            bs=2,
            req_pool_indices=self.req_pool_indices.clone(),
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            seq_lens_cpu=seq_lens,
        )
        # bs=2, num_padding=1, bs-num_padding=1
        # query_start_loc[:1] = [0], query_start_loc[1:] = [1, 1]
        self.assertEqual(meta.query_start_loc[0].item(), 0)
        self.assertTrue((meta.query_start_loc[1:] == 1).all().item())

    def test_target_verify_no_padding(self):
        """TARGET_VERIFY with no padding → query_start_loc = [0,3,6]."""
        seq_lens = torch.tensor([5, 10], dtype=torch.int32)
        spec_info = _make_spec_info(draft_token_num=3, topk=1)
        meta = self.backend._replay_metadata(
            bs=2,
            req_pool_indices=self.req_pool_indices.clone(),
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=spec_info,
            seq_lens_cpu=seq_lens,
        )
        self.assertTrue(
            torch.equal(
                meta.query_start_loc, torch.tensor([0, 3, 6], dtype=torch.int32)
            )
        )
        self.assertIsNotNone(meta.mamba_cache_indices_gdn)

    def test_target_verify_with_padding(self):
        """TARGET_VERIFY with padding → padded query_start_loc filled."""
        seq_lens = torch.tensor([5, 0], dtype=torch.int32)  # num_padding=1
        spec_info = _make_spec_info(draft_token_num=3, topk=1)
        meta = self.backend._replay_metadata(
            bs=2,
            req_pool_indices=self.req_pool_indices.clone(),
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=spec_info,
            seq_lens_cpu=seq_lens,
        )
        # bs=2, num_padding=1, bs-num_padding=1
        # query_start_loc[:1] = [0], query_start_loc[1:] = [1*3, 1*3] = [3, 3]
        self.assertEqual(meta.query_start_loc[0].item(), 0)
        self.assertTrue((meta.query_start_loc[1:] == 3).all().item())

    def test_target_verify_topk_gt_1(self):
        """TARGET_VERIFY with topk>1 → returns retrieve_next_token/sibling."""
        seq_lens = torch.tensor([5, 10], dtype=torch.int32)
        spec_info = _make_spec_info(draft_token_num=3, topk=2)
        meta = self.backend._replay_metadata(
            bs=2,
            req_pool_indices=self.req_pool_indices.clone(),
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=spec_info,
            seq_lens_cpu=seq_lens,
        )
        self.assertIsNotNone(meta.retrieve_next_token)
        self.assertIsNotNone(meta.retrieve_next_sibling)
        self.assertIsNone(meta.mamba_cache_indices_gdn)

    def test_mamba_track_indices_passed_through(self):
        """mamba_track_indices is passed through as mamba_track_indices in ForwardMetadata."""
        track_indices = torch.tensor([0, 1], dtype=torch.int32)
        meta = self.backend._replay_metadata(
            bs=2,
            req_pool_indices=self.req_pool_indices.clone(),
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            seq_lens_cpu=None,
            mamba_track_indices=track_indices,
        )
        self.assertTrue(torch.equal(meta.mamba_track_indices, track_indices))

    def test_mamba_track_indices_none(self):
        """mamba_track_indices=None → mamba_track_indices in ForwardMetadata is None."""
        meta = self.backend._replay_metadata(
            bs=2,
            req_pool_indices=self.req_pool_indices.clone(),
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            seq_lens_cpu=None,
            mamba_track_indices=None,
        )
        self.assertIsNone(meta.mamba_track_indices)

    def test_invalid_mode_raises(self):
        """EXTEND mode → raises ValueError."""
        with self.assertRaises(ValueError):
            self.backend._replay_metadata(
                bs=2,
                req_pool_indices=self.req_pool_indices.clone(),
                forward_mode=ForwardMode.EXTEND,
                spec_info=None,
                seq_lens_cpu=None,
            )

    def test_padding_zeros_req_pool_indices(self):
        """Padding requests have req_pool_indices zeroed."""
        seq_lens = torch.tensor([5, 0], dtype=torch.int32)  # num_padding=1
        req_pool = self.req_pool_indices.clone()
        self.backend._replay_metadata(
            bs=2,
            req_pool_indices=req_pool,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            seq_lens_cpu=seq_lens,
        )
        # req_pool[1] should be zeroed (bs - num_padding = 1)
        self.assertEqual(req_pool[1].item(), 0)


# ---------------------------------------------------------------------------
#  get_cuda_graph_seq_len_fill_value
# ---------------------------------------------------------------------------


class TestGetCudaGraphSeqLenFillValue(CustomTestCase):
    """Tests for AscendMambaAttnBackendBase.get_cuda_graph_seq_len_fill_value."""

    def test_returns_zero(self):
        """Fill value is 0 for mamba attention."""
        backend = _make_backend()
        self.assertEqual(backend.get_cuda_graph_seq_len_fill_value(), 0)


# ---------------------------------------------------------------------------
#  AscendHybridLinearAttnBackend
# ---------------------------------------------------------------------------


class TestHybridLinearAttnBackend(CustomTestCase):
    """Tests for AscendHybridLinearAttnBackend."""

    def test_init(self):
        """__init__ calls super with correct args."""
        full_backend = MagicMock()
        linear_backend = _make_backend()
        hybrid = AscendHybridLinearAttnBackend(
            full_backend, linear_backend, full_attn_layers=[0, 1]
        )
        self.assertEqual(hybrid.full_attn_layers, [0, 1])

    def test_update_verify_buffers_does_nothing(self):
        """update_verify_buffers_to_fill_after_draft is a no-op (pass)."""
        full_backend = MagicMock()
        linear_backend = _make_backend()
        hybrid = AscendHybridLinearAttnBackend(
            full_backend, linear_backend, full_attn_layers=[]
        )
        # Should not raise — it's just pass
        hybrid.update_verify_buffers_to_fill_after_draft(
            spec_info=MagicMock(), cuda_graph_bs=None
        )


# ---------------------------------------------------------------------------
#  update_mamba_state_after_mtp_verify
# ---------------------------------------------------------------------------


class TestUpdateMambaStateAfterMtpVerify(CustomTestCase):
    """Tests for AscendHybridLinearAttnBackend.update_mamba_state_after_mtp_verify."""

    def setUp(self):
        full_backend = MagicMock()
        self.linear_backend = _make_backend()
        self.hybrid = AscendHybridLinearAttnBackend(
            full_backend, self.linear_backend, full_attn_layers=[]
        )

        # Set up forward_metadata with mamba_cache_indices
        self.linear_backend.forward_metadata = MagicMock()
        self.linear_backend.forward_metadata.mamba_cache_indices = torch.tensor(
            [0, 1], dtype=torch.int32
        )

        # Set up mamba_caches mock
        mamba_caches = MagicMock()
        mamba_caches.conv = [torch.randn(4, 8, 16)]
        mamba_caches.temporal = torch.randn(4, 2, 8, 8)
        mamba_caches.intermediate_ssm = torch.randn(4, 2, 3, 8, 8)
        self.linear_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers = MagicMock(
            return_value=mamba_caches
        )
        self.mamba_caches = mamba_caches

    @patch(_MOD + ".move_intermediate_cache")
    @patch(_MOD + ".conv_state_rollback")
    def test_no_track_indices(self, mock_rollback, mock_move):
        """mamba_track_indices=None → only base move_intermediate_cache and conv_state_rollback."""
        last_correct = torch.tensor([2, 1])
        self.hybrid.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct,
            mamba_track_indices=None,
            mamba_steps_to_track=None,
            model=MagicMock(),
        )
        # move_intermediate_cache called once (base path)
        self.assertEqual(mock_move.call_count, 1)
        # conv_state_rollback called once (for dst_indices)
        self.assertEqual(mock_rollback.call_count, 1)

    @patch(_MOD + ".move_intermediate_cache")
    @patch(_MOD + ".conv_state_rollback")
    def test_with_track_indices(self, mock_rollback, mock_move):
        """mamba_track_indices not None → extra move_intermediate_cache and conv_state_rollback."""
        last_correct = torch.tensor([2, 1])
        track_indices = torch.tensor([0, 1])
        steps_to_track = torch.tensor([1, 0])
        self.hybrid.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct,
            mamba_track_indices=track_indices,
            mamba_steps_to_track=steps_to_track,
            model=MagicMock(),
        )
        # move_intermediate_cache called twice (base + track)
        self.assertEqual(mock_move.call_count, 2)
        # conv_state_rollback called twice (dst + track)
        self.assertEqual(mock_rollback.call_count, 2)

    @patch(
        "sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend.move_intermediate_cache"
    )
    @patch(
        "sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend.conv_state_rollback"
    )
    def test_track_indices_cast_to_int64(self, mock_rollback, mock_move):
        """mamba_track_indices and mamba_steps_to_track are cast to int64."""
        last_correct = torch.tensor([2, 1], dtype=torch.int32)
        track_indices = torch.tensor([0, 1], dtype=torch.int32)
        steps_to_track = torch.tensor([1, 0], dtype=torch.int32)
        self.hybrid.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct,
            mamba_track_indices=track_indices,
            mamba_steps_to_track=steps_to_track,
            model=MagicMock(),
        )
        # Second call to move_intermediate_cache should use int64 track_indices
        _, args, kwargs = mock_move.mock_calls[1]
        # move_intermediate_cache is called positionally: (ssm_states, intermediate, dst, src, last_steps)
        dst_arg = args[2] if len(args) > 2 else kwargs.get("dst_indices_tensor")
        self.assertEqual(dst_arg.dtype, torch.int64)

    @patch(_MOD + ".move_intermediate_cache")
    @patch(_MOD + ".conv_state_rollback")
    def test_track_with_negative_steps(self, mock_rollback, mock_move):
        """mamba_steps_to_track with negative values → track_mask filters them out."""
        last_correct = torch.tensor([2, 1])
        track_indices = torch.tensor([0, 1])
        steps_to_track = torch.tensor([-1, 0])  # first is negative
        self.hybrid.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct,
            mamba_track_indices=track_indices,
            mamba_steps_to_track=steps_to_track,
            model=MagicMock(),
        )
        # Should not crash, conv_states copy only happens for non-negative steps
        self.assertEqual(mock_rollback.call_count, 2)

    @patch(_MOD + ".move_intermediate_cache")
    @patch(_MOD + ".conv_state_rollback")
    def test_empty_request_number(self, mock_rollback, mock_move):
        """Empty last_correct_step_indices → no conv_state_rollback for dst."""
        last_correct = torch.tensor([], dtype=torch.int64)
        self.hybrid.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct,
            mamba_track_indices=None,
            mamba_steps_to_track=None,
            model=MagicMock(),
        )
        # move_intermediate_cache still called once (base path)
        self.assertEqual(mock_move.call_count, 1)
        # dst_indices empty → conv_state_rollback skipped
        self.assertEqual(mock_rollback.call_count, 0)

    @patch(_MOD + ".move_intermediate_cache")
    @patch(_MOD + ".conv_state_rollback")
    def test_all_negative_steps(self, mock_rollback, mock_move):
        """All mamba_steps_to_track negative → track_indices empty, no conv_states copy."""
        last_correct = torch.tensor([2, 1])
        track_indices = torch.tensor([0, 1])
        steps_to_track = torch.tensor([-1, -1])  # all negative
        self.hybrid.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct,
            mamba_track_indices=track_indices,
            mamba_steps_to_track=steps_to_track,
            model=MagicMock(),
        )
        # move_intermediate_cache called twice (base + track)
        self.assertEqual(mock_move.call_count, 2)
        # conv_state_rollback called twice (dst + track)
        self.assertEqual(mock_rollback.call_count, 2)

    @patch(_MOD + ".move_intermediate_cache")
    @patch(_MOD + ".conv_state_rollback")
    def test_empty_track_indices(self, mock_rollback, mock_move):
        """mamba_track_indices is empty tensor (not None) → track conv_state_rollback skipped."""
        last_correct = torch.tensor([2, 1])
        track_indices = torch.tensor([], dtype=torch.int64)
        steps_to_track = torch.tensor([], dtype=torch.int64)
        self.hybrid.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct,
            mamba_track_indices=track_indices,
            mamba_steps_to_track=steps_to_track,
            model=MagicMock(),
        )
        # move_intermediate_cache called twice (base + track, track is empty but not None)
        self.assertEqual(mock_move.call_count, 2)
        # conv_state_rollback called once (dst only; track skipped because numel()==0)
        self.assertEqual(mock_rollback.call_count, 1)


if __name__ == "__main__":
    unittest.main()
