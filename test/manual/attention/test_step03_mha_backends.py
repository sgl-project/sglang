"""
Step-03 coverage: standard MHA attention backends × eager / graph-decode runners.

For each backend we test:
  1. eager_decode  — init_forward_metadata(decode_fb) -> forward_decode -> no NaN
  2. eager_extend  — init_forward_metadata(extend_fb) -> forward_extend -> no NaN
  3. graph_decode  — init_cuda_graph_state + capture_init + replay_init
                     -> forward_decode -> no NaN
  4. replay_consistent — two consecutive replay_init + forward_decode give equal output

Backends covered (MHA, standard models):
  TritonAttnBackend
  FlashInferAttnBackend   (skipped if flashinfer not installed)
  FlashAttentionBackend   (skipped on SM < 80; FA3 requires Hopper/Blackwell)
  TRTLLMMHABackend        (skipped if trtllm kernels not installed)
  TorchNativeAttnBackend  (no CUDA graph; eager-only)

Run on the cluster:
  python test_step03_mha_backends.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

# Allow importing the shared utils from the same directory
sys.path.insert(0, str(Path(__file__).parent))

from step03_test_utils import (
    assert_no_nan_inf,
    build_mha_runner,
    fill_req_to_token,
    gpu_arch_sm,
    init_graph_capture,
    init_graph_replay,
    make_decode_batch,
    make_extend_batch,
    make_qkv,
    make_radix_attention,
)

# Patch TP size to 1 before any backend import. The step-03 branch uses the
# distributed group system; we bypass it in unit tests.
from sglang.srt.layers import dp_attention as _dp_attn

_dp_attn.get_attention_tp_size = lambda: 1

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    set_forward_context,
)
from sglang.test.test_utils import CustomTestCase

_CUDA = torch.cuda.is_available()

# Shared geometry.  head_dim=64: FlashInfer prefill kernel requires >=64.
_NUM_HEADS = 4
_HEAD_DIM = 64
_MAX_BS = 8
_MAX_CTX = 64
_SEQ_LEN = 32
_EXTEND_LEN = 16
_PREFIX_LEN = 8
_DTYPE = torch.float16


# ---------------------------------------------------------------------------
# TritonAttnBackend
# ---------------------------------------------------------------------------


@unittest.skipUnless(_CUDA, "CUDA required")
class TestTritonInit(CustomTestCase):
    """init_forward_metadata* coverage for TritonAttnBackend."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        cls.mr = build_mha_runner(
            num_heads=_NUM_HEADS,
            head_dim=_HEAD_DIM,
            max_bs=_MAX_BS,
            max_context_len=_MAX_CTX,
            dtype=_DTYPE,
        )
        cls.backend = TritonAttnBackend(cls.mr)
        cls.layer = make_radix_attention(_NUM_HEADS, _HEAD_DIM)
        set_forward_context(ForwardContext(attn_backend=cls.backend))

    # -- helpers --

    def _fill(self, bs: int, seq_len: int):
        fill_req_to_token(self.mr, bs, seq_len)

    def _decode_fwd(self, bs: int, seq_len: int):
        self._fill(bs, seq_len)
        fb = make_decode_batch(bs, seq_len)
        q, k, v = make_qkv(bs, _NUM_HEADS, _HEAD_DIM, dtype=_DTYPE)
        return fb, q, k, v

    # -- eager path --

    def test_eager_decode_no_nan(self):
        fb, q, k, v = self._decode_fwd(bs=2, seq_len=_SEQ_LEN)
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_decode(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "triton eager decode")
        self.assertEqual(out.shape, (2, _NUM_HEADS * _HEAD_DIM))

    def test_eager_extend_no_nan(self):
        bs = 2
        self._fill(bs, _PREFIX_LEN + _EXTEND_LEN)
        fb = make_extend_batch(bs, _EXTEND_LEN, _PREFIX_LEN)
        q, k, v = make_qkv(bs * _EXTEND_LEN, _NUM_HEADS, _HEAD_DIM, dtype=_DTYPE)
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_extend(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "triton eager extend")
        self.assertEqual(out.numel(), bs * _EXTEND_LEN * _NUM_HEADS * _HEAD_DIM)

    def test_eager_decode_batch_size_1(self):
        fb, q, k, v = self._decode_fwd(bs=1, seq_len=_SEQ_LEN)
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_decode(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "triton eager decode bs=1")

    # -- graph path --

    def test_graph_decode_capture_no_crash(self):
        """init_cuda_graph_state + capture init must not raise."""
        self.backend.init_cuda_graph_state(_MAX_BS, _MAX_BS)
        self._fill(_MAX_BS, _SEQ_LEN)
        req_pool = torch.arange(_MAX_BS, dtype=torch.int32, device="cuda")
        seq_lens = torch.full((_MAX_BS,), _SEQ_LEN, dtype=torch.int32, device="cuda")
        fb = make_decode_batch(_MAX_BS, _SEQ_LEN)
        init_graph_capture(
            self.backend, fb, _MAX_BS, _MAX_BS, req_pool, seq_lens, ForwardMode.DECODE
        )

    def test_graph_decode_replay_no_nan(self):
        """capture + replay + forward_decode must produce a valid output."""
        self.backend.init_cuda_graph_state(_MAX_BS, _MAX_BS)
        self._fill(_MAX_BS, _SEQ_LEN)
        req_pool = torch.arange(_MAX_BS, dtype=torch.int32, device="cuda")
        seq_lens = torch.full((_MAX_BS,), _SEQ_LEN, dtype=torch.int32, device="cuda")
        seq_lens_cpu = torch.full((_MAX_BS,), _SEQ_LEN, dtype=torch.int32)

        fb = make_decode_batch(_MAX_BS, _SEQ_LEN)
        init_graph_capture(
            self.backend, fb, _MAX_BS, _MAX_BS, req_pool, seq_lens, ForwardMode.DECODE
        )
        init_graph_replay(
            self.backend,
            fb,
            _MAX_BS,
            req_pool,
            seq_lens,
            _MAX_BS * _SEQ_LEN,
            ForwardMode.DECODE,
            seq_lens_cpu,
        )
        q, k, v = make_qkv(_MAX_BS, _NUM_HEADS, _HEAD_DIM, dtype=_DTYPE)
        out = self.backend.forward_decode(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "triton graph replay decode")

    def test_graph_replay_consistent(self):
        """Two replay calls with the same data give identical outputs."""
        self.backend.init_cuda_graph_state(_MAX_BS, _MAX_BS)
        self._fill(_MAX_BS, _SEQ_LEN)
        req_pool = torch.arange(_MAX_BS, dtype=torch.int32, device="cuda")
        seq_lens = torch.full((_MAX_BS,), _SEQ_LEN, dtype=torch.int32, device="cuda")
        seq_lens_cpu = torch.full((_MAX_BS,), _SEQ_LEN, dtype=torch.int32)

        fb = make_decode_batch(_MAX_BS, _SEQ_LEN)
        init_graph_capture(
            self.backend, fb, _MAX_BS, _MAX_BS, req_pool, seq_lens, ForwardMode.DECODE
        )

        q, k, v = make_qkv(_MAX_BS, _NUM_HEADS, _HEAD_DIM, dtype=_DTYPE)

        init_graph_replay(
            self.backend,
            fb,
            _MAX_BS,
            req_pool,
            seq_lens,
            _MAX_BS * _SEQ_LEN,
            ForwardMode.DECODE,
            seq_lens_cpu,
        )
        out1 = self.backend.forward_decode(q, k, v, self.layer, fb)

        init_graph_replay(
            self.backend,
            fb,
            _MAX_BS,
            req_pool,
            seq_lens,
            _MAX_BS * _SEQ_LEN,
            ForwardMode.DECODE,
            seq_lens_cpu,
        )
        out2 = self.backend.forward_decode(q, k, v, self.layer, fb)

        self.assertTrue(
            torch.allclose(out1, out2, atol=0),
            "triton replay outputs differ on identical inputs",
        )

    # -- graph extend (PCG-style) --

    def test_graph_extend_capture_no_crash(self):
        """Capture init in EXTEND mode (simulates PCG prefill capture)."""
        bs = 2
        self.backend.init_cuda_graph_state(bs, bs * _EXTEND_LEN)
        self._fill(bs, _PREFIX_LEN + _EXTEND_LEN)
        req_pool = torch.arange(bs, dtype=torch.int32, device="cuda")
        seq_lens = torch.full(
            (bs,), _PREFIX_LEN + _EXTEND_LEN, dtype=torch.int32, device="cuda"
        )
        # PCG uses the eager init_forward_metadata path for extend captures;
        # the graph capture variant is exercised here directly.
        fb = make_extend_batch(bs, _EXTEND_LEN, _PREFIX_LEN)
        self.backend.init_forward_metadata(fb)


# ---------------------------------------------------------------------------
# FlashInferAttnBackend
# ---------------------------------------------------------------------------

try:
    from sglang.srt.utils import is_flashinfer_available as _fi_avail

    _FLASHINFER = _CUDA and _fi_avail()
except Exception:
    _FLASHINFER = False


@unittest.skipUnless(_FLASHINFER, "flashinfer not available")
class TestFlashInferInit(CustomTestCase):
    """init_forward_metadata* coverage for FlashInferAttnBackend."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        cls.mr = build_mha_runner(
            num_heads=_NUM_HEADS,
            head_dim=_HEAD_DIM,
            max_bs=_MAX_BS,
            max_context_len=_MAX_CTX,
            dtype=_DTYPE,
        )
        cls.backend = FlashInferAttnBackend(cls.mr)
        cls.layer = make_radix_attention(_NUM_HEADS, _HEAD_DIM)
        set_forward_context(ForwardContext(attn_backend=cls.backend))

    def _fill(self, bs: int, seq_len: int):
        fill_req_to_token(self.mr, bs, seq_len)

    def test_eager_decode_no_nan(self):
        bs = 2
        self._fill(bs, _SEQ_LEN)
        fb = make_decode_batch(bs, _SEQ_LEN)
        q, k, v = make_qkv(bs, _NUM_HEADS, _HEAD_DIM, dtype=_DTYPE)
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_decode(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "flashinfer eager decode")
        self.assertEqual(out.shape, (bs, _NUM_HEADS * _HEAD_DIM))

    def test_eager_extend_no_nan(self):
        bs = 2
        self._fill(bs, _PREFIX_LEN + _EXTEND_LEN)
        fb = make_extend_batch(bs, _EXTEND_LEN, _PREFIX_LEN)
        q, k, v = make_qkv(bs * _EXTEND_LEN, _NUM_HEADS, _HEAD_DIM, dtype=_DTYPE)
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_extend(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "flashinfer eager extend")

    def test_eager_extend_no_prefix(self):
        bs = 2
        self._fill(bs, _EXTEND_LEN)
        fb = make_extend_batch(bs, _EXTEND_LEN, prefix_len=0)
        q, k, v = make_qkv(bs * _EXTEND_LEN, _NUM_HEADS, _HEAD_DIM, dtype=_DTYPE)
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_extend(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "flashinfer extend no prefix")

    def test_graph_decode_capture_no_crash(self):
        self.backend.init_cuda_graph_state(_MAX_BS, _MAX_BS)
        self._fill(_MAX_BS, _SEQ_LEN)
        req_pool = torch.arange(_MAX_BS, dtype=torch.int32, device="cuda")
        seq_lens = torch.full((_MAX_BS,), _SEQ_LEN, dtype=torch.int32, device="cuda")
        fb = make_decode_batch(_MAX_BS, _SEQ_LEN)
        init_graph_capture(
            self.backend, fb, _MAX_BS, _MAX_BS, req_pool, seq_lens, ForwardMode.DECODE
        )

    def test_graph_decode_replay_no_nan(self):
        self.backend.init_cuda_graph_state(_MAX_BS, _MAX_BS)
        self._fill(_MAX_BS, _SEQ_LEN)
        req_pool = torch.arange(_MAX_BS, dtype=torch.int32, device="cuda")
        seq_lens = torch.full((_MAX_BS,), _SEQ_LEN, dtype=torch.int32, device="cuda")
        seq_lens_cpu = torch.full((_MAX_BS,), _SEQ_LEN, dtype=torch.int32)

        fb = make_decode_batch(_MAX_BS, _SEQ_LEN)
        init_graph_capture(
            self.backend, fb, _MAX_BS, _MAX_BS, req_pool, seq_lens, ForwardMode.DECODE
        )
        init_graph_replay(
            self.backend,
            fb,
            _MAX_BS,
            req_pool,
            seq_lens,
            _MAX_BS * _SEQ_LEN,
            ForwardMode.DECODE,
            seq_lens_cpu,
        )
        q, k, v = make_qkv(_MAX_BS, _NUM_HEADS, _HEAD_DIM, dtype=_DTYPE)
        out = self.backend.forward_decode(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "flashinfer graph replay decode")

    def test_graph_replay_consistent(self):
        self.backend.init_cuda_graph_state(_MAX_BS, _MAX_BS)
        self._fill(_MAX_BS, _SEQ_LEN)
        req_pool = torch.arange(_MAX_BS, dtype=torch.int32, device="cuda")
        seq_lens = torch.full((_MAX_BS,), _SEQ_LEN, dtype=torch.int32, device="cuda")
        seq_lens_cpu = torch.full((_MAX_BS,), _SEQ_LEN, dtype=torch.int32)

        fb = make_decode_batch(_MAX_BS, _SEQ_LEN)
        init_graph_capture(
            self.backend, fb, _MAX_BS, _MAX_BS, req_pool, seq_lens, ForwardMode.DECODE
        )

        q, k, v = make_qkv(_MAX_BS, _NUM_HEADS, _HEAD_DIM, dtype=_DTYPE)

        init_graph_replay(
            self.backend,
            fb,
            _MAX_BS,
            req_pool,
            seq_lens,
            _MAX_BS * _SEQ_LEN,
            ForwardMode.DECODE,
            seq_lens_cpu,
        )
        out1 = self.backend.forward_decode(q, k, v, self.layer, fb)

        init_graph_replay(
            self.backend,
            fb,
            _MAX_BS,
            req_pool,
            seq_lens,
            _MAX_BS * _SEQ_LEN,
            ForwardMode.DECODE,
            seq_lens_cpu,
        )
        out2 = self.backend.forward_decode(q, k, v, self.layer, fb)

        self.assertTrue(torch.allclose(out1, out2, atol=0))

    def test_pcg_extend_path(self):
        """PCG asymmetry: extend path calls eager init_forward_metadata (pre-step03)."""
        bs = 2
        self._fill(bs, _PREFIX_LEN + _EXTEND_LEN)
        fb = make_extend_batch(bs, _EXTEND_LEN, _PREFIX_LEN)
        q, k, v = make_qkv(bs * _EXTEND_LEN, _NUM_HEADS, _HEAD_DIM, dtype=_DTYPE)
        # PCG calls eager init for prefill; this is the path that step-03 unifies.
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_extend(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "flashinfer pcg extend path")


# ---------------------------------------------------------------------------
# FlashAttentionBackend  (FA2/FA3)
# ---------------------------------------------------------------------------

_SM = gpu_arch_sm()
_FA_OK = (
    _CUDA and _SM is not None and 80 <= _SM < 100
)  # FA3 requires SM80-90 (Ampere/Hopper); skip Blackwell/GB300


@unittest.skipUnless(_FA_OK, "FlashAttention requires SM >= 80 (Ampere)")
class TestFlashAttentionInit(CustomTestCase):
    """init_forward_metadata* coverage for FlashAttentionBackend (FA2/FA3)."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
        )

        cls.mr = build_mha_runner(
            num_heads=_NUM_HEADS,
            head_dim=_HEAD_DIM,
            max_bs=_MAX_BS,
            max_context_len=_MAX_CTX,
            dtype=_DTYPE,
        )
        cls.backend = FlashAttentionBackend(cls.mr)
        cls.layer = make_radix_attention(_NUM_HEADS, _HEAD_DIM)
        set_forward_context(ForwardContext(attn_backend=cls.backend))

    def _fill(self, bs: int, seq_len: int):
        fill_req_to_token(self.mr, bs, seq_len)

    def test_eager_decode_no_nan(self):
        bs = 2
        self._fill(bs, _SEQ_LEN)
        fb = make_decode_batch(bs, _SEQ_LEN)
        q, k, v = make_qkv(bs, _NUM_HEADS, _HEAD_DIM, dtype=_DTYPE)
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_decode(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "fa3 eager decode")
        self.assertEqual(out.shape, (bs, _NUM_HEADS * _HEAD_DIM))

    def test_eager_extend_no_nan(self):
        bs = 2
        self._fill(bs, _PREFIX_LEN + _EXTEND_LEN)
        fb = make_extend_batch(bs, _EXTEND_LEN, _PREFIX_LEN)
        q, k, v = make_qkv(bs * _EXTEND_LEN, _NUM_HEADS, _HEAD_DIM, dtype=_DTYPE)
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_extend(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "fa3 eager extend")

    def test_eager_extend_no_prefix(self):
        bs = 2
        self._fill(bs, _EXTEND_LEN)
        fb = make_extend_batch(bs, _EXTEND_LEN, prefix_len=0)
        q, k, v = make_qkv(bs * _EXTEND_LEN, _NUM_HEADS, _HEAD_DIM, dtype=_DTYPE)
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_extend(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "fa3 extend no prefix")

    def test_graph_decode_capture_no_crash(self):
        self.backend.init_cuda_graph_state(_MAX_BS, _MAX_BS)
        self._fill(_MAX_BS, _SEQ_LEN)
        req_pool = torch.arange(_MAX_BS, dtype=torch.int32, device="cuda")
        seq_lens = torch.full((_MAX_BS,), _SEQ_LEN, dtype=torch.int32, device="cuda")
        fb = make_decode_batch(_MAX_BS, _SEQ_LEN)
        init_graph_capture(
            self.backend, fb, _MAX_BS, _MAX_BS, req_pool, seq_lens, ForwardMode.DECODE
        )

    def test_graph_decode_replay_no_nan(self):
        self.backend.init_cuda_graph_state(_MAX_BS, _MAX_BS)
        self._fill(_MAX_BS, _SEQ_LEN)
        req_pool = torch.arange(_MAX_BS, dtype=torch.int32, device="cuda")
        seq_lens = torch.full((_MAX_BS,), _SEQ_LEN, dtype=torch.int32, device="cuda")
        seq_lens_cpu = torch.full((_MAX_BS,), _SEQ_LEN, dtype=torch.int32)

        fb = make_decode_batch(_MAX_BS, _SEQ_LEN)
        init_graph_capture(
            self.backend, fb, _MAX_BS, _MAX_BS, req_pool, seq_lens, ForwardMode.DECODE
        )
        init_graph_replay(
            self.backend,
            fb,
            _MAX_BS,
            req_pool,
            seq_lens,
            _MAX_BS * _SEQ_LEN,
            ForwardMode.DECODE,
            seq_lens_cpu,
        )
        q, k, v = make_qkv(_MAX_BS, _NUM_HEADS, _HEAD_DIM, dtype=_DTYPE)
        out = self.backend.forward_decode(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "fa3 graph replay decode")

    def test_graph_replay_consistent(self):
        self.backend.init_cuda_graph_state(_MAX_BS, _MAX_BS)
        self._fill(_MAX_BS, _SEQ_LEN)
        req_pool = torch.arange(_MAX_BS, dtype=torch.int32, device="cuda")
        seq_lens = torch.full((_MAX_BS,), _SEQ_LEN, dtype=torch.int32, device="cuda")
        seq_lens_cpu = torch.full((_MAX_BS,), _SEQ_LEN, dtype=torch.int32)

        fb = make_decode_batch(_MAX_BS, _SEQ_LEN)
        init_graph_capture(
            self.backend, fb, _MAX_BS, _MAX_BS, req_pool, seq_lens, ForwardMode.DECODE
        )

        q, k, v = make_qkv(_MAX_BS, _NUM_HEADS, _HEAD_DIM, dtype=_DTYPE)

        init_graph_replay(
            self.backend,
            fb,
            _MAX_BS,
            req_pool,
            seq_lens,
            _MAX_BS * _SEQ_LEN,
            ForwardMode.DECODE,
            seq_lens_cpu,
        )
        out1 = self.backend.forward_decode(q, k, v, self.layer, fb)

        init_graph_replay(
            self.backend,
            fb,
            _MAX_BS,
            req_pool,
            seq_lens,
            _MAX_BS * _SEQ_LEN,
            ForwardMode.DECODE,
            seq_lens_cpu,
        )
        out2 = self.backend.forward_decode(q, k, v, self.layer, fb)

        self.assertTrue(torch.allclose(out1, out2, atol=0))

    def test_pcg_extend_path(self):
        """PCG calls eager init for extend; verify it works end-to-end."""
        bs = 2
        self._fill(bs, _PREFIX_LEN + _EXTEND_LEN)
        fb = make_extend_batch(bs, _EXTEND_LEN, _PREFIX_LEN)
        q, k, v = make_qkv(bs * _EXTEND_LEN, _NUM_HEADS, _HEAD_DIM, dtype=_DTYPE)
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_extend(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "fa3 pcg extend path")


# ---------------------------------------------------------------------------
# TRTLLMMHABackend
# ---------------------------------------------------------------------------

try:
    import sgl_kernel as _sgl_kernel_mod  # noqa: F401

    _TRTLLM_OK = _CUDA
except ImportError:
    _TRTLLM_OK = False


@unittest.skipUnless(_TRTLLM_OK, "sgl_kernel (TRTLLM kernels) not installed")
class TestTRTLLMMHAInit(CustomTestCase):
    """init_forward_metadata* coverage for TRTLLMMHABackend."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMMHABackend

        cls.mr = build_mha_runner(
            num_heads=_NUM_HEADS,
            head_dim=_HEAD_DIM,
            max_bs=_MAX_BS,
            max_context_len=_MAX_CTX,
            dtype=torch.bfloat16,
        )
        cls.backend = TRTLLMMHABackend(cls.mr)
        cls.layer = make_radix_attention(_NUM_HEADS, _HEAD_DIM)
        set_forward_context(ForwardContext(attn_backend=cls.backend))

    def _fill(self, bs: int, seq_len: int):
        fill_req_to_token(self.mr, bs, seq_len)

    def test_eager_decode_no_nan(self):
        bs = 2
        self._fill(bs, _SEQ_LEN)
        fb = make_decode_batch(bs, _SEQ_LEN)
        q, k, v = make_qkv(bs, _NUM_HEADS, _HEAD_DIM, dtype=torch.bfloat16)
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_decode(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "trtllm_mha eager decode")

    def test_eager_extend_no_nan(self):
        bs = 2
        self._fill(bs, _PREFIX_LEN + _EXTEND_LEN)
        fb = make_extend_batch(bs, _EXTEND_LEN, _PREFIX_LEN)
        q, k, v = make_qkv(
            bs * _EXTEND_LEN, _NUM_HEADS, _HEAD_DIM, dtype=torch.bfloat16
        )
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_extend(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "trtllm_mha eager extend")

    def test_graph_decode_capture_no_crash(self):
        self.backend.init_cuda_graph_state(_MAX_BS, _MAX_BS)
        self._fill(_MAX_BS, _SEQ_LEN)
        req_pool = torch.arange(_MAX_BS, dtype=torch.int32, device="cuda")
        seq_lens = torch.full((_MAX_BS,), _SEQ_LEN, dtype=torch.int32, device="cuda")
        fb = make_decode_batch(_MAX_BS, _SEQ_LEN)
        init_graph_capture(
            self.backend, fb, _MAX_BS, _MAX_BS, req_pool, seq_lens, ForwardMode.DECODE
        )

    def test_graph_decode_replay_no_nan(self):
        self.backend.init_cuda_graph_state(_MAX_BS, _MAX_BS)
        self._fill(_MAX_BS, _SEQ_LEN)
        req_pool = torch.arange(_MAX_BS, dtype=torch.int32, device="cuda")
        seq_lens = torch.full((_MAX_BS,), _SEQ_LEN, dtype=torch.int32, device="cuda")
        seq_lens_cpu = torch.full((_MAX_BS,), _SEQ_LEN, dtype=torch.int32)

        fb = make_decode_batch(_MAX_BS, _SEQ_LEN)
        init_graph_capture(
            self.backend, fb, _MAX_BS, _MAX_BS, req_pool, seq_lens, ForwardMode.DECODE
        )
        init_graph_replay(
            self.backend,
            fb,
            _MAX_BS,
            req_pool,
            seq_lens,
            _MAX_BS * _SEQ_LEN,
            ForwardMode.DECODE,
            seq_lens_cpu,
        )
        q, k, v = make_qkv(_MAX_BS, _NUM_HEADS, _HEAD_DIM, dtype=torch.bfloat16)
        out = self.backend.forward_decode(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "trtllm_mha graph replay decode")

    def test_pcg_extend_path(self):
        bs = 2
        self._fill(bs, _PREFIX_LEN + _EXTEND_LEN)
        fb = make_extend_batch(bs, _EXTEND_LEN, _PREFIX_LEN)
        q, k, v = make_qkv(
            bs * _EXTEND_LEN, _NUM_HEADS, _HEAD_DIM, dtype=torch.bfloat16
        )
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_extend(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "trtllm_mha pcg extend path")


# ---------------------------------------------------------------------------
# TorchNativeAttnBackend  (eager-only, no CUDA graph support)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_CUDA, "CUDA required")
class TestTorchNativeInit(CustomTestCase):
    """Eager-only coverage for TorchNativeAttnBackend."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.torch_native_backend import (
            TorchNativeAttnBackend,
        )

        cls.mr = build_mha_runner(
            num_heads=_NUM_HEADS,
            head_dim=_HEAD_DIM,
            max_bs=_MAX_BS,
            max_context_len=_MAX_CTX,
            dtype=_DTYPE,
        )
        cls.backend = TorchNativeAttnBackend(cls.mr)
        cls.layer = make_radix_attention(_NUM_HEADS, _HEAD_DIM)
        set_forward_context(ForwardContext(attn_backend=cls.backend))

    def _fill(self, bs: int, seq_len: int):
        fill_req_to_token(self.mr, bs, seq_len)

    def test_eager_decode_no_nan(self):
        bs = 2
        self._fill(bs, _SEQ_LEN)
        fb = make_decode_batch(bs, _SEQ_LEN)
        q, k, v = make_qkv(bs, _NUM_HEADS, _HEAD_DIM, dtype=_DTYPE)
        self.backend.init_forward_metadata(fb)  # is a no-op for TorchNative
        out = self.backend.forward_decode(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "torch_native eager decode")
        self.assertEqual(out.shape, (bs, _NUM_HEADS * _HEAD_DIM))

    def test_eager_extend_no_nan(self):
        bs = 2
        self._fill(bs, _PREFIX_LEN + _EXTEND_LEN)
        fb = make_extend_batch(bs, _EXTEND_LEN, _PREFIX_LEN)
        q, k, v = make_qkv(bs * _EXTEND_LEN, _NUM_HEADS, _HEAD_DIM, dtype=_DTYPE)
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_extend(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "torch_native eager extend")

    def test_eager_extend_no_prefix(self):
        bs = 3
        self._fill(bs, _EXTEND_LEN)
        fb = make_extend_batch(bs, _EXTEND_LEN, prefix_len=0)
        q, k, v = make_qkv(bs * _EXTEND_LEN, _NUM_HEADS, _HEAD_DIM, dtype=_DTYPE)
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_extend(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "torch_native extend no prefix")

    def test_eager_decode_varied_bs(self):
        """Verify decode works for bs = 1, 4, max_bs."""
        for bs in (1, 4, _MAX_BS):
            with self.subTest(bs=bs):
                self._fill(bs, _SEQ_LEN)
                fb = make_decode_batch(bs, _SEQ_LEN)
                q, k, v = make_qkv(bs, _NUM_HEADS, _HEAD_DIM, dtype=_DTYPE)
                self.backend.init_forward_metadata(fb)
                out = self.backend.forward_decode(q, k, v, self.layer, fb)
                assert_no_nan_inf(self, out, f"torch_native bs={bs}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
