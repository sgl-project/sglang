"""
Step-03 coverage: MLA attention backends × eager / graph-decode runners.

Backends covered:
  FlashInferMLAAttnBackend  (needs flashinfer)
  FlashMLAAttnBackend       (Hopper SM90 only)
  CutlassMLABackend         (Blackwell SM100 only)
  TRTLLMMLABackend          (needs sgl_kernel TRTLLM kernels)

For each backend:
  1. eager_decode  — init_forward_metadata(decode_fb) -> forward_decode -> no NaN
  2. eager_extend  — init_forward_metadata(extend_fb) -> forward_extend -> no NaN
  3. graph_decode  — capture + replay init -> forward_decode -> no NaN
  4. replay_consistent — two identical replays give equal outputs

Run on the cluster (GB300 has all necessary kernels):
  python test_step03_mla_backends.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from step03_test_utils import (
    assert_no_nan_inf,
    build_mla_runner,
    fill_req_to_token,
    gpu_arch_sm,
    init_graph_capture,
    init_graph_replay,
    make_decode_batch,
    make_extend_batch,
)

from sglang.srt.layers import dp_attention as _dp_attn

# TP=1 patch — all MLA backends read get_attention_tp_size() at construction.
_dp_attn.get_attention_tp_size = lambda: 1

# Global server args stub — FlashInferMLAAttnBackend calls get_global_server_args()
# in __init__. Set a minimal stub before any backend is constructed.
import sglang.srt.server_args as _sa_mod

if _sa_mod._global_server_args is None:
    _stub = type(
        "_StubServerArgs",
        (),
        {
            "disaggregation_mode": "null",
            "enable_dp_attention": False,
            "dllm_algorithm": None,
            "disable_chunked_prefix_cache": False,
            "flashinfer_mla_disable_ragged": False,
        },
    )()
    _sa_mod._global_server_args = _stub

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    set_forward_context,
)
from sglang.test.test_utils import CustomTestCase

_CUDA = torch.cuda.is_available()
_SM = gpu_arch_sm()

# MLA geometry for FlashInferMLA / FlashMLA / CutlassMLA (flexible dims)
_NUM_HEADS = 16  # tp_q_head_num with TP=1
_KV_LORA = 64
_QK_ROPE = 32
_QK_NOPE = 64
_V_HEAD = 64
_MAX_BS = 4
_MAX_CTX = 32
_SEQ_LEN = 16
_EXTEND_LEN = 8
_PREFIX_LEN = 4
_DTYPE = torch.bfloat16

# TRTLLM MLA geometry: kernel only supports kv_lora_rank∈{256,512}, qk_rope_head_dim=64
_TRTLLM_NUM_HEADS = 16
_TRTLLM_KV_LORA = 256
_TRTLLM_QK_ROPE = 64
_TRTLLM_QK_NOPE = 64
_TRTLLM_V_HEAD = 128
_TRTLLM_MAX_CTX = 32
_TRTLLM_PAGE_SIZE = 32  # TRTLLM MLA kernel requires block_size 32 or 64

try:
    from sglang.srt.utils import is_flashinfer_available as _fi_avail

    _FLASHINFER = _CUDA and _fi_avail()
except Exception:
    _FLASHINFER = False

try:
    import sgl_kernel as _sgl  # noqa: F401

    _TRTLLM_OK = _CUDA
except ImportError:
    _TRTLLM_OK = False

_HOPPER = _CUDA and _SM is not None and _SM == 90  # SM90 = H100/H200
_B200 = _CUDA and _SM is not None and _SM == 100  # SM100 = B200


def _make_mla_layer(
    num_heads: int = _NUM_HEADS, v_head_dim: int = _V_HEAD
) -> RadixAttention:
    return RadixAttention(
        num_heads=num_heads,
        head_dim=_KV_LORA + _QK_ROPE,
        scaling=(_KV_LORA + _QK_ROPE) ** -0.5,
        num_kv_heads=1,
        layer_id=0,
        v_head_dim=v_head_dim,
    )


def _mla_qkv(num_tokens: int):
    """Combined k for FlashInferMLA (absorbs rope into latent)."""
    head_dim = _KV_LORA + _QK_ROPE
    q = torch.randn(num_tokens, _NUM_HEADS, head_dim, dtype=_DTYPE, device="cuda")
    k = torch.randn(num_tokens, 1, head_dim, dtype=_DTYPE, device="cuda")
    v = torch.randn(num_tokens, 1, _KV_LORA + _QK_ROPE, dtype=_DTYPE, device="cuda")
    return q, k, v


def _trtllm_mla_qkv(num_tokens: int):
    """Q/K/V for TRTLLMMLABackend.

    TRTLLM MLA expects q in "absorbed" form where the W_UK weight is
    pre-multiplied into q: q_head_dim = kv_lora_rank + qk_rope_head_dim
    (not qk_nope + qk_rope as in the non-absorbed path).
    k is the nope component (kv_lora_rank), k_rope is the rope component.
    """
    q = torch.randn(
        num_tokens,
        _TRTLLM_NUM_HEADS,
        _TRTLLM_KV_LORA + _TRTLLM_QK_ROPE,  # absorbed form
        dtype=_DTYPE,
        device="cuda",
    )
    k_nope = torch.randn(num_tokens, 1, _TRTLLM_KV_LORA, dtype=_DTYPE, device="cuda")
    v = torch.randn(num_tokens, 1, _TRTLLM_KV_LORA, dtype=_DTYPE, device="cuda")
    k_rope = torch.randn(num_tokens, 1, _TRTLLM_QK_ROPE, dtype=_DTYPE, device="cuda")
    return q, k_nope, v, k_rope


# ---------------------------------------------------------------------------
# FlashInferMLAAttnBackend
# ---------------------------------------------------------------------------


@unittest.skipUnless(_FLASHINFER, "flashinfer not available")
class TestFlashInferMLAInit(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.flashinfer_mla_backend import (
            FlashInferMLAAttnBackend,
        )

        cls.mr = build_mla_runner(
            num_heads=_NUM_HEADS,
            kv_lora_rank=_KV_LORA,
            qk_rope_head_dim=_QK_ROPE,
            qk_nope_head_dim=_QK_NOPE,
            v_head_dim=_V_HEAD,
            max_bs=_MAX_BS,
            max_context_len=_MAX_CTX,
            dtype=_DTYPE,
        )
        cls.backend = FlashInferMLAAttnBackend(cls.mr)
        cls.layer = _make_mla_layer()
        set_forward_context(ForwardContext(attn_backend=cls.backend))

    @classmethod
    def tearDownClass(cls):
        set_forward_context(None)

    def _fill(self, bs: int, seq_len: int):
        fill_req_to_token(self.mr, bs, seq_len)

    def test_eager_decode_no_nan(self):
        bs = 2
        self._fill(bs, _SEQ_LEN)
        fb = make_decode_batch(bs, _SEQ_LEN)
        q, k, v = _mla_qkv(bs)
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_decode(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "flashinfer_mla eager decode")

    def test_eager_extend_no_nan(self):
        bs = 2
        self._fill(bs, _PREFIX_LEN + _EXTEND_LEN)
        fb = make_extend_batch(bs, _EXTEND_LEN, _PREFIX_LEN)
        q, k, v = _mla_qkv(bs * _EXTEND_LEN)
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_extend(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "flashinfer_mla eager extend")

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
        q, k, v = _mla_qkv(_MAX_BS)
        out = self.backend.forward_decode(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "flashinfer_mla graph replay")

    def test_graph_replay_consistent(self):
        # Use a fresh backend to avoid flashinfer workspace buffer pollution
        # from test_graph_decode_replay_no_nan which runs before this test.
        from sglang.srt.layers.attention.flashinfer_mla_backend import (
            FlashInferMLAAttnBackend,
        )

        fresh_mr = build_mla_runner(
            num_heads=_NUM_HEADS,
            kv_lora_rank=_KV_LORA,
            qk_rope_head_dim=_QK_ROPE,
            qk_nope_head_dim=_QK_NOPE,
            v_head_dim=_V_HEAD,
            max_bs=_MAX_BS,
            max_context_len=_MAX_CTX,
            dtype=_DTYPE,
        )
        # Force a fresh global_workspace_buffer so this backend is completely isolated
        # from any prior FlashInfer state set by MHA tests running earlier in the suite.
        import sglang.srt.layers.attention.flashinfer_mla_backend as _fi_mla_mod

        _saved_ws = _fi_mla_mod.global_workspace_buffer
        _fi_mla_mod.global_workspace_buffer = None  # force new allocation in __init__
        backend = FlashInferMLAAttnBackend(fresh_mr)
        _fi_mla_mod.global_workspace_buffer = _saved_ws  # restore for other tests
        layer = _make_mla_layer()
        set_forward_context(ForwardContext(attn_backend=backend))

        backend.init_cuda_graph_state(_MAX_BS, _MAX_BS)
        fill_req_to_token(fresh_mr, _MAX_BS, _SEQ_LEN)
        req_pool = torch.arange(_MAX_BS, dtype=torch.int32, device="cuda")
        seq_lens = torch.full((_MAX_BS,), _SEQ_LEN, dtype=torch.int32, device="cuda")
        seq_lens_cpu = torch.full((_MAX_BS,), _SEQ_LEN, dtype=torch.int32)

        fb = make_decode_batch(_MAX_BS, _SEQ_LEN)
        init_graph_capture(
            backend, fb, _MAX_BS, _MAX_BS, req_pool, seq_lens, ForwardMode.DECODE
        )

        q, k, v = _mla_qkv(_MAX_BS)

        def _replay():
            init_graph_replay(
                backend,
                fb,
                _MAX_BS,
                req_pool,
                seq_lens,
                _MAX_BS * _SEQ_LEN,
                ForwardMode.DECODE,
                seq_lens_cpu,
            )
            # save_kv_cache=False: skip KV writes so the cache state is identical
            # for both replays (KV writes would change cache content and affect output)
            return backend.forward_decode(q, k, v, layer, fb, save_kv_cache=False)

        out1 = _replay()
        out2 = _replay()
        self.assertTrue(torch.allclose(out1, out2, atol=0))


# ---------------------------------------------------------------------------
# FlashMLAAttnBackend  (Hopper SM90 only)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HOPPER, "FlashMLA requires Hopper (SM90)")
class TestFlashMLAInit(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.flashmla_backend import FlashMLAAttnBackend

        cls.mr = build_mla_runner(
            num_heads=_NUM_HEADS,
            kv_lora_rank=_KV_LORA,
            qk_rope_head_dim=_QK_ROPE,
            qk_nope_head_dim=_QK_NOPE,
            v_head_dim=_V_HEAD,
            max_bs=_MAX_BS,
            max_context_len=_MAX_CTX,
            dtype=_DTYPE,
        )
        cls.backend = FlashMLAAttnBackend(cls.mr)
        cls.layer = _make_mla_layer()
        set_forward_context(ForwardContext(attn_backend=cls.backend))

    def _fill(self, bs: int, seq_len: int):
        fill_req_to_token(self.mr, bs, seq_len)

    def test_eager_decode_no_nan(self):
        bs = 2
        self._fill(bs, _SEQ_LEN)
        fb = make_decode_batch(bs, _SEQ_LEN)
        q, k, v = _mla_qkv(bs)
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_decode(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "flashmla eager decode")

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
        q, k, v = _mla_qkv(_MAX_BS)
        out = self.backend.forward_decode(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "flashmla graph replay")


# ---------------------------------------------------------------------------
# CutlassMLABackend  (Blackwell SM100 only)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_B200, "CutlassMLA requires Blackwell (SM100)")
class TestCutlassMLAInit(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.cutlass_mla_backend import CutlassMLABackend

        cls.mr = build_mla_runner(
            num_heads=_NUM_HEADS,
            kv_lora_rank=_KV_LORA,
            qk_rope_head_dim=_QK_ROPE,
            qk_nope_head_dim=_QK_NOPE,
            v_head_dim=_V_HEAD,
            max_bs=_MAX_BS,
            max_context_len=_MAX_CTX,
            dtype=_DTYPE,
        )
        cls.backend = CutlassMLABackend(cls.mr)
        cls.layer = _make_mla_layer()
        set_forward_context(ForwardContext(attn_backend=cls.backend))

    def _fill(self, bs: int, seq_len: int):
        fill_req_to_token(self.mr, bs, seq_len)

    def test_eager_decode_no_nan(self):
        bs = 2
        self._fill(bs, _SEQ_LEN)
        fb = make_decode_batch(bs, _SEQ_LEN)
        q, k, v = _mla_qkv(bs)
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_decode(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "cutlass_mla eager decode")

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
        q, k, v = _mla_qkv(_MAX_BS)
        out = self.backend.forward_decode(q, k, v, self.layer, fb)
        assert_no_nan_inf(self, out, "cutlass_mla graph replay")


# ---------------------------------------------------------------------------
# TRTLLMMLABackend
# ---------------------------------------------------------------------------


@unittest.skipUnless(_TRTLLM_OK, "sgl_kernel not installed")
def _trtllm_mla_layer():
    """RadixAttention layer sized for TRTLLM MLA dimensions."""
    return RadixAttention(
        num_heads=_TRTLLM_NUM_HEADS,
        head_dim=_TRTLLM_KV_LORA + _TRTLLM_QK_ROPE,
        scaling=(_TRTLLM_KV_LORA + _TRTLLM_QK_ROPE) ** -0.5,
        num_kv_heads=1,
        layer_id=0,
        v_head_dim=_TRTLLM_V_HEAD,
    )


class TestTRTLLMMLAInit(CustomTestCase):
    """TRTLLM MLA backend — uses split k_nope / k_rope via _trtllm_mla_qkv.

    Uses TRTLLM-supported dimensions: kv_lora_rank=256, qk_rope_head_dim=64.
    """

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

        cls.mr = build_mla_runner(
            num_heads=_TRTLLM_NUM_HEADS,
            kv_lora_rank=_TRTLLM_KV_LORA,
            qk_rope_head_dim=_TRTLLM_QK_ROPE,
            qk_nope_head_dim=_TRTLLM_QK_NOPE,
            v_head_dim=_TRTLLM_V_HEAD,
            max_bs=_MAX_BS,
            max_context_len=_TRTLLM_MAX_CTX,
            dtype=_DTYPE,
            page_size=_TRTLLM_PAGE_SIZE,
        )
        cls.backend = TRTLLMMLABackend(cls.mr)
        cls.layer = _trtllm_mla_layer()
        set_forward_context(ForwardContext(attn_backend=cls.backend))

    @classmethod
    def tearDownClass(cls):
        # Reset forward context so other classes get a clean slate.
        set_forward_context(None)

    def _fill(self, bs: int, seq_len: int):
        fill_req_to_token(self.mr, bs, seq_len)

    def test_eager_decode_no_nan(self):
        bs = 2
        self._fill(bs, _SEQ_LEN)
        fb = make_decode_batch(bs, _SEQ_LEN)
        q, k_nope, v, k_rope = _trtllm_mla_qkv(bs)
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_decode(q, k_nope, v, self.layer, fb, k_rope=k_rope)
        assert_no_nan_inf(self, out, "trtllm_mla eager decode")

    @unittest.skip(
        "trtllm_ragged_attention_deepseek (prefill kernel) requires query.shape[2] "
        "to be exactly 192 (DSV3 R1) or 128 (smaller), but neither is satisfied "
        "with kv_lora_rank=256. Decode path is tested; extend is architecture-constrained."
    )
    def test_eager_extend_no_nan(self):
        bs = 2
        self._fill(bs, _PREFIX_LEN + _EXTEND_LEN)
        fb = make_extend_batch(bs, _EXTEND_LEN, _PREFIX_LEN)
        num_tokens = bs * _EXTEND_LEN
        q = torch.randn(
            num_tokens,
            _TRTLLM_NUM_HEADS,
            _TRTLLM_KV_LORA + _TRTLLM_QK_ROPE,
            dtype=_DTYPE,
            device="cuda",
        )
        k_nope = torch.randn(
            num_tokens, 1, _TRTLLM_KV_LORA, dtype=_DTYPE, device="cuda"
        )
        v = torch.randn(num_tokens, 1, _TRTLLM_KV_LORA, dtype=_DTYPE, device="cuda")
        k_rope = torch.randn(
            num_tokens, 1, _TRTLLM_QK_ROPE, dtype=_DTYPE, device="cuda"
        )
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_extend(q, k_nope, v, self.layer, fb, k_rope=k_rope)
        assert_no_nan_inf(self, out, "trtllm_mla eager extend")

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
        q, k_nope, v, k_rope = _trtllm_mla_qkv(_MAX_BS)
        out = self.backend.forward_decode(q, k_nope, v, self.layer, fb, k_rope=k_rope)
        assert_no_nan_inf(self, out, "trtllm_mla graph replay")

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
        q, k_nope, v, k_rope = _trtllm_mla_qkv(_MAX_BS)

        def _replay():
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
            return self.backend.forward_decode(
                q, k_nope, v, self.layer, fb, k_rope=k_rope
            )

        out1 = _replay()
        out2 = _replay()
        self.assertTrue(torch.allclose(out1, out2, atol=0))

    @unittest.skip(
        "Same as test_eager_extend_no_nan: trtllm_ragged_attention_deepseek "
        "dimension constraint not met with kv_lora_rank=256."
    )
    def test_pcg_extend_path(self):
        # TRTLLM MLA: extend also expects absorbed q: kv_lora + qk_rope per head
        bs = 2
        self._fill(bs, _PREFIX_LEN + _EXTEND_LEN)
        fb = make_extend_batch(bs, _EXTEND_LEN, _PREFIX_LEN)
        num_tokens = bs * _EXTEND_LEN
        q = torch.randn(
            num_tokens,
            _TRTLLM_NUM_HEADS,
            _TRTLLM_KV_LORA + _TRTLLM_QK_ROPE,  # absorbed form
            dtype=_DTYPE,
            device="cuda",
        )
        k_nope = torch.randn(
            num_tokens, 1, _TRTLLM_KV_LORA, dtype=_DTYPE, device="cuda"
        )
        v = torch.randn(num_tokens, 1, _TRTLLM_KV_LORA, dtype=_DTYPE, device="cuda")
        k_rope = torch.randn(
            num_tokens, 1, _TRTLLM_QK_ROPE, dtype=_DTYPE, device="cuda"
        )
        self.backend.init_forward_metadata(fb)
        out = self.backend.forward_extend(q, k_nope, v, self.layer, fb, k_rope=k_rope)
        assert_no_nan_inf(self, out, "trtllm_mla pcg extend path")


if __name__ == "__main__":
    unittest.main(verbosity=2)
