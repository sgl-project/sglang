"""Hermetic unit tests for DeepseekSparseAttnBackend._forward_trtllm's SM12x
kwarg selection to flashinfer's dispatcher.

On SM120/SM121 (consumer Blackwell) there is no trtllm-gen kernel; the dsa
"trtllm" backend must instead route to flashinfer's native sparse-MLA kernel:
`backend="auto"` (flashinfer's dispatcher then picks its "sparse" kernel for
cc==12 + sparse_mla_top_k>0), a `torch.uint8` view of the packed KV buffer,
`kv_scale_format="arbitrary_fp32"` (sglang's quantize_k_cache writes
amax/448 arbitrary fp32 tile scales, not pow2/ue8m0), and
`skip_softmax_threshold_scale_factor=None` (unsupported by the sparse
backend). On any other arch/config the call must stay byte-identical to
upstream ("trtllm-gen" / "auto" scale format / the raw kv view / the
skip-softmax env value).

This is exercised by monkeypatching flashinfer.decode's dispatcher (via
sys.modules, so the test does not require flashinfer to be installed) and
calling `_forward_trtllm` against a lightweight fake backend object -- no
GPU, no real model weights.
"""

from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import dsa_backend as db
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _install_fake_flashinfer_decode(captured: dict):
    """Replace flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla with a
    kwarg-capturing stub via sys.modules, independent of whether the real
    flashinfer package is installed in this environment."""

    def _fake_call(**kwargs):
        captured.update(kwargs)
        return torch.zeros(1)

    fake_decode = types.ModuleType("flashinfer.decode")
    fake_decode.trtllm_batch_decode_with_kv_cache_mla = _fake_call
    fake_flashinfer = types.ModuleType("flashinfer")
    fake_flashinfer.decode = fake_decode
    return mock.patch.dict(
        sys.modules, {"flashinfer": fake_flashinfer, "flashinfer.decode": fake_decode}
    )


def _fake_backend(*, dsa_kv_cache_store_fp8: bool):
    """A minimal stand-in for a DeepseekSparseAttnBackend instance, providing
    just enough surface for _forward_trtllm to reach the flashinfer call
    without touching a GPU or real model weights."""
    real_page_size = 1
    kv_cache_dim = 8
    num_tokens = 4  # decode: one token per sequence

    k_cache = torch.zeros(
        num_tokens, real_page_size, kv_cache_dim, dtype=torch.bfloat16
    )
    page_table = torch.zeros((num_tokens, 1), dtype=torch.int32)

    return SimpleNamespace(
        forward_metadata=SimpleNamespace(max_seq_len_k=128),
        # bf16, not fp8: keeps the (unrelated) fused rope+fp8-quantize branch
        # off regardless of _sparse_sm120, isolating the kwarg selection.
        kv_cache_dtype=torch.bfloat16,
        dsa_kv_cache_store_fp8=dsa_kv_cache_store_fp8,
        token_to_kv_pool=SimpleNamespace(get_key_buffer=lambda layer_id: k_cache),
        real_page_size=real_page_size,
        kv_cache_dim=kv_cache_dim,
        use_fused_topk=True,
        _get_fused_topk_page_table=lambda topk_indices: page_table,
        workspace_buffer=object(),
        qk_nope_head_dim=6,
        kv_lora_rank=2,
        qk_rope_head_dim=2,
        dsa_index_topk=2048,
    )


def _fake_layer():
    return SimpleNamespace(
        tp_q_head_num=1,
        head_dim=4,
        v_head_dim=4,
        scaling=1.0,
        is_cross_attention=False,
        layer_id=0,
    )


def _fake_forward_batch():
    return SimpleNamespace(attn_cp_metadata=None)


class TestForwardTrtllmSM12xKwargs(CustomTestCase):
    def _run(self, *, is_sm120: bool, dsa_kv_cache_store_fp8: bool):
        captured: dict = {}
        num_tokens = 4
        q = torch.zeros(num_tokens, 4)
        seq_lens = torch.ones(num_tokens, dtype=torch.int32)

        with (
            mock.patch.object(db, "_IS_SM120", is_sm120),
            mock.patch.object(db, "dsa_use_prefill_cp", lambda forward_batch: False),
            _install_fake_flashinfer_decode(captured),
        ):
            db.DeepseekSparseAttnBackend._forward_trtllm(
                _fake_backend(dsa_kv_cache_store_fp8=dsa_kv_cache_store_fp8),
                q=q,
                k=None,
                v=None,
                layer=_fake_layer(),
                forward_batch=_fake_forward_batch(),
                seq_lens=seq_lens,
                save_kv_cache=False,
                q_rope=None,
                k_rope=None,
            )
        return captured

    def test_sm12x_fp8_pool_selects_sparse_kwargs(self):
        captured = self._run(is_sm120=True, dsa_kv_cache_store_fp8=True)
        self.assertEqual(captured["backend"], "auto")
        self.assertEqual(captured["kv_scale_format"], "arbitrary_fp32")
        self.assertIsNone(captured["skip_softmax_threshold_scale_factor"])
        self.assertEqual(captured["kv_cache"].dtype, torch.uint8)

    def test_sm100_fp8_pool_keeps_upstream_kwargs(self):
        # Datacenter Blackwell: _IS_SM120 False -> byte-identical to upstream
        # even though the pool happens to hold the packed fp8 layout.
        captured = self._run(is_sm120=False, dsa_kv_cache_store_fp8=True)
        self.assertEqual(captured["backend"], "trtllm-gen")
        self.assertEqual(captured["kv_scale_format"], "auto")
        self.assertEqual(
            captured["skip_softmax_threshold_scale_factor"],
            envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
        )
        self.assertNotEqual(captured["kv_cache"].dtype, torch.uint8)

    def test_sm12x_bf16_pool_keeps_upstream_kwargs(self):
        # SM12x but dsa_kv_cache_store_fp8 False (e.g. bf16 KV cache): the
        # pool does not hold the packed layout the sparse kernel needs, so
        # the gate (_IS_SM120 AND dsa_kv_cache_store_fp8) must stay False.
        captured = self._run(is_sm120=True, dsa_kv_cache_store_fp8=False)
        self.assertEqual(captured["backend"], "trtllm-gen")
        self.assertEqual(captured["kv_scale_format"], "auto")
        self.assertEqual(
            captured["skip_softmax_threshold_scale_factor"],
            envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
        )
        self.assertNotEqual(captured["kv_cache"].dtype, torch.uint8)

    def test_sm12x_sparse_route_rejects_prefill_cp(self):
        # Prefill context parallelism has only been exercised against the
        # trtllm-gen fp8-quantize-fused-rope path, which the SM12x sparse
        # route skips entirely -- must fail loudly, not silently mis-route.
        # flashinfer is stubbed too even though the dispatcher is never
        # reached: `import flashinfer.decode` is the function's first
        # statement, executed before this guard.
        with (
            mock.patch.object(db, "_IS_SM120", True),
            mock.patch.object(db, "dsa_use_prefill_cp", lambda forward_batch: True),
            _install_fake_flashinfer_decode({}),
        ):
            with self.assertRaises(NotImplementedError):
                db.DeepseekSparseAttnBackend._forward_trtllm(
                    _fake_backend(dsa_kv_cache_store_fp8=True),
                    q=torch.zeros(4, 4),
                    k=None,
                    v=None,
                    layer=_fake_layer(),
                    forward_batch=_fake_forward_batch(),
                    seq_lens=torch.ones(4, dtype=torch.int32),
                    save_kv_cache=False,
                    q_rope=None,
                    k_rope=None,
                )


if __name__ == "__main__":
    unittest.main()
