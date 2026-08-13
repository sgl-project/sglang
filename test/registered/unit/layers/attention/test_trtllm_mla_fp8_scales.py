import math
import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.kernels.ops.attention import utils as attention_utils
from sglang.srt.layers.attention import trtllm_mla_backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _layer(k_scale=0.25, v_scale=0.25, scaling=0.125):
    return SimpleNamespace(
        layer_id=3,
        k_scale=torch.tensor(k_scale),
        v_scale=torch.tensor(v_scale),
        k_scale_float=k_scale,
        v_scale_float=v_scale,
        scaling=scaling,
    )


def test_require_mla_kv_cache_scales_rejects_invalid_shared_scales():
    with pytest.raises(RuntimeError, match="requires both"):
        trtllm_mla_backend.require_mla_kv_cache_scales(
            SimpleNamespace(layer_id=0), path="test"
        )

    for value in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(RuntimeError, match="finite and positive"):
            trtllm_mla_backend.require_mla_kv_cache_scales(
                _layer(k_scale=value, v_scale=value), path="test"
            )

    with pytest.raises(RuntimeError, match="must match"):
        trtllm_mla_backend.require_mla_kv_cache_scales(
            _layer(k_scale=0.25, v_scale=0.5), path="test"
        )


def test_trtllm_mla_decode_propagates_k_and_v_descales(monkeypatch):
    captured = {}

    def fake_decode(**kwargs):
        captured.update(kwargs)
        return torch.empty_like(kwargs["query"])

    monkeypatch.setattr(
        trtllm_mla_backend,
        "flashinfer",
        SimpleNamespace(
            decode=SimpleNamespace(
                trtllm_batch_decode_with_kv_cache_mla=fake_decode
            )
        ),
        raising=False,
    )

    backend = trtllm_mla_backend.TRTLLMMLABackend.__new__(
        trtllm_mla_backend.TRTLLMMLABackend
    )
    backend.data_type = torch.float8_e4m3fn
    backend.backend = "trtllm-gen"
    backend.workspace_buffer = torch.empty(1)
    backend.qk_nope_head_dim = 8
    backend.kv_lora_rank = 8
    backend.qk_rope_head_dim = 4
    backend._multi_ctas_kv_counter_buffer = torch.empty(1)

    query = torch.empty((1, 1, 1, 12), dtype=torch.float8_e4m3fn)
    kv_cache = torch.empty((1, 1, 1, 12), dtype=torch.float8_e4m3fn)
    backend._run_decode_kernel(
        query=query,
        kv_cache=kv_cache,
        block_tables=torch.zeros((1, 1, 1), dtype=torch.int32),
        seq_lens=torch.ones(1, dtype=torch.int64),
        max_seq_len=1,
        layer=_layer(k_scale=0.25, v_scale=0.25, scaling=0.125),
    )

    assert captured["bmm1_scale"] == pytest.approx(0.25 * 0.125)
    assert captured["bmm2_scale"] == pytest.approx(0.25)


def test_mla_no_rope_quantization_uses_reciprocal_descale():
    q_nope = torch.tensor([[[1.0, 2.0]]], dtype=torch.bfloat16)
    q_rope = torch.tensor([[[3.0]]], dtype=torch.bfloat16)
    k_nope = torch.tensor([[[0.5, 1.0]]], dtype=torch.bfloat16)
    k_rope = torch.tensor([[[1.5]]], dtype=torch.bfloat16)

    q, k, k_rope_out = attention_utils.mla_quantize_without_rope_for_fp8(
        q_nope, q_rope, k_nope, k_rope, kv_descale=0.5
    )

    torch.testing.assert_close(q.float(), torch.tensor([[[1.0, 2.0, 3.0]]]))
    torch.testing.assert_close(k.float(), k_nope.float() * 2.0)
    torch.testing.assert_close(k_rope_out.float(), k_rope.float() * 2.0)


def test_mla_rope_quantization_forwards_reciprocal_descale():
    captured = {}

    def fake_rope_quantize_fp8(**kwargs):
        captured.update(kwargs)

    flashinfer = types.ModuleType("flashinfer")
    flashinfer_rope = types.ModuleType("flashinfer.rope")
    flashinfer_rope.mla_rope_quantize_fp8 = fake_rope_quantize_fp8
    flashinfer.rope = flashinfer_rope

    inputs = {
        "q_nope": torch.zeros((1, 1, 2), dtype=torch.bfloat16),
        "q_rope": torch.zeros((1, 1, 2), dtype=torch.bfloat16),
        "k_nope": torch.zeros((1, 1, 2), dtype=torch.bfloat16),
        "k_rope": torch.zeros((1, 1, 2), dtype=torch.bfloat16),
        "pos_ids": torch.zeros(1, dtype=torch.int64),
        "cos_sin_cache": torch.zeros((1, 4), dtype=torch.bfloat16),
    }

    with mock.patch.dict(
        sys.modules,
        {"flashinfer": flashinfer, "flashinfer.rope": flashinfer_rope},
    ), mock.patch.object(attention_utils, "is_arch_support_pdl", return_value=False):
        attention_utils.mla_quantize_and_rope_for_fp8(
            **inputs,
            is_neox=False,
            kv_lora_rank=2,
            qk_rope_head_dim=2,
            kv_descale=0.25,
        )

    assert captured["quant_scale_q"] == 1.0
    assert captured["quant_scale_kv"] == pytest.approx(4.0)
