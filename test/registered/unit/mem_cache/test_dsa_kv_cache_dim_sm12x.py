"""Hermetic unit test for the SM12x carve-out in calculate_mla_kv_cache_dim.

On SM120/SM121 (consumer Blackwell: DGX Spark GB10, RTX PRO/50-series) the
dsa "trtllm" backend does not dispatch to the datacenter trtllm-gen kernel;
it is routed to flashinfer's native sparse-MLA kernel instead (see
DeepseekSparseAttnBackend._forward_trtllm), which consumes the 656-byte
packed inline-scale layout rather than the plain kv_lora_rank +
qk_rope_head_dim layout trtllm-gen expects. `calculate_mla_kv_cache_dim`
must therefore skip its trtllm early-return on SM12x (module-level
`_is_sm120`, mocked here) so the packed layout is computed, while leaving
SM100/SM103 byte-identical.

Pure Python (no GPU): `_is_sm120` is patched directly; no CUDA context is
touched. Runs on any PR-CI lane.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

# GlmMoeDsaForCausalLM is one of the is_deepseek_dsa-recognized architectures;
# index_topk must be present for is_deepseek_dsa to return True.
_DSA_HF_CONFIG = {"architectures": ["GlmMoeDsaForCausalLM"], "index_topk": 2048}
_KV_LORA_RANK = 512
_QK_ROPE_HEAD_DIM = 64
# 512 (fp8 nope) + 512 // 128 * 4 (fp32 tile scales) + 64 * 2 (bf16 rope) = 656
_PACKED_DIM = 656
_PLAIN_DIM = _KV_LORA_RANK + _QK_ROPE_HEAD_DIM  # 576


def _model_config():
    return SimpleNamespace(
        hf_config=_DSA_HF_CONFIG,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
    )


def _server_args(*, dsa_prefill_backend="trtllm", dsa_decode_backend="trtllm"):
    return SimpleNamespace(
        dsa_prefill_backend=dsa_prefill_backend,
        dsa_decode_backend=dsa_decode_backend,
    )


class TestCalculateMlaKvCacheDimSM12x(CustomTestCase):
    def test_sm12x_fp8_trtllm_uses_packed_layout(self):
        # The change under test: SM12x + fp8 KV + trtllm dsa backend must NOT
        # early-return the plain layout -- the sparse kernel needs the packed
        # inline-scale layout.
        with mock.patch.object(kcc, "_is_sm120", True):
            dim = kcc.calculate_mla_kv_cache_dim(
                model_config=_model_config(),
                kv_cache_dtype=torch.float8_e4m3fn,
                server_args=_server_args(),
            )
        self.assertEqual(dim, _PACKED_DIM)

    def test_sm12x_bf16_trtllm_stays_plain_layout(self):
        # bf16 KV cache never uses the packed fp8 layout, on any arch: the
        # trtllm early-return is skipped on SM12x, but the fall-through fp8
        # check below it is false, so the plain layout still results.
        with mock.patch.object(kcc, "_is_sm120", True):
            dim = kcc.calculate_mla_kv_cache_dim(
                model_config=_model_config(),
                kv_cache_dtype=torch.bfloat16,
                server_args=_server_args(),
            )
        self.assertEqual(dim, _PLAIN_DIM)

    def test_sm100_fp8_trtllm_keeps_plain_layout(self):
        # Datacenter Blackwell (SM100/103) uses the real trtllm-gen kernel,
        # which dequants via a scalar bmm1 k_scale and expects the plain
        # layout regardless of kv_cache_dtype -- byte-identical to upstream.
        with mock.patch.object(kcc, "_is_sm120", False):
            dim = kcc.calculate_mla_kv_cache_dim(
                model_config=_model_config(),
                kv_cache_dtype=torch.float8_e4m3fn,
                server_args=_server_args(),
            )
        self.assertEqual(dim, _PLAIN_DIM)

    def test_sm12x_non_trtllm_backend_unaffected(self):
        # _is_sm120 must only change behavior for the trtllm branch; a
        # non-trtllm dsa backend on SM12x falls through to the normal fp8
        # scaled-layout computation, same as upstream.
        with mock.patch.object(kcc, "_is_sm120", True):
            dim = kcc.calculate_mla_kv_cache_dim(
                model_config=_model_config(),
                kv_cache_dtype=torch.float8_e4m3fn,
                server_args=_server_args(
                    dsa_prefill_backend="flashinfer_gather",
                    dsa_decode_backend="flashinfer_gather",
                ),
            )
        self.assertEqual(dim, _PACKED_DIM)


if __name__ == "__main__":
    unittest.main()
