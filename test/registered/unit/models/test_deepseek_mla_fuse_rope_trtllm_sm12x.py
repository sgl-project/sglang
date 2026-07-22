"""Hermetic unit test for the SM12x carve-out in
DeepseekMLAForwardMixin._fuse_rope_for_trtllm_mla (dsa/nsa branch), the mixin
DeepseekV2AttentionMLA inherits its MLA forward methods from.

Regression guard for the FIRST LIVE DEPLOY LESSON: on SM120/SM121, the dsa
"trtllm" backend routes to flashinfer's native sparse-MLA kernel, which
requires a BF16 query and dequantizes the KV cache itself via inline
per-block scales. The datacenter-Blackwell fused rope+fp8-quantize path
(`mla_quantize_and_rope_for_fp8` in dsa_backend.py's _forward_trtllm) hands
that kernel an fp8 query, which it rejects
("... expects BF16 query, got torch.float8_e4m3fn"). This method must
therefore return False on SM12x for the dsa/nsa branch (rope stays in
forward_absorb_prepare, query reaches _forward_trtllm as bf16) regardless of
the dsa_decode_backend/dsa_prefill_backend/kv_cache_dtype combination that
upstream (SM100/103) uses to decide the fused path.

Pure Python (no GPU): `_IS_SM120` is patched directly.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla as fm,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _fake_self(current_attention_backend: str):
    return SimpleNamespace(current_attention_backend=current_attention_backend)


class TestFuseRopeForTrtllmMlaSM12x(CustomTestCase):
    def _call(self, *, is_sm120: bool, dsa_decode_backend: str, kv_cache_dtype):
        fake_server_args = SimpleNamespace(
            dsa_decode_backend=dsa_decode_backend,
            dsa_prefill_backend="not_trtllm",
        )
        fake_attn_backend = SimpleNamespace(kv_cache_dtype=kv_cache_dtype)
        with (
            mock.patch.object(fm, "_IS_SM120", is_sm120),
            mock.patch.object(fm, "get_server_args", lambda: fake_server_args),
            mock.patch.object(fm, "get_attn_backend", lambda: fake_attn_backend),
        ):
            return fm.DeepseekMLAForwardMixin._fuse_rope_for_trtllm_mla(
                _fake_self("dsa"), forward_batch=SimpleNamespace()
            )

    def test_sm12x_dsa_trtllm_fp8_stays_unfused(self):
        # The exact upstream-fused config (trtllm backend + fp8 kv cache) --
        # on SM12x this must now return False, not True.
        result = self._call(
            is_sm120=True,
            dsa_decode_backend="trtllm",
            kv_cache_dtype=torch.float8_e4m3fn,
        )
        self.assertFalse(result)

    def test_sm100_dsa_trtllm_fp8_keeps_fused_upstream(self):
        # Datacenter Blackwell: byte-identical to upstream -- fused path taken.
        result = self._call(
            is_sm120=False,
            dsa_decode_backend="trtllm",
            kv_cache_dtype=torch.float8_e4m3fn,
        )
        self.assertTrue(result)

    def test_sm12x_dsa_non_trtllm_stays_unfused(self):
        # Non-trtllm dsa backend: upstream would also return False here; make
        # sure the SM12x carve-out doesn't accidentally flip it to True.
        result = self._call(
            is_sm120=True,
            dsa_decode_backend="flashinfer_gather",
            kv_cache_dtype=torch.float8_e4m3fn,
        )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
