"""Which DeepSeek-V4.1 HIP prefill layers take the fp8 OPUS kernel, and that the model
hands a packed fp8 query only to a layer whose attention takes it."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.attention.dsv4.opus_sparse_prefill_hip import OpusPackedQ
from sglang.srt.environ import envs
from sglang.srt.layers.attention import deepseek_v4_backend_hip_radix as radix
from sglang.srt.models.deepseek_common.amd import deepseek_v4_hip
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _backend(swa_layout=KVLayout.V4, extra_layout=KVLayout.V4):
    pool = SimpleNamespace(
        get_swa_key_layout=lambda: swa_layout,
        get_extra_key_layout=lambda layer_id: extra_layout,
    )
    return SimpleNamespace(token_to_kv_pool=pool)


def _fp8_applies(backend, num_heads=32, compress_ratio=2, available=True):
    with patch.object(radix, "opus_fp8_available", return_value=available):
        return radix.DeepseekV4HipRadixBackend._opus_fp8_applies(
            backend, num_heads=num_heads, layer_id=3, compress_ratio=compress_ratio
        )


class TestOpusFp8Applies(CustomTestCase):
    def test_off_unless_switched_on(self):
        with envs.SGLANG_OPT_HIP_OPUS_FP8_PREFILL.override(False):
            self.assertFalse(_fp8_applies(_backend()))
        with envs.SGLANG_OPT_HIP_OPUS_FP8_PREFILL.override(True):
            self.assertTrue(_fp8_applies(_backend()))

    def test_layers_the_kernel_cannot_take_keep_bf16(self):
        with envs.SGLANG_OPT_HIP_OPUS_FP8_PREFILL.override(True):
            # the kernel takes at most 32 heads
            self.assertFalse(_fp8_applies(_backend(), num_heads=33))
            # the fp8 gather reads only the V4 cache record
            self.assertFalse(_fp8_applies(_backend(swa_layout=KVLayout.V41)))
            self.assertFalse(_fp8_applies(_backend(extra_layout=KVLayout.V41)))
            # an AITER build without the fp8 kernel
            self.assertFalse(_fp8_applies(_backend(), available=False))

    def test_swa_only_layer_ignores_the_compressed_layout(self):
        with envs.SGLANG_OPT_HIP_OPUS_FP8_PREFILL.override(True):
            backend = _backend(extra_layout=KVLayout.V41)
            self.assertTrue(_fp8_applies(backend, compress_ratio=0))


class TestWqBPackedForOpus(CustomTestCase):
    """The attention asserts that a packed query reaches only a layer that runs the fp8
    kernel, so the model must not pack for any other backend or layer."""

    def _call(self, attn_backend, packed=None):
        attn = SimpleNamespace(
            use_mega_attention=False,
            n_local_heads=32,
            layer_id=3,
            compress_ratio=2,
            wq_b=SimpleNamespace(quant_method=None),
        )
        apply = MagicMock(return_value=packed)
        with (
            patch.object(
                deepseek_v4_hip, "is_in_breakable_cuda_graph", return_value=False
            ),
            patch("sglang.srt.layers.quantization.fp8_hip.apply_dense_packed_q", apply),
        ):
            out = deepseek_v4_hip.wq_b_packed_for_opus(
                attn=attn,
                q=torch.empty(16, 8),
                forward_batch=None,
                attn_backend=attn_backend,
            )
        return out, apply

    def test_other_backends_get_no_packed_query(self):
        out, apply = self._call(attn_backend=SimpleNamespace())
        self.assertIsNone(out)
        apply.assert_not_called()

    def test_layer_whose_attention_declines_gets_no_packed_query(self):
        backend = MagicMock(spec=radix.DeepseekV4HipRadixBackend)
        backend.opus_fp8_takes_packed_q.return_value = False
        out, apply = self._call(attn_backend=backend)
        self.assertIsNone(out)
        apply.assert_not_called()

    def test_layer_that_takes_it_gets_the_gemm_pair(self):
        backend = MagicMock(spec=radix.DeepseekV4HipRadixBackend)
        backend.opus_fp8_takes_packed_q.return_value = True
        pair = (torch.empty(16, 32, 512), torch.empty(16, 32, 64))
        out, apply = self._call(attn_backend=backend, packed=pair)
        self.assertIsInstance(out, OpusPackedQ)
        self.assertIs(out.q_nope, pair[0])
        self.assertIs(out.q_rope, pair[1])
        self.assertEqual(
            backend.opus_fp8_takes_packed_q.call_args.kwargs["num_tokens"], 16
        )


if __name__ == "__main__":
    unittest.main()
