"""Hermetic unit tests for DeepSeek MLA attention-method dispatch on ROCm.

`_dispatch_mla_subtype` picks the forward method for MLA attention. On ROCm the
fused-decode-MLA + fused-RoPE fast path (`MLA_FUSED_ROPE_ROCM`) is only correct
for the aiter attention backend; taking it under the triton backend GPU-faults
on gfx95 (MI355). This test pins the dispatch table so the triton MLA path stays
on the plain `MLA` method.

Pure Python (no GPU, no model weights): `_is_hip` is patched and `attn` /
`forward_batch` are lightweight fakes. Runs on any PR-CI lane.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.models.deepseek_common import attention_backend_handler as abh
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mha_rocm as mha_rocm,
)
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _fake_forward_batch(is_decode: bool):
    return SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: is_decode))


def _fake_attn(backend: str, rocm_fused_decode_mla: bool = True):
    return SimpleNamespace(
        current_attention_backend=backend,
        rocm_fused_decode_mla=rocm_fused_decode_mla,
    )


class TestDispatchMLASubtype(CustomTestCase):
    def test_hip_aiter_decode_takes_fused_rope(self):
        # aiter + fused-decode + decode -> fused ROPE fast path (unchanged).
        with mock.patch.object(abh, "_is_hip", True):
            method = abh._dispatch_mla_subtype(
                _fake_attn("aiter"), _fake_forward_batch(is_decode=True)
            )
        self.assertEqual(method, AttnForwardMethod.MLA_FUSED_ROPE_ROCM)

    def test_hip_triton_decode_stays_plain_mla(self):
        # The fix: triton backend must NOT take the aiter-only fused path even
        # with rocm_fused_decode_mla set -- that path GPU-faults on gfx95.
        with mock.patch.object(abh, "_is_hip", True):
            method = abh._dispatch_mla_subtype(
                _fake_attn("triton"), _fake_forward_batch(is_decode=True)
            )
        self.assertEqual(method, AttnForwardMethod.MLA)

    def test_hip_aiter_extend_stays_plain_mla(self):
        # Fused path is decode-only; extend/prefill uses plain MLA.
        with mock.patch.object(abh, "_is_hip", True):
            method = abh._dispatch_mla_subtype(
                _fake_attn("aiter"), _fake_forward_batch(is_decode=False)
            )
        self.assertEqual(method, AttnForwardMethod.MLA)


class TestResolveRocmForwardMethod(CustomTestCase):
    """The generic MHA/MLA methods must never reach the CUDA forward paths on
    ROCm: those were stripped of their AMD branches when the AITER kernels moved
    into forward_mha_rocm.py / forward_mla_rocm.py."""

    def test_hip_routes_shared_methods_to_rocm(self):
        with mock.patch.object(abh, "_is_hip", True):
            self.assertEqual(
                abh.resolve_rocm_forward_method(AttnForwardMethod.MHA),
                AttnForwardMethod.MHA_ROCM,
            )
            self.assertEqual(
                abh.resolve_rocm_forward_method(AttnForwardMethod.MHA_ONE_SHOT),
                AttnForwardMethod.MHA_ONE_SHOT_ROCM,
            )
            self.assertEqual(
                abh.resolve_rocm_forward_method(AttnForwardMethod.MLA),
                AttnForwardMethod.MLA_ROCM,
            )

    def test_hip_leaves_platform_specific_methods_alone(self):
        with mock.patch.object(abh, "_is_hip", True):
            self.assertEqual(
                abh.resolve_rocm_forward_method(AttnForwardMethod.MLA_FUSED_ROPE_ROCM),
                AttnForwardMethod.MLA_FUSED_ROPE_ROCM,
            )

    def test_non_hip_is_identity(self):
        with mock.patch.object(abh, "_is_hip", False):
            for method in AttnForwardMethod:
                self.assertEqual(abh.resolve_rocm_forward_method(method), method)


def _fake_rocm_mha_attn():
    return SimpleNamespace(
        attn_mha=SimpleNamespace(layer_id=3),
        kv_lora_rank=2,
        qk_rope_head_dim=1,
    )


class TestRocmMhaKvBufferRouting(CustomTestCase):
    def test_gfx95_uses_mla_kv_pool(self):
        attn = _fake_rocm_mha_attn()
        pool = mock.Mock()
        batch = SimpleNamespace(out_cache_loc=object())
        latent_cache = torch.zeros(2, 1, 3)
        kv_a = torch.ones(2, 2)
        k_pe = torch.ones(2, 1, 1)
        kv_indices = torch.tensor([0, 1])
        pool.get_mla_kv_buffer.return_value = (kv_a.unsqueeze(1), k_pe)

        with (
            mock.patch.object(mha_rocm, "_use_aiter_gfx95", True),
            mock.patch.object(mha_rocm, "get_token_to_kv_pool", return_value=pool),
            mock.patch.object(
                mha_rocm,
                "filter_dcp_local_kv_indices",
                return_value=kv_indices,
            ) as mock_filter_indices,
        ):
            mha_rocm.DeepseekMHARocmForwardMixin._set_mla_kv_buffer_rocm(
                attn, latent_cache, kv_a, k_pe, batch
            )
            fetched_kv_a, fetched_k_pe = (
                mha_rocm.DeepseekMHARocmForwardMixin._get_mla_kv_buffer_rocm(
                    attn, kv_indices, torch.float32, batch
                )
            )

        pool.set_mla_kv_buffer.assert_called_once()
        args = pool.set_mla_kv_buffer.call_args.args
        self.assertIs(args[0], attn.attn_mha)
        self.assertIs(args[1], batch.out_cache_loc)
        torch.testing.assert_close(args[2], kv_a.unsqueeze(1))
        self.assertIs(args[3], k_pe)
        pool.set_kv_buffer.assert_not_called()
        mock_filter_indices.assert_called_once_with(kv_indices=kv_indices)
        pool.get_mla_kv_buffer.assert_called_once_with(
            attn.attn_mha, kv_indices, torch.float32
        )
        torch.testing.assert_close(fetched_kv_a, kv_a)
        self.assertIs(fetched_k_pe, k_pe)

    def test_non_gfx95_uses_combined_key_buffer(self):
        attn = _fake_rocm_mha_attn()
        pool = mock.Mock()
        batch = SimpleNamespace(out_cache_loc=object())
        latent_cache = torch.zeros(2, 1, 3)
        kv_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        k_pe = torch.tensor([[[5.0]], [[6.0]]])
        kv_indices = torch.tensor([0, 1])
        pool.get_key_buffer.return_value = latent_cache

        with (
            mock.patch.object(mha_rocm, "_use_aiter_gfx95", False),
            mock.patch.object(mha_rocm, "get_token_to_kv_pool", return_value=pool),
        ):
            mha_rocm.DeepseekMHARocmForwardMixin._set_mla_kv_buffer_rocm(
                attn, latent_cache, kv_a, k_pe, batch
            )
            fetched_kv_a, fetched_k_pe = (
                mha_rocm.DeepseekMHARocmForwardMixin._get_mla_kv_buffer_rocm(
                    attn, kv_indices, torch.float32, batch
                )
            )

        expected = torch.cat([kv_a.unsqueeze(1), k_pe], dim=-1)
        torch.testing.assert_close(latent_cache, expected)
        pool.set_kv_buffer.assert_called_once()
        args = pool.set_kv_buffer.call_args.args
        self.assertIs(args[0], attn.attn_mha)
        self.assertIs(args[1], batch.out_cache_loc)
        self.assertIs(args[2], latent_cache)
        self.assertIsNone(args[3])
        pool.set_mla_kv_buffer.assert_not_called()
        pool.get_key_buffer.assert_called_once_with(attn.attn_mha.layer_id)
        torch.testing.assert_close(fetched_kv_a, kv_a)
        torch.testing.assert_close(fetched_k_pe, k_pe)


if __name__ == "__main__":
    unittest.main()
