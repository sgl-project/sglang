import contextlib
import types
import unittest
from unittest import mock

import torch

from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestRocmMlaProjectionFusion(CustomTestCase):
    def _make_attn(self):
        # Deliberately omits q_lora_rank and v_head_dim: the fused kernel is
        # key-side only, so reading either would raise AttributeError here.
        return types.SimpleNamespace(
            current_attention_backend="aiter",
            use_dsa=False,
            w_kc=torch.empty(1, dtype=torch.uint8),
            w_scale_k=torch.empty(1),
            rotary_emb=object(),
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
        )

    def _make_decode_batch(self):
        return types.SimpleNamespace(
            forward_mode=types.SimpleNamespace(
                is_decode_or_idle=mock.Mock(return_value=True)
            )
        )

    @contextlib.contextmanager
    def _aiter_gfx95_env(self):
        with (
            mock.patch.object(forward_mla, "_is_hip", True),
            mock.patch.object(forward_mla, "_use_aiter_gfx95", True),
            mock.patch.object(
                forward_mla.envs.SGLANG_ROCM_USE_MULTI_STREAM,
                "get",
                side_effect=AssertionError(
                    "MLA projection fusion must not depend on MoE multi-stream mode"
                ),
            ),
            mock.patch.object(
                forward_mla.envs.SGLANG_ROCM_FUSE_MLA_PROJECTION_ROPE_CACHE,
                "get",
                return_value=True,
            ),
            mock.patch.object(forward_mla, "is_kv_b_lora_active", return_value=False),
            mock.patch.object(
                forward_mla,
                "get_parallel",
                return_value=types.SimpleNamespace(dcp_enabled=False),
            ),
        ):
            yield

    def _can_fuse(self, attn, forward_batch):
        return forward_mla.DeepseekMLAForwardMixin._can_fuse_rocm_mla_projection_rope_cache(
            attn, forward_batch, is_capture_mode=True
        )

    def test_projection_fusion_gate_targets_aiter_graph_decode(self):
        attn = self._make_attn()
        forward_batch = self._make_decode_batch()

        with self._aiter_gfx95_env():
            self.assertTrue(self._can_fuse(attn, forward_batch))

            attn.current_attention_backend = "flashinfer"
            self.assertFalse(self._can_fuse(attn, forward_batch))

    def test_projection_fusion_gate_requires_power_of_two_shapes(self):
        """fused_fp4_bmm_rope_cat_and_cache_mla tiles kv_lora_rank and
        qk_rope_head_dim as unmasked ``tl.arange`` tiles and quantizes the BMM K
        dim in 32-wide MXFP4 groups.  None of that is asserted kernel-side, so a
        merely-divisible-looking shape (384 is a multiple of 128, 96 a multiple
        of 32) must still be rejected here or the kernel reads out of bounds.
        """
        forward_batch = self._make_decode_batch()

        with self._aiter_gfx95_env():
            for field, value in (
                ("kv_lora_rank", 384),
                ("qk_nope_head_dim", 96),
                ("qk_nope_head_dim", 16),
                ("qk_nope_head_dim", 512),
                ("qk_rope_head_dim", 48),
            ):
                with self.subTest(field=field, value=value):
                    attn = self._make_attn()
                    setattr(attn, field, value)
                    self.assertFalse(self._can_fuse(attn, forward_batch))


if __name__ == "__main__":
    unittest.main()
