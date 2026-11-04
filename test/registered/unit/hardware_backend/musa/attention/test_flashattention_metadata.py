import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

with patch.dict(
    sys.modules,
    {
        module: MagicMock()
        for module in (
            "sgl_kernel",
            "sgl_kernel.quantization",
            "sgl_kernel.scalar_type",
        )
    },
):
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase

    with patch.dict(sys.modules, {"flash_attn_interface": MagicMock()}):
        from sglang.srt.hardware_backend.musa.attention import (
            flashattention_backend as fa_backend,
        )


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMusaFlashAttentionMetadata(CustomTestCase):
    @staticmethod
    def _forward_batch(
        *,
        batch_size=1,
        has_prefix=False,
        decode=False,
        target_verify=False,
        draft_extend=False,
    ):
        forward_mode = Mock()
        forward_mode.is_decode_or_idle.return_value = decode
        forward_mode.is_target_verify.return_value = target_verify
        forward_mode.is_draft_extend_v2.return_value = draft_extend
        forward_mode.is_extend_or_draft_extend_or_mixed.return_value = has_prefix
        return SimpleNamespace(
            batch_size=batch_size,
            extend_prefix_lens_cpu=[1] if has_prefix else [0],
            forward_mode=forward_mode,
        )

    @staticmethod
    def _backend():
        backend = object.__new__(fa_backend.MusaFlashAttentionBackend)
        backend._is_gemma4 = False
        backend._is_minimax_m2 = False
        backend.num_splits = 1
        backend.full_attention_num_splits = 1
        return backend

    def test_kvcache_metadata_modes(self):
        backend = self._backend()

        self.assertFalse(
            backend._needs_no_mla_kvcache_scheduler_metadata(self._forward_batch())
        )
        for forward_batch in (
            self._forward_batch(has_prefix=True),
            self._forward_batch(decode=True),
            self._forward_batch(target_verify=True),
            self._forward_batch(draft_extend=True),
        ):
            with self.subTest(forward_batch=forward_batch):
                self.assertTrue(
                    backend._needs_no_mla_kvcache_scheduler_metadata(forward_batch)
                )

    def test_gemma4_split_policy(self):
        backend = self._backend()
        backend._is_gemma4 = True

        self.assertEqual(
            backend._num_splits_for_no_mla_call(
                self._forward_batch(batch_size=4), is_swa_layer=True
            ),
            0,
        )
        self.assertEqual(
            backend._num_splits_for_no_mla_call(
                self._forward_batch(batch_size=8), is_swa_layer=True
            ),
            1,
        )
        self.assertEqual(
            backend._num_splits_for_no_mla_call(
                self._forward_batch(batch_size=4), is_swa_layer=False
            ),
            1,
        )

    def test_kvcache_wrapper_forwards_explicit_metadata(self):
        q = MagicMock()
        k_cache = MagicMock()
        v_cache = MagicMock()
        metadata = MagicMock()
        expected = MagicMock()

        with patch.object(
            fa_backend, "mate_flash_attn_with_kvcache", return_value=expected
        ) as kernel:
            result = fa_backend.flash_attn_with_kvcache(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                scheduler_metadata=metadata,
            )

        self.assertIs(result, expected)
        self.assertIs(kernel.call_args.kwargs["scheduler_metadata"], metadata)

    def test_kvcache_wrapper_rejects_missing_metadata(self):
        with self.assertRaisesRegex(AssertionError, "requires scheduler_metadata"):
            fa_backend.flash_attn_with_kvcache(
                q=MagicMock(),
                k_cache=MagicMock(),
                v_cache=MagicMock(),
            )


if __name__ == "__main__":
    unittest.main()
