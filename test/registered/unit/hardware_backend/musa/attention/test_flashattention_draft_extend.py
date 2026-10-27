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
        from sglang.srt.hardware_backend.musa.attention.flashattention_backend import (
            _should_use_kvcache_for_extend,
        )

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMusaFlashAttentionDraftExtend(CustomTestCase):
    @staticmethod
    def _forward_batch(*, has_prefix=False, target_verify=False, draft_extend=False):
        forward_mode = Mock()
        forward_mode.is_target_verify.return_value = target_verify
        forward_mode.is_draft_extend_v2.return_value = draft_extend
        return SimpleNamespace(
            extend_prefix_lens_cpu=[1] if has_prefix else [0],
            forward_mode=forward_mode,
        )

    def test_regular_extend_without_prefix_uses_varlen_attention(self):
        forward_batch = self._forward_batch()

        self.assertFalse(_should_use_kvcache_for_extend(forward_batch))

    def test_cached_prefix_uses_kvcache_attention(self):
        forward_batch = self._forward_batch(has_prefix=True)

        self.assertTrue(_should_use_kvcache_for_extend(forward_batch))

    def test_target_verify_uses_kvcache_attention(self):
        forward_batch = self._forward_batch(target_verify=True)

        self.assertTrue(_should_use_kvcache_for_extend(forward_batch))

    def test_draft_extend_uses_kvcache_attention(self):
        forward_batch = self._forward_batch(draft_extend=True)

        self.assertTrue(_should_use_kvcache_for_extend(forward_batch))


if __name__ == "__main__":
    unittest.main()
