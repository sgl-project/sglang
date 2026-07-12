"""Unit tests for the DeepSeek-V4 fused-MHC enable policy."""

from unittest.mock import patch

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepseekV4FusedMHCPolicy(CustomTestCase):
    def _is_enabled(
        self,
        *,
        fuse: bool,
        tilelang_pre: bool,
        tilelang_post: bool,
        sm120: bool,
    ) -> bool:
        with (
            envs.SGLANG_OPT_FUSE_MHC_POST_PRE.override(fuse),
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(tilelang_pre),
            envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(tilelang_post),
            patch.object(deepseek_v4, "is_sm120_supported", return_value=sm120),
        ):
            return deepseek_v4._is_fused_mhc_post_pre_enabled()

    def test_sm120_allows_fused_opt_in_with_standalone_pre_disabled(self):
        self.assertTrue(
            self._is_enabled(
                fuse=True,
                tilelang_pre=False,
                tilelang_post=True,
                sm120=True,
            )
        )

    def test_other_platform_still_requires_tilelang_pre(self):
        self.assertFalse(
            self._is_enabled(
                fuse=True,
                tilelang_pre=False,
                tilelang_post=True,
                sm120=False,
            )
        )
        self.assertTrue(
            self._is_enabled(
                fuse=True,
                tilelang_pre=True,
                tilelang_post=True,
                sm120=False,
            )
        )

    def test_fusion_opt_in_and_tilelang_post_remain_required(self):
        self.assertFalse(
            self._is_enabled(
                fuse=False,
                tilelang_pre=False,
                tilelang_post=True,
                sm120=True,
            )
        )
        self.assertFalse(
            self._is_enabled(
                fuse=True,
                tilelang_pre=False,
                tilelang_post=False,
                sm120=True,
            )
        )
