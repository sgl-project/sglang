"""Unit tests for the RecoverSSM (``--gdn-mtp-cache-mode=none``) contract."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (  # noqa: E402
    FlashInferGDNKernel,
    fi_recovery_kernel,
)
from sglang.srt.server_args import ServerArgs  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_server_args(
    *,
    gdn_mtp_cache_mode="none",
    enable_gdn_replayssm_spec=False,
    speculative_eagle_topk=None,
    disable_radix_cache=True,
    mamba_radix_cache_strategy="no_buffer",
):
    sa = ServerArgs.__new__(ServerArgs)
    sa.gdn_mtp_cache_mode = gdn_mtp_cache_mode
    sa.enable_gdn_replayssm_spec = enable_gdn_replayssm_spec
    sa.speculative_eagle_topk = speculative_eagle_topk
    sa.disable_radix_cache = disable_radix_cache
    sa.mamba_radix_cache_strategy = mamba_radix_cache_strategy
    return sa


class TestValidateGdnMtpCacheMode(CustomTestCase):
    def test_non_none_mode_bypasses_all_checks(self):
        for mode in ("full", "auto"):
            with self.subTest(mode=mode):
                sa = make_server_args(
                    gdn_mtp_cache_mode=mode,
                    enable_gdn_replayssm_spec=True,
                    speculative_eagle_topk=8,
                    disable_radix_cache=False,
                    mamba_radix_cache_strategy="extra_buffer",
                )
                self.assertIsNone(sa._validate_gdn_mtp_cache_mode())

    def test_none_mode_rejects_replayssm(self):
        sa = make_server_args(
            enable_gdn_replayssm_spec=True,
            speculative_eagle_topk=1,
        )
        with self.assertRaises(ValueError) as cm:
            sa._validate_gdn_mtp_cache_mode()
        self.assertIn(
            "--gdn-mtp-cache-mode=none is mutually exclusive with "
            "--enable-gdn-replayssm-spec",
            str(cm.exception),
        )

    def test_replayssm_guard_wins_over_topk_guard(self):
        sa = make_server_args(
            enable_gdn_replayssm_spec=True,
            speculative_eagle_topk=8,
        )
        with self.assertRaises(ValueError) as cm:
            sa._validate_gdn_mtp_cache_mode()
        msg = str(cm.exception)
        self.assertIn("mutually exclusive with --enable-gdn-replayssm-spec", msg)
        self.assertNotIn("linear draft chain", msg)

    def test_none_mode_rejects_topk_gt_one(self):
        sa = make_server_args(speculative_eagle_topk=2)
        with self.assertRaises(ValueError) as cm:
            sa._validate_gdn_mtp_cache_mode()
        msg = str(cm.exception)
        self.assertIn("--gdn-mtp-cache-mode=none requires a linear draft chain", msg)
        self.assertIn("--speculative-eagle-topk=2", msg)

    def test_none_mode_allows_linear_draft_chain(self):
        for topk in (None, 1):
            with self.subTest(topk=topk):
                sa = make_server_args(speculative_eagle_topk=topk)
                self.assertIsNone(sa._validate_gdn_mtp_cache_mode())

    def test_none_mode_rejects_mamba_extra_buffer(self):
        for strategy in ("extra_buffer", "extra_buffer_lazy"):
            with self.subTest(strategy=strategy):
                sa = make_server_args(
                    speculative_eagle_topk=1,
                    disable_radix_cache=False,
                    mamba_radix_cache_strategy=strategy,
                )
                with self.assertRaises(ValueError) as cm:
                    sa._validate_gdn_mtp_cache_mode()
                self.assertIn(
                    "--gdn-mtp-cache-mode=none is not compatible with mamba "
                    "extra_buffer",
                    str(cm.exception),
                )

    def test_extra_buffer_disabled_when_radix_cache_off(self):
        sa = make_server_args(
            speculative_eagle_topk=1,
            disable_radix_cache=True,
            mamba_radix_cache_strategy="extra_buffer",
        )
        self.assertIsNone(sa._validate_gdn_mtp_cache_mode())

    def test_valid_recover_ssm_config_no_buffer(self):
        sa = make_server_args(
            speculative_eagle_topk=None,
            disable_radix_cache=False,
            mamba_radix_cache_strategy="no_buffer",
        )
        self.assertIsNone(sa._validate_gdn_mtp_cache_mode())


class TestFiRecoveryKernel(CustomTestCase):
    @staticmethod
    def _backend(**dispatcher_fields):
        return SimpleNamespace(kernel_dispatcher=SimpleNamespace(**dispatcher_fields))

    @staticmethod
    def _fake_fi_kernel(use_state_pool):
        kernel = FlashInferGDNKernel.__new__(FlashInferGDNKernel)
        kernel.use_state_pool = use_state_pool
        return kernel

    def test_missing_kernel_dispatcher(self):
        self.assertIsNone(fi_recovery_kernel(SimpleNamespace()))

    def test_dispatcher_without_decode_kernel(self):
        self.assertIsNone(fi_recovery_kernel(self._backend(decode_kernel=None)))

    def test_non_flashinfer_decode_kernel(self):
        self.assertIsNone(fi_recovery_kernel(self._backend(decode_kernel=object())))

    def test_flashinfer_kernel_with_state_pool(self):
        kernel = self._fake_fi_kernel(use_state_pool=True)
        backend = self._backend(decode_kernel=kernel)
        self.assertIs(fi_recovery_kernel(backend), kernel)

    def test_flashinfer_kernel_without_state_pool(self):
        kernel = self._fake_fi_kernel(use_state_pool=False)
        self.assertIsNone(fi_recovery_kernel(self._backend(decode_kernel=kernel)))


if __name__ == "__main__":
    unittest.main()
