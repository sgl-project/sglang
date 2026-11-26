"""--prefill-only-disable-kv-cache admits out-of-tree platforms that declare
PlatformCapabilities.prefill_only_noop_kv_pool."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache import kv_cache_configurator as kcc  # noqa: E402
from sglang.srt.platforms.interface import PlatformCapabilities  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _platform(*, out_of_tree, **capabilities):
    return SimpleNamespace(
        is_out_of_tree=lambda: out_of_tree,
        capabilities=PlatformCapabilities(**capabilities),
    )


class TestPrefillOnlyNoopKVPoolCapability(CustomTestCase):
    def _validate(self, platform, *, use_mla_backend=False):
        cfg = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        cfg.is_draft_worker = False
        cfg.mambaish_config = None
        cfg.use_mla_backend = use_mla_backend
        cfg.is_hybrid_swa = False
        cfg.kv_cache_dtype = torch.bfloat16
        schedule = SimpleNamespace(prefill_only_disable_kv_cache=True)
        exec_cfg = SimpleNamespace(kernel=SimpleNamespace(attention_backend="fa3"))
        with patch.object(kcc, "get_schedule", return_value=schedule), patch.object(
            kcc, "get_exec", return_value=exec_cfg
        ):
            cfg._validate_prefill_only_disable_kv_cache_pool_family(
                is_dsa_model=False,
                is_compressed_kv_model=False,
                current_platform=platform,
            )

    def test_in_tree_platform_is_accepted(self):
        self._validate(_platform(out_of_tree=False))

    def test_out_of_tree_platform_is_rejected_by_default(self):
        with self.assertRaisesRegex(RuntimeError, "out-of-tree platform KV pool"):
            self._validate(_platform(out_of_tree=True))

    def test_out_of_tree_platform_declaring_the_capability_is_accepted(self):
        self._validate(_platform(out_of_tree=True, prefill_only_noop_kv_pool=True))

    def test_capability_keeps_the_other_pool_family_checks(self):
        with self.assertRaisesRegex(RuntimeError, "MLA KV pool"):
            self._validate(
                _platform(out_of_tree=True, prefill_only_noop_kv_pool=True),
                use_mla_backend=True,
            )


if __name__ == "__main__":
    unittest.main()
