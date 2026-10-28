"""Regression tests for issue #29149.

`--enable-deterministic-inference` crashed at startup on sm_89 (L40S,
RTX 4060 Laptop): the fixed float16 persistent-matmul config
(128x256x64, num_stages=3, num_warps=8) lowers to a 106496 B shared-memory
footprint on sm_8x, above the 101376 B per-block limit, so the first launch
raised ``triton OutOfResources: shared memory, Required: 106496, Hardware
limit: 101376``. The identical config fits under the identical 101376 B
limit on SM120, so the selection gates on (compute capability, limit), not
the limit alone.

Pure config-selection logic; no GPU kernels are launched (device properties
are mocked), so this runs on CPU-only machines.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
    _MATMUL_PERSISTENT_CONFIGS,
    _MATMUL_PERSISTENT_FP16_LOW_SMEM_CONFIG,
    _SM8X_FP16_SMEM_REQUIRED_BYTES,
    _get_matmul_persistent_configs,
    _matmul_persistent_configs_for_device,
    _select_matmul_persistent_configs,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ADA_LIMIT = 101376  # bytes/block on sm_86, sm_89 and, coincidentally, SM120
A100_LIMIT = 166912
H100_LIMIT = 232448

_PROPS_PATH = "torch.cuda.get_device_properties"


def _props(major, minor, smem):
    return SimpleNamespace(major=major, minor=minor, shared_memory_per_block_optin=smem)


class TestMatmulPersistentSmemConfig(CustomTestCase):
    def setUp(self):
        # Clear before as well as after: when this file runs in the same
        # process as tests that launch the real kernels (e.g. pytest on the
        # whole directory on a GPU box), the cache may already hold the real
        # device's configs.
        _matmul_persistent_configs_for_device.cache_clear()

    def tearDown(self):
        _matmul_persistent_configs_for_device.cache_clear()

    def test_sm89_fp16_degraded_to_two_stages(self):
        # The exact crash arch from #29149: L40S / RTX 4060 Laptop.
        configs = _select_matmul_persistent_configs((8, 9), ADA_LIMIT)
        self.assertEqual(configs[torch.float16]["num_stages"], 2)

    def test_sm86_fp16_degraded_to_two_stages(self):
        # Consumer Ampere shares the sm_8x codegen and the 99KB limit.
        configs = _select_matmul_persistent_configs((8, 6), ADA_LIMIT)
        self.assertEqual(configs[torch.float16]["num_stages"], 2)

    def test_sm89_bf16_and_fp32_unchanged(self):
        # Only the fp16 config exceeds the sm_8x limit; the others fit.
        configs = _select_matmul_persistent_configs((8, 9), ADA_LIMIT)
        self.assertEqual(
            configs[torch.bfloat16], _MATMUL_PERSISTENT_CONFIGS[torch.bfloat16]
        )
        self.assertEqual(
            configs[torch.float32], _MATMUL_PERSISTENT_CONFIGS[torch.float32]
        )

    def test_sm120_same_limit_keeps_default(self):
        # SM120 exposes the same 101376 B limit but the fp16 config compiles
        # and runs there (verified on RTX PRO 6000 Blackwell in #29149):
        # the 106496 B footprint is sm_8x-codegen-specific. Gating on the
        # limit alone would wrongly degrade SM120.
        configs = _select_matmul_persistent_configs((12, 0), ADA_LIMIT)
        self.assertIs(configs, _MATMUL_PERSISTENT_CONFIGS)

    def test_datacenter_parts_keep_default(self):
        for capability, limit in [((8, 0), A100_LIMIT), ((9, 0), H100_LIMIT)]:
            with self.subTest(capability=capability):
                configs = _select_matmul_persistent_configs(capability, limit)
                self.assertIs(configs, _MATMUL_PERSISTENT_CONFIGS)

    def test_unknown_limit_keeps_default(self):
        configs = _select_matmul_persistent_configs((8, 9), None)
        self.assertIs(configs, _MATMUL_PERSISTENT_CONFIGS)

    def test_low_smem_variant_only_changes_num_stages(self):
        # Batch invariance depends on this: reducing num_stages only changes
        # Triton's prefetch pipeline depth. Changing any block size could
        # change the accumulation order and must never happen here.
        default = _MATMUL_PERSISTENT_CONFIGS[torch.float16]
        low = _MATMUL_PERSISTENT_FP16_LOW_SMEM_CONFIG
        self.assertEqual(
            {k: v for k, v in low.items() if k != "num_stages"},
            {k: v for k, v in default.items() if k != "num_stages"},
        )
        self.assertLess(low["num_stages"], default["num_stages"])

    def test_threshold_matches_triton_reported_requirement(self):
        # triton reported exactly 106496 bytes in #29149, on two sm_89 parts.
        self.assertEqual(_SM8X_FP16_SMEM_REQUIRED_BYTES, 106496)

    def test_device_lookup_uses_mocked_props(self):
        with patch(_PROPS_PATH, return_value=_props(8, 9, ADA_LIMIT)):
            configs = _get_matmul_persistent_configs(torch.device("cuda", 0))
        self.assertEqual(configs[torch.float16]["num_stages"], 2)

    def test_device_lookup_cached_per_device(self):
        # The choice is made once per device index and must stay fixed for
        # the process lifetime; later calls may not re-query the driver.
        with patch(_PROPS_PATH, return_value=_props(8, 9, ADA_LIMIT)) as mock:
            first = _get_matmul_persistent_configs(torch.device("cuda", 0))
            second = _get_matmul_persistent_configs(torch.device("cuda", 0))
        self.assertIs(first, second)
        self.assertEqual(mock.call_count, 1)

    def test_props_query_failure_keeps_default(self):
        with patch(_PROPS_PATH, side_effect=RuntimeError("no driver")):
            configs = _get_matmul_persistent_configs(torch.device("cuda", 0))
        self.assertIs(configs, _MATMUL_PERSISTENT_CONFIGS)

    def test_non_cuda_device_keeps_default(self):
        self.assertIs(
            _get_matmul_persistent_configs(torch.device("cpu")),
            _MATMUL_PERSISTENT_CONFIGS,
        )


if __name__ == "__main__":
    unittest.main()
