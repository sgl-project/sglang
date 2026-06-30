"""Unit tests for srt/configs/mamba_utils.py

CPU-only, no server, no model loading. Covers every branch of the pure-logic
helpers in mamba_utils: dtype resolution, head-shard group math, and the
Mamba2 / KimiLinear state-shape + cache-size computations.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
    extra_groups_for_head_shards,
    mamba2_state_dtype,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_LOGGER_NAME = "sglang.srt.configs.mamba_utils"


class TestExtraGroupsForHeadShards(unittest.TestCase):
    def test_divisible_returns_zero(self):
        self.assertEqual(extra_groups_for_head_shards(4, 2), 0)
        self.assertEqual(extra_groups_for_head_shards(8, 4), 0)

    def test_single_group(self):
        # documented case: for n_groups == 1 this is exactly tp_size - 1
        self.assertEqual(extra_groups_for_head_shards(1, 4), 3)
        self.assertEqual(extra_groups_for_head_shards(1, 8), 7)

    def test_non_divisible_below_tp(self):
        self.assertEqual(extra_groups_for_head_shards(2, 4), 2)

    def test_nondiv_ngroups_gt_tp_current_behavior(self):
        # Documents CURRENT behavior; may need clarification. For n_groups >
        # tp_size and not divisible, this returns a NEGATIVE number (the inline
        # comment in the source only justifies the n_groups == 1 case). Not
        # asserting this is "correct" -- just locking current behavior. The PR
        # should ask: is extra_groups_for_head_shards(3, 2) == -1 intended?
        self.assertEqual(extra_groups_for_head_shards(3, 2), -1)
        self.assertEqual(extra_groups_for_head_shards(6, 4), -2)


class TestMamba2StateDtype(unittest.TestCase):
    def test_all_defaults(self):
        with envs.SGLANG_MAMBA_CONV_DTYPE.override(
            "bfloat16"
        ), envs.SGLANG_MAMBA_SSM_DTYPE.override(None):
            dt = mamba2_state_dtype(config=None)
        self.assertEqual(dt.conv, torch.bfloat16)
        self.assertEqual(dt.temporal, torch.float32)

    def test_conv_dtype_valid_env(self):
        with envs.SGLANG_MAMBA_CONV_DTYPE.override(
            "float16"
        ), envs.SGLANG_MAMBA_SSM_DTYPE.override(None):
            dt = mamba2_state_dtype(config=None)
        self.assertEqual(dt.conv, torch.float16)

    def test_conv_dtype_invalid_env_falls_back(self):
        # invalid conv dtype -> dict.get fallback to bfloat16 (silent, no warning)
        with envs.SGLANG_MAMBA_CONV_DTYPE.override(
            "int8"
        ), envs.SGLANG_MAMBA_SSM_DTYPE.override(None):
            dt = mamba2_state_dtype(config=None)
        self.assertEqual(dt.conv, torch.bfloat16)

    def test_ssm_from_root_config(self):
        # mamba_ssm_dtype on the root config (text-model path, not text_config)
        cfg = SimpleNamespace(mamba_ssm_dtype="bfloat16")
        with envs.SGLANG_MAMBA_SSM_DTYPE.override(None):
            dt = mamba2_state_dtype(cfg)
        self.assertEqual(dt.temporal, torch.bfloat16)

    def test_ssm_from_vl_text_config(self):
        cfg = SimpleNamespace(text_config=SimpleNamespace(mamba_ssm_dtype="float16"))
        with envs.SGLANG_MAMBA_SSM_DTYPE.override(None):
            dt = mamba2_state_dtype(cfg)
        self.assertEqual(dt.temporal, torch.float16)

    def test_ssm_config_without_attrs_uses_default(self):
        cfg = SimpleNamespace()
        with envs.SGLANG_MAMBA_SSM_DTYPE.override(None):
            dt = mamba2_state_dtype(cfg)
        self.assertEqual(dt.temporal, torch.float32)

    def test_ssm_invalid_config_warns_and_defaults(self):
        cfg = SimpleNamespace(mamba_ssm_dtype="int8")
        with envs.SGLANG_MAMBA_SSM_DTYPE.override(None):
            with self.assertLogs(_LOGGER_NAME, level="WARNING") as cm:
                dt = mamba2_state_dtype(cfg)
        self.assertEqual(dt.temporal, torch.float32)
        self.assertTrue(any("in config" in line for line in cm.output))

    def test_text_config_missing_attr_falls_to_root(self):
        # text_config exists but lacks mamba_ssm_dtype -> the `and` is False ->
        # elif reads root mamba_ssm_dtype.
        cfg = SimpleNamespace(text_config=SimpleNamespace(), mamba_ssm_dtype="bfloat16")
        with envs.SGLANG_MAMBA_SSM_DTYPE.override(None):
            dt = mamba2_state_dtype(cfg)
        self.assertEqual(dt.temporal, torch.bfloat16)

    def test_env_overrides_config(self):
        cfg = SimpleNamespace(mamba_ssm_dtype="bfloat16")
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("float32"):
            dt = mamba2_state_dtype(cfg)
        self.assertEqual(dt.temporal, torch.float32)  # env wins over config

    def test_env_unset_keeps_config(self):
        cfg = SimpleNamespace(mamba_ssm_dtype="float16")
        with envs.SGLANG_MAMBA_SSM_DTYPE.override(None):
            dt = mamba2_state_dtype(cfg)
        self.assertEqual(dt.temporal, torch.float16)

    def test_ssm_invalid_env_no_config_stays_default(self):
        # config=None: ssm_dtype was never moved off its float32 default, so the
        # warn-only invalid-env branch leaves it at float32. (It "stays default"
        # because the default already WAS float32 -- not because env reset it;
        # see the config case below.)
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("badval"):
            with self.assertLogs(_LOGGER_NAME, level="WARNING") as cm:
                dt = mamba2_state_dtype(config=None)
        self.assertEqual(dt.temporal, torch.float32)
        self.assertTrue(any("environment variable" in line for line in cm.output))

    def test_invalid_env_keeps_config_value_current_behavior(self):
        # Documents CURRENT behavior + a message/behavior mismatch: an invalid
        # env value only logs a warning, it does NOT reset ssm_dtype. So a
        # config value applied earlier is RETAINED -- even though the warning
        # text says "Using default 'float32'." PR should flag the misleading
        # message.
        cfg = SimpleNamespace(mamba_ssm_dtype="bfloat16")
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("badval"):
            with self.assertLogs(_LOGGER_NAME, level="WARNING"):
                dt = mamba2_state_dtype(cfg)
        self.assertEqual(dt.temporal, torch.bfloat16)  # config value kept, NOT f32


class TestMamba2StateShape(unittest.TestCase):
    def test_divisible_groups(self):
        s = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=4096,
            n_groups=8,
            num_heads=32,
            head_dim=128,
            state_size=128,
            conv_kernel=4,
        )
        # conv_dim = 4096 + 2*8*128 = 6144
        self.assertEqual(s.conv_dim, 6144)
        self.assertEqual(s.conv, [(6144, 3)])
        self.assertEqual(s.temporal, (32, 128, 128))
        self.assertEqual(s.ssm_state_size, 128)
        self.assertEqual(s.num_heads, 32)
        self.assertEqual(s.head_dim, 128)
        self.assertEqual(s.conv_kernel, 4)

    def test_non_divisible_groups_extended(self):
        s = Mamba2StateShape.create(
            tp_world_size=2,
            intermediate_size=8,
            n_groups=1,
            num_heads=4,
            head_dim=4,
            state_size=2,
            conv_kernel=4,
        )
        # n_groups 1 -> +1 extra -> 2 ; conv_dim = 8 + 2*2*2 = 16
        self.assertEqual(s.conv_dim, 16)
        self.assertEqual(s.conv, [(8, 3)])
        self.assertEqual(s.temporal, (2, 4, 2))

    def test_non_divisible_conv_dim_raises(self):
        # conv_dim = 8 + 2*3*2 = 20 ; divide(20, 3) asserts (the first divide).
        with self.assertRaises(AssertionError):
            Mamba2StateShape.create(
                tp_world_size=3,
                intermediate_size=8,
                n_groups=3,
                num_heads=4,
                head_dim=4,
                state_size=2,
                conv_kernel=4,
            )

    def test_non_divisible_num_heads_raises(self):
        # conv_dim = 6 + 2*3*2 = 18 is divisible by tp=3 (conv divide passes),
        # so the assert fires specifically on divide(num_heads=4, tp=3).
        with self.assertRaises(AssertionError):
            Mamba2StateShape.create(
                tp_world_size=3,
                intermediate_size=6,
                n_groups=3,
                num_heads=4,
                head_dim=4,
                state_size=2,
                conv_kernel=4,
            )


class TestMamba2CacheParams(unittest.TestCase):
    @staticmethod
    def _shape():
        return Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=4096,
            n_groups=8,
            num_heads=32,
            head_dim=128,
            state_size=128,
            conv_kernel=4,
        )

    def test_cache_per_req_explicit_dtype(self):
        params = Mamba2CacheParams(
            shape=self._shape(),
            dtype=Mamba2StateDType(conv=torch.float16, temporal=torch.float32),
            layers=[0, 1],
        )
        # (6144*3*2 + 32*128*128*4) * 2 = (36864 + 2097152) * 2 = 4268032
        self.assertEqual(params.mamba_cache_per_req, 4268032)

    def test_default_dtype_matches_explicit(self):
        # omitting dtype exercises the default_factory (mamba2_state_dtype(None));
        # with env at defaults it resolves to conv=bfloat16, temporal=float32.
        with envs.SGLANG_MAMBA_CONV_DTYPE.override(
            "bfloat16"
        ), envs.SGLANG_MAMBA_SSM_DTYPE.override(None):
            defaulted = Mamba2CacheParams(shape=self._shape(), layers=[0])
        explicit = Mamba2CacheParams(
            shape=self._shape(),
            dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
            layers=[0],
        )
        self.assertEqual(defaulted.dtype, explicit.dtype)
        self.assertEqual(defaulted.mamba_cache_per_req, explicit.mamba_cache_per_req)
        self.assertGreater(defaulted.mamba_cache_per_req, 0)


class TestKimiLinearStateShape(unittest.TestCase):
    def test_defaults_mirror_heads(self):
        s = KimiLinearStateShape.create(
            tp_world_size=1,
            num_heads=4,
            head_dim=8,
        )
        # num_k_heads/head_k_dim default to num_heads/head_dim
        self.assertEqual(s.num_k_heads, 4)
        self.assertEqual(s.head_k_dim, 8)
        # proj=32, proj_k=32 ; conv pre=(32,3),(32,3) -> reshuffle (3, 32+32*2)
        self.assertEqual(s.conv, [(3, 96)])
        self.assertEqual(s.temporal, (4, 8, 8))
        self.assertEqual(s.conv_kernel, 4)
        self.assertEqual(s.num_spec, 0)

    def test_explicit_k_heads_with_tp(self):
        s = KimiLinearStateShape.create(
            tp_world_size=2,
            num_heads=8,
            head_dim=8,
            num_k_heads=4,
            head_k_dim=4,
            conv_kernel_size=3,
            num_spec=2,
        )
        # proj=64 -> /2 =32 ; proj_k=16 -> /2 =8 ; reshuffle (2, 32+8*2)
        self.assertEqual(s.conv, [(2, 48)])
        self.assertEqual(s.temporal, (4, 8, 8))
        self.assertEqual(s.num_spec, 2)

    def test_non_divisible_raises(self):
        with self.assertRaises(AssertionError):
            KimiLinearStateShape.create(
                tp_world_size=3,
                num_heads=4,
                head_dim=8,
            )  # proj_size = 32, divide(32, 3) asserts


class TestKimiLinearCacheParams(unittest.TestCase):
    def test_cache_per_req(self):
        shape = KimiLinearStateShape.create(
            tp_world_size=1,
            num_heads=4,
            head_dim=8,
        )
        params = KimiLinearCacheParams(
            shape=shape,
            dtype=Mamba2StateDType(conv=torch.float16, temporal=torch.float32),
            layers=[0],
        )
        # (3*96*2 + 4*8*8*4) * 1 = (576 + 1024) = 1600
        self.assertEqual(params.mamba_cache_per_req, 1600)


if __name__ == "__main__":
    unittest.main()
