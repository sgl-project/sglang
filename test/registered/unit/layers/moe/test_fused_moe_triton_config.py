"""CPU unit tests for M_eff-driven Triton MoE tile configuration."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import triton

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.triton_utils import (
    fused_moe_triton_config as fmtc,
)
from sglang.srt.layers.moe.moe_runner.triton_utils import (
    override_config_lookup,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    get_default_config,
    get_moe_configs,
    try_get_optimal_moe_config,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Qwen3-30B-A3B at TP1: E=128, topk=8, N=768, H=2048.
QWEN3_MOE = dict(E=128, N=768, K=2048, topk=8)


def _default_config(M, dtype=None, is_marlin=False, block_shape=None, **overrides):
    shape = {**QWEN3_MOE, **overrides}
    return get_default_config(
        M=M,
        E=shape["E"],
        N=shape["N"],
        K=shape["K"],
        topk=shape["topk"],
        dtype=dtype,
        is_marlin=is_marlin,
        block_shape=block_shape,
    )


def _is_power_of_two(n: int) -> bool:
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


class TestFusedMoeTritonConfig(CustomTestCase):
    def setUp(self):
        get_moe_configs.cache_clear()
        self._ctx = get_context().override_server_args(
            enable_deterministic_inference=False
        )
        self._ctx.install()

    def tearDown(self):
        self._ctx.restore()
        get_moe_configs.cache_clear()

    def _ep(self, ep_size: int, a2a_none: bool = True):
        return (
            patch.object(
                fmtc, "get_parallel", return_value=SimpleNamespace(moe_ep_size=ep_size)
            ),
            patch.object(
                fmtc,
                "get_moe_a2a_backend",
                return_value=SimpleNamespace(is_none=lambda: a2a_none),
            ),
        )

    def test_block_size_m_boundaries(self):
        # M * topk / E rows per expert: <=4 -> 16, <=16 -> 32, else 64.
        cases = [(64, 16), (65, 32), (256, 32), (257, 64)]
        for dtype, block_shape in (
            (None, None),
            ("fp8_w8a8", None),
            ("fp8_w8a8", [128, 128]),
        ):
            for M, block_m in cases:
                with self.subTest(dtype=dtype, block_shape=block_shape, M=M):
                    cfg = _default_config(M=M, dtype=dtype, block_shape=block_shape)
                    self.assertEqual(cfg["BLOCK_SIZE_M"], block_m)

    def test_small_m_fp8_block_wise_enables_sm90_swap_ab(self):
        # should_enable_swap_ab needs BLOCK_SIZE_M < 64 and BLOCK_SIZE_N >= 64.
        for M in (1, 16, 64, 128, 256):
            cfg = _default_config(M=M, dtype="fp8_w8a8", block_shape=[128, 128])
            self.assertLess(cfg["BLOCK_SIZE_M"], 64)
            self.assertGreaterEqual(cfg["BLOCK_SIZE_N"], 64)

    def test_block_wise_k_tile_stays_in_one_scale_group(self):
        """Block-wise quantized MoE must use a K tile that divides its scale group."""
        for dtype in ("fp8_w8a8", "int8_w8a8"):
            for block_shape in ([128, 128], [128, 64], [128, 32]):
                for M in (1, 64, 256, 257, 4096):
                    with self.subTest(dtype=dtype, block_shape=block_shape, M=M):
                        cfg = _default_config(M=M, dtype=dtype, block_shape=block_shape)
                        self.assertEqual(block_shape[1] % cfg["BLOCK_SIZE_K"], 0)
                        # tl.dot needs K >= 32 for INT8 operands.
                        self.assertGreaterEqual(cfg["BLOCK_SIZE_K"], 32)

    def test_block_wise_int8_rejects_scale_group_below_dot_minimum(self):
        with self.assertRaisesRegex(ValueError, "block_shape\\[1\\] >= 32"):
            _default_config(M=64, dtype="int8_w8a8", block_shape=[128, 16])

    def test_float32_keeps_pre_meff_default_tiles(self):
        """FP32 MoE must keep the small-tile defaults that fit 99 KiB shared memory."""
        for M in (1, 128, 4096):
            with self.subTest(M=M):
                cfg = _default_config(M=M, dtype="float32")
                if M <= QWEN3_MOE["E"]:
                    self.assertEqual(
                        (cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]),
                        (16, 32, 64),
                    )
                else:
                    self.assertEqual(
                        (cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]),
                        (64, 64, 32),
                    )

    def test_block_n_never_exceeds_small_n(self):
        for low_smem in (False, True):
            with patch.object(fmtc, "_use_low_smem_fp8_default", return_value=low_smem):
                for dtype in (None, "fp8_w8a8"):
                    with self.subTest(low_smem=low_smem, dtype=dtype):
                        cfg = _default_config(M=4096, dtype=dtype, N=64)
                        self.assertLessEqual(cfg["BLOCK_SIZE_N"], 64)

    def test_marlin_small_batch_guard(self):
        cfg = _default_config(M=32, is_marlin=True, E=8, topk=2)
        self.assertEqual((cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"]), (16, 32))
        cfg = _default_config(M=33, is_marlin=True, E=8, topk=2)
        self.assertNotEqual(cfg["BLOCK_SIZE_N"], 32)

    def test_ep_aware_standard_dispatcher_vs_a2a(self):
        # E_local=32 of 128 experts at EP=4, M=64, topk=8.
        std_parallel, std_a2a = self._ep(ep_size=4, a2a_none=True)
        with std_parallel, std_a2a:
            # Non-local slots are masked to -1: 512 / 128 = 4 rows per expert.
            self.assertEqual(_default_config(M=64, E=32)["BLOCK_SIZE_M"], 16)
        a2a_parallel, a2a_backend = self._ep(ep_size=4, a2a_none=False)
        with a2a_parallel, a2a_backend:
            # A2A delivers rank-local tokens only: 512 / 32 = 16 rows per expert.
            self.assertEqual(_default_config(M=64, E=32)["BLOCK_SIZE_M"], 32)

    def test_override_config_lookup_uses_meff_default(self):
        with (
            envs.SGLANG_MOE_CONFIG_DIR.override("/nonexistent_sglang_moe_dir_test"),
            patch.object(fmtc, "get_device_name", return_value="NVIDIA H200"),
        ):
            with override_config_lookup(num_tokens=512, top_k=8):
                cfg = try_get_optimal_moe_config(
                    w1_shape=(128, 1536, 2048),
                    w2_shape=(128, 2048, 768),
                    top_k=2,
                    dtype=None,
                    M=1,
                )
            self.assertEqual(cfg["BLOCK_SIZE_M"], 64)

    def test_deterministic_inference_ignores_meff_and_json(self):
        with get_context().override_server_args(enable_deterministic_inference=True):
            self.assertEqual(_default_config(M=1, dtype="fp8_w8a8")["BLOCK_SIZE_M"], 64)
            self.assertIsNone(get_moe_configs(E=128, N=768, dtype=None))

    def test_down_moe_block_size_m_alignment_invariant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ver_dir = (
                Path(tmpdir)
                / "configs"
                / f"triton_{triton.__version__.replace('.', '_')}"
            )
            ver_dir.mkdir(parents=True, exist_ok=True)
            entry = {
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 3,
            }
            (ver_dir / "E=128,N=768,device_name=NVIDIA_H200.json").write_text(
                json.dumps({"16": {"BLOCK_SIZE_M": 16, **entry}})
            )
            (ver_dir / "E=128,N=768,device_name=NVIDIA_H200_down.json").write_text(
                json.dumps({"16": {"BLOCK_SIZE_M": 64, "USE_TMA": True, **entry}})
            )

            with (
                envs.SGLANG_MOE_CONFIG_DIR.override(tmpdir),
                patch.object(fmtc, "get_device_name", return_value="NVIDIA H200"),
            ):
                config, (down_config, max_block_m) = try_get_optimal_moe_config(
                    w1_shape=(128, 1536, 2048),
                    w2_shape=(128, 2048, 768),
                    top_k=8,
                    dtype=None,
                    M=16,
                    return_down_config=True,
                )
                # Both GEMMs share one moe_align_block_size sort.
                self.assertEqual(down_config["BLOCK_SIZE_M"], config["BLOCK_SIZE_M"])
                self.assertEqual(max_block_m, 64)
                # The clamp must copy, not mutate, the lru_cached down config.
                cached_down = get_moe_configs(E=128, N=768, dtype=None, down_moe=True)
                self.assertEqual(cached_down[16]["BLOCK_SIZE_M"], 64)

    def test_missing_config_dir_returns_none(self):
        """A config dir without configs/ falls back to the default config."""
        with (
            envs.SGLANG_MOE_CONFIG_DIR.override("/nonexistent_sglang_moe_dir_test"),
            patch.object(fmtc, "get_device_name", return_value="NVIDIA H200"),
        ):
            self.assertIsNone(get_moe_configs(E=128, N=768, dtype=None))
            self.assertIsNone(get_moe_configs(E=128, N=768, dtype=None, down_moe=True))

    def test_empty_config_dir_env_resolves_against_cwd(self):
        """An explicitly empty SGLANG_MOE_CONFIG_DIR keeps reading ./configs/."""
        entry = {"16": {"BLOCK_SIZE_M": 16}}
        version_dir = f"triton_{triton.__version__.replace('.', '_')}"
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(fmtc, "get_device_name", return_value="NVIDIA H200"),
            envs.SGLANG_MOE_CONFIG_DIR.override(""),
        ):
            target = Path(tmp) / "configs" / version_dir
            target.mkdir(parents=True)
            name = fmtc.get_config_file_name(128, 768, None)
            (target / name).write_text(json.dumps(entry), encoding="utf-8")
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                self.assertEqual(
                    get_moe_configs(E=128, N=768, dtype=None), {16: entry["16"]}
                )
            finally:
                os.chdir(old_cwd)

    def test_h200_triton_3_8_0_configs_are_valid(self):
        configs_dir = Path(fmtc.__file__).resolve().parent / "configs" / "triton_3_8_0"
        h200_files = sorted(configs_dir.glob("*NVIDIA_H200*.json"))
        self.assertGreater(len(h200_files), 0)
        for path in h200_files:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertGreater(len(data), 0, f"Empty config map in {path.name}")
            for batch_key, entry in data.items():
                where = f"{path.name}[{batch_key}]"
                self.assertGreater(int(batch_key), 0, where)
                for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps"):
                    self.assertTrue(_is_power_of_two(entry[k]), f"{where}[{k}]")
                self.assertGreater(entry["GROUP_SIZE_M"], 0, where)
                self.assertGreater(entry["num_stages"], 0, where)
                if "block_shape=[128, 128]" in path.name:
                    # One scale per K tile: block_k must be divisible by BLOCK_SIZE_K.
                    self.assertEqual(128 % entry["BLOCK_SIZE_K"], 0, where)


if __name__ == "__main__":
    unittest.main()
