"""Schema checks for the tuned W8A8 block-FP8 GEMM tables.

The tables under ``sglang/kernels/ops/gemm/configs`` are runtime data read by
``get_w8a8_block_fp8_configs``.  A malformed or misnamed file does not fail
loudly -- the lookup misses and the kernel silently falls back to the default
tile config, which is exactly the state these tables exist to avoid.  These
checks run on CPU so ``run-ci`` can catch a broken table without a GPU.

Scope note: the directory also holds int8 and channelwise FP8 tables that this
loader never opens, so every check below is restricted to files whose name
carries ``dtype=fp8_w8a8`` with a ``block_shape``.
"""

import json
import os
import re
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")

# Mirrors `get_w8a8_block_fp8_configs` in kernels/ops/gemm/fp8_kernel.py: the
# loader builds this exact name and joins it with the module's own `configs`
# directory, so a file that does not match is never opened.
CONFIGS_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "python",
        "sglang",
        "kernels",
        "ops",
        "gemm",
        "configs",
    )
)
BLOCK_FP8_RE = re.compile(
    r"^N=(?P<N>\d+),K=(?P<K>\d+),device_name=(?P<device>.+),"
    r"dtype=fp8_w8a8,block_shape=\[(?P<bn>\d+), (?P<bk>\d+)\]\.json$"
)
# The loader reads these unconditionally when it builds the kernel.
REQUIRED_KEYS = (
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
)


def _block_fp8_files():
    """Every tuned block-FP8 table, paired with its parsed filename."""
    found = []
    for name in sorted(os.listdir(CONFIGS_DIR)):
        match = BLOCK_FP8_RE.match(name)
        if match is not None:
            found.append((os.path.join(CONFIGS_DIR, name), match))
    return found


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class TestW8A8BlockFP8Configs(CustomTestCase):
    def test_configs_dir_exists(self):
        self.assertTrue(
            os.path.isdir(CONFIGS_DIR),
            f"configs directory not found at {CONFIGS_DIR}; the loader reads "
            "it relative to kernels/ops/gemm/fp8_kernel.py",
        )

    def test_block_fp8_tables_are_present(self):
        self.assertTrue(
            _block_fp8_files(), f"no block-FP8 tables found under {CONFIGS_DIR}"
        )

    def test_every_entry_has_the_fields_the_loader_reads(self):
        for path, _ in _block_fp8_files():
            data = _load(path)
            name = os.path.basename(path)
            self.assertTrue(data, f"{name} is empty")
            for m_key, cfg in data.items():
                self.assertEqual(
                    str(int(m_key)), m_key, f"{name}: key {m_key!r} is not an M value"
                )
                for field in REQUIRED_KEYS:
                    self.assertIn(field, cfg, f"{name}: M={m_key} lacks {field}")

    def test_block_sizes_satisfy_the_scale_stepping_invariant(self):
        # fp8_utils steps scales along K, so a BLOCK_SIZE_K that does not
        # divide ``block_k`` is clamped up at load time on CUDA -- the tuned
        # value would silently be replaced by one nobody benchmarked.  A
        # smaller-but-dividing BLOCK_SIZE_K is kept as written and is legal
        # (existing ROCm tables do it), so only divisibility is required here.
        for path, match in _block_fp8_files():
            block_k = int(match["bk"])
            name = os.path.basename(path)
            for m_key, cfg in _load(path).items():
                where = f"{name}: M={m_key}"
                self.assertEqual(
                    block_k % cfg["BLOCK_SIZE_K"],
                    0,
                    f"{where}: block_k must be a multiple of BLOCK_SIZE_K",
                )
                self.assertGreaterEqual(cfg["BLOCK_SIZE_M"], 16, where)
                self.assertGreaterEqual(cfg["BLOCK_SIZE_N"], 16, where)
                self.assertGreaterEqual(cfg["num_warps"], 1, where)
                self.assertGreaterEqual(cfg["num_stages"], 1, where)


class TestH20Block32Tables(CustomTestCase):
    """The H20 ``[32, 32]`` tables added for the DeepSeek-V4.x GEMM shapes."""

    SHAPES = [
        (512, 5120),
        (576, 5120),
        (1280, 5120),
        (1536, 5120),
        (1792, 5120),
        (4096, 1280),
        (5120, 288),
        (5120, 1024),
        (5120, 15360),
        (25600, 6144),
    ]
    M_TIERS = 27
    MAX_M = 8192

    def _path(self, n, k):
        return os.path.join(
            CONFIGS_DIR,
            f"N={n},K={k},device_name=NVIDIA_H20,dtype=fp8_w8a8,"
            "block_shape=[32, 32].json",
        )

    def test_all_shapes_present(self):
        missing = [
            f"N={n},K={k}"
            for n, k in self.SHAPES
            if not os.path.isfile(self._path(n, k))
        ]
        self.assertEqual(missing, [], f"H20 block-[32,32] tables missing for {missing}")

    def test_m_tiers_form_one_ascending_grid(self):
        # The lookup picks the closest M, so the grid is deliberately
        # irregular; what matters is that it is strictly ascending, covers
        # small M densely, and reaches the prefill sizes this kernel sees.
        for n, k in self.SHAPES:
            data = _load(self._path(n, k))
            tiers = sorted(int(m) for m in data)
            where = f"N={n},K={k}"
            self.assertEqual(len(tiers), self.M_TIERS, f"{where}: {tiers}")
            self.assertEqual(tiers[0], 1, where)
            self.assertEqual(tiers[-1], self.MAX_M, where)
            self.assertEqual(tiers, sorted(set(tiers)), f"{where}: duplicate tiers")

    def test_block_size_k_matches_the_tuned_block_k(self):
        # These tables were tuned for block_k = 32; a value below it would be
        # legal but would be used unscaled, so pin what was actually measured.
        for n, k in self.SHAPES:
            for m_key, cfg in _load(self._path(n, k)).items():
                self.assertEqual(
                    cfg["BLOCK_SIZE_K"],
                    32,
                    f"N={n},K={k}: M={m_key} was not tuned at block_k=32",
                )

    def test_at_most_one_dispatch_hint_per_entry(self):
        # SWAP_AB and SPLIT_K select different kernel variants; setting both
        # has no defined meaning.  (Some older tables in this directory do set
        # both -- these ten must not.)
        for n, k in self.SHAPES:
            for m_key, cfg in _load(self._path(n, k)).items():
                self.assertFalse(
                    bool(cfg.get("SWAP_AB")) and int(cfg.get("SPLIT_K", 1)) > 1,
                    f"N={n},K={k}: M={m_key} sets both SWAP_AB and SPLIT_K",
                )


if __name__ == "__main__":
    unittest.main()
