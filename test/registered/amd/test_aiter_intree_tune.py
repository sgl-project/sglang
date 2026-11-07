"""In-tree aiter#6004 / #6029 / #6030 (aiter_intree_tune).

Guards: the in-tree rows / JSON / mHC policy apply only while aiter ships the pre-PR
config; once aiter has the PR (simulated by handing aiter a table that already has the
PR rows) nothing is installed, aiter's merge does not see duplicate keys, and the
effective table is the PR's either way. The mHC tile_m=16 policy is bit-identical.
"""

import glob
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import pandas as pd

from sglang.srt.models.deepseek_common.amd import aiter_intree_tune as it
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")

try:
    import aiter  # noqa: F401
    from aiter.jit.core import AITER_CONFIGS, AITER_ROOT_DIR

    _HAS_AITER = True
except Exception:  # noqa: BLE001
    _HAS_AITER = False


def _intree():
    return pd.read_csv(it.INTREE_BPRESHUFFLE_CSV)


def _as_stock(intree):
    """The rows as aiter ships them before the PRs."""
    stock = intree.copy()
    for c in it._KERNEL_COLS:
        stock[c] = stock["base_" + c]
    return stock.drop(columns=it._INTREE_ONLY_COLS)


def _keys(df):
    return list(zip(*(df[c].astype(str) for c in it._KEY_COLS)))


def _kernels_by_key(df):
    return {k: it._norm_kernel(r) for k, (_, r) in zip(_keys(df), df.iterrows())}


class TestBpreshuffleRows(CustomTestCase):
    def test_rows_match_prs(self):
        df = _intree()
        self.assertEqual(len(df), 32)
        self.assertEqual(len(set(_keys(df))), 32)
        pr6004 = df[df.aiter_pr == 6004]
        self.assertEqual(len(pr6004), 8)
        self.assertTrue(((pr6004.N == 2048) & (pr6004.K == 7168)).all())
        pr6029 = df[df.aiter_pr == 6029]
        self.assertEqual(len(pr6029), 24)
        self.assertEqual(set(zip(pr6029.N, pr6029.K)), {(8192, 1536), (7168, 2048)})
        # only kernel choices change, never to a CK instance aiter may not have built
        self.assertTrue(df.libtype.isin(["triton", "asm"]).all())

    def _other(self):
        return pd.DataFrame(
            [["gfx950", 256, 1, 4096, 7168, "triton", 0, 0, 1.0, "triton", 1, 1, 0.0]],
            columns=[c for c in _as_stock(_intree()).columns],
        )

    def test_stock_aiter_rows_replaced_in_place(self):
        stock = pd.concat([_as_stock(_intree()), self._other()], ignore_index=True)
        out, stats = it.apply_bpreshuffle_rows(stock, _intree())
        self.assertEqual(stats[it.APPLIED], 32)
        self.assertEqual(_keys(out), _keys(stock))  # same keys, same order
        self.assertFalse(out.duplicated(subset=it._KEY_COLS).any())
        want = _kernels_by_key(_intree())
        got = _kernels_by_key(out)
        for k, v in want.items():
            self.assertEqual(got[k], v)

    def test_merged_aiter_is_left_alone(self):
        merged = _intree().drop(columns=it._INTREE_ONLY_COLS)
        out, stats = it.apply_bpreshuffle_rows(merged, _intree())
        self.assertEqual(stats[it.APPLIED], 0)
        self.assertEqual(stats[it.UPSTREAM], 32)
        pd.testing.assert_frame_equal(out, merged)

    def test_superseded_or_missing_rows_prefer_aiter(self):
        stock = _as_stock(_intree())
        stock.loc[0, "libtype"] = "asm"
        stock.loc[0, "kernelName"] = "some_newer_kernel"
        stock = stock.drop(index=1).reset_index(drop=True)
        out, stats = it.apply_bpreshuffle_rows(stock, _intree())
        self.assertEqual(stats[it.SUPERSEDED], 1)
        self.assertEqual(stats[it.MISSING], 1)
        self.assertEqual(stats[it.APPLIED], 30)
        self.assertEqual(out.loc[0, "kernelName"], "some_newer_kernel")


@unittest.skipUnless(_HAS_AITER, "aiter not installed")
class TestBpreshuffleWithAiter(CustomTestCase):
    """Runs aiter's real table resolution (default CSV + model_configs merge)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sglang_intree_test_")
        # keep aiter's merged-table output private to this test
        self._tempdir = tempfile.tempdir
        tempfile.tempdir = self.tmp
        self._env = os.environ.pop(it.BPRESHUFFLE_ENV, None)
        self._clear()

    def tearDown(self):
        os.environ.pop(it.BPRESHUFFLE_ENV, None)
        if self._env is not None:
            os.environ[it.BPRESHUFFLE_ENV] = self._env
        tempfile.tempdir = self._tempdir
        self._clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def _clear():
        type(AITER_CONFIGS).get_config_file.cache_clear()

    def _effective(self):
        path = AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE
        df = pd.read_csv(path)
        self.assertFalse(df.duplicated(subset=it._KEY_COLS).any())
        return path, _kernels_by_key(df)

    def _assert_pr_rows(self, kernels):
        for k, v in _kernels_by_key(_intree()).items():
            self.assertEqual(kernels[k], v, k)

    def test_install_on_pinned_aiter(self):
        report = it.IntreeReport()
        it.install_bpreshuffle_rows(report, out_dir=os.path.join(self.tmp, "out"))
        stats = report.csv_rows
        self.assertEqual(stats[it.SUPERSEDED] + stats[it.MISSING], 0, stats)
        self.assertEqual(stats[it.APPLIED] + stats[it.UPSTREAM], 32, stats)
        path, kernels = self._effective()
        self.assertEqual(path, report.csv_file)
        if stats[it.APPLIED]:
            self.assertEqual(os.environ[it.BPRESHUFFLE_ENV], path)
        self._assert_pr_rows(kernels)
        # idempotent: a second process inheriting the env changes nothing
        again = it.IntreeReport()
        it.install_bpreshuffle_rows(again, out_dir=os.path.join(self.tmp, "out"))
        self.assertEqual(again.csv_rows[it.APPLIED], 0)
        self.assertEqual(again.csv_file, path)

    def test_aiter_with_prs_merged(self):
        """Simulate aiter having merged #6004/#6029: its tables carry the PR rows."""
        cfg = os.path.join(AITER_ROOT_DIR, "aiter", "configs")
        name = "a8w8_blockscale_bpreshuffle_tuned_gemm"
        files = [os.path.join(cfg, f"{name}.csv")] + sorted(
            p
            for p in glob.glob(os.path.join(cfg, "model_configs", f"*{name}*.csv"))
            if "untuned" not in os.path.basename(p)
        )
        new = {k: r for k, (_, r) in zip(_keys(_intree()), _intree().iterrows())}
        copies, patched = [], 0
        for p in files:
            df = pd.read_csv(p)
            if "gfx" in df.columns:
                for i, k in enumerate(_keys(df)):
                    if k in new:
                        for c in it._KERNEL_COLS:
                            df.at[df.index[i], c] = new[k][c]
                        patched += 1
            dst = os.path.join(self.tmp, os.path.basename(p))
            df.to_csv(dst, index=False)
            copies.append(dst)
        self.assertEqual(patched, 32)
        os.environ[it.BPRESHUFFLE_ENV] = os.pathsep.join(copies)
        self._clear()

        report = it.IntreeReport()
        it.install_bpreshuffle_rows(report, out_dir=os.path.join(self.tmp, "out"))
        self.assertEqual(report.csv_rows[it.APPLIED], 0)
        self.assertEqual(report.csv_rows[it.UPSTREAM], 32)
        # nothing installed: aiter keeps resolving its own (merged) list
        self.assertEqual(os.environ[it.BPRESHUFFLE_ENV], os.pathsep.join(copies))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "out")))
        _, kernels = self._effective()
        self._assert_pr_rows(kernels)


class TestTritonJson(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sglang_intree_json_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _ours(self, name):
        with open(os.path.join(it._INTREE_TRITON_DIR, name)) as f:
            return json.load(f)

    def test_select(self):
        names = sorted(it.INTREE_TRITON_JSON)
        self.assertEqual(len(names), 3)
        stock = {"M_LEQ_1": {"BLOCK_SIZE_M": 8}}
        digests = {
            n: (pr, it._json_digest(stock))
            for n, (pr, _) in it.INTREE_TRITON_JSON.items()
        }
        contents = {
            names[0]: stock,  # pre-PR -> in-tree applies
            names[1]: self._ours(names[1]),  # aiter merged the PR
            # names[2] absent: aiter dropped / renamed the file
        }
        for n, obj in contents.items():
            with open(os.path.join(self.tmp, n), "w") as f:
                json.dump(obj, f, indent=1)
        report = it.IntreeReport()
        with mock.patch.dict(it.INTREE_TRITON_JSON, digests):
            overrides = it.select_triton_overrides(self.tmp, report)
        self.assertEqual(
            report.triton_json,
            {names[0]: it.APPLIED, names[1]: it.UPSTREAM, names[2]: it.MISSING},
        )
        self.assertEqual(
            overrides, {os.path.join(self.tmp, names[0]): self._ours(names[0])}
        )
        # re-tuned upstream differently -> aiter wins
        with open(os.path.join(self.tmp, names[0]), "w") as f:
            json.dump({"M_LEQ_1": {"BLOCK_SIZE_M": 64}}, f)
        report = it.IntreeReport()
        with mock.patch.dict(it.INTREE_TRITON_JSON, digests):
            self.assertEqual(it.select_triton_overrides(self.tmp, report), {})
        self.assertEqual(report.triton_json[names[0]], it.SUPERSEDED)

    @unittest.skipUnless(_HAS_AITER, "aiter not installed")
    def test_lookup_through_aiter(self):
        import aiter.ops.triton.utils.gemm_config_utils as gcu
        from aiter.ops.triton.utils._triton import arch_info

        try:
            arch = arch_info.get_arch()
        except Exception:  # noqa: BLE001
            arch = None
        if arch != it.TRITON_ARCH:
            self.skipTest("gfx950 only")
        orig = gcu.load_config_json
        report = it.IntreeReport()
        try:
            it.install_triton_json(report)
            self.assertEqual(len(report.triton_json), 3)
            for name, status in report.triton_json.items():
                self.assertIn(status, (it.APPLIED, it.UPSTREAM), name)
                n, k = (
                    int(x.split("=")[1])
                    for x in name[len(it.TRITON_CONFIG_NAME) + 1 : -5].split("-")
                )
                ours = self._ours(name)
                for bucket in (b for b in ours if b.startswith("M_LEQ_")):
                    m = int(bucket[len("M_LEQ_") :])
                    got, tuned = gcu.get_gemm_config(
                        it.TRITON_CONFIG_NAME, m, n, k, backend="triton"
                    )
                    self.assertTrue(tuned)
                    self.assertEqual(got, ours[bucket], (name, bucket))
        finally:
            gcu.load_config_json = orig
            gcu._get_gemm_config_cached.cache_clear()


def _stock_gfx950_policy(m, hidden_size, w_preshuffle_bf16=False, res_preshuffle=False):
    from aiter.ops.mhc import _mhc_fused_config_gfx950_256

    return _mhc_fused_config_gfx950_256(m, hidden_size, 256)


@unittest.skipUnless(_HAS_AITER, "aiter not installed")
class TestMhcTileM16(CustomTestCase):
    def test_policy(self):
        self.assertTrue(it.mhc_needs_tile_m16(_stock_gfx950_policy, "gfx950"))
        self.assertFalse(it.mhc_needs_tile_m16(_stock_gfx950_policy, "gfx942"))
        patched = it.make_mhc_tile_m16_config(_stock_gfx950_policy)
        self.assertFalse(it.mhc_needs_tile_m16(patched, "gfx950"))  # i.e. "upstream"
        saw_32 = False
        for m in list(range(1, 1025, 7)) + [1024, 1025, 2048, 4096]:
            for bf16 in (False, True):
                old = _stock_gfx950_policy(m, 7168, bf16)
                new = patched(m, 7168, w_preshuffle_bf16=bf16, res_preshuffle=False)
                saw_32 |= old[1] == 32 and not bf16 and m <= 1024
                want_m = 16 if (not bf16 and m <= 1024) else old[1]
                self.assertEqual(new, (old[0], want_m, old[2], old[3]), (m, bf16))
        self.assertTrue(saw_32)

    @unittest.skipUnless(__import__("torch").cuda.is_available(), "needs a GPU")
    def test_bit_identical_on_gpu(self):
        import aiter.ops.mhc as aiter_mhc
        import torch

        if aiter_mhc.get_gfx_runtime() != "gfx950":
            self.skipTest("gfx950 only")
        orig = aiter_mhc.get_mhc_fused_post_pre_config
        stock = getattr(orig, "__wrapped__", orig)
        hc, hidden = 4, 7168
        hc3 = hc * 2 + hc * hc
        g = torch.Generator(device="cuda").manual_seed(0)

        def rnd(*shape, dtype=torch.float32, scale=1.0):
            return (torch.randn(*shape, generator=g, device="cuda") * scale).to(dtype)

        try:
            for m in (80, 144, 192, 384, 640):
                if stock(m, hidden)[1] == 16:
                    continue  # this aiter already picks 16 here
                args = (
                    rnd(m, hidden, dtype=torch.bfloat16),
                    rnd(m, hc, hidden, dtype=torch.bfloat16),
                    rnd(m, hc, 1),
                    rnd(m, hc, hc),
                    rnd(hc3, hc * hidden),
                    rnd(3, scale=0.1),
                    rnd(hc3, scale=0.1),
                )
                outs = []
                for fn in (stock, it.make_mhc_tile_m16_config(stock)):
                    aiter_mhc.get_mhc_fused_post_pre_config = fn
                    outs.append(
                        aiter_mhc.mhc_fused_post_pre(*args, hc_post_mult_value=2.0)
                    )
                for a, b in zip(*outs):
                    self.assertTrue(torch.equal(a, b), m)
        finally:
            aiter_mhc.get_mhc_fused_post_pre_config = orig


if __name__ == "__main__":
    unittest.main()
