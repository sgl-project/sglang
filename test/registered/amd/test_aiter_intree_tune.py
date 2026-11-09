"""In-tree aiter rows and DeepSeek-V4's aiter patches apply only while aiter lacks them."""

import importlib.metadata
import json
import os
import re
import shutil
import tempfile
import unittest
from unittest import mock

import pandas as pd

from sglang.srt.models.deepseek_common.amd import aiter_intree_tune as it
from sglang.srt.utils import aiter_tuned_tables as tables
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd")

try:
    import aiter  # noqa: F401
    from aiter.jit.core import AITER_CONFIGS

    _HAS_AITER = True
except Exception:  # noqa: BLE001
    _HAS_AITER = False


class TestApplyRows(CustomTestCase):
    def test_rows_apply_only_while_aiter_lacks_them(self):
        """A row replaces aiter's only over its base kernel and adds only a missing shape,
        so aiter's merged or re-tuned rows win and no key repeats."""
        table = tables.A8W8_BLOCKSCALE_BPRESHUFFLE
        key = {"gfx": "gfx950", "cu_num": 256, "N": 2048, "K": 7168}
        ck = {"libtype": "ck", "kernelId": 7, "splitK": 0, "kernelName": "ck7"}
        ours = {"libtype": "triton", "kernelId": 0, "splitK": 0, "kernelName": "triton"}
        newer = {"libtype": "asm", "kernelId": 16, "splitK": 1, "kernelName": "newer"}
        base = {"base_" + c: v for c, v in ck.items()}
        aiter_rows = pd.DataFrame(
            [
                {**key, "M": 1, **ck},  # pre-PR -> replaced
                {**key, "M": 2, **ours},  # aiter merged the row
                {**key, "M": 3, **newer},  # aiter re-tuned the shape
                {**key, "M": 5, **ck},  # aiter has a row for an addition's key
            ]
        )
        intree = pd.DataFrame(
            [{**key, "M": m, **ours, **base} for m in (1, 2, 3, 4)]
            + [{**key, "M": m, **ours} for m in (5, 6)]
        )
        out, stats = tables.apply_intree_rows(
            aiter_rows, intree, table.key_cols, table.kernel_cols
        )
        self.assertEqual(
            stats,
            {
                tables.APPLIED: 1,
                tables.UPSTREAM: 1,
                tables.SUPERSEDED: 2,
                tables.MISSING: 1,
                tables.ADDED: 1,
            },
        )
        self.assertEqual(out["M"].tolist(), [1, 2, 3, 5, 6])
        self.assertEqual(
            out["kernelName"].tolist(), ["triton", "triton", "newer", "ck7", "triton"]
        )


def _write_rows(tmp, name, rows):
    path = os.path.join(tmp, name)
    pd.DataFrame(rows).to_csv(path, index=False)
    return [path]


class TestHipblasltRows(CustomTestCase):
    def test_another_rocm_skips_the_hipblaslt_rows(self):
        """ROCm 10's hipBLASLt rejects ROCm 7.2's solution indices at graph capture."""
        row = {"M": 1, "N": 50016, "K": 6144, "splitK": 0, "kernelName": ""}
        with tempfile.TemporaryDirectory() as tmp:
            csvs = _write_rows(
                tmp,
                "bf16.csv",
                [
                    {**row, "libtype": "hipblaslt", "solidx": 7, "rocm": "7.2"},
                    {**row, "M": 2, "libtype": "torch", "solidx": 0, "rocm": None},
                ],
            )
            for rocm, kept in (((7, 2, 26015), [1, 2]), ((10, 0, 0), [2])):
                with (
                    self.subTest(rocm=rocm),
                    mock.patch(
                        "sglang.srt.utils.device.get_hip_version", return_value=rocm
                    ),
                ):
                    out = tables.read_intree_rows(tables.GEMM_BF16, csvs)
                    self.assertEqual(out["M"].tolist(), kept)


def _aiter_commit() -> str:
    try:
        version = importlib.metadata.version("amd-aiter")
    except importlib.metadata.PackageNotFoundError:
        return ""
    match = re.search(r"\+g([0-9a-f]+)", version)
    return match.group(1) if match else ""


# the aiter the in-tree rows were checked against
_TABLES_AITER = "acf8fdf93"


@unittest.skipUnless(_HAS_AITER, "aiter not installed")
class TestInTreeTables(CustomTestCase):
    """On the pinned aiter a re-tuned (superseded) or dropped (missing) shape means the
    in-tree rows need updating."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sglang_intree_tables_test_")
        self._tempdir = tempfile.tempdir
        tempfile.tempdir = self.tmp
        self._env = {t.env: os.environ.pop(t.env, None) for t in tables.TABLES}
        type(AITER_CONFIGS).get_config_file.cache_clear()

    def tearDown(self):
        for env, value in self._env.items():
            os.environ.pop(env, None)
            if value is not None:
                os.environ[env] = value
        tempfile.tempdir = self._tempdir
        type(AITER_CONFIGS).get_config_file.cache_clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_rows_are_unique_and_resolve_on_the_pinned_aiter(self):
        pinned = _aiter_commit().startswith(_TABLES_AITER)
        for table in tables.TABLES:
            csvs = table.csv_paths()
            if not csvs:
                continue
            with self.subTest(table.name):
                rows = tables.read_intree_rows(table, csvs)
                self.assertFalse(rows.duplicated(subset=list(table.key_cols)).any())
                stats, path = tables.install_intree_table(
                    table, csvs, out_dir=os.path.join(self.tmp, "out")
                )
                self.assertEqual(path, getattr(AITER_CONFIGS, table.attr))
                if pinned:
                    # a key aiter repeats counts as missing, so this also rules it out
                    self.assertEqual(
                        stats[tables.SUPERSEDED] + stats[tables.MISSING], 0, stats
                    )


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
            {
                names[0]: tables.APPLIED,
                names[1]: tables.UPSTREAM,
                names[2]: tables.MISSING,
            },
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
        self.assertEqual(report.triton_json[names[0]], tables.SUPERSEDED)


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

    def test_wraps_an_aiter_without_preshuffled_layouts(self):
        """An aiter whose mHC config takes only (m, hidden_size) raised TypeError."""
        patched = it.make_mhc_tile_m16_config(lambda m, h: _stock_gfx950_policy(m, h))
        self.assertEqual(patched(8, 7168)[1], 16)
        self.assertEqual(patched(4096, 7168), _stock_gfx950_policy(4096, 7168))
        with self.assertRaises(TypeError):
            patched(8, 7168, w_preshuffle_bf16=True)


if __name__ == "__main__":
    unittest.main()
