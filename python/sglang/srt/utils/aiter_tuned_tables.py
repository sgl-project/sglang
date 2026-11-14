"""In-tree rows for aiter's tuned-config tables, for shapes the aiter pin lacks.

``aiter_tuned_configs/<table>/<model>.csv`` holds rows in one table's columns, keyed by exact
shape like aiter's ``model_configs/``, so they apply to whichever model runs that shape. A
table whose ``AITER_CONFIG_*`` the user set is left alone.
"""

from __future__ import annotations

import functools
import glob
import hashlib
import logging
import os
import tempfile
from typing import Optional, Sequence

import msgspec

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "aiter_tuned_configs"
)

APPLIED, UPSTREAM, SUPERSEDED, MISSING, ADDED = (
    "applied",
    "upstream",
    "superseded",
    "missing",
    "added",
)


class AiterTable(msgspec.Struct, frozen=True):
    """One aiter tuned-config table: its in-tree folder, ``AITER_CONFIG_*`` variable, and
    the columns that key a shape and name its kernel."""

    name: str
    env: str
    key_cols: tuple
    kernel_cols: tuple

    @property
    def attr(self) -> str:
        """The ``AITER_CONFIGS`` property that resolves the table."""
        return self.env + "_FILE"

    def csv_paths(self) -> list:
        return sorted(glob.glob(os.path.join(CONFIG_DIR, self.name, "*.csv")))


_GEMM_KERNEL_COLS = ("libtype", "kernelId", "splitK", "kernelName")

A8W8_BLOCKSCALE_BPRESHUFFLE = AiterTable(
    name="a8w8_blockscale_bpreshuffle",
    env="AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
    key_cols=("gfx", "cu_num", "M", "N", "K"),
    kernel_cols=_GEMM_KERNEL_COLS,
)
A8W8_BPRESHUFFLE = AiterTable(
    name="a8w8_bpreshuffle",
    env="AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE",
    key_cols=("gfx", "cu_num", "M", "N", "K", "q_dtype_w"),
    kernel_cols=_GEMM_KERNEL_COLS,
)
GEMM_BF16 = AiterTable(
    name="bf16",
    env="AITER_CONFIG_GEMM_BF16",
    key_cols=(
        "gfx",
        "cu_num",
        "M",
        "N",
        "K",
        "bias",
        "dtype",
        "outdtype",
        "scaleAB",
        "bpreshuffle",
    ),
    kernel_cols=("libtype", "solidx", "splitK", "kernelName"),
)
FMOE = AiterTable(
    name="fmoe",
    env="AITER_CONFIG_FMOE",
    # without gfx, which aiter derives from cu_num for the rows of its default table
    key_cols=(
        "cu_num",
        "token",
        "model_dim",
        "inter_dim",
        "expert",
        "topk",
        "act_type",
        "dtype",
        "q_dtype_a",
        "q_dtype_w",
        "q_type",
        "use_g1u1",
        "doweight_stage1",
    ),
    kernel_cols=("block_m", "ksplit", "kernelName1", "kernelName2", "run_1stage"),
)

TABLES = (A8W8_BLOCKSCALE_BPRESHUFFLE, A8W8_BPRESHUFFLE, GEMM_BF16, FMOE)


def _norm(v):
    # a column with a missing value reads back as float: 256 and 256.0 are one key
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return str(v).strip()


def norm_kernel(row, kernel_cols: Sequence[str]) -> tuple:
    return tuple(_norm(row[c]) for c in kernel_cols)


def row_keys(df, key_cols: Sequence[str]) -> list:
    return list(zip(*(df[c].map(_norm) for c in key_cols)))


def apply_intree_rows(table, intree, key_cols, kernel_cols):
    """Apply the in-tree rows to ``table`` (aiter's resolved table).

    A row with ``base_*`` columns replaces aiter's row only while aiter still picks that
    base kernel; one without adds a shape aiter has no row for. Returns
    ``(new_table, {status: count})``; no key appears twice, which aiter refuses to load.
    """
    stats = {APPLIED: 0, UPSTREAM: 0, SUPERSEDED: 0, MISSING: 0, ADDED: 0}
    if any(c not in table.columns for c in [*key_cols, *kernel_cols]):
        stats[MISSING] = len(intree)
        return table, stats
    table = table.copy()
    index = {}
    for i, key in enumerate(row_keys(table, key_cols)):
        index.setdefault(key, []).append(table.index[i])
    shared = [c for c in table.columns if c in intree.columns]
    base_cols = ["base_" + c for c in kernel_cols]
    has_base = all(c in intree.columns for c in base_cols)
    added = []
    for key, (_, row) in zip(row_keys(intree, key_cols), intree.iterrows()):
        replaces = has_base and not row[base_cols].isna().any()
        hits = index.get(key, [])
        if key not in index and not replaces:
            added.append(row[shared])
            index[key] = []  # a second in-tree row for this key counts as missing
            stats[ADDED] += 1
            continue
        if len(hits) != 1:
            stats[MISSING] += 1
            continue
        current = norm_kernel(table.loc[hits[0]], kernel_cols)
        if current == norm_kernel(row, kernel_cols):
            stats[UPSTREAM] += 1
        elif replaces and current == norm_kernel(
            {c: row["base_" + c] for c in kernel_cols}, kernel_cols
        ):
            for c in shared:
                table.at[hits[0], c] = row[c]
            stats[APPLIED] += 1
        else:
            stats[SUPERSEDED] += 1
    if added:
        import pandas as pd

        table = pd.concat([table, pd.DataFrame(added)], ignore_index=True)[
            table.columns
        ]
    return table, stats


@functools.cache
def _aiter_has_flydsl() -> bool:
    try:
        import aiter.ops.flydsl  # noqa: F401
    except ImportError:
        return False
    return True


def uses_flydsl(rows, kernel_cols):
    """Rows whose kernel aiter builds with FlyDSL: libtype flydsl, or a flydsl_ kernel name."""
    import pandas as pd

    flydsl = (
        rows["libtype"] == "flydsl"
        if "libtype" in rows.columns
        else pd.Series(False, index=rows.index)
    )
    for c in kernel_cols:
        if "kernelName" in c and c in rows.columns:
            flydsl = flydsl | rows[c].astype(str).str.startswith("flydsl_")
    return flydsl


def read_intree_rows(table: AiterTable, csvs: Sequence[str]):
    """The rows of ``csvs`` this runtime can use: a hipBLASLt row's solution index belongs
    to the ROCm in its ``rocm`` column, and a FlyDSL row needs an aiter that imports FlyDSL."""
    import pandas as pd

    from sglang.srt.utils.device import get_hip_version

    rows = pd.concat(
        [pd.read_csv(p, dtype={"rocm": str}) for p in csvs], ignore_index=True
    )
    if not _aiter_has_flydsl():
        flydsl = uses_flydsl(rows, table.kernel_cols)
        if flydsl.any():
            logger.info(
                "aiter %s: %d FlyDSL rows skipped; this aiter cannot import FlyDSL",
                table.name,
                int(flydsl.sum()),
            )
            rows = rows[~flydsl]
    if "rocm" in rows.columns:
        rocm = ".".join(map(str, get_hip_version()[:2]))
        other = rows["rocm"].notna() & (rows["rocm"] != rocm)
        if other.any():
            logger.info(
                "aiter %s: %d hipBLASLt rows of another ROCm skipped (this is ROCm %s)",
                table.name,
                int(other.sum()),
                rocm,
            )
            rows = rows[~other]
    return rows


def intree_out_dir() -> str:
    return os.path.join(tempfile.gettempdir(), f"sglang_aiter_intree_{os.getuid()}")


def _write_atomic(df, out_dir: str, stem: str) -> str:
    text = df.to_csv(index=False)
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}.{digest}.csv")
    if not os.path.exists(path):
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    return path


def install_intree_table(
    table: AiterTable, csvs: Sequence[str], out_dir: Optional[str] = None
):
    """Hand aiter its resolved ``table`` with the rows of ``csvs`` applied. Returns
    ``({status: count}, the file aiter reads)``."""
    import pandas as pd
    from aiter.jit.core import AITER_CONFIGS

    out_dir = out_dir or intree_out_dir()
    base = getattr(AITER_CONFIGS, table.attr)
    merged, stats = apply_intree_rows(
        pd.read_csv(base),
        read_intree_rows(table, csvs),
        table.key_cols,
        table.kernel_cols,
    )
    if stats[APPLIED] + stats[ADDED] == 0:
        return stats, base
    path = _write_atomic(merged, out_dir, table.name)
    prev = os.environ.get(table.env)
    # one path: aiter skips its own merge, which refuses duplicate keys
    os.environ[table.env] = path
    # get_config_file is lru-cached; drop a resolution made before the env change
    cache_clear = getattr(type(AITER_CONFIGS).get_config_file, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
    resolved = getattr(AITER_CONFIGS, table.attr)
    if resolved != path:
        if prev is None:
            os.environ.pop(table.env, None)
        else:
            os.environ[table.env] = prev
        if cache_clear is not None:
            cache_clear()
        raise RuntimeError(f"aiter resolved {resolved}, expected {path}")
    return stats, path


_report: Optional[dict] = None


def install_aiter_tuned_tables() -> Optional[dict]:
    """Idempotent; call before the first aiter GEMM or fused-MoE config lookup. Returns
    ``{table: outcome}``, or None when aiter is off or missing."""
    global _report
    if _report is not None:
        return _report
    from sglang.srt.environ import envs
    from sglang.srt.utils.device import is_hip

    if not (is_hip() and envs.SGLANG_USE_AITER.get()):
        return None
    if not envs.SGLANG_AITER_INTREE_GEMM_TUNE.get():
        return None
    try:
        import aiter  # noqa: F401
    except ImportError:
        return None
    out_dir = intree_out_dir()
    report = {}
    for table in TABLES:
        csvs = table.csv_paths()
        if not csvs:
            continue
        user = os.environ.get(table.env)
        if user and os.path.dirname(os.path.abspath(user)) != os.path.abspath(out_dir):
            report[table.name] = f"user {table.env}"
            continue
        try:
            report[table.name] = install_intree_table(table, csvs, out_dir)
        except Exception as err:  # noqa: BLE001 - keep aiter's stock tables
            report[table.name] = repr(err)
            logger.warning("aiter in-tree table %s skipped: %r", table.name, err)
    _report = report
    logger.info("aiter in-tree tables: %s", report)
    return report
