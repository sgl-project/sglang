"""AITER's bf16 GEMM table with the rows tuned for Kimi-K3.1's shapes on MI350X."""

from __future__ import annotations

import hashlib
import os
import tempfile

K31_BF16_TUNED_GEMM = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kimi_k31_bf16_tuned_gemm.csv"
)
# aiter's dedup key for bf16 GEMM rows
_KEY_COLS = [
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
]


def write_k31_bf16_gemm_table() -> str:
    """aiter's resolved table plus the K3.1 rows for shapes it does not tune, as one file.

    aiter writes its merged table to a fixed path, so it must get this single file rather
    than a list naming that path; a shape aiter already tunes keeps aiter's row.
    """
    import pandas as pd
    from aiter.jit.core import AITER_CONFIGS

    base = pd.read_csv(AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE)
    k31 = pd.read_csv(K31_BF16_TUNED_GEMM)
    known = set(map(tuple, base[_KEY_COLS].astype(str).values))
    new_rows = k31[[tuple(r) not in known for r in k31[_KEY_COLS].astype(str).values]]
    text = pd.concat([base, new_rows], ignore_index=True).to_csv(index=False)
    out_dir = os.path.join(tempfile.gettempdir(), f"sglang_k31_aiter_{os.getuid()}")
    os.makedirs(out_dir, exist_ok=True)
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    path = os.path.join(out_dir, f"bf16_tuned_gemm.{digest}.csv")
    if not os.path.exists(path):
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    return path
