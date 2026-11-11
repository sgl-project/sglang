"""AITER settings Kimi-K3.1 is served with on ROCm."""

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


def apply_kimi_k31_rocm_env_defaults() -> None:
    """AITER settings Kimi-K3.1 is served with on ROCm, unless the user set them."""
    # AITER's opus MoE stage 2 otherwise stores per-slot down outputs in fp8 before the
    # top-k sum: 11x the error of the bf16 partials the reference keeps.
    os.environ.setdefault("AITER_OPUS_STAGE2_FP8", "0")
    # the same experts as AITER's biased top-k, in under half its time
    os.environ.setdefault("SGLANG_K3_RADIX4_TOPK", "1")
    # AgentX target verify (up to 64 requests x 6 rows) takes the column-split tail
    os.environ.setdefault("SGLANG_ROCM_K3_COLUMN_TAIL_MAX_TOKENS", "384")
    if "AITER_CONFIG_GEMM_BF16" not in os.environ:
        os.environ["AITER_CONFIG_GEMM_BF16"] = write_k31_bf16_gemm_table()
