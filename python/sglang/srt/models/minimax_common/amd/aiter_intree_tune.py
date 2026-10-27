"""aiter tuned rows for MiniMax-M3.1 at TP 4 on gfx950, applied through DeepSeek-V4's in-tree
mechanism (``deepseek_common/amd/aiter_intree_tune``): the rows add shapes aiter has no row
for, and where aiter has a row a measured kernel beats, the row carries aiter's kernel in
``base_*`` columns and replaces it only while aiter still picks that kernel.

* a8w8 blockscale bpreshuffle: the block-fp8 copy of the MXFP8 projections above
  ``MXFP8_DENSE_PTPC_DECODE_MAX_M`` rows, up to M = 32768.
* a8w8 bpreshuffle: their rowwise-fp8 copy at decode sizes, (split-K) FlyDSL where >= 3%
  faster than aiter's minimax_m3 rows with cold weights.
* bf16: lm_head (50016 x 6144) and the DSpark fc (1536 x 30720) up to M = 1024. Its hipBLASLt
  solution indices are ROCm 7.2's; ROCm 10's hipBLASLt rejects them.
* fused MoE: the routed experts (128 x top-4, hidden 6144, intermediate 768, MXFP4,
  SwiGLU-OAI).

A table whose ``AITER_CONFIG_*`` variable the user set is left to that setting.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from sglang.srt.models.deepseek_common.amd.aiter_intree_tune import (
    IntreeTable,
    install_intree_table,
    intree_out_dir,
)
from sglang.srt.runtime_context import get_model

logger = logging.getLogger(__name__)

_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "aiter_intree_configs"
)
_GEMM_KERNEL_COLS = ("libtype", "kernelId", "splitK", "kernelName")

M31_TABLES = (
    IntreeTable(
        env="AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
        attr="AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE",
        csv=os.path.join(
            _CONFIG_DIR, "minimax_m31_tp4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
        ),
        stem="minimax_m31_a8w8_blockscale_bpreshuffle_tuned_gemm",
        key_cols=("gfx", "cu_num", "M", "N", "K"),
        kernel_cols=_GEMM_KERNEL_COLS,
    ),
    IntreeTable(
        env="AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE",
        attr="AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE",
        csv=os.path.join(
            _CONFIG_DIR, "minimax_m31_tp4_a8w8_bpreshuffle_tuned_gemm.csv"
        ),
        stem="minimax_m31_a8w8_bpreshuffle_tuned_gemm",
        key_cols=("gfx", "cu_num", "M", "N", "K", "q_dtype_w"),
        kernel_cols=_GEMM_KERNEL_COLS,
    ),
    IntreeTable(
        env="AITER_CONFIG_GEMM_BF16",
        attr="AITER_CONFIG_GEMM_BF16_FILE",
        csv=os.path.join(_CONFIG_DIR, "minimax_m31_tp4_bf16_tuned_gemm.csv"),
        stem="minimax_m31_bf16_tuned_gemm",
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
    ),
    IntreeTable(
        env="AITER_CONFIG_FMOE",
        attr="AITER_CONFIG_FMOE_FILE",
        csv=os.path.join(_CONFIG_DIR, "minimax_m31_tp4_mxfp4_tuned_fmoe.csv"),
        stem="minimax_m31_tuned_fmoe",
        # aiter's fused-MoE index without gfx, which aiter derives from cu_num for the
        # rows of its default table
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
    ),
)

_installed: Optional[dict] = None


def _m31_tables(hf_config) -> tuple:
    from sglang.srt.arg_groups.model_overrides.minimax_m3_hip import (
        is_m31_quark_export,
    )

    quantization = get_model().quantization
    if quantization == "minimax_m3_nvfp4" or (
        quantization == "quark" and is_m31_quark_export(hf_config)
    ):
        return M31_TABLES
    if quantization == "quark_mxfp4":
        # the MXFP8-base source shares the M3.1 checkpoint's dense projections; the
        # other tables were measured on the M3.1 checkpoints alone
        return M31_TABLES[:1]
    return ()


def install_m31_aiter_tables(hf_config) -> Optional[dict]:
    """Idempotent; call before the first aiter GEMM or fused-MoE config lookup."""
    global _installed
    if _installed is not None:
        return _installed
    out_dir = intree_out_dir()
    report = {}
    for spec in _m31_tables(hf_config):
        user = os.environ.get(spec.env)
        if user and os.path.dirname(os.path.abspath(user)) != os.path.abspath(out_dir):
            report[spec.stem] = f"user {spec.env}"
            continue
        try:
            report[spec.stem] = install_intree_table(spec, out_dir)
        except Exception as err:  # noqa: BLE001 - keep aiter's stock tables
            report[spec.stem] = repr(err)
            logger.warning("MiniMax-M3.1 aiter table %s skipped: %r", spec.stem, err)
    _installed = report
    if report:
        logger.info("MiniMax-M3.1 aiter in-tree tables: %s", report)
    return report
