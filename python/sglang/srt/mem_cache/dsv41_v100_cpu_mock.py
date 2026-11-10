"""CPU mock: v1 launch shape without constructing the 189 GiB Engram graph.

DoD: imports and config/quant selection succeed on CPU; the dry-run allocator
dies with a **budget** error (exit 1) or NUMA error (exit 2), never an SM90
import error. Does not load safetensor shards.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

EXIT_OK = 0
EXIT_BUDGET = 1
EXIT_NUMA = 2
EXIT_SM90 = 3
EXIT_CONFIG = 4

OFFICIAL_CONFIG = os.path.expanduser("~/models/DeepSeek-V4.1-Flash/config.json")
INDEX_CANDIDATES = (
    "/tmp/dsv41-flash-index/model.safetensors.index.json",
    os.path.expanduser("~/models/DeepSeek-V4.1-Flash/model.safetensors.index.json"),
)
SERVE_SCRIPT = "scripts/serve_dsv41_v100.sh"

# Frozen inference-format names from the official index (2026-09-12).
_REMAP_CASES: Tuple[Tuple[str, str], ...] = (
    ("embed.weight", "model.embed_tokens.weight"),
    ("head.weight", "lm_head.weight"),
    ("norm.weight", "model.norm.weight"),
    ("layers.6.attn_norm.weight", "model.layers.6.input_layernorm.weight"),
    ("layers.6.ffn_norm.weight", "model.layers.6.post_attention_layernorm.weight"),
    ("layers.6.attn.wq_a.weight", "model.layers.6.self_attn.wq_a.weight"),
    ("layers.6.attn.wq_a.scale", "model.layers.6.self_attn.wq_a.weight_scale_inv"),
    ("layers.6.attn.wo_a.weight", "model.layers.6.self_attn.wo_a.weight"),
    ("layers.6.attn.wo_a.scale", "model.layers.6.self_attn.wo_a.weight_scale_inv"),
    ("layers.6.attn.wo_b.weight", "model.layers.6.self_attn.wo_b.weight"),
    ("layers.6.attn.attn_sink", "model.layers.6.self_attn.attn_sink"),
    ("layers.6.attn.wkv.weight", "model.layers.6.self_attn.wkv.weight"),
    ("layers.6.ffn.experts.0.w1.weight", "model.layers.6.mlp.experts.0.gate_proj.weight"),
    (
        "layers.6.ffn.experts.0.w1.scale",
        "model.layers.6.mlp.experts.0.gate_proj.weight_scale_inv",
    ),
    ("layers.6.ffn.experts.0.w2.weight", "model.layers.6.mlp.experts.0.down_proj.weight"),
    ("layers.6.ffn.experts.0.w3.weight", "model.layers.6.mlp.experts.0.up_proj.weight"),
    (
        "layers.6.ffn.shared_experts.w1.weight",
        "model.layers.6.mlp.shared_experts.gate_proj.weight",
    ),
    ("layers.6.ffn.gate.weight", "model.layers.6.mlp.gate.weight"),
    ("layers.6.ffn.gate.bias", "model.layers.6.mlp.gate.e_score_correction_bias"),
    ("layers.6.ffn.gate.bias_vl", "model.layers.6.mlp.gate.bias_vl"),
    ("layers.6.hc_attn_base", "model.layers.6.hc_attn_base"),
    ("layers.6.hc_attn_fn", "model.layers.6.hc_attn_fn"),
    ("layers.6.hc_attn_scale", "model.layers.6.hc_attn_scale"),
    (
        "layers.14.attn.compressor.wkv.weight",
        "model.layers.14.self_attn.compressor.wkv.weight",
    ),
    (
        "layers.14.attn.compressor.wgate.weight",
        "model.layers.14.self_attn.compressor.wgate.weight",
    ),
    (
        "layers.14.attn.indexer.wq_b.weight",
        "model.layers.14.self_attn.indexer.wq_b.weight",
    ),
    (
        "layers.14.attn.indexer.wq_b.scale",
        "model.layers.14.self_attn.indexer.wq_b.weight_scale_inv",
    ),
    (
        "layers.14.attn.indexer.weights_proj.weight",
        "model.layers.14.self_attn.indexer.weights_proj.weight",
    ),
    ("layers.14.engram.embed.weight", "model.layers.14.engram.embed.weight"),
    ("layers.14.engram.embed.scale", "model.layers.14.engram.embed.scale"),
    ("layers.14.engram.q_weight", "model.layers.14.engram.q_weight"),
    ("layers.14.engram.k_weight", "model.layers.14.engram.k_weight"),
    ("layers.14.engram.wkv.weight", "model.layers.14.engram.wkv.weight"),
    ("layers.14.engram.wkv.scale", "model.layers.14.engram.wkv.weight_scale_inv"),
)


@dataclass
class MockReport:
    ok: bool
    exit_code: int
    lines: List[str]


def _log(lines: List[str], msg: str) -> None:
    lines.append(msg)
    print(msg, flush=True)


def check_imports(lines: List[str]) -> Optional[int]:
    try:
        from sglang.srt.configs.deepseek_v41 import (
            DeepseekV41Config,
            normalize_deepseek_v41_config,
        )
        from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
            DeepseekV41Compressor,
            DeepseekV41Indexer,
        )
        from sglang.srt.layers.engram import Engram, build_engram_layout
        from sglang.srt.layers.quantization.fp8 import Fp8Config
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
    except Exception as exc:  # noqa: BLE001 — DoD is "not an SM90 import error"
        msg = str(exc)
        _log(lines, f"IMPORT_FAIL {type(exc).__name__}: {msg}")
        if "sm90" in msg.lower() or "deep_gemm" in msg.lower() or "flashinfer_mxfp4" in msg.lower():
            return EXIT_SM90
        return EXIT_CONFIG
    _log(
        lines,
        f"IMPORT_OK {DeepseekV41Config.model_type} "
        f"remap={normalize_deepseek_v41_config.__name__} "
        f"model={DeepseekV4ForCausalLM.__name__} "
        f"fp8={Fp8Config.__name__} engram={Engram.__name__} "
        f"layout={build_engram_layout.__name__} "
        f"compressor={DeepseekV41Compressor.__name__} "
        f"indexer={DeepseekV41Indexer.__name__}",
    )
    return None


def check_official_config(lines: List[str], config_path: str = OFFICIAL_CONFIG) -> Optional[int]:
    from sglang.srt.configs.deepseek_v41 import (
        DeepseekV41Config,
        normalize_deepseek_v41_config,
    )
    from sglang.srt.layers.quantization.fp8 import Fp8Config

    if not os.path.isfile(config_path):
        _log(lines, f"CONFIG_SKIP missing {config_path}")
        # Still prove the official quant block without the file.
        cfg = Fp8Config.from_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [32, 32],
                "scale_fmt": "ue8m0",
                "expert_dtype": "fp4",
            }
        )
        if not (cfg.use_mxfp8 and cfg.is_fp4_experts):
            _log(lines, "CONFIG_FAIL fixture quant is not MXFP8+FP4 experts")
            return EXIT_CONFIG
        _log(lines, "CONFIG_OK fixture MXFP8+FP4 (official file not on disk yet)")
        return None

    raw = json.loads(Path(config_path).read_text())
    values = normalize_deepseek_v41_config(raw)
    if values.get("architectures") != ["DeepseekV4ForCausalLM"]:
        _log(lines, f"CONFIG_FAIL architectures={values.get('architectures')}")
        return EXIT_CONFIG
    hf = DeepseekV41Config(**raw)
    if hf.hidden_size != 5120 or hf.n_routed_experts != 384:
        _log(
            lines,
            f"CONFIG_FAIL hidden={hf.hidden_size} experts={hf.n_routed_experts}",
        )
        return EXIT_CONFIG
    q = raw.get("quantization_config") or {}
    cfg = Fp8Config.from_config(q)
    if not (cfg.use_mxfp8 and cfg.is_fp4_experts and list(cfg.weight_block_size) == [32, 32]):
        _log(lines, f"CONFIG_FAIL quant {cfg.use_mxfp8=} {cfg.is_fp4_experts=} {cfg.weight_block_size}")
        return EXIT_CONFIG
    _log(
        lines,
        f"CONFIG_OK {config_path} layers={hf.num_hidden_layers} "
        f"engram={list(hf.engram_layer_ids)} mxfp8={cfg.use_mxfp8} fp4_experts={cfg.is_fp4_experts}",
    )
    return None


def check_weight_remap(lines: List[str]) -> Optional[int]:
    from sglang.srt.models.deepseek_v4 import (
        DeepseekV4ForCausalLM,
        _skip_dsv41_language_only_weight,
    )

    bad: List[str] = []
    for src, want in _REMAP_CASES:
        got = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(src)
        if got != want:
            bad.append(f"{src} -> {got!r} want {want!r}")
    if bad:
        _log(lines, "REMAP_FAIL " + "; ".join(bad[:8]))
        return EXIT_CONFIG
    if not _skip_dsv41_language_only_weight("vision.patch_embed.proj.weight"):
        _log(lines, "REMAP_FAIL vision weights are not skipped")
        return EXIT_CONFIG
    if not _skip_dsv41_language_only_weight("layers.6.mlp.gate.bias_vl"):
        _log(lines, "REMAP_FAIL gate.bias_vl is not skipped")
        return EXIT_CONFIG
    _log(lines, f"REMAP_OK {len(_REMAP_CASES)} frozen official-index names")
    return None


def check_index_if_present(lines: List[str]) -> Optional[int]:
    from sglang.srt.models.deepseek_v4 import (
        DeepseekV4ForCausalLM,
        _skip_dsv41_language_only_weight,
    )

    index_path = next((p for p in INDEX_CANDIDATES if os.path.isfile(p)), None)
    if index_path is None:
        _log(lines, "INDEX_SKIP no model.safetensors.index.json yet")
        return None
    weight_map = json.loads(Path(index_path).read_text())["weight_map"]
    unknown: List[str] = []
    skipped = 0
    for name in weight_map:
        remapped = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(name)
        if _skip_dsv41_language_only_weight(name) or _skip_dsv41_language_only_weight(
            remapped
        ):
            skipped += 1
            continue
        if remapped.startswith("mtp."):
            skipped += 1
            continue
        if not (
            remapped.startswith("model.")
            or remapped.startswith("lm_head.")
        ):
            unknown.append(f"{name} -> {remapped}")
            if len(unknown) >= 12:
                break
    if unknown:
        _log(lines, "INDEX_FAIL " + "; ".join(unknown[:8]))
        return EXIT_CONFIG
    _log(
        lines,
        f"INDEX_OK {index_path} n={len(weight_map)} skipped_lm_only_or_mtp={skipped}",
    )
    return None


def check_sm70_fail_closed(lines: List[str]) -> Optional[int]:
    import torch

    from sglang.srt.layers.attention import dsa_backend
    from sglang.srt.layers.attention.dsv4.v41_indexer import (
        is_sm100_or_newer,
        make_candidate_indexer,
    )
    from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
        DenseBlocksBackend,
    )
    from sglang.srt.runtime_context import get_platform, override_platform

    sm70 = dict(
        is_cuda=True,
        is_hip=False,
        is_sm70=True,
        is_sm90=False,
        is_sm100=False,
        is_sm120=False,
        device_sm=70,
    )
    # DeepGEMM candidate indexing is SM100+. Below that the factory keeps the
    # dense block backend for both prefill and decode.
    with override_platform(**sm70):
        platform = get_platform()
        if platform.is_sm90 or platform.is_sm100 or platform.device_sm >= 100:
            _log(lines, "SM90_GATE_FAIL override_platform still reports SM90")
            return EXIT_SM90
        if is_sm100_or_newer():
            _log(lines, "SM90_GATE_FAIL device capability is SM100+")
            return EXIT_SM90
        prefill, decode = make_candidate_indexer(
            token_to_kv_pool=object(),
            req_to_token=torch.empty(0, dtype=torch.int32),
            page_size=64,
            candidate_topk_blocks=1,
            candidate_block_size=64,
        )
        if not isinstance(prefill, DenseBlocksBackend) or not isinstance(
            decode, DenseBlocksBackend
        ):
            _log(
                lines,
                "SM90_GATE_FAIL make_candidate_indexer built a DeepGEMM "
                "indexer on SM70",
            )
            return EXIT_SM90
    _log(
        lines,
        f"SM90_GATE_OK candidate indexer closed below SM100; "
        f"dsa_backend.deep_gemm={dsa_backend.deep_gemm is not None}",
    )
    return None


def check_budget(lines: List[str], *, allow_numa_split: bool = True) -> Optional[int]:
    from sglang.srt.mem_cache.dsv41_v100_budget import main as budget_main

    argv = ["--spill-gb", "10", "--chunked-prefill-size", "2048"]
    if allow_numa_split:
        argv.append("--allow-numa-split")
    code = budget_main(argv)
    if code == 0:
        _log(lines, "BUDGET_OK dry-run HBM <= 31 GiB (v1 10 GiB spill, host Engram)")
        return None
    if code == 1:
        _log(lines, "BUDGET_FAIL HBM over 31 GiB")
        return EXIT_BUDGET
    if code == 2:
        _log(lines, "BUDGET_FAIL Engram NUMA (pass --allow-numa-split for bring-up)")
        return EXIT_NUMA
    _log(lines, f"BUDGET_FAIL allocator exit {code}")
    return EXIT_BUDGET


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / SERVE_SCRIPT).is_file():
            return parent
    return here.parents[4]


def check_serve_script(lines: List[str], repo_root: Optional[str] = None) -> Optional[int]:
    root = Path(repo_root) if repo_root else _repo_root()
    script = root / SERVE_SCRIPT
    text = script.read_text()
    required = (
        "--tp 8",
        "--ep-size 8",
        "--dtype float16",
        "--moe-runner-backend marlin",
        "--attention-backend dsv4",
        "--mem-fraction-static",
        "SGLANG_DSV41_MEM_FRACTION:-0.99",
        "--chunked-prefill-size 2048",
        "--disable-custom-all-reduce",
        "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1",
        "SGLANG_DSV41_EXPERT_SPILL_GB",
        "SGLANG_DSV41_EXPERT_SPILL_APPLY",
    )
    missing = [item for item in required if item not in text]
    if missing:
        _log(lines, "SERVE_FAIL missing " + ", ".join(missing))
        return EXIT_CONFIG
    if "SGLANG_DSV41_EXPERT_SPILL_APPLY:-1" not in text and "APPLY:-1}" not in text:
        # default must be on for v1 HBM
        if re.search(r"EXPERT_SPILL_APPLY:-1", text) is None:
            _log(lines, "SERVE_FAIL APPLY default is not 1")
            return EXIT_CONFIG
    _log(lines, f"SERVE_OK {script} mem-fraction-static default 0.99 APPLY default on")
    return None


def run_cpu_mock(
    *,
    config_path: str = OFFICIAL_CONFIG,
    allow_numa_split: bool = True,
) -> MockReport:
    lines: List[str] = []
    for fn in (
        lambda: check_imports(lines),
        lambda: check_official_config(lines, config_path),
        lambda: check_weight_remap(lines),
        lambda: check_index_if_present(lines),
        lambda: check_sm70_fail_closed(lines),
        lambda: check_serve_script(lines),
        lambda: check_budget(lines, allow_numa_split=allow_numa_split),
    ):
        code = fn()
        if code is not None:
            return MockReport(ok=False, exit_code=code, lines=lines)
    _log(lines, "CPU_MOCK_OK v1 launch shape is selected; no 476 GiB load")
    return MockReport(ok=True, exit_code=EXIT_OK, lines=lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    del argv  # flags are env / defaults; keep CLI torch-free besides sglang imports
    report = run_cpu_mock()
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
