"""Build a private attention-launcher variant from the installed public sources."""

import hashlib
import shutil
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from flashinfer.jit import env as jit_env
from flashinfer.jit.attention import gen_trtllm_gen_fmha_module
from flashinfer.jit.cubin_loader import setup_cubin_loader


def build_variant():
    original = jit_env.FLASHINFER_INCLUDE_DIR
    relative = Path("flashinfer/trtllm/fmha/fmhaKernels.cuh")
    text = (original / relative).read_text()
    start = text.index("  void selectGqGenerationKernel(")
    end = text.index("  // Select a kernel based on the heuristic.", start)
    block = text[start:end]
    insertion = block.rfind("  }")
    override = """
    if (params.mHeadDimQk == 256 && params.mHeadDimV == 256 &&
        params.mMaxSeqLenQ > 1) {
      auto const tileEnv = std::getenv("SGLANG_Q35MF_ATTN_PROBE_TILE");
      auto const modeEnv = std::getenv("SGLANG_Q35MF_ATTN_PROBE_REDUCTION");
      int tile = tileEnv ? std::atoi(tileEnv) : 64;
      int mode = modeEnv ? std::atoi(modeEnv) : 1;
      selectKernelParams.mTileSizeQ = tile;
      selectKernelParams.mKernelType = tile >= 64 ? FmhaKernelType::KeepsMmaAbForGeneration
                                                 : FmhaKernelType::SwapsMmaAbForGeneration;
      selectKernelParams.mHeadDimPerCtaV = params.mHeadDimV;
      selectKernelParams.mForceGmemReduction = true;
      if (isMultiCtasKvEnabled(selectKernelParams.mMultiCtasKvMode)) {
        selectKernelParams.mMultiCtasKvMode = static_cast<MultiCtasKvMode>(mode);
      }
      syncGqaGenerationTraitsForKernelHash(params, selectKernelParams);
      FLASHINFER_CHECK(static_cast<int>(selectKernelParams.mMultiCtasKvMode) == mode,
                       "Requested reduction mode is unavailable for this shape.");
    }
"""
    updated = (
        text[:start] + block[:insertion] + override + block[insertion:] + text[end:]
    )
    updated = updated.replace(
        "#pragma once",
        "// Modified by NVIDIA: expose a tunable attention launch configuration.\n\n#pragma once\n#include <cstdlib>",
        1,
    )
    fingerprint = hashlib.sha256(updated.encode()).hexdigest()[:16]
    root = Path("/tmp") / "attention-launcher" / fingerprint
    include = root / "include"
    if not (include / relative).exists():
        shutil.copytree(original, include, dirs_exist_ok=True)
        (include / relative).write_text(updated)
    with patch.object(jit_env, "FLASHINFER_INCLUDE_DIR", include):
        spec = gen_trtllm_gen_fmha_module()
        spec = replace(spec, name=f"fmha_variant_{fingerprint}")
        module = spec.build_and_load()
        setup_cubin_loader(spec.get_library_path())
    return module, fingerprint
