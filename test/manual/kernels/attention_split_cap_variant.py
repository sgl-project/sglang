"""Build an isolated launcher with a tunable limit on KV splits."""

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
    marker = "      // Update the numCtasX.\n      numCtasX *= numCtasPerSeqKv;"
    assert text.count(marker) == 1
    override = """
      if (params.mHeadDimQk == 256 && params.mHeadDimV == 256) {
        auto const capText = std::getenv("Q35MF_ATTENTION_SPLIT_CAP");
        int const cap = capText ? std::atoi(capText) : 0;
        if (cap > 0) {
          numCtasPerSeqKv = std::min(numCtasPerSeqKv, cap);
        }
      }
"""
    updated = text.replace(marker, override + marker).replace(
        "#pragma once",
        "// Modified by NVIDIA: allow a tunable limit on KV splits.\n\n#pragma once\n#include <cstdlib>",
        1,
    )
    fingerprint = hashlib.sha256(updated.encode()).hexdigest()[:16]
    include = Path("/tmp") / "attention-split-cap" / fingerprint / "include"
    if not (include / relative).exists():
        shutil.copytree(original, include, dirs_exist_ok=True)
        (include / relative).write_text(updated)
    with patch.object(jit_env, "FLASHINFER_INCLUDE_DIR", include):
        spec = replace(
            gen_trtllm_gen_fmha_module(), name=f"fmha_split_cap_{fingerprint}"
        )
        module = spec.build_and_load()
        setup_cubin_loader(spec.get_library_path())
    return module, fingerprint
