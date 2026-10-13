# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Optional Kimi-K3 AITER BF16 GEMM profile.

The Kimi-K3 model calls this while it is built. AITER reads
``AITER_CONFIG_GEMM_BF16`` when it loads its BF16 GEMM table on the first tuned
GEMM, not at import, so the profile applies in every process that builds the
model, including single-process tools such as ``bench_one_batch``.
"""

import functools
import importlib.util
import logging
import os
import sys
from pathlib import Path

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def _aiter_bf16_table_loaded() -> bool:
    tuned_gemm = sys.modules.get("aiter.tuned_gemm")
    cache_info = getattr(
        getattr(tuned_gemm, "get_GEMM_A16W16_config_", None), "cache_info", None
    )
    return cache_info is not None and cache_info().currsize > 0


@functools.cache
def maybe_set_aiter_m16384_profile() -> None:
    if torch.version.hip is None:
        return
    if not envs.SGLANG_ROCM_K3_AITER_M16384_PROFILE.get():
        return
    if "AITER_CONFIG_GEMM_BF16" in os.environ:
        return
    aiter_spec = importlib.util.find_spec("aiter")
    if aiter_spec is None or aiter_spec.origin is None:
        return
    aiter_root = Path(aiter_spec.origin).resolve().parent
    base = aiter_root / "configs" / "bf16_tuned_gemm.csv"
    model_configs = sorted(
        (aiter_root / "configs" / "model_configs").glob("*bf16_tuned_gemm*.csv")
    )
    profile = (
        Path(__file__).resolve().parents[2]
        / "kernels"
        / "ops"
        / "gemm"
        / "configs"
        / "kimik3_m16384_profile.csv"
    )
    paths = [base, *model_configs, profile]
    if not all(path.is_file() for path in paths):
        return
    if _aiter_bf16_table_loaded():
        logger.warning(
            "AITER already loaded its BF16 GEMM table in this process; "
            "SGLANG_ROCM_K3_AITER_M16384_PROFILE will not apply here."
        )
    os.environ["AITER_CONFIG_GEMM_BF16"] = os.pathsep.join(map(str, paths))
