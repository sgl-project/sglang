# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
#
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
"""Config-time override declarations for ERNIE 5.5 Flash."""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    register_dspark_dp_a2a_backends,
    resolving_view,
)
from sglang.srt.arg_groups.model_overrides.deepseek_v4 import (
    DSV4_DEFAULT_SWA_FULL_TOKENS_RATIO,
    DSV4_PAGE_SIZE,
)
from sglang.srt.configs.ernie55 import ERNIE5_5_MOE_ARCH
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# The MLA latent sub-pool is allocated as e4m3 or bf16,
# and the memory budget follows the same choice, so no other KV dtype is servable.
_SUPPORTED_KV_CACHE_DTYPES = ("auto", "fp8_e4m3", "bfloat16")

# DSpark with DP attention is validated on DeepEP here; MegaMoE is rejected below.
register_dspark_dp_a2a_backends(ERNIE5_5_MOE_ARCH, ("none", "deepep"))


def _validate_features(cfg: Any) -> None:
    if cfg.kv_cache_dtype not in _SUPPORTED_KV_CACHE_DTYPES:
        raise ValueError(
            f"ERNIE 5.5 Flash supports --kv-cache-dtype "
            f"{', '.join(_SUPPORTED_KV_CACHE_DTYPES)}, got {cfg.kv_cache_dtype!r}"
        )
    if (
        cfg.enable_eplb
        or cfg.ep_num_redundant_experts
        or cfg.init_expert_location != "trivial"
    ):
        # The router does not remap logical experts to physical ones.
        raise ValueError(
            "ERNIE 5.5 Flash does not support EPLB, redundant experts or a "
            "non-trivial --init-expert-location"
        )
    if cfg.enable_unified_cache_external_linker:
        # The linker's device pool group has no region for the MLA latent sub-pool.
        raise ValueError(
            "ERNIE 5.5 Flash does not support --enable-unified-cache-external-linker"
        )
    for flag, enabled in (
        ("--enable-encoder-swa-bounded-replay", cfg.enable_encoder_swa_bounded_replay),
        ("--enable-decoder-swa-bounded-replay", cfg.enable_decoder_swa_bounded_replay),
    ):
        # The request window and its budget assume every layer keeps SWA rows,
        # while the MLA layers keep none.
        if enabled:
            raise ValueError(f"ERNIE 5.5 Flash does not support {flag}")
    if cfg.enable_hisparse:
        # HiSparse knows the compressed-KV sub-pools, not the MLA layers' latent one.
        raise ValueError("ERNIE 5.5 Flash does not support --enable-hisparse")
    if cfg.moe_a2a_backend == "megamoe":
        # The MegaMoE kernel hardcodes SwiGLU; these experts use SiTU.
        raise ValueError(
            "ERNIE 5.5 Flash does not support --moe-a2a-backend megamoe; "
            "use deepep or none"
        )
    for flag, value in (
        ("--dsa-prefill-backend", cfg.dsa_prefill_backend),
        ("--dsa-decode-backend", cfg.dsa_decode_backend),
    ):
        # Only flashmla_sparse takes the learnable sink; the backend pins it.
        if value not in (None, "flashmla_sparse"):
            raise ValueError(
                f"ERNIE 5.5 Flash's MLA layers run flashmla_sparse only, got "
                f"{flag} {value}"
            )
    if cfg.enable_two_batch_overlap or cfg.enable_single_batch_overlap:
        # Each attention module carries the output gate between two of the parent's
        # hooks, which a second micro-batch entering the same module would clobber.
        raise ValueError(
            "ERNIE 5.5 Flash does not support batch overlap; drop "
            "--enable-two-batch-overlap / --enable-single-batch-overlap"
        )
    if cfg.dcp_size > 1:
        # The memory budget does not price the DSA index buffer DCP enlarges.
        raise ValueError("ERNIE 5.5 Flash does not support decode context parallelism")
    if cfg.disaggregation_mode != "null" and cfg.pp_size > 1:
        # The PD slicer treats the MLA layers as SWA layers when the two sides'
        # pipeline layouts differ, which would corrupt the transferred KV.
        raise ValueError(
            "ERNIE 5.5 Flash does not support PD disaggregation with --pp-size > 1"
        )
    if (
        envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
        or envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
    ):
        # The router emits two experts' worth of logits per token ([T, 2E]),
        # so simulated ids would land past the real expert count.
        raise ValueError(
            "ERNIE 5.5 Flash does not support SGLANG_SIMULATE_UNIFORM_EXPERTS / "
            "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS"
        )


def _validate_speculative(cfg: Any) -> None:
    # Runs before NEXTN is resolved to EAGLE, so it sees the raw CLI spelling.
    algo = (cfg.speculative_algorithm or "").upper()
    is_mtp = algo in ("EAGLE", "NEXTN")
    if algo not in ("", "DSPARK") and not is_mtp:
        raise ValueError(
            "ERNIE 5.5 Flash speculative decoding supports DSPARK and EAGLE (MTP), "
            f"got {cfg.speculative_algorithm!r}"
        )
    # --speculative-adaptive leaves the chain unset here, then retunes
    # speculative_num_steps at runtime, which the draft backends are not built for.
    if is_mtp and cfg.speculative_adaptive:
        raise ValueError(
            "ERNIE 5.5 Flash MTP does not support --speculative-adaptive; "
            "set --speculative-num-steps instead."
        )
    # DeepseekV4AttnBackend, which the HCA layers run on, asserts topk <= 1.
    if is_mtp and cfg.speculative_eagle_topk not in (None, 1):
        raise ValueError(
            "ERNIE 5.5 Flash MTP supports --speculative-eagle-topk 1 only, "
            f"got {cfg.speculative_eagle_topk}"
        )


@_register_for(ERNIE5_5_MOE_ARCH)
def _ernie55_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    _validate_features(cfg)
    _validate_speculative(cfg)
    overrides: Dict[str, Any] = {
        "attention_backend": "ernie55",
        "page_size": DSV4_PAGE_SIZE,
    }
    if cfg.kv_cache_dtype == "auto":
        logger.warning(
            "ERNIE 5.5 Flash defaults --kv-cache-dtype to fp8_e4m3; "
            "pass --kv-cache-dtype bfloat16 to keep the latent cache in bf16."
        )
        overrides["kv_cache_dtype"] = "fp8_e4m3"
    if cfg.swa_full_tokens_ratio is None:
        # The 128-token HCA window needs only a small SWA share.
        overrides["swa_full_tokens_ratio"] = DSV4_DEFAULT_SWA_FULL_TOKENS_RATIO
    logger.info(
        "Use ernie55 attention backend for ERNIE 5.5 Flash, with %s.",
        ", ".join(f"{k}={v}" for k, v in overrides.items() if k != "attention_backend"),
    )
    return overrides
