# Copyright 2026 SGLang Team
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
"""Validate SynthID configuration during ServerArgs resolution."""

from __future__ import annotations

import logging
from typing import Callable

from sglang.srt.arg_groups.overrides import ResolvedView
from sglang.srt.environ import envs
from sglang.srt.watermarking.synthid.state import load_synthid_keys

logger = logging.getLogger(__name__)


def validate_synthid_watermark(
    *,
    view: ResolvedView,
    key_file: str | None,
    ngram_len: int,
    top_k: int,
    context_history_size: int,
    rl_on_policy_target: str | None,
    speculative_algorithm: str | None,
    cuda_is_available: Callable[[], bool],
) -> None:
    if envs.SGLANG_RETURN_WATERMARKED_LOGPROB.get():
        if envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get():
            raise ValueError(
                "SGLANG_RETURN_WATERMARKED_LOGPROB and "
                "SGLANG_RETURN_ORIGINAL_LOGPROB are mutually exclusive."
            )
        if key_file is None:
            raise ValueError(
                "SGLANG_RETURN_WATERMARKED_LOGPROB requires "
                "--synthid-watermark-key-file."
            )
        if view.speculative_algorithm:
            raise ValueError(
                "SGLANG_RETURN_WATERMARKED_LOGPROB requires non-speculative decoding."
            )
    if key_file is None:
        return

    if ngram_len < 2:
        raise ValueError("--synthid-watermark-ngram-len must be at least 2.")
    if not 2 <= top_k <= 65536:
        raise ValueError("--synthid-watermark-top-k must be in [2, 65536].")
    if context_history_size < 1:
        raise ValueError("--synthid-watermark-context-history-size must be positive.")
    if view.device != "cuda" or not cuda_is_available():
        raise ValueError("SynthID watermarking requires NVIDIA CUDA.")
    if view.is_embedding or view.dllm_algorithm is not None:
        raise ValueError(
            "SynthID watermarking requires autoregressive text generation."
        )
    if view.disaggregation_mode != "null":
        raise ValueError("SynthID watermarking does not support PD disaggregation.")
    if rl_on_policy_target is not None:
        raise ValueError(
            "SynthID watermarking is incompatible with --rl-on-policy-target "
            "because it intentionally changes the rollout distribution."
        )
    if view.sampling_backend not in ("flashinfer", "pytorch"):
        raise ValueError(
            "SynthID watermarking currently requires the built-in "
            "flashinfer or pytorch sampling backend."
        )

    keys = load_synthid_keys(key_file)
    if len(keys) > 64:
        raise ValueError("SynthID supports at most 64 watermark keys.")

    algorithm = (view.speculative_algorithm or "").upper()
    if algorithm:
        if view.pp_size > 1:
            raise ValueError("SynthID does not support pipeline-parallel speculation.")
        supported = {"EAGLE", "STANDALONE"}
        if algorithm not in supported:
            raise ValueError(
                "SynthID watermarking currently supports non-speculative "
                "decoding and linear EAGLE-family speculative decoding; got "
                f"--speculative-algorithm={speculative_algorithm}."
            )
        if view.speculative_eagle_topk != 1:
            raise ValueError(
                "SynthID watermarking with speculative decoding requires "
                "--speculative-eagle-topk=1."
            )
        if view.speculative_token_map is not None:
            raise ValueError(
                "SynthID watermarking does not support --speculative-token-map."
            )
        if view.enable_multi_layer_eagle:
            raise ValueError(
                "SynthID watermarking does not yet support --enable-multi-layer-eagle."
            )
        if view.disaggregation_mode != "null":
            raise ValueError(
                "SynthID watermarking with speculative decoding does not yet "
                "support PD disaggregation."
            )
        if not view.speculative_use_rejection_sampling:
            raise ValueError(
                "SynthID watermarking with speculative decoding requires "
                "--speculative-use-rejection-sampling."
            )

    logger.info(
        "Enabled SynthID Text watermarking "
        "(ngram_len=%d, depth=%d, top_k=%d, history=%d).",
        view.synthid_watermark_ngram_len,
        len(keys),
        view.synthid_watermark_top_k,
        view.synthid_watermark_context_history_size,
    )
