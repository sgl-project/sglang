# SPDX-License-Identifier: Apache-2.0
"""Launch and request constraints for single-model Clover speculation."""

import json
import logging
import math

from sglang.srt.arg_groups.model_override_base import model_config_of
from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
from sglang.srt.speculative.clover_penalty import validate_clover_penalties

logger = logging.getLogger(__name__)


def configure_clover(server_args):
    cfg = resolving_view(server_args)
    model = model_config_of(server_args)
    if model.hf_config.architectures != ["CloverMoeModelLM"]:
        raise ValueError("CLOVER speculative decoding requires CloverMoeModelLM")
    if cfg.dllm_algorithm is not None:
        raise ValueError("CLOVER cannot be combined with --dllm-algorithm")
    if cfg.speculative_draft_model_path is not None:
        raise ValueError("CLOVER shares the target model; do not specify a draft model")
    if cfg.speculative_adaptive:
        raise ValueError("CLOVER does not support adaptive speculation")
    if cfg.enable_lora or cfg.enable_two_batch_overlap or cfg.enable_torch_compile:
        raise ValueError(
            "CLOVER does not support LoRA, two-batch overlap or torch.compile"
        )
    if cfg.enable_watermark:
        raise ValueError("CLOVER does not support watermarking")
    if cfg.linear_attn_verify_backend not in (None, "triton", "flashinfer"):
        raise ValueError("CLOVER requires Triton or FlashInfer linear-attention verify")
    if (
        cfg.enable_linear_replayssm
        or cfg.enable_linear_replayssm_spec
        or cfg.enable_mis
    ):
        raise ValueError(
            "CLOVER preview requires materialized recurrent state; ReplaySSM and MIS are unsupported"
        )
    if cfg.enable_return_hidden_states or cfg.enable_return_routed_experts:
        raise ValueError(
            "CLOVER does not support returning hidden states or routed experts"
        )
    checkpoint_width = model.hf_config.block_size
    width = (
        checkpoint_width
        if cfg.speculative_num_draft_tokens is None
        else cfg.speculative_num_draft_tokens
    )
    if width < 2 or checkpoint_width % width:
        raise ValueError(
            "CLOVER verify width must be >= 2 and divide checkpoint block_size"
        )
    if cfg.speculative_clover_accept_rule not in ("threshold", "exact"):
        raise ValueError("CLOVER accept rule must be threshold or exact")
    ratio = cfg.speculative_clover_accept_ratio
    delta = cfg.speculative_clover_accept_delta
    if not 0 <= ratio <= 1:
        raise ValueError("CLOVER accept ratio must be finite and in [0, 1]")
    if not math.isfinite(delta) or delta < 0:
        raise ValueError("CLOVER accept delta must be finite and non-negative")
    for name in ("speculative_num_steps", "speculative_eagle_topk"):
        value = getattr(cfg, name)
        if value is not None and int(value) != 1:
            logger.warning(
                "CLOVER only supports %s == 1, overriding %s to 1.", name, value
            )
    if cfg.enable_mixed_chunk:
        logger.warning("CLOVER does not support mixed chunked prefill, disabling it.")
    preferred = cfg.preferred_sampling_params or {}
    if isinstance(preferred, str):
        preferred = json.loads(preferred)
    declare_resolution(
        server_args,
        "CLOVER",
        preferred_sampling_params={
            "presence_penalty": 0.9,
            "repetition_penalty": 1.05,
            **preferred,
        },
        speculative_num_draft_tokens=width,
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        disable_overlap_schedule=True,
        enable_mixed_chunk=False,
    )


def validate_clover_request(req):
    sp = req.sampling_params
    unsupported = []
    for name, default in (("min_p", 0.0),):
        if getattr(sp, name) != default:
            unsupported.append(name)
    for name in ("json_schema", "regex", "ebnf", "structural_tag", "logit_bias"):
        if getattr(sp, name):
            unsupported.append(name)
    for name in (
        "return_logprob",
        "return_sampling_mask",
        "custom_logit_processor",
        "input_embeds",
        "positional_embed_overrides",
        "return_hidden_states",
    ):
        value = getattr(req, name)
        if value is not None and value is not False:
            unsupported.append(name)
    if unsupported:
        return "CLOVER does not support: " + ", ".join(unsupported)
    return validate_clover_penalties(req)
