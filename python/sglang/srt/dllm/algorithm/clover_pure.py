"""Single-request Clover pure block diffusion on the separate-context scheduler."""

import math

from copy import copy

import torch

from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.clover_sampling import (
    CloverPenaltyState,
    greedy_with_confidence,
    resolve_stop_ids,
    validate_penalties,
)
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase


def scheduled_transfer(confidence, pending, threshold, target):
    # Count committed positions towards the target without synchronizing the device.
    scores = confidence.masked_fill(~pending, torch.inf)
    slots = scores.argsort(descending=True, stable=True)[:target]
    transfer = confidence > threshold
    transfer[slots] = True
    return transfer & pending


class CloverPureBlock(DllmAlgorithm):
    supported_architectures = ("CloverMoeModelLM",)
    requires_separate_context_encoding = True
    requires_request_inputs = True
    required_attention_backend = "flashinfer"

    def __init__(self, config):
        super().__init__(config)
        self.decode_defaults = config.algorithm_config

    @classmethod
    def configure_server_args(cls, server_args):
        cfg = resolving_view(server_args)
        if (
            cfg.dllm_fdfo
            or cfg.dp_size != 1
            or cfg.tp_size != 1
            or cfg.max_running_requests not in (None, 1)
        ):
            raise ValueError(
                "CloverPureBlock MVP requires synchronous single-request TP=1"
            )
        if cfg.enable_unified_memory or cfg.kv_cache_dtype not in ("auto", "bfloat16"):
            raise ValueError("CloverPureBlock MVP requires a static BF16 KV pool")
        if (
            cfg.enable_linear_replayssm
            or cfg.enable_linear_replayssm_spec
            or cfg.enable_mis
        ):
            raise ValueError(
                "CloverPureBlock requires materialized single-sequence GDN state"
            )
        if cfg.enable_tf32_matmul or cfg.enable_fp32_lm_head:
            raise ValueError(
                "CloverPureBlock uses the checkpoint BF16 projection settings"
            )
        graph = with_phase(
            cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
        )
        declare_resolution(
            server_args,
            cls.__name__,
            max_running_requests=1,
            disable_radix_cache=True,
            page_size=1,
            chunked_prefill_size=-1,
            cuda_graph_config=with_phase(graph, Phase.DECODE, backend=Backend.DISABLED),
        )

    @classmethod
    def validate_request(cls, req):
        sp = req.sampling_params
        if (
            sp.top_k != 1
            or req.stream
            or req.return_logprob
            or sp.ignore_eos
            or sp.stop_strs
            or sp.stop_regex_strs
            or sp.logit_bias
            or sp.watermark
            or sp.min_p
            or sp.top_p != 1
            or sp.n != 1
            or (sp.beam_width or 1) != 1
            or req.custom_logit_processor
            or sp.regex
            or sp.json_schema
            or sp.ebnf
            or sp.structural_tag
            or sp.min_new_tokens
        ):
            return "CloverPureBlock MVP supports non-streaming greedy generation with token stops"

    def run(self, model_runner, forward_batch, algo_states=None):
        if not forward_batch.forward_mode.is_dllm_extend():
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], None, None, out.can_run_graph

        (req,) = algo_states
        block = forward_batch.input_ids.view(1, self.block_size)
        block.fill_(self.mask_id)
        gen = copy(model_runner.model_config.hf_generation_config)
        options = {**self.decode_defaults, **(req.sampling_params.custom_params or {})}
        for name, value in options.items():
            if name.startswith("dllm_"):
                setattr(gen, name, value)
        for name in ("presence_penalty", "frequency_penalty", "repetition_penalty"):
            setattr(gen, name, getattr(req.sampling_params, name))
        validate_penalties(gen, model_runner.model_config.vocab_size)
        stops = set(req.sampling_params.stop_token_ids or ())
        default_stops, pad = resolve_stop_ids(gen, model_runner.model_config.hf_config)
        stops.update(default_stops)
        penalties = CloverPenaltyState(
            gen, model_runner.model_config.vocab_size, block.device, list(stops) + [pad]
        )
        history = torch.as_tensor(
            list(req.output_ids), dtype=torch.long, device=block.device
        )
        penalties.commit(history)
        threshold = options.get(
            "threshold",
            model_runner.model_config.hf_config.denoising_config["threshold"],
        )
        positions = torch.arange(self.block_size, device=block.device)
        budget = options.get("max_denoise_steps")
        refine = options.get("refine_uncertain", False)
        uncertain = torch.zeros_like(block[0], dtype=torch.bool) if refine else None

        def forward_logits():
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits = out.logits_output.full_logits.float()
            logits[:, self.mask_id] = -torch.inf
            penalties.apply(logits, block, self.mask_id)
            return out, logits

        for step in range(budget or self.block_size):
            out, logits = forward_logits()
            if step == 0:
                # No page, position, or recurrent slot changes within this block.
                forward_batch.mark_forward_metadata_ready(replan_equivalent=True)
            pred, confidence = greedy_with_confidence(logits)
            confidence.masked_fill_(block[0] != self.mask_id, -torch.inf)
            if budget is None:
                transfer = confidence > threshold
            else:
                # A small number of anchors first, then increasing parallel commits.
                # Explicit final target avoids sensitivity to cos(pi/2) rounding.
                target = (
                    self.block_size
                    if step + 1 == budget
                    else math.ceil(
                        self.block_size
                        * (1 - math.cos(math.pi * (step + 1) / (2 * budget)))
                    )
                )
                transfer = scheduled_transfer(
                    confidence, block[0] == self.mask_id, threshold, target
                )
            fallback = (positions == confidence.argmax()) & ~transfer.any()
            update = transfer | fallback
            if refine:
                uncertain |= update & (confidence <= threshold)
            block[0].copy_(torch.where(update, pred, block[0]))
            if not (block == self.mask_id).any():
                break
        else:
            raise RuntimeError("Unresolved Clover masks")
        if refine:
            for parity in (0, 1):
                revisit = (positions % 2 == parity) & uncertain
                if not revisit.any():
                    continue
                block[0].masked_fill_(revisit, self.mask_id)
                out, logits = forward_logits()
                pred = logits.argmax(-1)
                block[0].copy_(torch.where(revisit, pred, block[0]))
        limit = min(
            self.block_size, req.sampling_params.max_new_tokens - len(req.output_ids)
        )
        for i, token in enumerate(block[0, :limit].tolist()):
            if token in stops:
                limit = i + 1
                break
        return out.logits_output, [block[0, :limit]], None, None, out.can_run_graph


Algorithm = CloverPureBlock
