# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Greedy block diffusion with a causal next-token seed."""

import math

import torch

from sglang.srt.arg_groups.model_override_base import model_config_of, resolving_view
from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.dllm.algorithm.base import DllmAlgorithm


class NemotronFastDiffuser(DllmAlgorithm):
    supported_architectures = ("NemotronLabsDiffusionModel",)
    requires_separate_context_encoding = True
    required_attention_backend = "flashinfer"
    supported_attention_backends = ("flashinfer", "fa4")

    @classmethod
    def configure_server_args(cls, server_args):
        cfg = resolving_view(server_args)
        if cfg.device != "cuda":
            raise ValueError("NemotronFastDiffuser requires a CUDA GPU")
        if not cfg.dllm_fdfo:
            raise ValueError("NemotronFastDiffuser requires FDFO scheduling")
        if cfg.pp_size > 1 or cfg.dcp_size > 1 or cfg.attn_cp_size > 1:
            raise ValueError("NemotronFastDiffuser does not support PP or CP")
        if cfg.enable_dp_attention:
            raise ValueError("NemotronFastDiffuser does not support attention DP")
        if cfg.enable_lora or model_config_of(server_args).quantization is not None:
            raise ValueError(
                "NemotronFastDiffuser does not support LoRA or quantized weights"
            )
        if cfg.speculative_algorithm is not None or cfg.enable_two_batch_overlap:
            raise ValueError(
                "NemotronFastDiffuser does not support speculative or two-batch overlap decoding"
            )
        declare_resolution(
            server_args,
            cls.__name__,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
        )

    @classmethod
    def validate_request(cls, req):
        sp = req.sampling_params
        if (
            sp.top_k != 1
            or sp.frequency_penalty != 0
            or sp.presence_penalty != 0
            or sp.repetition_penalty != 1
            or sp.min_new_tokens != 0
            or sp.min_p != 0
            or sp.logit_bias
            or sp.json_schema is not None
            or sp.regex is not None
            or sp.ebnf is not None
            or sp.structural_tag is not None
            or req.custom_logit_processor
            or req.return_logprob
            or req.return_sampling_mask
        ):
            return "NemotronFastDiffuser supports greedy decoding without penalties, grammars, custom processors or logprobs; use temperature=0"
        return None

    def __init__(self, config):
        super().__init__(config)
        options = config.algorithm_config
        unknown = options.keys() - {"block_size", "threshold"}
        if unknown:
            raise ValueError(f"Unknown NemotronFastDiffuser options: {sorted(unknown)}")
        self.threshold = options.get("threshold", 0.9)
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not math.isfinite(self.threshold)
            or not 0 <= self.threshold <= 1
        ):
            raise ValueError("threshold must be between 0 and 1")
        if (
            isinstance(self.block_size, bool)
            or not isinstance(self.block_size, int)
            or self.block_size < 2
        ):
            raise ValueError("NemotronFastDiffuser block_size must be an integer >= 2")

    def run(self, model_runner, forward_batch, algo_states=None):
        if not forward_batch.forward_mode.is_dllm_extend():
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            seeds = out.logits_output.next_token_logits.argmax(dim=-1)
            states = [{"seed": seed, "started": False, "steps": 0} for seed in seeds]
            return out.logits_output, [], None, states, out.can_run_graph
        if algo_states is None or any(state is None for state in algo_states):
            raise RuntimeError(
                "NemotronFastDiffuser is missing its causal context seed"
            )
        return super().run(model_runner, forward_batch, algo_states)

    def prepare_inputs(self, model_runner, forward_batch, states):
        blocks = forward_batch.input_ids.view(-1, self.block_size)
        for block, state in zip(blocks, states):
            if not state["started"]:
                block.fill_(self.mask_id)
                block[0] = state["seed"]
                state["started"] = True

        forward_batch.input_embeds = model_runner.model.get_input_embeddings()(
            forward_batch.input_ids
        )

    def step(self, forward_batch, full_logits, states):
        blocks = forward_batch.input_ids.view(-1, self.block_size)
        logits = full_logits.view(-1, self.block_size, full_logits.shape[-1])
        predictions = logits.argmax(dim=-1)
        confidence = (
            logits.softmax(dim=-1).gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
        )
        masked = blocks == self.mask_id
        confidence = confidence.masked_fill(~masked, -torch.inf)
        best = confidence.argmax(dim=-1, keepdim=True)
        commit = masked & (confidence >= self.threshold)
        commit.scatter_(1, best, masked.gather(1, best))
        blocks.copy_(torch.where(commit, predictions, blocks))
        done = (blocks != self.mask_id).all(dim=-1).tolist()
        for index, state in enumerate(states):
            state["steps"] += 1
            if not done[index] and state["steps"] >= self.block_size:
                raise RuntimeError(
                    "NemotronFastDiffuser could not resolve its mask tokens"
                )
        return done


Algorithm = NemotronFastDiffuser
