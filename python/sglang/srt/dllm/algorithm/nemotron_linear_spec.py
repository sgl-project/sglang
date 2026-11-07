# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Greedy diffusion proposals verified by the same model with causal attention."""

import torch

from sglang.srt.dllm.algorithm.nemotron_fast_diffuser import NemotronFastDiffuser


class NemotronLinearSpec(NemotronFastDiffuser):
    retains_causal_prefix = True

    def __init__(self, config):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.0)

    def run(self, model_runner, forward_batch, algo_states=None):
        if not forward_batch.forward_mode.is_dllm_extend():
            return super().run(model_runner, forward_batch, algo_states)
        if algo_states is None or any(state is None for state in algo_states):
            raise RuntimeError("NemotronLinearSpec is missing its causal context seed")

        for _ in range(self.block_size):
            self.prepare_inputs(model_runner, forward_batch, algo_states)
            draft = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            done = self.step(
                forward_batch, draft.logits_output.full_logits, algo_states
            )
            if all(done):
                break

        self.prepare_inputs(model_runner, forward_batch, algo_states)
        forward_batch.dllm_is_causal = True
        try:
            verified = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        finally:
            forward_batch.dllm_is_causal = False

        blocks = forward_batch.input_ids.view(-1, self.block_size)
        predictions = verified.logits_output.full_logits.view(
            -1, self.block_size, verified.logits_output.full_logits.shape[-1]
        ).argmax(dim=-1)
        matches = blocks[:, 1:] == predictions[:, :-1]
        accepted = matches.to(torch.int64).cumprod(dim=-1).sum(dim=-1) + 1
        seeds = predictions.gather(1, (accepted - 1).unsqueeze(-1)).squeeze(-1)
        states = [{"seed": seed, "started": False, "steps": 0} for seed in seeds]
        # The next seed is emitted in the next round, after its causal KV exists.
        return (
            verified.logits_output,
            blocks.tolist(),
            accepted.tolist(),
            states,
            verified.can_run_graph,
        )


Algorithm = NemotronLinearSpec
