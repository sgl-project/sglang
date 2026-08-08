from __future__ import annotations

import hashlib
from typing import Any, List, Optional

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm, DllmRunOutput
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class Gemma4Renoise(DllmAlgorithm):
    def __init__(self, config: DllmConfig):
        super().__init__(config)
        algorithm_config = config.algorithm_config or {}
        self.max_denoising_steps = algorithm_config.get("max_denoising_steps", 48)
        sampler_config = algorithm_config.get("sampler_config", {})
        self.entropy_bound = sampler_config.get("entropy_bound", 0.1)
        temperature_config = algorithm_config.get("temperature_schedule", {})
        self.t_min = temperature_config.get("t_min", 0.4)
        self.t_max = temperature_config.get("t_max", 0.8)
        stopping_config = algorithm_config.get("stopping_config", {})
        self.confidence_threshold = stopping_config.get("confidence_threshold", 0.005)
        self.stability_threshold = stopping_config.get("stability_threshold", 1)
        self.seed = algorithm_config.get("seed")
        self.vocab_size = None
        self.embed_tokens = None

        if self.max_denoising_steps < 1:
            raise ValueError("max_denoising_steps must be at least 1")
        if self.entropy_bound <= 0:
            raise ValueError("entropy_bound must be positive")
        if self.t_min < 0 or self.t_max <= self.t_min:
            raise ValueError("temperature_schedule must satisfy 0 <= t_min < t_max")
        if self.stability_threshold < 0 or self.confidence_threshold <= 0:
            raise ValueError("invalid stopping_config")

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        algo_states: Optional[List[Any]] = None,
    ) -> DllmRunOutput:
        if forward_batch.dllm_is_encoder:
            if forward_batch.input_ids.numel() == 0:
                return None, [], None, None, False
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], None, None, out.can_run_graph

        self.vocab_size = model_runner.model_config.hf_config.text_config.vocab_size
        self.embed_tokens = model_runner.model.get_input_embeddings()
        return super().run(model_runner, forward_batch, algo_states)

    def _block_start_list(self, forward_batch: ForwardBatch) -> List[int]:
        return [0] * forward_batch.batch_size

    def max_steps(self, block_size: int) -> int:
        return self.max_denoising_steps

    def _temperature(self, step: int) -> float:
        return self.t_min + (self.t_max - self.t_min) * (
            step / self.max_denoising_steps
        )

    def _request_seed(self, forward_batch: ForwardBatch, index: int) -> int:
        if self.seed is not None:
            return self.seed
        rid = forward_batch.rids[index] if forward_batch.rids else str(index)
        return int.from_bytes(
            hashlib.blake2b(rid.encode(), digest_size=8).digest(), "little"
        )

    def init_step_state(self, forward_batch: ForwardBatch) -> List[Any]:
        device = forward_batch.input_ids.device
        states = []
        for index in range(forward_batch.batch_size):
            generator = torch.Generator(device=device)
            generator.manual_seed(self._request_seed(forward_batch, index))
            current = torch.randint(
                self.vocab_size,
                (self.block_size,),
                device=device,
                generator=generator,
            )
            states.append(
                {
                    "step": self.max_denoising_steps,
                    "current": current,
                    "argmax": current,
                    "history": [],
                    "self_conditioning": None,
                    "rng_state": generator.get_state(),
                    "finished": False,
                }
            )
        return states

    def prepare_inputs(self, forward_batch: ForwardBatch, states: List[Any]) -> None:
        self._write_inputs(forward_batch, states)

    def _write_inputs(self, forward_batch: ForwardBatch, states: List[Any]) -> None:
        current = torch.stack([state["current"] for state in states])
        forward_batch.input_ids.copy_(current.view(-1))

        signals = [state["self_conditioning"] for state in states]
        signal = next((value for value in signals if value is not None), None)
        if signal is None:
            forward_batch.dllm_self_conditioning_embeds = None
            return
        batched = torch.zeros(
            (len(states), self.block_size, signal.shape[-1]),
            dtype=signal.dtype,
            device=signal.device,
        )
        for index, value in enumerate(signals):
            if value is not None:
                batched[index].copy_(value)
        forward_batch.dllm_self_conditioning_embeds = batched.view(
            len(states) * self.block_size, -1
        )

    def _soft_embeddings(self, processed_logits: torch.Tensor) -> torch.Tensor:
        weight = self.embed_tokens.weight
        logits = processed_logits.to(weight.dtype)
        probabilities = logits.softmax(dim=-1, dtype=torch.float32).to(weight.dtype)
        scale = torch.as_tensor(
            self.embed_tokens.embed_scale, dtype=weight.dtype, device=weight.device
        )
        return torch.matmul(probabilities, weight) * scale

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        logits = full_logits.view(
            forward_batch.batch_size, self.block_size, self.vocab_size
        )
        done = []

        for index, state in enumerate(states):
            if state["finished"]:
                done.append(True)
                continue

            processed_logits = logits[index] / self._temperature(state["step"])
            distribution = torch.distributions.Categorical(logits=processed_logits)
            probabilities = distribution.probs
            generator = torch.Generator(device=processed_logits.device)
            generator.set_state(state["rng_state"])
            denoiser = torch.multinomial(
                probabilities, num_samples=1, generator=generator
            ).squeeze(-1)
            token_entropy = distribution.entropy()
            sorted_entropy, indices = torch.sort(token_entropy)
            cumulative_entropy = torch.cumsum(sorted_entropy, dim=-1)
            selected = cumulative_entropy - sorted_entropy <= self.entropy_bound
            selected = torch.zeros_like(selected).scatter(-1, indices, selected)
            random_canvas = torch.randint(
                self.vocab_size,
                (self.block_size,),
                device=processed_logits.device,
                generator=generator,
            )
            current = torch.where(selected, denoiser, random_canvas)
            argmax = torch.argmax(processed_logits, dim=-1)

            history = state["history"]
            stable = self.stability_threshold == 0 or (
                len(history) == self.stability_threshold
                and all(torch.equal(previous, argmax) for previous in history)
            )
            history.append(argmax)
            if len(history) > self.stability_threshold:
                history.pop(0)
            confident = bool(token_entropy.mean() < self.confidence_threshold)

            state["step"] -= 1
            state["argmax"] = argmax
            state["rng_state"] = generator.get_state()
            state["finished"] = (stable and confident) or state["step"] == 0
            if state["finished"]:
                state["current"] = argmax
                state["self_conditioning"] = None
            else:
                state["current"] = current
                state["self_conditioning"] = self._soft_embeddings(processed_logits)
            done.append(state["finished"])

        self._write_inputs(forward_batch, states)
        return done


Algorithm = Gemma4Renoise
