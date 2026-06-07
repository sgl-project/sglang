"""Uniform-state (renoising) block-diffusion algorithm for DiffusionGemma.

Two phases, distinguished by `forward_batch.dllm_is_encoder`:
  * encoder/prefill: one forward to write the context KV cache (no denoising).
  * decoder/denoise: for one canvas, run `max_denoising_steps` reverse steps,
    feeding the previous step's logits back as self-conditioning.

Ported from the HF `diffusion_gemma` EntropyBoundSampler + linear temperature
schedule + StableAndConfident stopping. Defaults follow the checkpoint's
generation_config.json and may be overridden via --dllm-algorithm-config.
"""

from typing import List, Tuple, Union

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class Gemma4Renoise(DllmAlgorithm):
    def __init__(self, config: DllmConfig):
        super().__init__(config)

        ac = config.algorithm_config or {}
        self.max_denoising_steps = ac.get("max_denoising_steps", 48)
        if self.max_denoising_steps < 1:
            raise ValueError("max_denoising_steps must be >= 1")
        s = ac.get("sampler_config", {})
        self.entropy_bound = s.get("entropy_bound", 0.1)
        t = ac.get("temperature_schedule", {})
        self.t_min = t.get("t_min", 0.4)
        self.t_max = t.get("t_max", 0.8)
        st = ac.get("stopping_config", {})
        self.confidence_threshold = st.get("confidence_threshold", 0.005)
        self.stability_threshold = st.get("stability_threshold", 1)

        self.vocab_size = None
        self._base_seed = ac.get("seed", None)
        self._generator = None

    def _temperature(self, step: int) -> float:
        return self.t_min + (self.t_max - self.t_min) * (
            step / self.max_denoising_steps
        )

    def _seed_generator(self, device) -> torch.Generator:
        # Re-seed per decode. Under TP, broadcast rank 0's seed so all ranks draw
        # identical canvases from the all-gathered logits.
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device)
        if self._base_seed is not None:
            seed = int(self._base_seed)
        else:
            seed = int(torch.randint(0, torch.iinfo(torch.int32).max, ()).item())
        tp_group = get_tp_group()
        if tp_group.world_size > 1:
            t = torch.tensor([seed], dtype=torch.int64, device=device)
            tp_group.broadcast(t, src=0)
            seed = int(t.item())
        self._generator.manual_seed(seed)
        return self._generator

    def _init_canvas(self, shape, device, gen) -> torch.Tensor:
        return torch.randint(
            low=0, high=self.vocab_size, size=shape, device=device, generator=gen
        )

    def _renoise(self, denoiser, token_entropy, gen) -> torch.Tensor:
        # EntropyBound: keep the lowest-entropy positions within entropy_bound,
        # re-noise the complement with fresh uniform tokens.
        sorted_e, idx = torch.sort(token_entropy, dim=-1, descending=False)
        cum = torch.cumsum(sorted_e, dim=-1)
        sel = (cum - sorted_e) <= self.entropy_bound
        mask = torch.zeros_like(sel).scatter(-1, idx, sel)
        rand = self._init_canvas(denoiser.shape, denoiser.device, gen)
        return torch.where(mask, denoiser, rand)

    def _stop(self, history, argmax, token_entropy) -> torch.Tensor:
        # Per-request [bs] bool: each request stops on its own stability/confidence.
        bs = argmax.shape[0]
        if len(history) == self.stability_threshold:
            stable = torch.ones(bs, dtype=torch.bool, device=argmax.device)
            for c in history:
                stable &= (argmax == c).all(dim=1)
        else:
            stable = torch.zeros(bs, dtype=torch.bool, device=argmax.device)
        history.append(argmax)
        if len(history) > self.stability_threshold:
            history.pop(0)
        confident = token_entropy.mean(dim=1) < self.confidence_threshold
        return stable & confident

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        if self.vocab_size is None:
            self.vocab_size = model_runner.model_config.hf_config.text_config.vocab_size

        if forward_batch.dllm_is_encoder:
            # Skip an empty (fully-cached) encode round, rope can't reshape 0 tokens.
            if forward_batch.input_ids.numel() == 0:
                return None, [], False
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        bs = forward_batch.batch_size
        L = self.block_size
        device = forward_batch.input_ids.device
        gen = self._seed_generator(device)

        current = self._init_canvas((bs, L), device, gen)
        forward_batch.input_ids[:] = current.reshape(-1)
        self_cond = None
        history: List[torch.Tensor] = []
        out = None
        # Each request freezes its canvas once it stops while the rest keep denoising.
        done = torch.zeros(bs, dtype=torch.bool, device=device)
        # Emit the greedy argmax (not the multinomial canvas), frozen per-item on stop.
        argmax = current.clone()

        for step in reversed(range(1, self.max_denoising_steps + 1)):
            forward_batch.dllm_self_conditioning_logits = self_cond
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)

            logits = (out.logits_output.full_logits / self._temperature(step)).view(
                bs, L, -1
            )
            token_entropy = torch.distributions.Categorical(logits=logits).entropy()
            probs = torch.softmax(logits, dim=-1)
            denoiser = torch.multinomial(
                probs.reshape(bs * L, -1), num_samples=1, generator=gen
            ).view(bs, L)

            keep = done.view(bs, 1)
            argmax = torch.where(keep, argmax, torch.argmax(logits, dim=-1))
            current = torch.where(
                keep, current, self._renoise(denoiser, token_entropy, gen)
            )
            forward_batch.input_ids[:] = current.reshape(-1)
            if self_cond is None:
                self_cond = logits.reshape(bs * L, -1)
            else:
                self_cond = torch.where(
                    done.view(bs, 1, 1), self_cond.view(bs, L, -1), logits
                ).reshape(bs * L, -1)

            done |= self._stop(history, argmax, token_entropy) & (~done)
            if bool(done.all()):
                break

        next_token_ids = [argmax[i] for i in range(bs)]
        return out.logits_output, next_token_ids, out.can_run_graph


Algorithm = Gemma4Renoise
