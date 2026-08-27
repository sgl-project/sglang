from typing import Any, List, Tuple

import torch
from torch.nn import functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def top_p_logits(logits: torch.Tensor, top_p: float | None = None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits: torch.Tensor, top_k: int | None = None):
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    margin_confidence: bool = False,
    neg_entropy: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match Dream's official ``sample_tokens`` interface and return order."""
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1.0:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = F.softmax(logits, dim=-1)

    if temperature > 0:
        x0 = torch.distributions.Categorical(probs=probs).sample()
        confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[..., 0] - sorted_probs[..., 1]

    if neg_entropy:
        confidence = torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

    return confidence, x0


class Dream(DllmAlgorithm):
    """Dream's official denoising schedules."""

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        configured_steps = config.algorithm_config.get("steps") or 512
        self.steps = max(1, int(configured_steps))
        self.eps = float(config.algorithm_config.get("eps", 1e-3))
        self.alg = config.algorithm_config.get("alg", "origin")
        self.alg_temp = config.algorithm_config.get("alg_temp", None)

    def max_steps(self, block_size: int) -> int:
        return self.steps

    def init_step_state(self, forward_batch: ForwardBatch) -> List[Any]:
        return [{"step": 0} for _ in range(forward_batch.batch_size)]

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        lengths = forward_batch.extend_seq_lens_cpu
        if lengths is None:
            raise RuntimeError("Dream requires CPU sequence lengths")
        input_parts = forward_batch.input_ids.split(lengths)
        logits_parts = full_logits.split(lengths)
        sampling = forward_batch.sampling_info
        timesteps = torch.linspace(
            1.0,
            self.eps,
            self.steps + 1,
            device=full_logits.device,
        )
        done = []

        for i, (ids, logits) in enumerate(zip(input_parts, logits_parts)):
            prompt_len = states[i]["prompt_len"]
            generation_ids = ids[prompt_len:]
            mask = generation_ids.eq(self.mask_id)
            num_mask = int(mask.sum().item())
            if num_mask == 0:
                done.append(True)
                continue

            step = states[i]["step"]
            token_logits = logits[prompt_len:][mask]
            temperature = float(sampling.original_temperatures[i].item())
            top_k = int(sampling.original_top_ks[i].item())
            top_p = float(sampling.top_ps[i].item())
            if top_k < 0 or top_k >= token_logits.shape[-1]:
                top_k = None
            # Dream's official schedule is linspace(1, eps, S + 1);
            # transfer the fraction 1-s/t and resolve all remaining masks
            # on the final step.
            t = timesteps[step]
            s = timesteps[step + 1]
            p_transfer = 1.0 if step >= self.steps - 1 else 1.0 - s / t
            mask_positions = mask.nonzero(as_tuple=False).flatten()

            if self.alg == "origin":
                # This is the official origin algorithm: choose each masked
                # position independently, then sample tokens only for the
                # selected positions. alg_temp is intentionally unused.
                transfer_mask = (
                    torch.rand(
                        num_mask,
                        device=token_logits.device,
                        dtype=torch.float32,
                    )
                    < p_transfer
                )
                x0 = (
                    torch.zeros(
                        num_mask,
                        device=token_logits.device,
                        dtype=torch.long,
                    )
                    + self.mask_id
                )
                if transfer_mask.any():
                    _, x0[transfer_mask] = sample_tokens(
                        token_logits[transfer_mask],
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
                generation_ids[mask_positions] = x0.clone()
                states[i]["step"] = step + 1
                done.append(step >= self.steps - 1)
                continue

            if self.alg == "maskgit_plus":
                confidence, x0 = sample_tokens(
                    token_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            elif self.alg == "topk_margin":
                confidence, x0 = sample_tokens(
                    token_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    margin_confidence=True,
                )
            elif self.alg == "entropy":
                confidence, x0 = sample_tokens(
                    token_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    neg_entropy=True,
                )
            else:
                raise RuntimeError(f"Unknown alg: {self.alg}")
            if step >= self.steps - 1:
                num_transfer = num_mask
            else:
                num_transfer = min(
                    int(num_mask * p_transfer.item()),
                    num_mask,
                )
            states[i]["step"] = step + 1

            if num_transfer == 0:
                done.append(False)
                continue
            if self.alg_temp is None or float(self.alg_temp) == 0:
                chosen = torch.topk(confidence, num_transfer).indices
            else:
                transfer_probs = F.softmax(confidence / float(self.alg_temp), dim=-1)
                chosen = torch.multinomial(transfer_probs, num_samples=num_transfer)
            generation_ids[mask_positions[chosen]] = x0[chosen]
            # Dream discards this request's KV instead of inserting it into
            # radix cache, so the final transfer can be emitted immediately.
            done.append(step >= self.steps - 1)

        return done


Algorithm = Dream
