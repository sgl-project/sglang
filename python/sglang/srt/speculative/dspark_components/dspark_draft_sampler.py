from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    SampleStepTokens,
)
from sglang.srt.environ import DsparkFoldedSampling, envs
from sglang.srt.utils import get_available_gpu_memory

logger = logging.getLogger(__name__)

# Same free-memory floor init_cuda_graphs requires before draft capture.
_CAPTURE_HEADROOM_GB = 1.0


def greedy_step_sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
    del step_idx
    return torch.argmax(step_logits, dim=-1)


class DsparkDraftSampler:
    """Draft proposal head folded into the draft graph as a tail hook; with
    folded_sampling it also Gumbel-samples non-greedy rows in-graph."""

    def __init__(
        self,
        *,
        model,
        gamma,
        max_bs,
        device,
        confidence_fn=None,
        out=None,
        folded_sampling: bool = True,
    ):
        self.model = model
        self.markov_head = model.markov_head
        self.gamma = int(gamma)
        max_bs = int(max_bs)
        if out is not None:
            assert out.shape == (max_bs * self.gamma,) and out.dtype == torch.int64
            self.out = out
        else:
            self.out = torch.empty(
                (max_bs * self.gamma,), dtype=torch.int64, device=device
            )
        self.confidence_fn = confidence_fn
        self.confidence_out = (
            torch.empty((max_bs, self.gamma), dtype=torch.float32, device=device)
            if confidence_fn is not None
            else None
        )
        self.folded_sampling = folded_sampling
        self.temperatures = None
        self.greedy_mask = None
        self.kernel_greedy_mask = None
        self.exp_noise = None
        self.corrected_out = None
        self.write_corrected_logits = None
        self._staged_all_greedy = True
        if folded_sampling:
            vocab = int(model.lm_head.org_vocab_size)
            self.temperatures = torch.ones(
                (max_bs,), dtype=torch.float32, device=device
            )
            self.greedy_mask = torch.ones((max_bs,), dtype=torch.bool, device=device)
            # Keep the dtype consumed by the Triton kernel resident. Converting
            # the bool request mask inside the captured Markov loop would add
            # one graph op for every proposal step.
            self.kernel_greedy_mask = torch.ones(
                (max_bs,), dtype=torch.int32, device=device
            )
            self.exp_noise = torch.empty(
                (max_bs, vocab), dtype=torch.float32, device=device
            )
            self.corrected_out = torch.empty(
                (max_bs * self.gamma, vocab),
                dtype=model.lm_head.weight.dtype,
                device=device,
            )
            self.write_corrected_logits = torch.zeros(
                (), dtype=torch.int32, device=device
            )

    def stage_sampling_params(self, *, bs: int, sampling_info) -> None:
        """Host-side refresh of the static sampling params; must run before
        the draft graph replay that consumes them."""
        if not self.folded_sampling:
            return
        all_greedy = sampling_info is None or sampling_info.is_all_greedy
        if all_greedy:
            # These buffers are initialized to the greedy state and remain
            # valid across graph replays. Only restore them once after a
            # stochastic batch instead of launching three tiny device ops on
            # every greedy decode iteration.
            if not self._staged_all_greedy:
                self.greedy_mask.fill_(True)
                self.kernel_greedy_mask.fill_(1)
                self.write_corrected_logits.zero_()
                self._staged_all_greedy = True
            return

        torch.clamp(
            sampling_info.temperatures.view(-1)[:bs].to(torch.float32),
            min=1e-5,
            out=self.temperatures[:bs],
        )
        live_greedy_mask = (sampling_info.top_ks <= 1).view(-1)[:bs]
        self.greedy_mask[:bs].copy_(live_greedy_mask)
        self.kernel_greedy_mask[:bs].copy_(live_greedy_mask)
        # CUDA-graph buckets can be larger than the live batch. Reset their
        # tail rows so a smaller request cannot inherit stochastic flags from
        # a previous larger replay and pay full-vocabulary noise/log work.
        self.greedy_mask[bs:].fill_(True)
        self.kernel_greedy_mask[bs:].fill_(1)
        if self._staged_all_greedy:
            self.write_corrected_logits.fill_(1)
        self._staged_all_greedy = False
        # Refresh stochastic noise immediately before replay instead of baking
        # RNG into every captured proposal.
        self.exp_noise[:bs].exponential_(1)

    def __call__(self, hidden_states, input_ids):
        bs = hidden_states.shape[0] // self.gamma
        base_logits, confidence_tap = self.model.compute_base_logits(hidden_states)
        base_logits = base_logits.view(bs, self.gamma, -1)
        anchor = input_ids.view(bs, self.gamma)[:, 0]

        if self.folded_sampling:

            def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
                return SampleStepTokens.execute(
                    step_logits=step_logits,
                    temperatures=self.temperatures[:bs],
                    greedy_mask=self.kernel_greedy_mask[:bs],
                    exp_noise=self.exp_noise[:bs],
                    corrected_logits_out=self.corrected_out.view(
                        -1, self.gamma, step_logits.shape[-1]
                    )[:bs, step_idx, :],
                    write_corrected_logits=self.write_corrected_logits,
                )

        else:
            sampler = greedy_step_sampler

        if not self.folded_sampling and getattr(
            self.markov_head, "keeps_base_logits_tp_sharded", False
        ):
            draft_tokens = self.markov_head.sample_greedy_block_sharded(
                base_logits, first_prev_tokens=anchor
            )
            corrected_logits = None
        else:
            draft_tokens, corrected_logits = self.markov_head.sample_block(
                base_logits,
                first_prev_tokens=anchor,
                hidden_states=hidden_states.view(bs, self.gamma, -1),
                sampler=sampler,
                return_corrected_logits=False,
            )
        self.out[: draft_tokens.numel()].copy_(draft_tokens.reshape(-1))
        assert corrected_logits is None
        if self.confidence_out is not None:
            confidence = self.confidence_fn(
                draft_hidden=hidden_states.view(bs, self.gamma, -1),
                anchor_tokens=anchor,
                draft_tokens=draft_tokens,
                confidence_tap=confidence_tap,
            )
            self.confidence_out[:bs].copy_(confidence)


def _resolve_folded_sampling(*, model, gamma, max_bs, device, tp_rank) -> bool:
    """The sampling buffers are baked into the captured draft graph, so AUTO
    must decide before capture from a free-memory probe."""
    mode = envs.SGLANG_DSPARK_FOLDED_SAMPLING.get()
    if mode == DsparkFoldedSampling.OFF:
        return False
    if mode == DsparkFoldedSampling.FORCE:
        return True
    if getattr(model.markov_head, "keeps_base_logits_tp_sharded", False):
        # AUTO favors the greedy-only distributed top-1 graph when the dense
        # Markov head follows lm_head's vocab shard. Sampling requests still
        # use the exact eager fallback; FORCE retains the mixed in-graph path.
        if tp_rank == 0:
            logger.info(
                "DSpark folded sampling AUTO selected greedy-only TP-sharded "
                "proposal; stochastic requests use the eager proposal path."
            )
        return False
    vocab = int(model.lm_head.org_vocab_size)
    noise_bytes = max_bs * vocab * 4
    logits_bytes = max_bs * gamma * vocab * model.lm_head.weight.dtype.itemsize
    need_gb = (noise_bytes + logits_bytes) / (1 << 30)
    available_gb = get_available_gpu_memory(
        device, torch.get_device_module().current_device()
    )
    if available_gb - need_gb >= _CAPTURE_HEADROOM_GB:
        return True
    if tp_rank == 0:
        logger.warning(
            "DSpark folded sampling disabled: its static buffers need %.2f GB "
            "but only %.2f GB GPU memory is free; sampling batches will take "
            "the eager proposal path. Set SGLANG_DSPARK_FOLDED_SAMPLING=%d "
            "to force.",
            need_gb,
            available_gb,
            int(DsparkFoldedSampling.FORCE),
        )
    return False


def maybe_build_draft_sampler(
    *,
    draft_model,
    gamma: int,
    max_bs: int,
    device,
    tp_rank: int,
    confidence_fn=None,
    out=None,
) -> Optional[DsparkDraftSampler]:
    """Build the graph-folded draft sampler, or None (reason logged) when the
    proposal must stay eager."""

    def _eager(reason):
        if tp_rank == 0:
            logger.info("DSpark draft proposal kept eager (reason=%s).", reason)
        return None

    if gamma <= 0:
        return _eager("gamma<=0")
    if not hasattr(draft_model, "compute_base_logits"):
        return _eager("no compute_base_logits")
    if getattr(draft_model, "markov_head", None) is None:
        return _eager("no markov head")
    folded_sampling = _resolve_folded_sampling(
        model=draft_model, gamma=gamma, max_bs=max_bs, device=device, tp_rank=tp_rank
    )
    if tp_rank == 0:
        logger.info(
            "DSpark draft proposal (%s) folded into the draft cuda graph.",
            "greedy + sampling" if folded_sampling else "greedy only",
        )
    return DsparkDraftSampler(
        model=draft_model,
        gamma=gamma,
        max_bs=max_bs,
        device=device,
        confidence_fn=confidence_fn,
        out=out,
        folded_sampling=folded_sampling,
    )
