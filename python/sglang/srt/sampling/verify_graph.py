from types import SimpleNamespace

import torch


class VerifySamplingBuffers:
    """Graph-stable sampling inputs shared by speculative acceptance adapters."""

    def __init__(
        self,
        max_bs,
        width,
        vocab_size,
        device,
        *,
        with_draft=False,
        with_logits_adjustments=False,
    ):
        self.max_bs, self.width, self.vocab_size = max_bs, width, vocab_size
        self.temperatures = torch.ones((max_bs, 1), device=device)
        self.top_ks = torch.full(
            (max_bs,), vocab_size, dtype=torch.int32, device=device
        )
        self.top_ps = torch.ones(max_bs, device=device)
        self.greedy_mask = torch.ones(max_bs, dtype=torch.bool, device=device)
        self.vocab_mask = torch.full(
            (max_bs * width, (vocab_size + 31) // 32),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.draft_distribution = (
            torch.zeros((max_bs, width - 1, vocab_size), device=device)
            if with_draft
            else None
        )
        self.min_ps = torch.zeros(max_bs, device=device)
        self.need_min_p_sampling = True
        self.sampling_seed = torch.zeros(max_bs, dtype=torch.int64, device=device)
        self.use_sampling_seed = torch.zeros((), dtype=torch.bool, device=device)
        self.acc_additive_penalties = None
        self.acc_scaling_penalties = None
        self.logit_bias = None
        if with_logits_adjustments:
            self.acc_additive_penalties = torch.zeros(
                (max_bs, vocab_size), device=device
            )
            self.acc_scaling_penalties = torch.ones((max_bs, vocab_size), device=device)
            self.logit_bias = torch.zeros((max_bs, vocab_size), device=device)
        self.need_top_k_sampling = self.need_top_p_sampling = True

    def stage(self, info, bs, *, temperatures=None, greedy_mask=None):
        if not 0 < bs <= self.max_bs:
            raise ValueError(f"Sampling batch {bs} exceeds capacity {self.max_bs}")
        temperatures = info.temperatures if temperatures is None else temperatures
        for name, value in (
            ("temperatures", temperatures),
            ("top_ks", info.top_ks),
            ("top_ps", info.top_ps),
        ):
            if value.numel() != bs:
                raise ValueError(f"{name} must contain one value per request")
        if greedy_mask is not None and greedy_mask.numel() != bs:
            raise ValueError("greedy_mask must contain one value per request")
        self.min_ps.zero_()
        if getattr(info, "need_min_p_sampling", False):
            self.min_ps[:bs].copy_(info.min_ps.reshape(bs))
        seed = getattr(info, "sampling_seed", None)
        self.use_sampling_seed.fill_(seed is not None)
        self.sampling_seed.zero_()
        if seed is not None:
            self.sampling_seed[:bs].copy_(seed.reshape(bs))
        for name, neutral in (
            ("acc_additive_penalties", 0),
            ("acc_scaling_penalties", 1),
            ("logit_bias", 0),
        ):
            dest = getattr(self, name)
            if dest is None:
                continue
            src = getattr(info, name, None)
            if src is not None and tuple(src.shape) != (bs, self.vocab_size):
                raise ValueError(f"{name} must have shape {(bs, self.vocab_size)}")
            dest.fill_(neutral)
            if src is not None:
                dest[:bs].copy_(src)
        # Padding executes the same sampling kernels but must have valid distributions.
        self.temperatures[bs:].fill_(1)
        self.top_ks[bs:].fill_(1)
        self.top_ps[bs:].fill_(1)
        self.greedy_mask[bs:].fill_(True)
        self.temperatures[:bs].copy_(temperatures.reshape(bs, 1))
        self.top_ks[:bs].copy_(info.top_ks.reshape(bs))
        self.top_ps[:bs].copy_(info.top_ps.reshape(bs))
        self.greedy_mask[:bs].copy_(
            info.top_ks.reshape(bs) == 1
            if greedy_mask is None
            else greedy_mask.reshape(bs)
        )

    def stage_draft(self, distribution, bs):
        if self.draft_distribution is None:
            raise RuntimeError("Draft distribution storage was not allocated")
        if distribution is None:
            raise RuntimeError("Sampling requires a draft distribution")
        expected = (bs, self.width - 1, self.vocab_size)
        if not 0 < bs <= self.max_bs or tuple(distribution.shape) != expected:
            raise ValueError(
                f"Expected draft proposal shape {expected}, got {tuple(distribution.shape)}"
            )
        self.draft_distribution[:bs].copy_(distribution)

    def for_batch(self, bs, *, is_all_greedy):
        return SimpleNamespace(
            temperatures=self.temperatures[:bs],
            top_ks=self.top_ks[:bs],
            top_ps=self.top_ps[:bs],
            is_all_greedy=is_all_greedy,
            min_ps=self.min_ps[:bs],
            need_min_p_sampling=True,
            sampling_seed=self.sampling_seed[:bs],
            use_sampling_seed=self.use_sampling_seed,
            **{
                name: getattr(self, name)[:bs]
                if getattr(self, name) is not None
                else None
                for name in (
                    "acc_additive_penalties",
                    "acc_scaling_penalties",
                    "logit_bias",
                )
            },
            need_top_k_sampling=True,
            need_top_p_sampling=True,
        )


def verify_logits_adjustments_are_noop(sampling_info, *, allow_grammar=False) -> bool:
    if sampling_info is None:
        return True
    if sampling_info.has_custom_logit_processor:
        return False
    for field in (
        "acc_linear_penalties",
        "acc_additive_penalties",
        "acc_scaling_penalties",
    ):
        if getattr(sampling_info, field, None) is not None:
            return False
    penalizer = getattr(sampling_info, "penalizer_orchestrator", None)
    if penalizer is not None and penalizer.is_required:
        return False
    if not allow_grammar and getattr(sampling_info, "grammar_mask", None) is not None:
        return False
    if getattr(sampling_info, "logit_bias", None) is not None:
        return False
    return True


def apply_verify_logits_adjustments(logits, info, width):
    """Broadcast request penalties over each verification chain in native order."""
    from sglang.srt.sampling.penaltylib.repetition_penalty import (
        apply_scaling_penalties,
    )

    rows = logits.view(-1, width, logits.shape[-1])
    gate = getattr(info, "logits_adjustment_gate", None)
    for name, neutral in (
        ("acc_additive_penalties", 0),
        ("acc_scaling_penalties", 1),
        ("logit_bias", 0),
    ):
        values = getattr(info, name)
        if values is None:
            continue
        # Fallback replays must leave logits untouched for the eager sampler.
        if gate is not None:
            values = torch.where(gate[:, None], values, neutral)
        if name == "acc_scaling_penalties":
            apply_scaling_penalties(rows, values[:, None, :])
        else:
            rows.add_(values[:, None, :])
