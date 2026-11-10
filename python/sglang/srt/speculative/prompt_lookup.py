"""Experimental prompt lookup for single-request, greedy Qwen MTP.

Only draft proposals change. The existing target verifier, state commit and
draft extend remain authoritative. CPU history comes from verified GPU outputs,
not the scheduler's potentially one-round-old overlap snapshot.
"""

import logging

import torch

logger = logging.getLogger(__name__)


class PromptSuffixIndex:
    def __init__(self, prompt, width=3, match_length=8):
        self.width = width
        self.match_length = match_length
        self.continuations = {}
        for end in range(match_length, len(prompt) - width + 1):
            self.continuations[tuple(prompt[end - match_length : end])] = tuple(
                prompt[end : end + width]
            )
        self.suffix = list(prompt[-match_length:])

    def accept(self, tokens):
        self.suffix = (self.suffix + list(tokens))[-self.match_length :]

    def propose(self):
        return self.continuations.get(tuple(self.suffix))


class GreedyPromptLookup:
    def __init__(self, width, device):
        self.width = width
        self.device = device
        self.request = None
        self.index = None
        self.pending = None
        self.cpu_tokens = torch.empty(
            (1, width + 1), dtype=torch.int32, pin_memory=True
        )
        self.cpu_length = torch.empty((1,), dtype=torch.int32, pin_memory=True)
        self.hits = self.attempts = 0

    def _drain(self):
        if self.pending is not None:
            event, tokens, lengths = self.pending
            event.synchronize()
            count = int(self.cpu_length[0])
            if not 1 <= count <= self.width + 1:
                raise RuntimeError(f"Invalid prompt-lookup acceptance count: {count}")
            self.index.accept(self.cpu_tokens[0, :count].tolist())
            # Keep copy sources alive until the event has completed.
            self.pending = None

    def reset(self):
        self._drain()
        if self.request is not None:
            logger.info(
                "Prompt lookup: %s, hits=%d, rounds=%d",
                self.request.rid,
                self.hits,
                self.attempts,
            )
        self.request = self.index = None
        self.hits = self.attempts = 0

    def draft(self, batch, bonus_tokens):
        self._drain()
        if (
            batch.forward_mode.is_idle()
            or len(batch.reqs) != 1
            or not batch.sampling_info.is_all_greedy
            or batch.has_grammar
            or batch.reqs[0].multimodal_inputs is not None
        ):
            self.reset()
            return None
        req = batch.reqs[0]
        # Opt-out permits interleaved A/B on one resident model. It cannot
        # enable lookup unless the server experiment is enabled at startup.
        custom = req.sampling_params.custom_params
        if isinstance(custom, dict) and custom.get("prompt_lookup") is False:
            self.reset()
            return None
        if req is not self.request:
            self.reset()
            self.request = req
            self.index = PromptSuffixIndex(req.origin_input_ids, self.width)
            self.index.accept(bonus_tokens.cpu().tolist())
        self.attempts += 1
        proposal = self.index.propose()
        if proposal is None:
            return None
        self.hits += 1
        return torch.tensor([proposal], dtype=torch.int64, device=self.device)

    def observe(self, batch, accepted_tokens, accept_lens):
        if (
            self.request is None
            or len(batch.reqs) != 1
            or batch.reqs[0] is not self.request
        ):
            return
        assert self.pending is None
        self.cpu_tokens.copy_(accepted_tokens, non_blocking=True)
        self.cpu_length.copy_(accept_lens, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self.pending = (event, accepted_tokens, accept_lens)
