"""Pointer head of Kev decision checkpoints.

A Kev checkpoint is a Qwen3.5 backbone saved without an LM head, with its head
in ``head.pt``. Each question runs as its own prompt after the shared state,
and the head scores every option by the final hidden state at the option's
closing token against the one at the prompt's last token.
"""

from __future__ import annotations

import math
from typing import List

import torch
from torch import nn

from sglang.srt.layers.joint_schema_head import parse_decision_layout
from sglang.srt.layers.pooler import EmbeddingPoolerOutput


class PointerPooler(nn.Module):
    """Option logits per request, from a one-question decision layout whose
    question span ends at the prompt's last token and whose option spans end
    at the options' closing tokens."""

    def __init__(self, hidden_size: int, meta: dict) -> None:
        super().__init__()
        head_dim = meta["head_dim"]
        # The checkpoint trains and serves the head in FP32.
        self.q = nn.Linear(hidden_size, head_dim, dtype=torch.float32)
        self.k = nn.Linear(hidden_size, head_dim, dtype=torch.float32)
        self.scale = 1 / math.sqrt(head_dim)
        self.temperature = float(meta["temperature"])

    def forward(
        self, hidden_states: torch.Tensor, forward_batch
    ) -> EmbeddingPoolerOutput:
        lengths = forward_batch.extend_seq_lens_cpu
        embeddings: List[torch.Tensor] = [
            hidden_states.new_empty(0, dtype=torch.float32)
            for _ in range(forward_batch.batch_size)
        ]
        if forward_batch.decision_layouts is None or lengths is None:
            return EmbeddingPoolerOutput(embeddings=embeddings)
        prefix_lengths = forward_batch.extend_prefix_lens_cpu or [0] * len(lengths)
        scored, decide, options, owners = [], [], [], []
        offset = 0
        for i, (layout, prefix, length) in enumerate(
            zip(forward_batch.decision_layouts, prefix_lengths, lengths)
        ):
            # A layout that no longer fits its prompt gets no scores, which the
            # decision route reports as a server error.
            questions = None
            if layout is not None and prefix == 0:
                try:
                    questions = parse_decision_layout(layout, length)
                except ValueError:
                    pass
            if questions is not None and len(questions) == 1:
                question = questions[0]
                owners += [len(scored)] * len(question.option_spans)
                scored.append((i, len(question.option_spans)))
                decide.append(offset + question.question_span[1] - 1)
                options += [offset + end - 1 for _, end in question.option_spans]
            offset += length
        if not scored:
            return EmbeddingPoolerOutput(embeddings=embeddings)
        device = hidden_states.device
        h_decide = hidden_states[torch.tensor(decide, device=device)].float()
        h_options = hidden_states[torch.tensor(options, device=device)].float()
        owner = torch.tensor(owners, device=device)
        logits = (self.k(h_options) * self.q(h_decide)[owner]).sum(-1) * self.scale
        logits = logits / self.temperature
        for (i, _), values in zip(scored, logits.split([n for _, n in scored])):
            embeddings[i] = values
        return EmbeddingPoolerOutput(embeddings=embeddings)
