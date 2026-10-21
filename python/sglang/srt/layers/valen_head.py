"""Valen's FP32 role mixer and packed per-request span layout.

Each candidate reads context, instruction, candidate and decision roles. The
backbone runs once for all questions; the head batches candidates across requests.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from sglang.srt.layers.pooler import EmbeddingPoolerOutput


@dataclass(frozen=True)
class ValenQuestion:
    instruction: tuple[int, int]
    candidates: tuple[tuple[int, int], ...]
    decision: int


def pack_valen_layout(length: int, context_end: int, questions) -> list[int]:
    """Pack pre-expansion positions into the request's integer layout.

    Header: prompt length, context end, question count. Each question stores
    instruction start/end, decision position, candidate count, then span pairs.
    """
    layout = [length, context_end, len(questions)]
    for q in questions:
        layout.extend((*q.instruction, q.decision, len(q.candidates)))
        for span in q.candidates:
            layout.extend(span)
    return layout


def parse_valen_layout(layout: Sequence[int], length: int):
    """Validate a layout and place its roles after image placeholder expansion."""
    if len(layout) < 3 or any(type(v) is not int for v in layout):
        raise ValueError("Valen decision_layout must contain integers")
    original, context, count = layout[:3]
    shift = length - original
    if not 0 < context < original or shift < 0 or count < 1:
        raise ValueError("Valen decision_layout does not match the prompt")
    cursor, questions = 3, []
    for _ in range(count):
        if cursor + 4 > len(layout):
            raise ValueError("Valen decision_layout is truncated")
        start, end, decision, n = layout[cursor : cursor + 4]
        cursor += 4
        if not context <= start < end <= decision < original or not 1 <= n <= 255:
            raise ValueError("Valen decision_layout has an invalid question")
        spans = []
        for _ in range(n):
            if cursor + 2 > len(layout):
                raise ValueError("Valen decision_layout is truncated")
            a, b = layout[cursor : cursor + 2]
            cursor += 2
            if not end <= a < b <= decision:
                raise ValueError("Valen decision_layout has an invalid candidate")
            spans.append((a + shift, b + shift))
        questions.append(
            ValenQuestion((start + shift, end + shift), tuple(spans), decision + shift)
        )
    if cursor != len(layout):
        raise ValueError("Valen decision_layout has trailing values")
    # Media expands inside the context, not before its first token.
    return context + shift, questions


class MixerBlock(nn.Module):
    def __init__(self, width, token_hidden, channel_hidden):
        super().__init__()
        self.token_norm = nn.LayerNorm(width, dtype=torch.float32)
        self.token_mlp = nn.Sequential(
            nn.Linear(4, token_hidden, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(token_hidden, 4, dtype=torch.float32),
        )
        self.channel_norm = nn.LayerNorm(width, dtype=torch.float32)
        self.channel_mlp = nn.Sequential(
            nn.Linear(width, channel_hidden, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(channel_hidden, width, dtype=torch.float32),
        )

    def forward(self, x):
        x = x + self.token_mlp(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channel_mlp(self.channel_norm(x))


class ValenMixerHead(nn.Module):
    def __init__(self, hidden_size, config):
        super().__init__()
        defaults = dict(
            head_width=256,
            head_layers=2,
            head_token_hidden_dim=16,
            head_channel_hidden_dim=1024,
        )
        settings = {k: config.get(k, v) for k, v in defaults.items()}
        if config.get("head_type", "mixer") != "mixer" or any(
            type(v) is not int or v <= 0 for v in settings.values()
        ):
            raise ValueError(
                "Valen requires a mixer head with positive integer dimensions"
            )
        width = settings["head_width"]
        self.input_norm = nn.LayerNorm(hidden_size, dtype=torch.float32)
        self.projection = nn.Linear(hidden_size, width, dtype=torch.float32)
        self.roles = nn.Parameter(torch.zeros(1, 4, width, dtype=torch.float32))
        self.blocks = nn.Sequential(
            *[
                MixerBlock(
                    width,
                    settings["head_token_hidden_dim"],
                    settings["head_channel_hidden_dim"],
                )
                for _ in range(settings["head_layers"])
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(width, dtype=torch.float32),
            nn.Linear(width, 1, dtype=torch.float32),
        )

    def forward(self, features):
        x = self.projection(self.input_norm(features.float())) + self.roles
        return self.output(self.blocks(x)[:, 2]).squeeze(-1)


class ValenPooler(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    @torch.no_grad()
    def forward(self, hidden_states, forward_batch):
        outputs = [
            hidden_states.new_empty(0, dtype=torch.float32)
            for _ in range(forward_batch.batch_size)
        ]
        if (
            forward_batch.decision_layouts is None
            or forward_batch.extend_seq_lens_cpu is None
        ):
            return EmbeddingPoolerOutput(embeddings=outputs)
        lengths = forward_batch.extend_seq_lens_cpu
        prefixes = forward_batch.extend_prefix_lens_cpu or [0] * len(lengths)
        features, owners, counts = [], [], []
        offset = 0
        for i, (layout, length, prefix) in enumerate(
            zip(forward_batch.decision_layouts, lengths, prefixes)
        ):
            hidden = hidden_states[offset : offset + length]
            offset += length
            if layout is None:  # Health probe.
                continue
            if prefix:
                continue
            try:
                context_end, questions = parse_valen_layout(layout, length)
            except ValueError:
                continue
            context = hidden[:context_end].float().mean(0)
            count = 0
            for q in questions:
                instruction = (
                    hidden[q.instruction[0] : q.instruction[1]].float().mean(0)
                )
                decision = hidden[q.decision].float()
                for a, b in q.candidates:
                    features.append(
                        torch.stack(
                            (
                                context,
                                instruction,
                                hidden[a:b].float().mean(0),
                                decision,
                            )
                        )
                    )
                    count += 1
            owners.append(i)
            counts.append(count)
        if features:
            logits = self.head(torch.stack(features))
            for owner, values in zip(owners, logits.split(counts)):
                outputs[owner] = values
        return EmbeddingPoolerOutput(embeddings=outputs)
