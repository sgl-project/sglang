"""Joint schema head of Clef decision checkpoints.

A Clef checkpoint is a Qwen3.5 backbone saved with ``joint_head_config.json``
and ``joint_head.safetensors``. The head reads the backbone's final hidden
state at every prompt token and scores every allowed option of every question
together: each option gathers evidence from the prompt, the questions attend
to each other and to the prompt, and a lexical prior built from the LM head
rows of each option's tokens keeps the option's meaning.

One head pass serves a whole prefill batch. Token-wise steps run on the flat
batch, attention is variable-length per request, and span means come from
per-request prefix sums over the schema region only, so a long state costs no
more than its memory projection.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import msgspec
import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.pooler import EmbeddingPoolerOutput

# Question types in the order of the head's type embedding.
QUESTION_TYPES = ("noul", "choice", "score")

# Bounds of one head pass, so a large prefill batch is scored in a few passes.
_MAX_GROUP_TOKENS = 65536
_MAX_GROUP_SLOTS = 1 << 18


class LayoutQuestion(msgspec.Struct, frozen=True):
    question_type: int
    question_span: Tuple[int, int]
    option_spans: Tuple[Tuple[int, int], ...]


def max_joint_prompt_tokens(
    *,
    context_len: int,
    num_reserved_tokens: int,
    max_req_input_len: int,
    max_prefill_tokens: int,
) -> int:
    """The longest prompt, images expanded, that the server runs as one prefill.

    The tokenizer manager refuses a prompt of context_len - num_reserved_tokens
    tokens, the scheduler one of max_req_input_len, and without chunked prefill
    activation memory is reserved for max_prefill_tokens.
    """
    return min(
        context_len - num_reserved_tokens - 1,
        max_req_input_len - 1,
        max_prefill_tokens,
    )


def pack_decision_layout(
    prompt_length: int, questions: Sequence[LayoutQuestion]
) -> List[int]:
    """Flatten a prompt's questions into the ints a request carries.

    The layout is ``[prompt_length, num_questions]`` followed, per question,
    by ``type, question_start, question_end, num_options`` and one
    ``option_start, option_end`` pair per option. Positions index the prompt
    before its image placeholders expand.
    """
    layout = [prompt_length, len(questions)]
    for question in questions:
        layout += [
            question.question_type,
            *question.question_span,
            len(question.option_spans),
        ]
        for start, end in question.option_spans:
            layout += [start, end]
    return layout


def parse_decision_layout(
    layout: Sequence[int], num_tokens: int
) -> List[LayoutQuestion]:
    """The questions of a layout, placed on the sequence the model reads.

    Every span follows the prompt's images, so expanding their placeholders
    shifts all spans by the number of added tokens. Spans come in prompt order
    without overlap, each question before its options, as encode_joint_schema
    writes them. Raises ValueError when the layout does not describe the
    sequence, or needs more option slots than one head pass holds.
    """
    if len(layout) < 2 or any(type(value) is not int for value in layout):
        raise ValueError("decision_layout must be a list of ints")
    prompt_length, num_questions = layout[0], layout[1]
    shift = num_tokens - prompt_length
    if prompt_length < 1 or shift < 0 or num_questions < 1:
        raise ValueError("decision_layout does not match the prompt length")
    questions = []
    cursor = 2
    previous_end = 0
    for _ in range(num_questions):
        if cursor + 4 > len(layout):
            raise ValueError("decision_layout is truncated")
        question_type, start, end, num_options = layout[cursor : cursor + 4]
        cursor += 4
        if (
            not 0 <= question_type < len(QUESTION_TYPES)
            or num_options < 1
            or not previous_end <= start < end <= prompt_length
            or cursor + 2 * num_options > len(layout)
        ):
            raise ValueError("decision_layout has an invalid question")
        previous_end = end
        spans = []
        for _ in range(num_options):
            option_start, option_end = layout[cursor : cursor + 2]
            cursor += 2
            if not previous_end <= option_start < option_end <= prompt_length:
                raise ValueError("decision_layout has an invalid option span")
            previous_end = option_end
            spans.append((option_start + shift, option_end + shift))
        questions.append(
            LayoutQuestion(question_type, (start + shift, end + shift), tuple(spans))
        )
    if cursor != len(layout):
        raise ValueError("decision_layout has trailing values")
    # The head pads every question to the widest one, and _groups never splits
    # a request, so one layout must fit one pass.
    slots = num_questions * max(len(question.option_spans) for question in questions)
    if slots > _MAX_GROUP_SLOTS:
        raise ValueError(
            f"decision_layout needs {slots} option slots, "
            f"but one head pass holds {_MAX_GROUP_SLOTS}"
        )
    return questions


class EvidenceRoutingLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(width)
        # Identity slots keep the checkpoint's Sequential indices (dropout at 2 and 4).
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(feedforward, width),
            nn.Identity(),
        )


class JointSchemaHead(nn.Module):
    """Parameters named as in the checkpoint's ``joint_head.safetensors``."""

    def __init__(
        self,
        hidden_size: int,
        width: int,
        routing_layers: int,
        layers: int,
        heads: int,
        feedforward: int,
    ) -> None:
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.question_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = nn.Embedding(len(QUESTION_TYPES), width)
        self.evidence_layers = nn.ModuleList(
            EvidenceRoutingLayer(width, heads, feedforward)
            for _ in range(routing_layers)
        )
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=feedforward,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        )
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.residual_scorer = nn.Sequential(
            nn.Linear(width * 4, width), nn.GELU(), nn.Identity(), nn.Linear(width, 1)
        )
        self.prior_logit_scale = nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = nn.Parameter(torch.zeros(()))
        self.residual_gate = nn.Parameter(torch.zeros(()))

    @torch.no_grad()
    def forward_batch(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        lengths: List[int],
        layouts: List[List[LayoutQuestion]],
        output_embedding_weight: torch.Tensor,
    ) -> List[torch.Tensor]:
        """One FP32 logit vector per request, its questions' options in layout order.

        ``hidden`` holds the final hidden states of the requests back to back.
        """
        outputs: List[torch.Tensor] = []
        start = 0
        for group in _groups(lengths, layouts):
            group_lengths = [lengths[i] for i in group]
            end = start + sum(group_lengths)
            outputs += self._forward_group(
                hidden[start:end],
                input_ids[start:end],
                group_lengths,
                [layouts[i] for i in group],
                output_embedding_weight,
            )
            start = end
        return outputs

    def _forward_group(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        lengths: List[int],
        layouts: List[List[LayoutQuestion]],
        output_embedding_weight: torch.Tensor,
    ) -> List[torch.Tensor]:
        device, dtype = hidden.device, hidden.dtype
        batch = len(layouts)
        question_bounds, option_bounds = [], []
        type_ids, owner, slot, question_request, last_tokens = [], [], [], [], []
        regions, region_offsets = [], [0]
        token_cu, question_cu, option_cu, option_counts = [0], [0], [0], []
        for request, (length, questions) in enumerate(zip(lengths, layouts)):
            offset = token_cu[-1]
            low = min(min(q.question_span[0], q.option_spans[0][0]) for q in questions)
            high = max(
                max(q.question_span[1], *(e for _, e in q.option_spans))
                for q in questions
            )
            # Prefix row of span position p in this request's schema region.
            base = region_offsets[-1] + request - low
            regions.append((offset + low, offset + high))
            request_options = 0
            for question in questions:
                count = len(question.option_spans)
                question_bounds += (
                    base + question.question_span[0],
                    base + question.question_span[1],
                )
                for option_slot, (option_start, option_end) in enumerate(
                    question.option_spans
                ):
                    option_bounds += (base + option_start, base + option_end)
                    owner.append(len(type_ids))
                    slot.append(option_slot)
                type_ids.append(question.question_type)
                question_request.append(request)
                request_options += count
            last_tokens.append(offset + length - 1)
            token_cu.append(offset + length)
            region_offsets.append(region_offsets[-1] + high - low)
            question_cu.append(len(type_ids))
            option_cu.append(option_cu[-1] + request_options)
            option_counts.append(request_options)

        num_questions = len(type_ids)
        widest = max(len(q.option_spans) for questions in layouts for q in questions)
        sizes = [
            2 * num_questions,
            len(option_bounds),
            num_questions,
            len(owner),
            len(slot),
            num_questions,
            batch,
            3 * (batch + 1),
        ]
        packed = torch.tensor(
            question_bounds
            + option_bounds
            + type_ids
            + owner
            + slot
            + question_request
            + last_tokens
            + token_cu
            + question_cu
            + option_cu,
            dtype=torch.long,
        ).to(device, non_blocking=True)
        (
            question_bounds_t,
            option_bounds_t,
            type_ids_t,
            owner_t,
            slot_t,
            question_request_t,
            last_tokens_t,
            cu_t,
        ) = torch.split(packed, sizes)
        question_bounds_t = question_bounds_t.view(-1, 2)
        option_bounds_t = option_bounds_t.view(-1, 2)
        token_cu_t, question_cu_t, option_cu_t = (
            cu_t.to(torch.int32).view(3, batch + 1).unbind(0)
        )
        token_bounds = (token_cu_t, max(lengths))
        question_bounds_cu = (question_cu_t, max(len(q) for q in layouts))
        option_bounds_cu = (option_cu_t, max(option_counts))

        normalized = self.hidden_norm(hidden)
        # Running sums over each schema region, after one zero row per request.
        region_total = region_offsets[-1] + batch
        hidden_prefix = torch.zeros(
            region_total, normalized.shape[-1], dtype=torch.float32, device=device
        )
        lexical_prefix = torch.zeros(
            region_total,
            output_embedding_weight.shape[-1],
            dtype=torch.float32,
            device=device,
        )
        for request, (low, high) in enumerate(regions):
            row = region_offsets[request] + request + 1
            torch.cumsum(
                normalized[low:high].float(),
                dim=0,
                out=hidden_prefix[row : row + high - low],
            )
            torch.cumsum(
                output_embedding_weight[input_ids[low:high]].float(),
                dim=0,
                out=lexical_prefix[row : row + high - low],
            )

        def span_means(prefix: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
            sums = prefix[bounds[:, 1]] - prefix[bounds[:, 0]]
            return (sums / (bounds[:, 1] - bounds[:, 0]).unsqueeze(-1)).to(dtype)

        memory = self.memory_projection(normalized)
        global_vectors = normalized[last_tokens_t]
        question_vectors = span_means(hidden_prefix, question_bounds_t)
        option_contexts = span_means(hidden_prefix, option_bounds_t)
        lexical = span_means(lexical_prefix, option_bounds_t)

        routed = (
            self.option_context_projection(option_contexts)
            + self.option_lexical_projection(lexical)
            + self.option_question_projection(question_vectors)[owner_t]
        )
        for layer in self.evidence_layers:
            routed = routed + _attention(
                layer.attention,
                layer.query_norm(routed),
                layer.memory_norm(memory),
                option_bounds_cu,
                token_bounds,
            )
            routed = routed + layer.feedforward(layer.feedforward_norm(routed))

        base_fields = self.question_projection(question_vectors)
        scores = torch.matmul(
            routed.unsqueeze(1), base_fields[owner_t].unsqueeze(2)
        ).view(-1) / math.sqrt(routed.shape[-1])
        padded_scores = torch.full(
            (num_questions, widest), float("-inf"), dtype=torch.float32, device=device
        )
        padded_scores[owner_t, slot_t] = scores.float()
        weights = torch.softmax(padded_scores, dim=1).to(dtype)
        padded_options = routed.new_zeros(num_questions, widest, routed.shape[-1])
        padded_options[owner_t, slot_t] = routed
        summaries = torch.sum(weights.unsqueeze(-1) * padded_options, dim=1)
        fields = (
            base_fields
            + self.option_summary_norm(summaries)
            + self.global_projection(global_vectors)[question_request_t]
            + self.type_embedding(type_ids_t)
        )
        for layer in self.layers:
            fields = fields + _attention(
                layer.self_attn,
                layer.norm1(fields),
                None,
                question_bounds_cu,
                question_bounds_cu,
            )
            fields = fields + _attention(
                layer.multihead_attn,
                layer.norm2(fields),
                memory,
                question_bounds_cu,
                token_bounds,
            )
            fields = fields + layer.linear2(
                layer.activation(layer.linear1(layer.norm3(fields)))
            )
        fields = self.field_norm(fields)

        prior_scale = self.prior_logit_scale.clamp(max=math.log(100.0)).exp()
        joint_scale = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
        anchor = F.normalize(
            question_vectors + global_vectors[question_request_t], dim=-1
        )
        prior = prior_scale * torch.matmul(
            F.normalize(lexical, dim=-1).unsqueeze(1), anchor[owner_t].unsqueeze(2)
        ).view(-1)
        options = self.option_norm(routed)
        repeated = fields[owner_t]
        cosine = F.cosine_similarity(repeated, options, dim=-1)
        features = torch.cat(
            [repeated, options, repeated * options, torch.abs(repeated - options)],
            dim=-1,
        )
        residual = self.residual_scorer(features).squeeze(-1)
        logits = prior + torch.sigmoid(self.residual_gate) * (
            joint_scale * cosine + residual
        )
        return list(torch.split(logits.float(), option_counts))


def _attention(
    module: nn.MultiheadAttention,
    queries: torch.Tensor,
    keys: Optional[torch.Tensor],
    query_bounds: Tuple[torch.Tensor, int],
    key_bounds: Tuple[torch.Tensor, int],
) -> torch.Tensor:
    """nn.MultiheadAttention over packed sequences, keys and values from ``keys``
    (or from ``queries`` when None), attending within each request."""
    width = queries.shape[-1]
    heads = module.num_heads
    if keys is None:
        query, key, value = F.linear(
            queries, module.in_proj_weight, module.in_proj_bias
        ).chunk(3, dim=-1)
    else:
        query_weight, key_value_weight = module.in_proj_weight.split([width, 2 * width])
        query_bias, key_value_bias = module.in_proj_bias.split([width, 2 * width])
        query = F.linear(queries, query_weight, query_bias)
        key, value = F.linear(keys, key_value_weight, key_value_bias).chunk(2, dim=-1)
    head_dim = width // heads
    output = torch.ops.aten._flash_attention_forward(
        query.reshape(-1, heads, head_dim),
        key.reshape(-1, heads, head_dim),
        value.reshape(-1, heads, head_dim),
        query_bounds[0],
        key_bounds[0],
        query_bounds[1],
        key_bounds[1],
        0.0,
        False,
        False,
    )[0]
    return module.out_proj(output.reshape(-1, width))


def _groups(lengths: List[int], layouts: List[List[LayoutQuestion]]) -> List[range]:
    """Consecutive requests per head pass, bounded in tokens and padded option slots."""
    groups, start, tokens, questions, widest = [], 0, 0, 0, 0
    for index, (length, layout) in enumerate(zip(lengths, layouts)):
        layout_widest = max(len(question.option_spans) for question in layout)
        grown = (questions + len(layout)) * max(widest, layout_widest)
        if index > start and (
            tokens + length > _MAX_GROUP_TOKENS or grown > _MAX_GROUP_SLOTS
        ):
            groups.append(range(start, index))
            start, tokens, questions, widest = index, 0, 0, 0
        tokens += length
        questions += len(layout)
        widest = max(widest, layout_widest)
    groups.append(range(start, len(lengths)))
    return groups


class JointSchemaPooler(nn.Module):
    """Pooler of a Clef checkpoint: the joint schema head's option logits per request."""

    def __init__(
        self, head: JointSchemaHead, output_embedding: Callable[[], torch.Tensor]
    ) -> None:
        super().__init__()
        self.head = head
        # A callable, so the LM head is not registered twice as a parameter.
        self._output_embedding = output_embedding

    def forward(
        self, hidden_states: torch.Tensor, forward_batch
    ) -> EmbeddingPoolerOutput:
        if (
            forward_batch.decision_layouts is None
            or forward_batch.extend_seq_lens_cpu is None
        ):
            return EmbeddingPoolerOutput(
                embeddings=[
                    hidden_states.new_empty(0, dtype=torch.float32)
                    for _ in range(forward_batch.batch_size)
                ]
            )
        lengths = list(forward_batch.extend_seq_lens_cpu)
        prefix_lengths = forward_batch.extend_prefix_lens_cpu or [0] * len(lengths)
        raw_layouts = forward_batch.decision_layouts
        embeddings: List[torch.Tensor] = [
            hidden_states.new_empty(0, dtype=torch.float32) for _ in lengths
        ]
        # The tokenizer manager checks every layout against its prompt, and the
        # server runs each prompt in one uncached prefill. A layout that no longer
        # fits its prompt, as after a truncation, gets no scores, which the
        # decision route reports as a server error.
        layouts = {}
        for i, (layout, prefix, length) in enumerate(
            zip(raw_layouts, prefix_lengths, lengths)
        ):
            if layout is None or prefix != 0:
                continue
            try:
                layouts[i] = parse_decision_layout(layout, length)
            except ValueError:
                continue
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        runs: List[List[int]] = []
        for i in layouts:
            if runs and runs[-1][-1] == i - 1:
                runs[-1].append(i)
            else:
                runs.append([i])
        for run in runs:
            start, end = offsets[run[0]], offsets[run[-1] + 1]
            logits = self.head.forward_batch(
                hidden_states[start:end],
                forward_batch.input_ids[start:end],
                [lengths[i] for i in run],
                [layouts[i] for i in run],
                self._output_embedding(),
            )
            for i, values in zip(run, logits):
                embeddings[i] = values
        return EmbeddingPoolerOutput(embeddings=embeddings)
