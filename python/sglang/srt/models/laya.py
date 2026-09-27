# SPDX-License-Identifier: Apache-2.0
"""Laya: a non-autoregressive decision model served as an SGLang embedding model.

Laya is *not* a generative LLM. It is a bidirectional ModernBERT encoder
(28 layers, hidden 1024) followed by a small transformer decision head. Every
answer option is represented by a ``[MASK]`` marker token in the prompt; the
head scores each marker and the softmax over those marker scores is the answer
distribution. A second head predicts whether the agent should escalate to a
human.

One request carries exactly one question, and the returned vector is
``[p_0, ..., p_{K-1}, act_escalate, act_not_escalate]`` where ``K`` is the
number of options. ``K`` varies per request, so the pooler returns one tensor
per request (``EmbeddingPoolerOutput.embeddings`` is a list).

The prompt layout (``[CLS] <type> question: <instructions> [SEP]`` followed by
``[MASK] <option text>`` per option, then ``[SEP] <state> [SEP]``) is built by
the client. The question type is recovered from the token right after ``[CLS]``.

Numerical semantics deliberately follow the checkpoint's own reference
implementation (``rl_common.DecisionModel`` in the model repository) rather than
any re-derivation: the marker mask uses ``-1e4``, the entropy normaliser is
``clamp(min=2)``, and the temperature is clamped to ``[0.5, 5.0]`` the way the
released SDK does (set ``SGLANG_LAYA_TEMPERATURE_CLAMP=0`` to score with the raw
fitted bucket temperatures).

Attention note: ModernBERT alternates global and *sliding* (local) bidirectional
attention, and the local layers keep keys within
``|i - j| <= local_attention / 2``. The encoder computes that mask itself rather
than delegating to the attention backend, because the windowed backend path does
not reproduce the same mask on every backend (some apply it only for causal
attention, and the FlashAttention window measures differently). The mask is
built from device tensors whose values are refreshed on every step, and every
loop bound comes from a shape, so the encoder body is safe to capture in a
prefill CUDA graph. Set ``SGLANG_LAYA_ATTENTION_BACKEND=1`` to route attention
through the backend instead, for comparison.
"""

import logging
import math
from typing import Any, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)

# Question type -> index, matching ``QTYPES`` in the model repository.
QTYPE_NAMES = ("choice", "score", "noul")

# The released SDK refuses to apply a temperature that sharpens the logits this
# hard. The shipped ``choice:11+`` bucket is 0.1006, which multiplies them ~10x
# and turns a 0.24 top probability into a published 0.99; clamping keeps served
# probabilities identical to the official runtime.
TEMP_MIN = 0.5
TEMP_MAX = 5.0

# Bounds for the self-computed attention. The query axis is processed in chunks
# so the score tensor stays within a fixed budget regardless of prompt length;
# both numbers are shape-derived, so a captured CUDA graph keeps a fixed number
# of iterations for a given capture bucket.
_SDPA_QUERY_CHUNK = 1024
_SDPA_SCORE_BUDGET = 1 << 24


class LayaEmbeddings(nn.Module):
    """Token embeddings + LayerNorm; positions come from RoPE, not from a table."""

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tok_embeddings = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("tok_embeddings", prefix),
        )
        self.norm = nn.LayerNorm(
            config.hidden_size,
            eps=getattr(config, "norm_eps", 1e-5),
            bias=getattr(config, "norm_bias", False),
        )

    def forward(self, input_ids: torch.Tensor, inputs_embeds=None) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.tok_embeddings(input_ids)
        return self.norm(inputs_embeds)


class LayaAttention(nn.Module):
    """Bidirectional multi-head self attention over the packed prefill batch."""

    def __init__(
        self,
        config: Any,
        layer_id: int,
        layer_type: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.total_num_heads
        self.scaling = self.head_dim**-0.5
        self.sliding_window_size = _sliding_window_size(config, layer_type)

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.out_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
        )
        # Global and local layers rotate with different bases, so the RoPE module
        # is per layer rather than shared.
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            base=_rope_theta(config, layer_type),
            is_neox_style=True,
        )
        self.attn = RadixAttention(
            num_heads=self.total_num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.total_num_heads,
            layer_id=layer_id,
            sliding_window_size=self.sliding_window_size,
            attn_type=AttentionType.ENCODER_ONLY,
            prefix=add_prefix("attn", prefix),
        )

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        token_req: torch.Tensor,
    ) -> torch.Tensor:
        """Bidirectional attention with an explicit local window, per request.

        ModernBERT alternates full and sliding (local) layers, and the local
        layers keep keys with ``|i - j| <= sliding_window``. Computing that mask
        here keeps the numerics identical to the reference implementation on
        every attention backend, which matters because the windowed path is not
        implemented consistently across them.

        Everything is derived from tensors whose values are refreshed on every
        step, and every loop bound comes from a shape, so the whole block is
        safe to capture in a CUDA graph: a host value read here would be frozen
        at capture time and silently reused for later batches.
        """
        total = q.shape[0]
        heads, head_dim = self.total_num_heads, self.head_dim
        qh = q.view(total, heads, head_dim).transpose(0, 1)
        kh = k.view(total, heads, head_dim).transpose(0, 1)
        vh = v.view(total, heads, head_dim).transpose(0, 1)
        window = self.sliding_window_size

        # Chunk the query axis so the score matrix stays bounded. The chunk size
        # is derived from shapes only, so the number of iterations is fixed for
        # a given capture bucket.
        chunk = max(1, min(_SDPA_QUERY_CHUNK, _SDPA_SCORE_BUDGET // max(total, 1)))
        out = torch.empty_like(qh)
        for offset in range(0, total, chunk):
            rows = slice(offset, offset + chunk)
            mask = token_req[rows][:, None] == token_req[None, :]
            if window is not None and window > -1:
                mask = mask & (
                    (positions[rows][:, None] - positions[None, :]).abs() <= window
                )
            out[:, rows] = F.scaled_dot_product_attention(
                qh[:, rows], kh, vh, attn_mask=mask[None]
            )
        return out.transpose(0, 1).reshape(total, -1)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        token_req: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [
                self.qkv_proj.q_proj_shard_size,
                self.qkv_proj.kv_proj_shard_size,
                self.qkv_proj.v_proj_shard_size,
            ],
            dim=-1,
        )
        q, k = self.rotary_emb(positions, q, k)
        if envs.SGLANG_LAYA_ATTENTION_BACKEND.get():
            output = self.attn(q, k, v, forward_batch)
        else:
            output = self._sdpa_attention(
                q=q, k=k, v=v, positions=positions, token_req=token_req
            )
        output, _ = self.out_proj(output)
        return output


class LayaMLP(nn.Module):
    """GeGLU feed-forward: ``Wo(gelu(Wi(x)[:d]) * Wi(x)[d:])``."""

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.Wi = ColumnParallelLinear(
            hidden_size,
            2 * intermediate_size,
            bias=getattr(config, "mlp_bias", False),
            quant_config=quant_config,
            prefix=add_prefix("Wi", prefix),
        )
        self.Wo = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=getattr(config, "mlp_bias", False),
            quant_config=quant_config,
            prefix=add_prefix("Wo", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected, _ = self.Wi(hidden_states)
        # Reference: ``input, gate = self.Wi(hidden_states).chunk(2, dim=-1)``
        # then ``self.act(input) * gate`` -- first half activated, second half
        # is the gate (not interleaved).
        activation, gate = projected.chunk(2, dim=-1)
        output, _ = self.Wo(F.gelu(activation) * gate)
        return output


class LayaEncoderLayer(nn.Module):
    """Pre-norm ModernBERT layer.

    The first layer has no ``attn_norm``: ``embeddings.norm`` already normalised
    the input, so the checkpoint ships only ``num_hidden_layers - 1`` of them.
    """

    def __init__(
        self,
        config: Any,
        layer_id: int,
        layer_type: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        eps = getattr(config, "norm_eps", 1e-5)
        norm_bias = getattr(config, "norm_bias", False)
        self.attn_norm = (
            nn.Identity()
            if layer_id == 0
            else nn.LayerNorm(hidden_size, eps=eps, bias=norm_bias)
        )
        self.mlp_norm = nn.LayerNorm(hidden_size, eps=eps, bias=norm_bias)
        self.attn = LayaAttention(
            config=config,
            layer_id=layer_id,
            layer_type=layer_type,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.mlp = LayaMLP(
            config=config, quant_config=quant_config, prefix=add_prefix("mlp", prefix)
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        token_req: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            positions=positions,
            hidden_states=self.attn_norm(hidden_states),
            forward_batch=forward_batch,
            token_req=token_req,
        )
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class LayaEncoder(nn.Module):
    """ModernBERT encoder: token embeddings, alternating global/local layers, final norm."""

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embeddings = LayaEmbeddings(
            config, quant_config=quant_config, prefix=add_prefix("embeddings", prefix)
        )
        layer_types = _layer_types(config, config.num_hidden_layers)
        self.layers = nn.ModuleList(
            [
                LayaEncoderLayer(
                    config=config,
                    layer_id=layer_id,
                    layer_type=layer_types[layer_id],
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(
            config.hidden_size,
            eps=getattr(config, "norm_eps", 1e-5),
            bias=getattr(config, "norm_bias", False),
        )

    @staticmethod
    def _token_request_ids(positions: torch.Tensor) -> torch.Tensor:
        """Which request each packed token belongs to, as a device tensor.

        Derived from ``positions``, which restarts at zero for every request and
        is refreshed in full on each step. Per-request lengths are deliberately
        *not* used: inside a captured graph the batch-size-length buffers are
        static views, and a shorter live batch only refreshes their prefix, so
        their tail keeps the values from capture time. Positions carry no such
        tail: the padded region sits after every real token, where an extra
        boundary cannot shift the mapping of a real one.
        """
        return torch.cumsum((positions == 0).to(torch.int64), dim=0) - 1

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids, inputs_embeds)
        token_req = self._token_request_ids(positions)
        for layer in self.layers:
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                token_req=token_req,
            )
        return self.final_norm(hidden_states)


class LayaForDecision(nn.Module):
    """Laya decision model: ModernBERT encoder + marker scorer + act head.

    Served as an embedding model; each request yields one ``K + 2`` vector.
    """

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size

        tp_size = get_parallel().tp_size
        if tp_size != 1:
            raise NotImplementedError(
                "LayaForDecision does not shard the decision head across tensor "
                f"parallel ranks (got tp_size={tp_size}); serve with --tp-size 1."
            )

        self.model = LayaEncoder(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        self.head_layers = int(getattr(config, "head_layers", 2))
        self.num_act = int(getattr(config, "n_act", 2))
        num_heads = max(1, hidden_size // 64)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_heads,
            4 * hidden_size,
            0.1,
            batch_first=True,
            norm_first=True,
        )
        self.head = nn.TransformerEncoder(
            encoder_layer, self.head_layers, enable_nested_tensor=False
        )
        self.type_emb = nn.Embedding(3, hidden_size)
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.act_head = nn.Sequential(
            nn.Linear(hidden_size + 4, 256),
            nn.GELU(),
            nn.Linear(256, self.num_act),
        )

        # The checkpoint's ``temperature`` buffer is all ones and unused at
        # inference; the fitted values live in ``temperature_by_options``.
        name_to_qtype = {name: idx for idx, name in enumerate(QTYPE_NAMES)}
        type_token_ids = {
            name: int(token_id)
            for name, token_id in (
                getattr(config, "type_token_ids", None) or {}
            ).items()
            if name in name_to_qtype
        }
        if len(set(type_token_ids.values())) != len(type_token_ids):
            raise ValueError(
                "config.type_token_ids maps distinct question types onto the same "
                f"token id, so the question type cannot be recovered: {type_token_ids}"
            )
        self.type_token_ids = {
            token_id: name_to_qtype[name] for name, token_id in type_token_ids.items()
        }
        self.temperature = _temperature_list(config)
        self.temperature_by_options = dict(
            getattr(config, "temperature_by_options", None) or {}
        )
        self.mask_token_id = int(getattr(config, "mask_token_id", 50284))
        # Latched so a marker-free prompt is reported once, not per step.
        self._warned_no_markers = False

    # ------------------------------------------------------------------ helpers
    def _answer_temperature(self, qtype: int, num_options: int) -> float:
        key = _temperature_bucket(QTYPE_NAMES[qtype], num_options)
        value = self.temperature_by_options.get(key, self.temperature[qtype])
        if envs.SGLANG_LAYA_TEMPERATURE_CLAMP.get():
            return _clamp_temperature(value)
        return float(value)

    def _marker_positions(self, ids: torch.Tensor, num_tokens: int) -> List[int]:
        return (
            (ids[:num_tokens] == self.mask_token_id).nonzero(as_tuple=True)[0].tolist()
        )

    def _request_slices(
        self, forward_batch: ForwardBatch, total_tokens: int
    ) -> List[Tuple[int, int]]:
        """Per-request ``(start, end)`` token ranges inside the packed batch.

        Read the device tensor rather than ``extend_seq_lens_cpu``: this runs in
        the eager tail of a captured prefill graph, whose static batch carries
        the host list from capture time, while the device tensor is refreshed on
        every step.
        """
        seq_lens = forward_batch.extend_seq_lens
        if seq_lens is not None:
            lengths = seq_lens.tolist()
        else:
            lengths = forward_batch.extend_seq_lens_cpu
        slices: List[Tuple[int, int]] = []
        start = 0
        for length in lengths:
            end = min(start + int(length), total_tokens)
            slices.append((start, end))
            start = end
        return slices

    # ------------------------------------------------------------------ forward
    def _request_layout(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        num_tokens: int,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[List[int]], List[int]]:
        """Per-request token ranges, lengths, ``[MASK]`` positions and question types."""
        slices = self._request_slices(forward_batch, num_tokens)
        lengths: List[int] = []
        markers: List[List[int]] = []
        qtypes: List[int] = []
        for start, end in slices:
            ids = input_ids[start:end]
            length = end - start
            lengths.append(length)
            qtypes.append(
                self.type_token_ids.get(int(ids[1]), 0) if ids.numel() > 1 else 0
            )
            markers.append(self._marker_positions(ids=ids, num_tokens=length))

        if any(not row for row in markers) and not self._warned_no_markers:
            # A prompt with no [MASK] marker has no options to score. It is
            # malformed for Laya, but it must not abort the engine: SGLang's own
            # startup warm-up sends a marker-free prompt, and any client can.
            # Such a request gets the act head alone (an empty answer
            # distribution), which is what the reference scalar path produces.
            logger.warning(
                "Laya request without any [MASK] marker token (marker counts %s); "
                "returning the act head only. The answer part will be empty.",
                [len(row) for row in markers],
            )
            self._warned_no_markers = True
        return slices, lengths, markers, qtypes

    def _head_encode(
        self,
        hidden_states: torch.Tensor,
        lengths: List[int],
        qtypes: List[int],
    ) -> torch.Tensor:
        """Run the decision head over the step's prompts, padded to a rectangle.

        The encoder output arrives packed request after request, so the head needs
        a rectangle; padding it costs one masked attention pass per step.
        """
        device = hidden_states.device
        batch = len(lengths)
        max_len = max(lengths)
        row_index = np.zeros((batch, max_len), dtype=np.int64)
        valid = np.zeros((batch, max_len), dtype=bool)
        cursor = 0
        for index, length in enumerate(lengths):
            row_index[index, :length] = np.arange(cursor, cursor + length)
            valid[index, :length] = True
            cursor += length
        valid_t = torch.from_numpy(valid).to(device)
        padded = hidden_states.index_select(
            dim=0, index=torch.from_numpy(row_index.reshape(-1)).to(device)
        ).view(batch, max_len, -1)

        # The type embedding is per request, broadcast over its own tokens.
        type_weight = self.type_emb.weight.index_select(
            dim=0, index=torch.tensor(qtypes, dtype=torch.long, device=device)
        )
        padded = padded + type_weight.unsqueeze(1)
        # Zero the pad rows (which alias row 0 above) so they cannot leak into
        # the head through a residual path.
        padded = padded.masked_fill(~valid_t.unsqueeze(-1), 0.0)
        return self.head(padded, src_key_padding_mask=~valid_t)

    def _marker_logits(
        self, encoded: torch.Tensor, markers: List[List[int]]
    ) -> torch.Tensor:
        """Score every request's ``[MASK]`` markers with one gather."""
        total_markers = sum(len(row) for row in markers)
        if not total_markers:
            return encoded.new_zeros(0, dtype=torch.float32)
        marker_row = np.zeros(total_markers, dtype=np.int64)
        marker_col = np.zeros(total_markers, dtype=np.int64)
        offset = 0
        for index, row in enumerate(markers):
            for position in row:
                marker_row[offset] = index
                marker_col[offset] = position
                offset += 1
        device = encoded.device
        picked = encoded[
            torch.from_numpy(marker_row).to(device),
            torch.from_numpy(marker_col).to(device),
        ]
        return self.scorer(picked).squeeze(-1).float()

    def _answer_summary(self, logits: torch.Tensor, count: int) -> torch.Tensor:
        """The act head's four features: top-1, margin, entropy and option count."""
        if not count:
            # No options: the act head sees a zeroed answer summary.
            zero = logits.new_zeros(())
            return torch.stack([zero, zero, zero, logits.new_tensor(2.0 / 255.0)])

        probs = torch.softmax(logits.detach(), dim=-1)
        # Reference: ``k = marker_mask.sum(-1).clamp(min=2).float()`` and the same
        # clamped value feeds both the entropy and the count feature.
        k = float(max(count, 2))
        entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum() / math.log(k)
        if probs.numel() >= 2:
            top2 = probs.topk(2).values
            top1, margin = top2[0], top2[0] - top2[1]
        else:
            top1 = probs.max()
            margin = top1
        return torch.stack([top1, margin, entropy, logits.new_tensor(k / 255.0)])

    def _request_outputs(
        self,
        encoded: torch.Tensor,
        markers: List[List[int]],
        qtypes: List[int],
        logits_flat: torch.Tensor,
    ) -> List[torch.Tensor]:
        """One ``[p_0 .. p_{K-1}, act_escalate, act_not_escalate]`` per request."""
        outputs: List[torch.Tensor] = []
        cursor = 0
        for index, row in enumerate(markers):
            count = len(row)
            logits = logits_flat[cursor : cursor + count]
            cursor += count

            feats = self._answer_summary(logits=logits, count=count)
            # The act head reads the pooled [CLS] row *after* the head stack.
            act_in = torch.cat([encoded[index, 0].float(), feats]).to(
                self.act_head[0].weight.dtype
            )
            act = torch.softmax(self.act_head(act_in), dim=-1)

            if count:
                answer = torch.softmax(
                    logits / self._answer_temperature(qtypes[index], count), dim=-1
                )
            else:
                answer = logits
            outputs.append(torch.cat([answer, act]).float())
        return outputs

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
    ) -> EmbeddingPoolerOutput:
        assert get_embedding, (
            "LayaForDecision is an embedding/decision model; it is always served "
            "with embedding mode enabled."
        )

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            inputs_embeds=input_embeds,
        )

        _, lengths, markers, qtypes = self._request_layout(
            input_ids=input_ids,
            forward_batch=forward_batch,
            num_tokens=hidden_states.shape[0],
        )
        encoded = self._head_encode(
            hidden_states=hidden_states, lengths=lengths, qtypes=qtypes
        )
        logits_flat = self._marker_logits(encoded=encoded, markers=markers)
        outputs = self._request_outputs(
            encoded=encoded,
            markers=markers,
            qtypes=qtypes,
            logits_flat=logits_flat,
        )
        return EmbeddingPoolerOutput(embeddings=outputs)

    # ------------------------------------------------------------------ weights
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        # The checkpoint names the encoder ``encoder.*`` and stores the attention
        # projections under their transformers names; everything else already
        # matches our module names. ``mlp.Wo`` keeps its name -- only the
        # *attention* ``Wo`` becomes ``out_proj``.
        rename = (
            ("encoder.", "model."),
            (".attn.Wqkv.", ".attn.qkv_proj."),
            (".attn.Wo.", ".attn.out_proj."),
        )
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if name == "temperature":
                # All-ones buffer, unused at inference.
                continue
            if not name.startswith("encoder.") and not name.startswith(
                ("head.", "scorer.", "act_head.", "type_emb.")
            ):
                logger.warning("Unexpected Laya checkpoint tensor %r; skipped", name)
                continue
            mapped = name
            for old, new in rename:
                # Every rule applies; the encoder prefix and the attention
                # projection names are independent rewrites.
                if old in mapped:
                    mapped = mapped.replace(old, new)
            param = params_dict.get(mapped)
            if param is None:
                logger.warning(
                    "No parameter for Laya checkpoint tensor %r; skipped", name
                )
                continue
            default_weight_loader(param, loaded_weight)
            loaded_params.add(mapped)
        return loaded_params


def _clamp_temperature(value: Any) -> float:
    """Mirror of ``laya.common.clamp_temperature``: confine to [TEMP_MIN, TEMP_MAX]."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0
    if value != value or value in (float("inf"), float("-inf")):
        return 1.0
    return min(TEMP_MAX, max(TEMP_MIN, value))


def _temperature_bucket(qtype_name: str, num_options: int) -> str:
    """Mirror of ``rl_common.temp_bucket``."""
    if num_options <= 2:
        size = "2"
    elif num_options <= 5:
        size = "3-5"
    elif num_options <= 10:
        size = "6-10"
    else:
        size = "11+"
    return f"{qtype_name}:{size}"


def _layer_types(config: Any, num_layers: int) -> List[str]:
    """Per-layer attention type, tolerating both ModernBERT config generations."""
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        return list(layer_types)[:num_layers]
    every = int(getattr(config, "global_attn_every_n_layers", 3))
    return [
        "full_attention" if layer_id % every == 0 else "sliding_attention"
        for layer_id in range(num_layers)
    ]


def _rope_theta(config: Any, layer_type: str) -> float:
    """RoPE theta for one layer type (global and local use different bases)."""
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    per_type = (
        rope_parameters.get(layer_type) if isinstance(rope_parameters, dict) else None
    )
    if isinstance(per_type, dict) and per_type.get("rope_theta") is not None:
        return float(per_type["rope_theta"])
    if layer_type == "sliding_attention":
        local = getattr(config, "local_rope_theta", None)
        if local is not None:
            return float(local)
    return float(getattr(config, "global_rope_theta", 10000.0))


def _temperature_list(config: Any) -> List[float]:
    """Per-question-type fallback temperatures.

    The value cannot live under the plain ``temperature`` key: ``PretrainedConfig``
    consumes ``temperature`` (and ``top_p``, ...) as generation defaults while
    building the config object, so it never reaches the model (verified on
    transformers 5.12.1). The servable config therefore carries
    ``laya_temperature``; ``temperature`` is still honoured when it happens to be
    a list, for configs written before that was known.
    """
    for key in ("laya_temperature", "temperature"):
        value = getattr(config, key, None)
        if isinstance(value, (list, tuple)) and len(value) >= len(QTYPE_NAMES):
            return [float(item) for item in value[: len(QTYPE_NAMES)]]
    return [1.0] * len(QTYPE_NAMES)


def _sliding_window_size(config: Any, layer_type: str) -> int:
    """Window radius handed to the attention backend, or -1 for full attention.

    ``config.sliding_window`` is the half window (``local_attention // 2``, i.e.
    64 for the shipped checkpoint) and the attention mask keeps keys with
    ``abs(q - k) <= sliding_window``. The FlashAttention backends take the same
    inclusive radius directly as ``window_size=(w, w)`` for encoder-only layers,
    so the value is passed through unchanged. (transformers passes
    ``config.sliding_window + 1`` because its own FA wrapper subtracts one.)
    """
    if layer_type != "sliding_attention":
        return -1
    window = getattr(config, "sliding_window", None)
    if window is None:
        local_attention = getattr(config, "local_attention", None)
        if local_attention is None:
            raise ValueError(
                "ModernBERT config exposes neither `sliding_window` nor "
                "`local_attention`; cannot derive the local attention window."
            )
        window = int(local_attention) // 2
    return int(window)


EntryClass = LayaForDecision
