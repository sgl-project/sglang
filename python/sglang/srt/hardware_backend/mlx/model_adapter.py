"""Optional Gemma 4 target outputs needed by the MLX MTP proposer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class MlxTargetForwardOutput:
    logits: mx.array
    hidden_states: mx.array | None = None


@dataclass(frozen=True)
class MlxTargetSeed:
    """Request-private assistant seed from a committed target query row."""

    token_id: int
    hidden_state: mx.array
    token_embedding: mx.array


def _extract_logits(model_output: Any) -> mx.array:
    return model_output[0] if isinstance(model_output, tuple) else model_output


class Gemma4TargetAdapter:
    """Capture Gemma 4 backbone output without changing normal target math."""

    def __init__(self, model: Any):
        self.model = model
        self.causal_model = getattr(model, "language_model", model)
        self.backbone = getattr(self.causal_model, "model", None)
        if self.backbone is None or not hasattr(self.backbone, "embed_tokens"):
            raise TypeError(
                "Gemma4TargetAdapter requires an mlx-lm Gemma 4 text model "
                "or conditional-generation wrapper"
            )

    @property
    def hidden_size(self) -> int:
        return int(self.backbone.config.hidden_size)

    @property
    def vocab_size(self) -> int:
        return int(self.backbone.config.vocab_size)

    @property
    def embed_scale(self) -> float:
        return float(getattr(self.backbone, "embed_scale", 1.0))

    def input_embeddings(self, input_ids: mx.array) -> mx.array:
        """Return the scaled target-trunk embeddings used by normal forward."""

        return self.backbone.embed_tokens(input_ids) * self.embed_scale

    def forward(
        self,
        input_ids: mx.array,
        *,
        cache: list[Any] | None = None,
        collect_hidden_states: bool = False,
    ) -> MlxTargetForwardOutput:
        if not collect_hidden_states:
            return MlxTargetForwardOutput(
                logits=_extract_logits(self.model(input_ids, cache=cache))
            )

        # This is the same trunk and LM-head sequence as gemma4_text.Model;
        # only the already-computed normalized trunk output is retained.
        hidden = self.backbone(input_ids, cache=cache)
        if bool(getattr(self.causal_model, "tie_word_embeddings", False)):
            logits = self.backbone.embed_tokens.as_linear(hidden)
        else:
            logits = self.causal_model.lm_head(hidden)

        softcap = getattr(self.causal_model, "final_logit_softcapping", None)
        if softcap is not None:
            # Keep this expression identical to mlx-lm's compiled logit_softcap.
            logits = mx.tanh(logits / softcap) * softcap
        return MlxTargetForwardOutput(logits=logits, hidden_states=hidden)

    def make_seed(
        self,
        output: MlxTargetForwardOutput,
        *,
        hidden_row_index: int,
        emitted_token_id: int,
    ) -> MlxTargetSeed:
        if output.hidden_states is None:
            raise ValueError("target output did not collect hidden states")
        hidden = output.hidden_states
        if hidden.ndim != 3:
            raise ValueError(
                "target hidden states must have shape [batch, query, hidden]"
            )
        if hidden_row_index < 0 or hidden_row_index >= hidden.shape[1]:
            raise IndexError(
                f"hidden row {hidden_row_index} is outside query width {hidden.shape[1]}"
            )
        if emitted_token_id < 0 or emitted_token_id >= self.vocab_size:
            raise ValueError("assistant seed token is outside the target vocabulary")

        hidden_row = hidden[:, hidden_row_index : hidden_row_index + 1, :]
        token_ids = mx.array([[emitted_token_id]], dtype=mx.int32)
        embedding = self.input_embeddings(token_ids)
        return MlxTargetSeed(
            token_id=int(emitted_token_id),
            hidden_state=hidden_row,
            token_embedding=embedding,
        )
