"""
VocabAnchor: maps NERSchema class labels to single token IDs in a given tokenizer.

The core challenge: most class labels (e.g. "book_flight", "departure_city") tokenise to
multiple sub-word tokens.  We need a stable, single-token proxy for each label so we can
slice exactly one logit column per class.

Strategies
----------
first_token   (default)
    Take the first token of the label's encoding.  Fast, always works.
    Risk: "booking" and "book" may share the same first token — see collision detection.

single_token
    Raise ValueError if any label tokenises to more than one token.
    Use when you control the label strings and want a strict guarantee.

explicit
    Caller provides a pre-built Dict[str, int].  VocabAnchor skips tokenisation entirely.
    Use when you know the exact token IDs (best for production).

Collision detection (all strategies)
--------------------------------------
If two labels in the same head (intent or a single slot) map to the same token ID, the
classification is ambiguous.  VocabAnchor.anchor() raises ValueError on any collision.
You can either:
  - Rename the conflicting labels to tokens that differ in the first sub-word
  - Switch to the "explicit" strategy and hand-pick token IDs
  - Add the labels as new special tokens to the tokenizer
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sglang.srt.logit_slicing.schema import NERSchema

logger = logging.getLogger(__name__)


# Strategies exposed as module-level constants so callers can avoid magic strings.
STRATEGY_FIRST_TOKEN = "first_token"
STRATEGY_SINGLE_TOKEN = "single_token"
STRATEGY_EXPLICIT = "explicit"


class VocabAnchor:
    """
    Anchors a NERSchema to a specific tokenizer, filling label_to_token_id on
    every IntentSchema and SlotSchema in-place, then returns the schema for
    method chaining.

    Usage
    -----
    anchored_schema = VocabAnchor().anchor(schema, tokenizer)
    config = anchored_schema.to_anchor_config()          # pass to custom_params
    """

    def anchor(
        self,
        schema: NERSchema,
        tokenizer,
        strategy: str = STRATEGY_FIRST_TOKEN,
        intent_override: Optional[Dict[str, int]] = None,
        slot_overrides: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> NERSchema:
        """
        Anchor all labels in *schema* to token IDs.

        Parameters
        ----------
        schema          : NERSchema to anchor (mutated in-place; also returned).
        tokenizer       : Any HuggingFace-compatible tokenizer.
        strategy        : "first_token" | "single_token" | "explicit"
        intent_override : Override mapping for the intent head (strategy="explicit").
        slot_overrides  : {slot_name: {label: token_id}} overrides per slot.
        """
        slot_overrides = slot_overrides or {}

        # ── Intent head ────────────────────────────────────────────────────────
        if strategy == STRATEGY_EXPLICIT and intent_override is None:
            raise ValueError(
                'strategy="explicit" requires intent_override to be provided.'
            )
        schema.intents.label_to_token_id = self._anchor_labels(
            labels=schema.intents.labels,
            tokenizer=tokenizer,
            strategy=strategy,
            explicit_map=intent_override,
            head_name="intent",
        )

        # ── Entity slots ────────────────────────────────────────────────────────
        for slot in schema.slots:
            override = slot_overrides.get(slot.name)
            if strategy == STRATEGY_EXPLICIT and override is None:
                raise ValueError(
                    f'strategy="explicit" requires slot_overrides["{slot.name}"] to be provided.'
                )
            slot.label_to_token_id = self._anchor_labels(
                labels=slot.labels,
                tokenizer=tokenizer,
                strategy=strategy,
                explicit_map=override,
                head_name=f'slot "{slot.name}"',
            )

        logger.info(
            "VocabAnchor: anchored %d intent labels and %d slots.",
            len(schema.intents.labels),
            len(schema.slots),
        )
        return schema

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _anchor_labels(
        self,
        labels: List[str],
        tokenizer,
        strategy: str,
        explicit_map: Optional[Dict[str, int]],
        head_name: str,
    ) -> Dict[str, int]:
        if strategy == STRATEGY_EXPLICIT:
            mapping = dict(explicit_map)  # shallow copy
        else:
            mapping = {
                label: self._label_to_token_id(label, tokenizer, strategy)
                for label in labels
            }

        missing = set(labels) - set(mapping.keys())
        if missing:
            raise ValueError(
                f"VocabAnchor [{head_name}]: labels {missing} have no token ID mapping."
            )

        self._check_collisions(mapping, head_name)
        self._check_vocab_range(mapping, tokenizer, head_name)
        self._log_mapping(mapping, head_name)
        return mapping

    def _label_to_token_id(self, label: str, tokenizer, strategy: str) -> int:
        tokens = tokenizer.encode(label, add_special_tokens=False)
        if not tokens:
            raise ValueError(
                f"VocabAnchor: label '{label}' encoded to an empty token list. "
                "Check tokenizer compatibility."
            )
        if strategy == STRATEGY_SINGLE_TOKEN and len(tokens) > 1:
            raise ValueError(
                f"VocabAnchor: label '{label}' tokenises to {len(tokens)} tokens "
                f"({tokens}) but strategy='single_token' requires exactly one. "
                "Hint: shorten the label or switch to strategy='first_token'."
            )
        if len(tokens) > 1:
            logger.warning(
                "VocabAnchor: label '%s' tokenises to %d tokens; using first token %d only.",
                label,
                len(tokens),
                tokens[0],
            )
        return tokens[0]

    def _check_collisions(self, mapping: Dict[str, int], head_name: str) -> None:
        token_to_labels: Dict[int, List[str]] = {}
        for label, tid in mapping.items():
            token_to_labels.setdefault(tid, []).append(label)
        collisions = {
            tid: lbls for tid, lbls in token_to_labels.items() if len(lbls) > 1
        }
        if collisions:
            details = "; ".join(
                f"token_id={tid} ← {lbls}" for tid, lbls in collisions.items()
            )
            raise ValueError(
                f"VocabAnchor [{head_name}]: token ID collision(s) detected — {details}. "
                "Two labels map to the same token, making classification ambiguous. "
                "Use strategy='explicit' and hand-pick unique token IDs."
            )

    def _check_vocab_range(
        self, mapping: Dict[str, int], tokenizer, head_name: str
    ) -> None:
        if tokenizer is None:
            # No tokenizer available (e.g. STRATEGY_EXPLICIT with tokenizer=None in
            # build_phase_b_config).  Caller is responsible for providing valid IDs.
            return
        vocab_size = tokenizer.vocab_size
        out_of_range = {
            label: tid for label, tid in mapping.items() if not (0 <= tid < vocab_size)
        }
        if out_of_range:
            raise ValueError(
                f"VocabAnchor [{head_name}]: token IDs out of vocab range [0, {vocab_size}): "
                f"{out_of_range}"
            )

    def _log_mapping(self, mapping: Dict[str, int], head_name: str) -> None:
        lines = "  " + "\n  ".join(
            f"{lbl!r:30s} → {tid}" for lbl, tid in mapping.items()
        )
        logger.debug("VocabAnchor [%s]:\n%s", head_name, lines)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────────────────────────────────────


def build_anchor_config(
    schema: NERSchema,
    tokenizer,
    strategy: str = STRATEGY_FIRST_TOKEN,
    intent_override: Optional[Dict[str, int]] = None,
    slot_overrides: Optional[Dict[str, Dict[str, int]]] = None,
    eos_token_id: Optional[int] = None,
) -> dict:
    """
    One-shot helper: anchor the schema and return the custom_params dict.

    The returned dict is ready to be passed as SamplingParams.custom_params.

    Example
    -------
        custom_params = build_anchor_config(schema, tokenizer)
        sgl.gen(..., sampling_params={"custom_params": custom_params, "max_new_tokens": 1})
    """
    anchored = VocabAnchor().anchor(
        schema,
        tokenizer,
        strategy=strategy,
        intent_override=intent_override,
        slot_overrides=slot_overrides,
    )
    config = {"schema": anchored.to_anchor_config()}
    if eos_token_id is not None:
        config["eos_token_id"] = eos_token_id
    elif hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        config["eos_token_id"] = tokenizer.eos_token_id
    return config


def build_phase_b_config(
    schema: NERSchema,
    tokenizer,
    embedding_matrix,
    strategy: str = STRATEGY_FIRST_TOKEN,
    intent_override: Optional[Dict[str, int]] = None,
    slot_overrides: Optional[Dict[str, Dict[str, int]]] = None,
    eos_token_id: Optional[int] = None,
) -> dict:
    """
    One-shot helper for Phase B (SMIELPWithHiddenStates).

    Extends build_anchor_config by also extracting embedding vectors for each
    anchored label token from the model's input embedding matrix, enabling
    cosine-similarity classification in hidden-state space.

    Parameters
    ----------
    schema           : NERSchema to anchor.
    tokenizer        : HuggingFace-compatible tokenizer.
    embedding_matrix : Tensor[vocab_size, hidden_dim] — typically
                       ``model.embed_tokens.weight.detach()`` from the HF model.
                       Rows are indexed by token ID.
    strategy, intent_override, slot_overrides, eos_token_id : forwarded to
                       build_anchor_config unchanged.

    Returns
    -------
    dict with keys:
        "schema"           → same as build_anchor_config
        "eos_token_id"     → same as build_anchor_config
        "label_embeddings" → {
            "intent":     Tensor[num_intents, hidden_dim],
            slot.name:    Tensor[num_tags, hidden_dim],   # one per slot
            ...
        }

    Example
    -------
        custom_params = build_phase_b_config(schema, tokenizer, model.embed_tokens.weight.detach())
        SamplingParams(
            max_new_tokens=1,
            return_hidden_states=True,
            custom_params=custom_params,
            custom_logit_processor=SMIELPWithHiddenStates.to_str(),
        )
    """
    import torch as _torch

    config = build_anchor_config(
        schema,
        tokenizer,
        strategy=strategy,
        intent_override=intent_override,
        slot_overrides=slot_overrides,
        eos_token_id=eos_token_id,
    )
    schema_cfg = config["schema"]
    label_embs: Dict[str, Any] = {}

    intent_ids = _torch.tensor(schema_cfg["intent_token_ids"], dtype=_torch.long)
    label_embs["intent"] = embedding_matrix[intent_ids].detach().clone().cpu()

    for slot in schema_cfg["entity_slots"]:
        bio_ids = _torch.tensor(slot["bio_token_ids"], dtype=_torch.long)
        label_embs[slot["name"]] = embedding_matrix[bio_ids].detach().clone().cpu()

    config["label_embeddings"] = label_embs
    logger.info(
        "build_phase_b_config: extracted embeddings for %d intents and %d slots (hidden_dim=%d).",
        len(schema_cfg["intent_token_ids"]),
        len(schema_cfg["entity_slots"]),
        embedding_matrix.shape[1],
    )
    return config
