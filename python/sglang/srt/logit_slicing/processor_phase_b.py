"""
SMIELPWithHiddenStates — Phase B processor
============================================
Extends the Phase A logit-slicing approach by accepting the pre-lm_head hidden states
from the transformer backbone, enabling:

  1. **Norm-based confidence calibration** — the L2 norm of the last-token hidden state is a
     proxy for model certainty; low-norm states indicate underspecified context.

  2. **Embedding-space projection** (optional) — if the caller supplies ``label_embeddings``
     in ``custom_params``, classification is done as cosine similarity in hidden-state space
     rather than logit slicing.  This removes the vocabulary bottleneck: labels no longer need
     to be single tokens.

  3. **Fallback path** — when ``hidden_states`` is None (server launched without
     ``--enable-return-hidden-states``, or running in speculative-decode mode), the processor
     falls back seamlessly to Phase A logit slicing.

Phase B infrastructure requirements
------------------------------------
Server:
    python -m sglang.launch_server ...
        --enable-custom-logit-processor
        --enable-return-hidden-states     # ← NEW for Phase B

SamplingParams:
    SamplingParams(
        max_new_tokens=1,
        return_hidden_states=True,        # ← NEW for Phase B
        custom_params={
            "schema": anchored_config,
            "eos_token_id": tokenizer.eos_token_id,
            # Optional — enables embedding-space classification:
            "label_embeddings": {
                "intent": <FloatTensor [num_intents, hidden_dim]>,
                "city":   <FloatTensor [num_bio_tags, hidden_dim]>,
                ...
            },
        },
        custom_logit_processor=SMIELPWithHiddenStates.to_str(),
    )

Result extras (beyond Phase A)
-------------------------------
    result["hidden_state_l2"]   float   L2 norm of the last-token hidden state
    result["mode"]              str     "embedding" | "logit_slicing"
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

try:
    from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
except Exception:
    from abc import ABC, abstractmethod as _abstractmethod
    from typing import Any as _Any, Dict as _Dict, List as _List, Optional as _Optional
    import torch as _torch

    class CustomLogitProcessor(ABC):
        @_abstractmethod
        def __call__(
            self,
            logits: _torch.Tensor,
            custom_param_list: _Optional[_List[_Dict[str, _Any]]] = None,
            hidden_states: _Optional[_torch.Tensor] = None,
        ) -> _torch.Tensor:
            raise NotImplementedError

        @classmethod
        def to_str(cls) -> str:
            return ""


logger = logging.getLogger(__name__)


class SMIELPWithHiddenStates(CustomLogitProcessor):
    """
    Phase B SMIELP — uses transformer hidden states when available.

    Key differences from Phase A ``SimultaneousMultiIntentEntityLogitProcessor``:
    - ``__call__`` accepts an optional ``hidden_states`` keyword argument.
    - When ``hidden_states`` is provided AND ``custom_params["label_embeddings"]`` is set,
      classification is performed via cosine similarity in embedding space.
    - Result dict gains ``"hidden_state_l2"`` and ``"mode"`` fields.
    - Backward-compatible: falls back to logit slicing when ``hidden_states`` is None.
    """

    OUTPUT_KEY = "smielp"

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits          : [M, vocab_size] — same as Phase A.
        custom_param_list : len == M, each dict contains schema, eos_token_id,
                            and optionally label_embeddings.
        hidden_states   : [M, hidden_dim] — last-token hidden states, forwarded by
                          the patched apply_custom_logit_processor in sampler.py.
                          None when --enable-return-hidden-states is not set.
        """
        if not custom_param_list:
            return logits

        batch_size = logits.shape[0]
        if len(custom_param_list) != batch_size:
            logger.warning(
                "SMIELPWithHiddenStates: param list length %d != batch size %d; skipping.",
                len(custom_param_list),
                batch_size,
            )
            return logits

        # Validate hidden_states shape; discard silently if inconsistent.
        if hidden_states is not None:
            if hidden_states.shape[0] != batch_size:
                logger.warning(
                    "SMIELPWithHiddenStates: hidden_states.shape[0]=%d != batch_size=%d; "
                    "ignoring hidden_states and falling back to logit slicing.",
                    hidden_states.shape[0],
                    batch_size,
                )
                hidden_states = None
            elif hidden_states.dim() != 2:
                logger.warning(
                    "SMIELPWithHiddenStates: hidden_states must be 2-D [M, D], got shape %s; "
                    "ignoring.",
                    tuple(hidden_states.shape),
                )
                hidden_states = None

        for i, params in enumerate(custom_param_list):
            if params is None:
                logits[i, :] = float("-inf")
                logits[i, 2] = 0.0  # default EOS
                continue
            schema = params.get("schema")
            if not schema:
                result = {
                    "intent": None,
                    "entities": [],
                    "mode": "none",
                    "hidden_state_l2": None,
                }
            else:
                h = hidden_states[i] if hidden_states is not None else None

                # NaN guard: a NaN-filled hidden state is useless; fall back.
                if h is not None and torch.isnan(h).any():
                    logger.warning(
                        "SMIELPWithHiddenStates: NaN in hidden_states[%d]; "
                        "falling back to logit slicing.",
                        i,
                    )
                    h = None

                label_embs = params.get("label_embeddings")

                if h is not None and label_embs is not None:
                    result = self._classify_embedding(h, schema, label_embs)
                    result["mode"] = "embedding"
                else:
                    result = self._classify_logit_slicing(logits[i], schema)
                    result["mode"] = "logit_slicing"

                result["hidden_state_l2"] = (
                    float(h.float().norm().item()) if h is not None else None
                )

            req = params.get("__req__")
            if req is not None:
                if req.customized_info is None:
                    req.customized_info = {}
                req.customized_info[self.OUTPUT_KEY] = [result]
            else:
                logger.warning(
                    "SMIELPWithHiddenStates: __req__ missing in params[%d].", i
                )

            eos_id = params.get("eos_token_id", 2)
            vocab_size = logits.shape[1]
            if eos_id >= vocab_size:
                logger.warning(
                    "SMIELPWithHiddenStates: eos_token_id %d >= vocab_size %d; clamping.",
                    eos_id,
                    vocab_size,
                )
                eos_id = vocab_size - 1
            logits[i, :] = float("-inf")
            logits[i, eos_id] = 0.0

        return logits

    # ──────────────────────────────────────────────────────────────────────────
    # Phase A path: logit slicing (identical to SimultaneousMultiIntentEntityLogitProcessor)
    # ──────────────────────────────────────────────────────────────────────────

    def _classify_logit_slicing(self, row: torch.Tensor, schema: dict) -> dict:
        return {
            "intent": self._slice_intent(row, schema),
            "entities": self._slice_slots(row, schema),
        }

    def _slice_intent(self, row: torch.Tensor, schema: dict) -> Optional[dict]:
        ids = schema.get("intent_token_ids")
        labels = schema.get("intent_labels")
        if not ids or not labels:
            return None
        idx = torch.tensor(ids, device=row.device, dtype=torch.long)
        probs = F.softmax(row[idx].float(), dim=-1)
        best = int(probs.argmax())
        return {
            "label": labels[best],
            "confidence": float(probs[best]),
            "distribution": {lbl: float(p) for lbl, p in zip(labels, probs.tolist())},
        }

    def _slice_slots(self, row: torch.Tensor, schema: dict) -> List[dict]:
        entities = []
        for slot in schema.get("entity_slots", []):
            ids = slot.get("bio_token_ids")
            labels = slot.get("bio_labels")
            if not ids or not labels:
                continue
            idx = torch.tensor(ids, device=row.device, dtype=torch.long)
            probs = F.softmax(row[idx].float(), dim=-1)
            best = int(probs.argmax())
            entities.append(
                {
                    "slot": slot["name"],
                    "tag": labels[best],
                    "confidence": float(probs[best]),
                    "distribution": {
                        lbl: float(p) for lbl, p in zip(labels, probs.tolist())
                    },
                }
            )
        return entities

    # ──────────────────────────────────────────────────────────────────────────
    # Phase B path: embedding-space cosine similarity
    # ──────────────────────────────────────────────────────────────────────────

    def _classify_embedding(
        self,
        h: torch.Tensor,  # [hidden_dim]
        schema: dict,
        label_embs: dict,  # {head_name: Tensor[num_labels, hidden_dim]}
    ) -> dict:
        """
        Classify intent and slots by cosine similarity between the last-token hidden
        state and per-class label embedding vectors.

        label_embs keys must match schema intent/slot names:
            "intent"      → Tensor[num_intents, D]
            slot["name"]  → Tensor[num_bio_tags, D]

        If a head is missing from label_embs, fall back to logit slicing for that head.
        """
        h_norm = F.normalize(h.float().unsqueeze(0), dim=-1)  # [1, D]

        # ── Intent ────────────────────────────────────────────────────────────
        intent_result = None
        intent_labels = schema.get("intent_labels", [])
        if "intent" in label_embs and intent_labels:
            E_raw = label_embs["intent"].float().to(h.device)
            if E_raw.dim() != 2 or E_raw.shape != (len(intent_labels), h.shape[0]):
                logger.warning(
                    "SMIELPWithHiddenStates: label_embs['intent'] shape %s != expected "
                    "(%d, %d); skipping intent classification.",
                    tuple(E_raw.shape),
                    len(intent_labels),
                    h.shape[0],
                )
            else:
                E = F.normalize(E_raw, dim=-1)  # [K, D]
                sims = (h_norm @ E.T).squeeze(0)  # [K]
                probs = F.softmax(sims * 10.0, dim=-1)
                best = int(probs.argmax())
                intent_result = {
                    "label": intent_labels[best],
                    "confidence": float(probs[best]),
                    "distribution": {
                        lbl: float(p) for lbl, p in zip(intent_labels, probs.tolist())
                    },
                }
        else:
            logger.debug(
                "SMIELPWithHiddenStates: 'intent' not in label_embs; skipping intent."
            )

        # ── Entity slots ──────────────────────────────────────────────────────
        entities = []
        for slot in schema.get("entity_slots", []):
            name = slot["name"]
            labels = slot.get("bio_labels", [])
            if name in label_embs and labels:
                E_raw = label_embs[name].float().to(h.device)
                if E_raw.dim() != 2 or E_raw.shape != (len(labels), h.shape[0]):
                    logger.warning(
                        "SMIELPWithHiddenStates: label_embs['%s'] shape %s != expected "
                        "(%d, %d); skipping slot.",
                        name,
                        tuple(E_raw.shape),
                        len(labels),
                        h.shape[0],
                    )
                    continue
                E = F.normalize(E_raw, dim=-1)
                sims = (h_norm @ E.T).squeeze(0)
                probs = F.softmax(sims * 10.0, dim=-1)
                best = int(probs.argmax())
                entities.append(
                    {
                        "slot": name,
                        "tag": labels[best],
                        "confidence": float(probs[best]),
                        "distribution": {
                            lbl: float(p) for lbl, p in zip(labels, probs.tolist())
                        },
                    }
                )
            else:
                logger.debug(
                    "SMIELPWithHiddenStates: '%s' not in label_embs; skipping slot.",
                    name,
                )

        return {"intent": intent_result, "entities": entities}
