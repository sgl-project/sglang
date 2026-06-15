"""
SimultaneousMultiIntentEntityLogitProcessor
============================================
Performs single-forward-pass structured extraction by intercepting the model's
logit tensor at the one decode step, slicing intent and entity-slot dimensions
simultaneously, and writing a structured result dict to the response meta_info —
all with zero modifications to sglang core.

How it works
------------
1. The LLM processes the full prompt in a single prefill pass, building up a rich
   contextual representation of the input text.

2. At the single decode step (max_new_tokens=1) the logit tensor has shape
   [M, vocab_size].  M is the number of requests in this batch that use this
   processor (not necessarily the full batch size).

3. For each request we:
   a. Slice intent logit columns → softmax → argmax → predicted intent label
   b. Slice BIO-tag logit columns for each slot → softmax → argmax → tag per slot
      All slices happen in a single vectorised pass over logits[i].

4. Results are written to req.customized_info["smielp"].  sglang's output-streamer
   picks this up automatically and surfaces it in the HTTP response under
   meta_info["smielp"] — no changes to the sglang scheduler or tokenizer manager.

5. The processor forces the logit for the EOS token to 0.0 (all others to -inf),
   so the sampler cleanly terminates generation after exactly one token.

Returned structure (per request, in meta_info["smielp"][0])
------------------------------------------------------------
{
    "intent": {
        "label":        str,
        "confidence":   float,          # max softmax probability
        "distribution": {label: prob},  # full softmax over intent labels
    },
    "entities": [
        {
            "slot":         str,
            "tag":          str,
            "confidence":   float,
            "distribution": {label: prob},
        },
        ...
    ],
}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

try:
    from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
except Exception:
    # Fallback when running tests outside a full sglang install.
    from abc import ABC, abstractmethod as _abstractmethod
    from typing import Any as _Any, Dict as _Dict, List as _List, Optional as _Optional
    import torch as _torch

    class CustomLogitProcessor(ABC):
        @_abstractmethod
        def __call__(
            self,
            logits: _torch.Tensor,
            custom_param_list: _Optional[_List[_Dict[str, _Any]]] = None,
        ) -> _torch.Tensor:
            raise NotImplementedError

        @classmethod
        def to_str(cls) -> str:
            return ""


logger = logging.getLogger(__name__)


class SimultaneousMultiIntentEntityLogitProcessor(CustomLogitProcessor):
    """
    Single-forward-pass NER/NLU via logit slicing.

    Server startup requirement
    --------------------------
        python -m sglang.launch_server ... --enable-custom-logit-processor

    Minimal SamplingParams
    ----------------------
        from sglang.srt.logit_slicing import NERSchema, IntentSchema, SlotSchema, VocabAnchor
        from sglang.srt.logit_slicing.vocab_anchor import build_anchor_config

        schema = NERSchema(
            intents=IntentSchema(labels=["book_flight", "cancel", "check_status"]),
            slots=[
                SlotSchema(name="city", labels=["O", "B", "I"]),
            ],
        )
        custom_params = build_anchor_config(schema, tokenizer)

        SamplingParams(
            max_new_tokens=1,
            custom_params=custom_params,
            custom_logit_processor=SimultaneousMultiIntentEntityLogitProcessor.to_str(),
        )

    Reading results
    ---------------
        response["meta_info"]["smielp"][0]
        # → {"intent": {"label": "book_flight", ...}, "entities": [...]}
    """

    # Key written into req.customized_info and surfaced in meta_info.
    OUTPUT_KEY = "smielp"

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits          : [M, vocab_size] GPU tensor — only the M requests that use
                          this processor (not the whole batch).
        custom_param_list : len == M.  Each dict contains:
                            "__req__"      → live Req object (injected by sglang)
                            "schema"       → anchored config from NERSchema.to_anchor_config()
                            "eos_token_id" → int, defaults to 2
        """
        if not custom_param_list:
            return logits

        batch_size = logits.shape[0]
        if len(custom_param_list) != batch_size:
            logger.warning(
                "SMIELP: custom_param_list length %d != logits batch size %d; skipping.",
                len(custom_param_list),
                batch_size,
            )
            return logits

        results = self._classify_batch(logits, custom_param_list)

        for i, (params, result) in enumerate(zip(custom_param_list, results)):
            if params is None:
                logits[i, :] = float("-inf")
                logits[i, 2] = 0.0  # default EOS
                continue
            # ── Write to customized_info (zero-modification output path) ──────
            req = params.get("__req__")
            if req is not None:
                if req.customized_info is None:
                    req.customized_info = {}
                # One element per output token; we always generate exactly 1.
                req.customized_info[self.OUTPUT_KEY] = [result]
            else:
                logger.warning("SMIELP: __req__ not found in custom_params[%d].", i)

            # ── Force EOS to terminate generation after this single step ──────
            eos_id = params.get("eos_token_id", 2)
            vocab_size = logits.shape[1]
            if eos_id >= vocab_size:
                logger.warning(
                    "SMIELP: eos_token_id %d >= logit vocab_size %d (model has padded vocab); "
                    "clamping to last logit column %d.",
                    eos_id,
                    vocab_size,
                    vocab_size - 1,
                )
                eos_id = vocab_size - 1
            logits[i, :] = float("-inf")
            logits[i, eos_id] = 0.0

        return logits

    # ──────────────────────────────────────────────────────────────────────────
    # Core classification logic
    # ──────────────────────────────────────────────────────────────────────────

    def _classify_batch(
        self,
        logits: torch.Tensor,
        custom_param_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run simultaneous logit slicing for all M requests."""
        results = []
        for i, params in enumerate(custom_param_list):
            if params is None:
                results.append({"intent": None, "entities": []})
                continue
            schema = params.get("schema")
            if not schema:
                logger.warning(
                    "SMIELP: no 'schema' key in custom_params[%d]; returning empty result.",
                    i,
                )
                results.append({"intent": None, "entities": []})
                continue

            row = logits[i]  # [vocab_size]
            results.append(
                {
                    "intent": self._classify_intent(row, schema),
                    "entities": self._classify_slots(row, schema),
                }
            )
        return results

    def _classify_intent(
        self, row: torch.Tensor, schema: dict
    ) -> Optional[Dict[str, Any]]:
        intent_token_ids = schema.get("intent_token_ids")
        intent_labels = schema.get("intent_labels")
        if not intent_token_ids or not intent_labels:
            return None

        idx = torch.tensor(intent_token_ids, device=row.device, dtype=torch.long)
        sliced = row[idx].float()  # [num_intents]
        probs = F.softmax(sliced, dim=-1)  # numerically stable
        best = int(probs.argmax().item())

        return {
            "label": intent_labels[best],
            "confidence": float(probs[best]),
            "distribution": {
                label: float(p) for label, p in zip(intent_labels, probs.tolist())
            },
        }

    def _classify_slots(self, row: torch.Tensor, schema: dict) -> List[Dict[str, Any]]:
        entities = []
        for slot in schema.get("entity_slots", []):
            bio_token_ids = slot.get("bio_token_ids")
            bio_labels = slot.get("bio_labels")
            if not bio_token_ids or not bio_labels:
                continue

            idx = torch.tensor(bio_token_ids, device=row.device, dtype=torch.long)
            sliced = row[idx].float()  # [num_tags]
            probs = F.softmax(sliced, dim=-1)
            best = int(probs.argmax().item())

            entities.append(
                {
                    "slot": slot["name"],
                    "tag": bio_labels[best],
                    "confidence": float(probs[best]),
                    "distribution": {
                        label: float(p) for label, p in zip(bio_labels, probs.tolist())
                    },
                }
            )
        return entities
