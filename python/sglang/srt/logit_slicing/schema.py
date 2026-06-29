"""
NER/NLU schema definitions for the Simultaneous Multi-Intent & Entity Logit Processor.

A NERSchema declares:
  - Which intent classes to classify (IntentSchema)
  - Which entity slots to fill, each with its own set of BIO/categorical labels (SlotSchema)

After vocab anchoring (VocabAnchor.anchor), each label is mapped to a single token ID in
the target model's vocabulary.  The schema is then serialised to a plain dict and shipped
to the GPU worker via SamplingParams.custom_params.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class IntentSchema:
    """Classification labels for the intent head."""

    labels: List[str]
    # Filled by VocabAnchor.anchor(); empty until anchored.
    label_to_token_id: Dict[str, int] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if len(self.labels) < 2:
            raise ValueError("IntentSchema requires at least 2 labels.")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"IntentSchema has duplicate labels: {self.labels}")

    @property
    def is_anchored(self) -> bool:
        return len(self.label_to_token_id) == len(self.labels)

    def token_ids(self) -> List[int]:
        """Return token IDs in the same order as self.labels."""
        if not self.is_anchored:
            raise RuntimeError(
                "IntentSchema has not been anchored yet. Call VocabAnchor.anchor()."
            )
        return [self.label_to_token_id[lbl] for lbl in self.labels]


@dataclasses.dataclass
class SlotSchema:
    """A single entity slot with its own classification labels.

    Labels are typically BIO tags (B, I, O) or domain-specific categories
    (e.g. ["none", "present"] for binary slot presence detection).
    """

    name: str
    labels: List[str]
    # Filled by VocabAnchor.anchor(); empty until anchored.
    label_to_token_id: Dict[str, int] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("SlotSchema.name must not be empty.")
        if len(self.labels) < 2:
            raise ValueError(f"SlotSchema '{self.name}' requires at least 2 labels.")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(
                f"SlotSchema '{self.name}' has duplicate labels: {self.labels}"
            )

    @property
    def is_anchored(self) -> bool:
        return len(self.label_to_token_id) == len(self.labels)

    def token_ids(self) -> List[int]:
        """Return token IDs in the same order as self.labels."""
        if not self.is_anchored:
            raise RuntimeError(
                f"SlotSchema '{self.name}' has not been anchored. Call VocabAnchor.anchor()."
            )
        return [self.label_to_token_id[lbl] for lbl in self.labels]


@dataclasses.dataclass
class NERSchema:
    """
    Top-level schema bundling intent classification and entity slot filling.

    Typical usage:
        schema = NERSchema(
            intents=IntentSchema(labels=["book_flight", "cancel", "status_check"]),
            slots=[
                SlotSchema(name="departure_city", labels=["O", "B", "I"]),
                SlotSchema(name="destination_city", labels=["O", "B", "I"]),
                SlotSchema(name="date", labels=["O", "B", "I"]),
            ],
        )
        anchored = VocabAnchor().anchor(schema, tokenizer)
        custom_params = {"schema": anchored.to_anchor_config(), "eos_token_id": tokenizer.eos_token_id}
    """

    intents: IntentSchema
    slots: List[SlotSchema] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        slot_names = [s.name for s in self.slots]
        if len(set(slot_names)) != len(slot_names):
            raise ValueError(f"NERSchema has duplicate slot names: {slot_names}")

    @property
    def is_anchored(self) -> bool:
        return self.intents.is_anchored and all(s.is_anchored for s in self.slots)

    def to_anchor_config(self) -> dict:
        """
        Serialise the fully-anchored schema to a plain dict suitable for
        SamplingParams.custom_params["schema"].

        The dict is intentionally primitive (lists of ints/strings) so it
        can be JSON-serialised and is safe to pass across process boundaries.
        """
        if not self.is_anchored:
            raise RuntimeError(
                "Schema must be anchored before calling to_anchor_config(). "
                "Run VocabAnchor().anchor(schema, tokenizer) first."
            )
        return {
            "intent_labels": list(self.intents.labels),
            "intent_token_ids": self.intents.token_ids(),
            "entity_slots": [
                {
                    "name": slot.name,
                    "bio_labels": list(slot.labels),
                    "bio_token_ids": slot.token_ids(),
                }
                for slot in self.slots
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NERSchema":
        """Reconstruct from a plain dict (e.g. after JSON deserialisation)."""
        intents = IntentSchema(
            labels=d["intent_labels"],
            label_to_token_id=dict(zip(d["intent_labels"], d["intent_token_ids"])),
        )
        slots = [
            SlotSchema(
                name=slot["name"],
                labels=slot["bio_labels"],
                label_to_token_id=dict(zip(slot["bio_labels"], slot["bio_token_ids"])),
            )
            for slot in d.get("entity_slots", [])
        ]
        return cls(intents=intents, slots=slots)
