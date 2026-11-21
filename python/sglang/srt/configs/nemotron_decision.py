# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Checkpoint metadata for Nemotron Decision's bidirectional reads."""

import json
import math
from pathlib import Path

from transformers import PretrainedConfig
from transformers.utils import cached_file


def is_nemotron_decision(config: PretrainedConfig) -> bool:
    return "NemotronForDecision" in (config.architectures or [])


def load_nemotron_decision_config(model_path: str, revision: str | None) -> dict:
    config_path = cached_file(model_path, "decision.json", revision=revision)
    template_path = cached_file(
        model_path, "decision_template.jinja", revision=revision
    )
    config = json.loads(Path(config_path).read_text())
    config["chat_template"] = Path(template_path).read_text()
    if config.get("noul_options", ["false", "true"]) != ["false", "true"]:
        raise ValueError("Nemotron Decision requires noul_options in false, true order")
    for name in ("max_questions", "max_input_tokens", "max_candidates_per_read"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"decision.json {name} must be a positive integer")
    for kind in ("choice", "score", "noul"):
        labels = config["labels"][kind]
        if (
            not isinstance(labels, list)
            or len(labels) < 2
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError(
                f"decision.json requires distinct nonempty labels for {kind}"
            )
        temperature = config.get("temperature", {}).get(kind, 1.0)
        if (
            not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("decision.json temperatures must be finite and positive")
    if len(config["labels"]["noul"]) != 2:
        raise ValueError("Nemotron Decision requires two noul labels")
    return config


def validate_packed_decision_layout(layout, input_ids, mask_token_id, max_questions):
    if (
        not isinstance(layout, list)
        or len(layout) < 2
        or len(layout) > max_questions + 1
        or any(type(length) is not int or length < 1 for length in layout)
        or sum(layout) != len(input_ids)
    ):
        raise ValueError(
            "decision_layout needs a positive shared length and valid question lengths covering the prompt"
        )
    slots, end = [], layout[0]
    for length in layout[1:]:
        end += length
        slots.append(end - 1)
    if [i for i, token in enumerate(input_ids) if token == mask_token_id] != slots:
        raise ValueError("each packed question must end with its only mask token")
    return slots
