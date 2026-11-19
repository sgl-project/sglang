# Copyright 2026 SGLang Team
# Copyright the HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
"""Render Nemotron Decision's checkpoint-owned prompt and answer labels."""

import json
import math
import re

from tokenizers import Tokenizer

from sglang.srt.configs.nemotron_decision import is_nemotron_decision
from sglang.srt.entrypoints.openai.protocol import (
    DecisionChoiceQuestion,
    DecisionScoreQuestion,
)
from sglang.srt.entrypoints.systemone.protocol import NemotronDecisionRequest


class NemotronDecideEncoder:
    # Version 1 uses the generic /v1/decisions prompt, not the checkpoint template.
    prompt_format_version = 2

    def __init__(self, *, config, metadata, tokenizer, max_prompt_tokens):
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.mask_token_id = config.mask_token_id
        self.max_prompt_tokens = min(max_prompt_tokens, metadata["max_input_tokens"])
        self.label_ids = {}
        for kind, labels in metadata["labels"].items():
            ids = [
                tokenizer.encode(label, add_special_tokens=False) for label in labels
            ]
            if any(len(item) != 1 for item in ids) or len(
                {item[0] for item in ids}
            ) != len(ids):
                raise ValueError(
                    f"Nemotron Decision {kind} labels must use distinct single tokens"
                )
            self.label_ids[kind] = [item[0] for item in ids]
        control_tokens = tuple(
            token.content
            for token in tokenizer.added_tokens_decoder.values()
            if token.special
            or (
                token.content in metadata["chat_template"]
                and not token.content.isspace()
            )
        )
        self.control_tokens = re.compile(
            "|".join(
                re.escape(token)
                for token in sorted(control_tokens, key=len, reverse=True)
            )
            or "(?!)"
        )
        self.markers = re.compile(
            "\U000f0000("
            + self.control_tokens.pattern
            + ")\U000f0001|[\U000f0000\U000f0001]"
        )
        # Splitting control tokens must not mutate the shared tokenizer during concurrent requests.
        self.text_tokenizer = Tokenizer.from_str(tokenizer.backend_tokenizer.to_str())
        self.text_tokenizer.encode_special_tokens = True
        self.newline_ids = tokenizer.encode("\n", add_special_tokens=False)

    def validate_request(self, request):
        if len(request.questions) > self.metadata["max_questions"]:
            return f"Nemotron Decision supports at most {self.metadata['max_questions']} questions"
        for name, question in request.questions.items():
            if "instructions" not in question.model_fields_set:
                return f"question {name!r} requires instructions"
            count = 2 if question.type == "noul" else len(question.criteria)
            limit = min(
                len(self.label_ids[question.type]),
                self.metadata["max_candidates_per_read"],
            )
            if not 2 <= count <= limit:
                return f"question {name!r} needs 2–{limit} options or levels"
        return None

    def encode(self, state, question, chat_template_kwargs):
        if chat_template_kwargs:
            raise ValueError(
                "Nemotron Decision's trained prompt takes no chat_template_kwargs"
            )
        kind = question.type
        if kind == "noul":
            criteria = (
                question.criteria.model_dump(exclude_unset=True)
                if question.criteria is not None
                else {}
            )
            entries = [
                ("false", criteria.get("false", "No, the condition does not hold.")),
                ("true", criteria.get("true", "Yes, the condition holds.")),
            ]
        elif kind == "score":
            entries = list(
                zip(map(str, range(len(question.criteria))), question.criteria)
            )
        else:
            entries = list(question.criteria.items())
        labels = self.metadata["labels"][kind][: len(entries)]
        if (
            len(labels) != len(entries)
            or len(labels) > self.metadata["max_candidates_per_read"]
        ):
            raise ValueError(f"Nemotron Decision has too many {kind} candidates")
        options = [
            dict(code=code, label=name, meaning=description)
            for code, (name, description) in zip(labels, entries)
        ]
        content = json.dumps(
            dict(state=state, question=question.instructions, options=options),
            ensure_ascii=False,
            allow_nan=False,
        )
        content = content.replace("\U000f0000", "").replace("\U000f0001", "")
        marked = self.control_tokens.sub(
            lambda match: "\U000f0000" + match[0] + "\U000f0001", content
        )
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            chat_template=self.metadata["chat_template"],
            decision_content=marked,
            tokenize=False,
            add_generation_prompt=False,
        )
        ids = self._encode_marked_prompt(rendered)
        if (
            not ids
            or ids[-1] != self.mask_token_id
            or ids.count(self.mask_token_id) != 1
        ):
            raise ValueError(
                "The Decision template must produce exactly one trailing mask token"
            )
        if len(ids) > self.max_prompt_tokens:
            raise ValueError(
                f"the prompt has {len(ids)} tokens, limit {self.max_prompt_tokens}; a bidirectional read cannot be chunked"
            )
        label_ids = self.label_ids[kind][: len(labels)]
        # The API's boolean view is yes/no; the trained head is false/true.
        return ids, label_ids[::-1] if kind == "noul" else label_ids, []

    def encode_decisions(self, request, chat_template_kwargs):
        questions = {}
        for question in request.questions:
            if isinstance(question, DecisionChoiceQuestion):
                kind = "choice"
                criteria = {
                    option.name: option.description for option in question.options
                }
            elif isinstance(question, DecisionScoreQuestion):
                kind, criteria = "score", question.levels
            else:
                kind = "noul"
                criteria = {
                    {"yes": "true", "no": "false"}[name]: value
                    for name, value in question.model_dump(
                        include={"yes", "no"}, exclude_unset=True
                    ).items()
                }
            questions[question.id] = dict(
                type=kind, instructions=question.question, criteria=criteria
            )
        adapted = NemotronDecisionRequest(
            model=request.model,
            state=request.input,
            questions=questions,
            chat_template_kwargs=chat_template_kwargs,
        )
        error = self.validate_request(adapted)
        if error is not None:
            raise ValueError(error)
        for name, question in adapted.questions.items():
            try:
                yield self.encode(adapted.state, question, adapted.chat_template_kwargs)
            except ValueError as error:
                raise ValueError(f"question {name!r}: {error}") from error

    def calibrate(self, kind, probabilities, logprobs, temperature=1.0):
        model_temperature = self.metadata.get("temperature", {}).get(kind, 1.0)
        if model_temperature == temperature == 1.0:
            return probabilities
        maximum = max(logprobs)
        weights = [
            math.exp((value - maximum) / model_temperature / temperature)
            for value in logprobs
        ]
        total = math.fsum(weights)
        return [weight / total for weight in weights]

    def _encode_marked_prompt(self, rendered):
        # Match Transformers' sanitization while leaving template-owned control tokens intact.
        pieces = self.markers.split(rendered)
        ids = self.tokenizer.encode(pieces[0], add_special_tokens=False)
        for index, piece in enumerate(pieces[1:], start=1):
            if not piece:
                continue
            as_text = index % 2 == 1
            encoded = self._encode_piece("\n" + piece, as_text)
            if encoded[: len(self.newline_ids)] == self.newline_ids:
                encoded = encoded[len(self.newline_ids) :]
            else:
                encoded = self._encode_piece(piece, as_text)
            if as_text and self.tokenizer.convert_tokens_to_ids(piece) in encoded:
                raise ValueError(f"Cannot encode {piece!r} as plain text")
            ids.extend(encoded)
        return ids

    def _encode_piece(self, text, as_text):
        if as_text:
            return self.text_tokenizer.encode(text, add_special_tokens=False).ids
        return self.tokenizer.encode(text, add_special_tokens=False)


def create_decision_encoder(model_config, tokenizer, max_prompt_tokens):
    if tokenizer is None or not is_nemotron_decision(model_config.hf_config):
        return None
    return NemotronDecideEncoder(
        config=model_config.hf_config,
        metadata=model_config.nemotron_decision_config,
        tokenizer=tokenizer,
        max_prompt_tokens=max_prompt_tokens,
    )
