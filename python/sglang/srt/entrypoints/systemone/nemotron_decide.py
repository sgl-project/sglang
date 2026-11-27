# Copyright 2026 SGLang Team
# Copyright the HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
"""Render Nemotron Decision's checkpoint-owned prompt and answer labels."""

import json
import math
import re

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
        controls = sorted(
            tokenizer.get_added_vocab(), key=lambda token: (-len(token), token)
        )
        self.control_tokens = re.compile("|".join(map(re.escape, controls)) or "(?!)")
        self.state_end_token_id = vars(config).get("state_end_token_id")
        self.state_end_token = ""
        if self.state_end_token_id is not None:
            token_id = self.state_end_token_id
            added = {
                value: token for token, value in tokenizer.get_added_vocab().items()
            }
            if (
                type(token_id) is not int
                or token_id not in added
                or token_id == self.mask_token_id
            ):
                raise ValueError(
                    "state_end_token_id must name an added tokenizer token other than the mask token"
                )
            self.state_end_token = added[token_id]
            self._check_prompt(
                self._encode_content(dict(state=None, question="", options=[]))
            )

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
        ids = self._encode_content(
            dict(state=state, question=question.instructions, options=options)
        )
        self._check_prompt(ids)
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

    def _json_content(self, value):
        text = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if self.control_tokens.search(text) is None:
            return text
        return re.sub(r'"((?:[^"\\]|\\.)*)"', self._escape_literal, text)

    def _escape_literal(self, literal):
        text = literal[1]
        while True:
            escapes = list(re.finditer(r"\\(?:u[0-9a-fA-F]{4}|.)", text))
            pieces, last, position = [], 0, 0
            for match in self.control_tokens.finditer(text):
                start = match.start()
                while position < len(escapes) and escapes[position].end() <= start:
                    position += 1
                if position < len(escapes) and escapes[position].start() <= start:
                    continue
                # JSON escapes preserve the value without producing a control token.
                pieces.extend(
                    (
                        text[last:start],
                        f"\\u{ord(text[start]):04x}",
                        text[start + 1 : match.end()],
                    )
                )
                last = match.end()
            if not pieces:
                return '"' + text + '"'
            text = "".join((*pieces, text[last:]))

    def _encode_content(self, content):
        state = self._json_content(dict(state=content["state"]))[:-1]
        question = self._json_content(
            {key: value for key, value in content.items() if key != "state"}
        )[1:]
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            chat_template=self.metadata["chat_template"],
            decision_content=state + self.state_end_token + ", " + question,
            tokenize=False,
            add_generation_prompt=False,
        )
        return self.tokenizer.encode(rendered, add_special_tokens=False)

    def _check_prompt(self, ids):
        if (
            not ids
            or ids[-1] != self.mask_token_id
            or ids.count(self.mask_token_id) != 1
        ):
            raise ValueError(
                "The Decision template must produce exactly one trailing mask token"
            )
        if (
            self.state_end_token_id is not None
            and ids.count(self.state_end_token_id) != 1
        ):
            raise ValueError(
                "The Decision template must preserve exactly one state-end token"
            )


def create_decision_encoder(model_config, tokenizer, max_prompt_tokens):
    if tokenizer is None or not is_nemotron_decision(model_config.hf_config):
        return None
    return NemotronDecideEncoder(
        config=model_config.hf_config,
        metadata=model_config.nemotron_decision_config,
        tokenizer=tokenizer,
        max_prompt_tokens=max_prompt_tokens,
    )
