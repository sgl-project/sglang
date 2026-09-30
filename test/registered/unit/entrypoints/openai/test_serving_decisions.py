"""Unit tests for /v1/decisions and /v1/systemone: request limits, prompt text, answer labels, and scoring."""

import asyncio
import json
import math
import string
import unittest
from types import SimpleNamespace

import torch
from pydantic import ValidationError
from transformers import AddedToken, AutoTokenizer

from sglang.srt.entrypoints.openai import chat_encoding
from sglang.srt.entrypoints.openai.protocol import DecisionRequest
from sglang.srt.entrypoints.openai.serving_decisions import (
    PROMPT_FORMAT_VERSION,
    OpenAIServingDecisions,
    _decision_view,
    _encode_labels,
    _render_question,
)
from sglang.srt.entrypoints.systemone.protocol import SystemOneRequest
from sglang.srt.entrypoints.systemone.serving import (
    SystemOneServing,
    _answer,
    _choice_confidence,
    _score_confidence,
    _view,
)
from sglang.srt.managers.tokenizer_manager_score_mixin import TokenizerManagerScoreMixin
from sglang.srt.parser.template_detection import (
    ReasoningToggleConfig,
    detect_reasoning_parser,
    detect_reasoning_pattern,
)
from sglang.srt.runtime_context import publish, restore_context, snapshot_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

# Tokenizer files only, of a chat template that thinks by default.
TOKENIZER = "Qwen/Qwen3.5-35B-A3B"


def _question(kind, entries=None, question="Q"):
    """Options as a name to description map, levels as a list, or yes and no text."""
    body = {"type": kind, "question": question}
    if kind == "choice" and entries is not None:
        body["options"] = [
            {"name": name} if text is None else {"name": name, "description": text}
            for name, text in entries.items()
        ]
    elif kind == "score" and entries is not None:
        body["levels"] = entries
    elif entries is not None:
        body.update(entries)
    return body


def _request(input_, questions, **kwargs):
    """Questions as an id to question map, in order."""
    return DecisionRequest(
        input=input_,
        questions=[{"id": qid, **question} for qid, question in questions.items()],
        **kwargs,
    )


def _by_id(request, question_id):
    return next(q for q in request.questions if q.id == question_id)


def _encoded(handler, request):
    """Prompt and label ids for every question, as the handler scores them."""
    encoded, _ = handler._convert_to_internal_request(request)
    prompts, label_ids = zip(*encoded)
    return list(prompts), list(label_ids)


def _handler(
    manager,
    reasoning_config=None,
    reasoning_parser=None,
    lossy=False,
    serving_class=OpenAIServingDecisions,
):
    """Build the handler over the chat serving state the server builds at startup."""
    template = manager.tokenizer.chat_template
    force_reasoning, detected = detect_reasoning_pattern(template)
    if reasoning_config is None:
        reasoning_config = detected
    template_manager = SimpleNamespace(
        chat_template_name=None,
        reasoning_config=reasoning_config,
        suggested_reasoning_parser=detect_reasoning_parser(
            template, manager.tokenizer, detected, force_reasoning
        ),
    )
    chat_serving = SimpleNamespace(
        tokenizer_manager=manager,
        template_manager=template_manager,
        default_chat_template_kwargs={},
        chat_encoding_spec=chat_encoding.resolve_chat_encoding_spec(
            hf_config=manager.model_config.hf_config,
            tokenizer=manager.tokenizer,
            tool_call_parser=None,
        ),
        _prompt_text_round_trip_is_lossy=lossy,
        reasoning_parser=reasoning_parser,
    )
    return serving_class(chat_serving)


class UnknownTokenizer:
    """Encodes every character as the same unknown token."""

    def encode(self, text, add_special_tokens=False):
        return [0] * len(text)


class PlainTokenizer:
    """A tokenizer outside transformers, without added-token metadata."""

    def __init__(self, tokenizer):
        self.chat_template = tokenizer.chat_template
        self.encode = tokenizer.encode
        self.apply_chat_template = tokenizer.apply_chat_template
        self.vocab_size = len(tokenizer)

    def __len__(self):
        return self.vocab_size


class ScoringManager(TokenizerManagerScoreMixin):
    """Replace only model execution with fixed full-vocabulary logprobs."""

    def __init__(
        self,
        tokenizer,
        is_generation=True,
        context_len=4096,
        architecture=None,
        **server_args,
    ):
        self.server_args = ServerArgs(model_path="dummy", **server_args)
        publish(self.server_args, role="test")
        self.tokenizer = tokenizer
        self.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=[architecture or "Qwen3_5MoeForConditionalGeneration"],
                model_type="qwen3_5_moe",
            )
        )
        self.is_generation = is_generation
        self.context_len = context_len
        self.num_reserved_tokens = 0
        self.request_logger = SimpleNamespace(log_requests=False)
        generator = torch.Generator().manual_seed(0)
        logits = torch.randn(len(tokenizer), generator=generator, dtype=torch.float64)
        self.logprobs = torch.log_softmax(logits * 4, dim=0)
        self.requests = []
        self.served_model_name = "served-model"

    def config_value(self, name):
        return None

    async def generate_request(self, request, raw_request):
        self.requests.append(request)
        request.normalize_batch_and_arguments()
        results = []
        for ids, labels in zip(request.input_ids, request.token_ids_logprob):
            logprobs = [(self.logprobs[token].item(), token, None) for token in labels]
            meta = {"prompt_tokens": len(ids), "output_token_ids_logprobs": [logprobs]}
            results.append({"meta_info": meta})
        yield results


# Rendered lines of test_prompt_text by prompt format version.
PROMPT_FIXTURES = {
    1: {
        "choice": [
            'Question: {"question":"Which team?"}',
            "A: billing - Payments",
            "B: sales",
            'C: other - {"k":1}',
            "Answer with the letter of one option only.",
        ],
        "score": [
            "Question: Mood?",
            "0: Calm",
            "1: Angry",
            "Answer with the number of one level only.",
        ],
        "yes_no": [
            "Is the following true? Urgent",
            "no: Can wait",
            "Answer with yes or no only.",
        ],
    },
}


class TestDecisions(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    def setUp(self):
        self.addCleanup(restore_context, snapshot_context())

    def test_request_limits_follow_the_label_alphabets(self):
        options = {name: None for name in string.ascii_lowercase}
        rejected = {
            "unknown type": {"type": "rank", "question": "Q"},
            "one option": _question("choice", {"a": None}),
            "27 options": _question("choice", {**options, "extra": None}),
            "blank option name": _question("choice", {"a": None, " ": None}),
            "repeated option name": _question("choice", {"a": None, " A": None}),
            "option name with a line break": _question(
                "choice", {"a\nB: b": None, "c": None}
            ),
            "one level": _question("score", ["low"]),
            "11 levels": _question("score", [str(i) for i in range(11)]),
            "blank level": _question("score", ["low", " "]),
            "no question text": {"type": "yes_no"},
            "blank question text": _question("yes_no", question=" "),
            "empty object question text": _question("yes_no", question={}),
            "misspelled yes_no description": _question("yes_no", {"true": "Urgent"}),
        }
        for case, question in rejected.items():
            with self.subTest(case), self.assertRaises(ValidationError) as caught:
                _request("s", {"q1": question})
            self.assertIn("questions.0", str(caught.exception))
        question = [{"id": "q1", **_question("yes_no")}]
        for payload in (
            {"input": "s", "questions": []},
            {"input": "s", "questions": question, "temperature": 0},
            {"input": " ", "questions": question},
            {"input": [], "questions": question},
            {"input": "s", "questions": question, "top_p": 0.5},
            {"input": "s", "questions": [{"id": " ", **_question("yes_no")}]},
            {"input": "s", "questions": question * 2},
        ):
            with self.subTest(payload), self.assertRaises(ValidationError):
                DecisionRequest(**payload)
        request = _request(
            "s",
            {
                "choice": _question("choice", options),
                "score": _question("score", [str(i) for i in range(10)]),
                "yes_no": _question("yes_no"),
            },
        )
        self.assertEqual(request.chat_template_kwargs, {})

    def test_prompt_text(self):
        request = _request(
            {"ticket": "Refund please", "tags": ["billing"]},
            {
                "choice": _question(
                    "choice",
                    {"billing": "Payments", "sales": None, "other": {"k": 1}},
                    question={"question": "Which team?"},
                ),
                "score": _question("score", ["Calm", "Angry"], question="Mood?"),
                "yes_no": _question("yes_no", {"no": "Can wait"}, question="Urgent"),
            },
        )
        text = '{"ticket":"Refund please","tags":["billing"]}'
        # A wording change needs a new PROMPT_FORMAT_VERSION and its own fixture.
        self.assertIn(PROMPT_FORMAT_VERSION, PROMPT_FIXTURES)
        labels = {
            "choice": ["A", "B", "C"],
            "score": ["0", "1"],
            "yes_no": ["yes", "no"],
        }
        for question_id, lines in PROMPT_FIXTURES[PROMPT_FORMAT_VERSION].items():
            rendered = _render_question(
                text=text,
                view=_decision_view(_by_id(request, question_id)),
                labels=labels[question_id],
            )
            self.assertEqual(rendered, "\n".join([text, "", *lines]))
        # Renderings only /v1/systemone reaches: no question text, criteria-only
        # yes or no questions, and two-letter labels beyond 26 options.
        systemone = SystemOneRequest(
            state="s",
            model="m",
            questions={
                "choice": {
                    "type": "choice",
                    "criteria": {"billing": "Payments", "sales": None},
                },
                "score": {"type": "score", "criteria": ["Calm", "Angry"]},
                "noul": {
                    "type": "noul",
                    "criteria": {"true": "Reply today", "false": "Can wait"},
                },
            },
        )
        cases = {
            "choice": (
                ["AA", "AB"],
                [
                    "AA: billing - Payments",
                    "AB: sales",
                    "Answer with the letter of one option only.",
                ],
            ),
            "score": (
                ["0", "1"],
                ["0: Calm", "1: Angry", "Answer with the number of one level only."],
            ),
            "noul": (
                ["yes", "no"],
                [
                    "Is the following true?",
                    "yes: Reply today",
                    "no: Can wait",
                    "Answer with yes or no only.",
                ],
            ),
        }
        for question_id, (labels, lines) in cases.items():
            rendered = _render_question(
                text="s", view=_view(systemone.questions[question_id]), labels=labels
            )
            self.assertEqual(rendered, "\n".join(["s", "", *lines]))

    def test_labels_are_vocabulary_tokens_after_the_non_thinking_prompt(self):
        handler = _handler(ScoringManager(self.tokenizer))
        questions = {
            "choice": _question("choice", {c: None for c in string.ascii_lowercase}),
            "score": _question("score", [str(i) for i in range(10)]),
            "yes_no": _question("yes_no"),
        }
        labels = [
            list(string.ascii_uppercase),
            [str(i) for i in range(10)],
            ["yes", "no"],
        ]
        label_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in labels]
        # Other kwargs and a server default that thinks keep reasoning off.
        for kwargs, server_kwargs in (
            ({}, {}),
            ({"chat_template_kwargs": {}}, {}),
            ({"chat_template_kwargs": {"custom_flag": 1}}, {}),
            ({}, {"enable_thinking": True}),
        ):
            with self.subTest(kwargs=kwargs, server_kwargs=server_kwargs):
                handler.default_chat_template_kwargs = server_kwargs
                request = _request("s", questions, **kwargs)
                prompts, ids = _encoded(handler, request)
                self.assertEqual(ids, label_ids)
                for prompt in prompts:
                    self.assertTrue(
                        self.tokenizer.decode(prompt).endswith("</think>\n\n")
                    )
                    # The suffix check agrees with the whole-prompt check.
                    text = self.tokenizer.decode(prompt)
                    for question_labels, question_ids in zip(labels, label_ids):
                        self.assertEqual(
                            _encode_labels(
                                tokenizer=self.tokenizer,
                                prompt=text,
                                prompt_ids=prompt,
                                labels=question_labels,
                                added_tokens={},
                            ),
                            question_ids,
                        )

    def test_request_kwargs_override_the_server_defaults(self):
        handler = _handler(ScoringManager(self.tokenizer))
        handler.default_chat_template_kwargs = {"note": "server"}
        original_template = self.tokenizer.chat_template
        self.addCleanup(setattr, self.tokenizer, "chat_template", original_template)
        self.tokenizer.chat_template = "{{ messages[0]['content'] }} {{ note }}\n\n"
        question = {"q": _question("choice", {"a": None, "b": None})}
        for kwargs, note in (({}, "server"), ({"note": "request"}, "request")):
            with self.subTest(kwargs):
                request = _request("s", question, chat_template_kwargs=kwargs)
                prompts, _ = _encoded(handler, request)
                self.assertTrue(
                    self.tokenizer.decode(prompts[0]).endswith(f" {note}\n")
                )

    def test_label_check_skips_the_text_before_the_last_added_token(self):
        encoded = []
        tokenizer = SimpleNamespace(
            encode=lambda text, **kwargs: (
                encoded.append(text) or self.tokenizer.encode(text, **kwargs)
            )
        )
        added_tokens = {i: t for t, i in self.tokenizer.get_added_vocab().items()}
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "state " * 2000}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        label_ids = _encode_labels(
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_ids=prompt_ids,
            labels=["A", "B"],
            added_tokens=added_tokens,
        )
        self.assertEqual(label_ids, self.tokenizer.convert_tokens_to_ids(["A", "B"]))
        self.assertLess(max(len(text) for text in encoded), 16)

    def test_label_check_falls_back_when_the_suffix_does_not_split(self):
        # An added token that absorbs the following space tokenizes differently
        # from the suffix on its own, so the whole prompt is checked.
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        tokenizer.add_tokens([AddedToken("<mark>", rstrip=True)])
        prompt = "state <mark> "
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        added_tokens = {i: t for t, i in tokenizer.get_added_vocab().items()}
        self.assertEqual(
            _encode_labels(
                tokenizer=tokenizer,
                prompt=prompt,
                prompt_ids=prompt_ids,
                labels=["A"],
                added_tokens=added_tokens,
            ),
            tokenizer.encode(prompt + "A", add_special_tokens=False)[-1:],
        )

    def test_labels_must_be_distinct_tokens(self):
        with self.assertRaisesRegex(ValueError, "label 'B' is not one distinct"):
            _encode_labels(
                tokenizer=UnknownTokenizer(),
                prompt="p",
                prompt_ids=[0],
                labels=["A", "B"],
                added_tokens={},
            )

    async def test_questions_yield_to_other_requests(self):
        noul = {"type": "noul", "instructions": "x"}
        for route, handler, request in (
            (
                "decisions",
                _handler(ScoringManager(self.tokenizer)),
                _request("s", {q: _question("yes_no") for q in "abc"}),
            ),
            (
                "systemone",
                _handler(
                    ScoringManager(self.tokenizer), serving_class=SystemOneServing
                ),
                _systemone_request({q: noul for q in "abc"}),
            ),
        ):
            with self.subTest(route):
                events = []
                encode = handler._encode_question

                def recorded(**kwargs):
                    events.append("encode")
                    return encode(**kwargs)

                async def other_request():
                    for _ in range(3):
                        events.append("other")
                        await asyncio.sleep(0)

                handler._encode_question = recorded
                other = asyncio.create_task(other_request())
                response = await handler.handle_request(request, None)
                await other
                self.assertEqual(response.status_code, 200)
                self.assertEqual(events, ["encode", "other"] * 3)

    async def test_all_questions_are_scored_in_one_call(self):
        manager = ScoringManager(self.tokenizer)
        request = _request(
            "The integration keeps failing and I am losing sales.",
            {
                "department": _question(
                    "choice", {"billing": None, "technical": None, "sales": None}
                ),
                "frustration": _question("score", ["Calm", "Civil", "Angry"]),
                "is_urgent": _question("yes_no"),
            },
            temperature=2.0,
            prompt_format_version=PROMPT_FORMAT_VERSION,
        )
        response = await _handler(manager).handle_request(request, None)
        body = json.loads(response.body)
        self.assertEqual(len(manager.requests), 1)
        self.assertEqual(body["object"], "decisions")
        self.assertEqual(body["prompt_format_version"], PROMPT_FORMAT_VERSION)
        prompt_tokens = sum(len(ids) for ids in manager.requests[0].input_ids)
        self.assertEqual(body["usage"]["prompt_tokens"], prompt_tokens)
        self.assertEqual(body["usage"]["total_tokens"], prompt_tokens)
        self.assertEqual(body["usage"]["completion_tokens"], 0)
        cases = {
            "department": (["billing", "technical", "sales"], ["A", "B", "C"]),
            "frustration": (["0", "1", "2"], ["0", "1", "2"]),
            "is_urgent": (["yes", "no"], ["yes", "no"]),
        }
        self.assertEqual(list(body["answers"]), list(cases))
        for question_id, (names, tokens) in cases.items():
            answer = body["answers"][question_id]
            logprobs = manager.logprobs[self.tokenizer.convert_tokens_to_ids(tokens)]
            probabilities = torch.softmax(logprobs / 2.0, dim=0)
            self.assertEqual(list(answer["probabilities"]), names)
            torch.testing.assert_close(
                torch.tensor(
                    list(answer["probabilities"].values()), dtype=torch.float64
                ),
                probabilities,
            )
            # Relative only, label mass under these logits is far below 1e-7.
            torch.testing.assert_close(
                torch.tensor(answer["label_mass"], dtype=torch.float64),
                logprobs.exp().sum(),
                rtol=1e-9,
                atol=0,
            )
            self.assertEqual(answer["type"], _by_id(request, question_id).type)
            self.assertNotIn("prompt_token_ids", answer)
        department = body["answers"]["department"]
        self.assertEqual(
            department["choice"],
            max(department["probabilities"], key=department["probabilities"].get),
        )
        self.assertNotIn("score", department)
        frustration = body["answers"]["frustration"]["probabilities"]
        self.assertAlmostEqual(
            body["answers"]["frustration"]["score"],
            sum(int(level) * p for level, p in frustration.items()),
        )
        self.assertEqual(
            set(body["answers"]["is_urgent"]),
            {"type", "probabilities", "label_mass"},
        )

    async def test_returned_ids_are_the_scored_ids(self):
        manager = ScoringManager(self.tokenizer)
        request = _request(
            "s",
            {
                "team": _question("choice", {"a": None, "b": None}),
                "urgent": _question("yes_no"),
            },
            return_prompt_token_ids=True,
        )
        response = await _handler(manager).handle_request(request, None)
        answers = json.loads(response.body)["answers"]
        scored = manager.requests[0]
        self.assertEqual(
            [answers[q]["prompt_token_ids"] for q in ("team", "urgent")],
            scored.input_ids,
        )
        self.assertEqual(
            [answers[q]["label_token_ids"] for q in ("team", "urgent")],
            scored.token_ids_logprob,
        )

    async def test_refusals_name_the_question_and_skip_scoring(self):
        request = _request("s", {"first": _question("choice", {"a": None, "b": None})})
        prompt_len = len(
            _encoded(_handler(ScoringManager(self.tokenizer)), request)[0][0]
        )
        at_limit = ScoringManager(self.tokenizer, context_len=prompt_len + 3)
        at_limit.num_reserved_tokens = 3
        # The label merges with the trailing space, or follows a split double space.
        trailing_space = "{{ messages[0]['content'] }}\nAnswer: "
        double_space = "{{ messages[0]['content'] }}\nAnswer:  "
        cases = {
            "generation model": (
                ScoringManager(self.tokenizer, is_generation=False),
                None,
            ),
            "context length": (at_limit, None),
            "label 'A' is not one distinct token": (
                ScoringManager(self.tokenizer),
                trailing_space,
            ),
            "label 'A' is not one distinct token after": (
                ScoringManager(self.tokenizer),
                double_space,
            ),
        }
        original_template = self.tokenizer.chat_template
        self.addCleanup(setattr, self.tokenizer, "chat_template", original_template)
        for message, (manager, template) in cases.items():
            with self.subTest(message):
                self.tokenizer.chat_template = template or original_template
                response = await _handler(manager).handle_request(request, None)
                self.assertEqual(response.status_code, 400)
                error = json.loads(response.body)["message"]
                self.assertIn(message, error)
                if message != "generation model":
                    self.assertIn("question 'first'", error)
                self.assertEqual(manager.requests, [])
        self.tokenizer.chat_template = original_template
        below_limit = ScoringManager(self.tokenizer, context_len=prompt_len + 4)
        below_limit.num_reserved_tokens = 3
        response = await _handler(below_limit).handle_request(request, None)
        self.assertEqual(response.status_code, 200)

    async def test_refusals_keep_the_answer_outside_reasoning(self):
        question = {"first": _question("yes_no")}
        original_template = self.tokenizer.chat_template
        self.addCleanup(setattr, self.tokenizer, "chat_template", original_template)
        # Reasoning that the detected toggle does not control.
        open_block = "{{ messages[0]['content'] }}\nassistant\n<think>\n"
        cases = {
            "always reason": (None, ReasoningToggleConfig(special_case="always"), {}),
            "sets 'enable_thinking' to True": (None, None, {"enable_thinking": True}),
            "sets 'enable_thinking' to None": (None, None, {"enable_thinking": None}),
            "sets 'enable_thinking' to 0": (None, None, {"enable_thinking": 0}),
            "leaves a reasoning block open": (open_block, None, {}),
        }
        for message, (template, config, kwargs) in cases.items():
            with self.subTest(message):
                manager = ScoringManager(self.tokenizer)
                handler = _handler(manager, config)
                self.tokenizer.chat_template = template or original_template
                request = _request("s", question, chat_template_kwargs=kwargs)
                response = await handler.handle_request(request, None)
                self.tokenizer.chat_template = original_template
                self.assertEqual(response.status_code, 400)
                self.assertIn(message, json.loads(response.body)["message"])
                self.assertEqual(manager.requests, [])
        # A parser whose answers start inside reasoning needs a closed block, and
        # its advice comes first when the template's own replies also open one.
        no_block = (
            "{% for m in messages %}{% if m['role'] == 'user' %}"
            "{{ m['content'] }}\nassistant:\n\n{% else %}<think></think>"
            "{{ m['content'] }}{% endif %}{% endfor %}"
        )
        closed_block = (
            "{{ messages[0]['content'] }}\nassistant:\n<think>\n\n</think>\n\n"
        )
        for template, status in ((no_block, 400), (closed_block, 200)):
            with self.subTest(template=template[-24:]):
                manager = ScoringManager(self.tokenizer)
                handler = _handler(manager, reasoning_parser="deepseek-r1")
                self.tokenizer.chat_template = template
                request = _request(
                    "s", {"first": _question("choice", {"a": None, "b": None})}
                )
                response = await handler.handle_request(request, None)
                self.tokenizer.chat_template = original_template
                self.assertEqual(response.status_code, status)
                if status == 400:
                    self.assertIn(
                        "expects answers to start with a reasoning block",
                        json.loads(response.body)["message"],
                    )
        # A toggle that only the parser names, in its plain or explicit form, is
        # turned off and checked like a detected one.
        choice = {"first": _question("choice", {"a": None, "b": None})}
        for parser, toggle in (
            ("qwen3", "enable_thinking"),
            ("deepseek-v3", "thinking"),
        ):
            with self.subTest(parser=parser):
                self.tokenizer.chat_template = (
                    f"{{% if {toggle} is defined and {toggle} %}}/think\n{{% endif %}}"
                    "{{ messages[0]['content'] }}\nassistant:\n\n"
                )
                handler = _handler(
                    ScoringManager(self.tokenizer), reasoning_parser=parser
                )
                handler.default_chat_template_kwargs = {toggle: True}
                prompts, _ = _encoded(handler, _request("s", choice))
                thinking = _request("s", choice, chat_template_kwargs={toggle: True})
                response = await handler.handle_request(thinking, None)
                self.tokenizer.chat_template = original_template
                self.assertNotIn("/think", self.tokenizer.decode(prompts[0]))
                self.assertEqual(response.status_code, 400)
                self.assertIn(
                    f"sets '{toggle}' to True", json.loads(response.body)["message"]
                )
        # A template that opens reasoning before every answer is refused.
        for reply, status in (("<think></think>", 400), ("", 200)):
            with self.subTest(reply=reply):
                self.tokenizer.chat_template = (
                    "{% for m in messages %}{% if m['role'] == 'user' %}"
                    "{{ m['content'] }}\nassistant:\n\n{% else %}"
                    + reply
                    + "{{ m['content'] }}\n{% endif %}{% endfor %}"
                )
                manager = ScoringManager(self.tokenizer)
                handler = _handler(manager, reasoning_parser="qwen3")
                response = await handler.handle_request(_request("s", choice), None)
                self.tokenizer.chat_template = original_template
                self.assertEqual(response.status_code, status)
                if status == 400:
                    self.assertIn(
                        "starts every answer with a reasoning block",
                        json.loads(response.body)["message"],
                    )
                    self.assertEqual(manager.requests, [])
        # A reasoning tag inside the input does not count, with or without a
        # reasoning block after the message.
        request = _request(
            "<think> draft", {"first": _question("choice", {"a": None, "b": None})}
        )
        no_block = "{{ messages[0]['content'] }}\nassistant:\n\n"
        for template in (original_template, no_block):
            with self.subTest(template=template[-20:]):
                handler = _handler(ScoringManager(self.tokenizer))
                self.tokenizer.chat_template = template
                response = await handler.handle_request(request, None)
                self.tokenizer.chat_template = original_template
                self.assertEqual(response.status_code, 200)

    async def test_refusals_for_unsupported_serving_setups(self):
        def named_template():
            handler = _handler(ScoringManager(self.tokenizer))
            handler.template_manager.chat_template_name = "chatml"
            return handler

        # Each handler is built in its own case, since ScoringManager publishes
        # its server args globally.
        cases = {
            "model names the LoRA adapter 'adapter'": (
                lambda: _handler(ScoringManager(self.tokenizer)),
                {"model": "base:adapter"},
            ),
            # The version is checked before the reasoning settings.
            "uses version 1": (
                lambda: _handler(ScoringManager(self.tokenizer)),
                {
                    "prompt_format_version": 2,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            ),
            "'dsv4' encoder": (
                lambda: _handler(
                    ScoringManager(self.tokenizer, architecture="DeepseekV4ForCausalLM")
                ),
                {},
            ),
            "built-in chat template 'chatml'": (named_template, {}),
            "does not encode back to the same ids": (
                lambda: _handler(ScoringManager(self.tokenizer), lossy=True),
                {},
            ),
            "--enable-mis": (
                lambda: _handler(ScoringManager(self.tokenizer, enable_mis=True)),
                {},
            ),
            "--dllm-algorithm": (
                lambda: _handler(
                    ScoringManager(self.tokenizer, dllm_algorithm="LowConfidence")
                ),
                {},
            ),
        }
        for message, (build, kwargs) in cases.items():
            with self.subTest(message):
                handler = build()
                request = _request("s", {"q": _question("yes_no")}, **kwargs)
                response = await handler.handle_request(request, None)
                self.assertEqual(response.status_code, 400)
                self.assertIn(message, json.loads(response.body)["message"])
                self.assertEqual(handler.tokenizer_manager.requests, [])

    async def test_tokenizers_without_added_tokens_use_the_full_prompt(self):
        request = _request("s", {"q": _question("choice", {"a": None, "b": None})})
        plain = _handler(ScoringManager(PlainTokenizer(self.tokenizer)))
        self.assertEqual(plain.added_tokens, {})
        _, plain_ids = _encoded(plain, request)
        _, ids = _encoded(_handler(ScoringManager(self.tokenizer)), request)
        self.assertEqual(plain_ids, ids)

    def test_label_after_a_final_added_token_starts_a_new_segment(self):
        added_tokens = {i: t for t, i in self.tokenizer.get_added_vocab().items()}
        prompt = "state</think>"
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        self.assertIn(prompt_ids[-1], added_tokens)
        self.assertEqual(
            _encode_labels(
                tokenizer=self.tokenizer,
                prompt=prompt,
                prompt_ids=prompt_ids,
                labels=["A"],
                added_tokens=added_tokens,
            ),
            self.tokenizer.encode("A", add_special_tokens=False),
        )


def _systemone_request(questions, **kwargs):
    return SystemOneRequest(
        state="The integration keeps failing.",
        model="jev-latest",
        questions=questions,
        **kwargs,
    )


class TestSystemOne(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    def setUp(self):
        self.addCleanup(restore_context, snapshot_context())

    def _serving(self, manager=None, **kwargs):
        return _handler(
            manager or ScoringManager(self.tokenizer),
            serving_class=SystemOneServing,
            **kwargs,
        )

    async def test_answers_follow_the_published_shapes(self):
        manager = ScoringManager(self.tokenizer)
        legend_level = {"label": "Angry", "examples": ["caps"]}
        request = _systemone_request(
            {
                "urgent": {"type": "noul", "instructions": "Needs an answer today"},
                "team": {
                    "type": "choice",
                    "criteria": {"billing": None, "technical": "Bugs", "sales": None},
                },
                "mood": {"type": "score", "criteria": ["Calm", "Civil", legend_level]},
            }
        )
        response = await self._serving(manager).handle_request(request, None)
        self.assertEqual(response.status_code, 200)
        body = json.loads(response.body)
        self.assertEqual(set(body), {"model", "answers", "usage"})
        # The served model answers, never the requested alias.
        self.assertEqual(body["model"], "served-model")
        self.assertEqual(list(body["answers"]), ["urgent", "team", "mood"])
        prompt_tokens = sum(len(ids) for ids in manager.requests[0].input_ids)
        self.assertEqual(
            body["usage"], {"input_tokens": prompt_tokens, "output_tokens": 0}
        )

        urgent = body["answers"]["urgent"]
        self.assertEqual(set(urgent), {"type", "noul", "x_label_mass"})
        yes_no = manager.logprobs[self.tokenizer.convert_tokens_to_ids(["yes", "no"])]
        torch.testing.assert_close(
            torch.tensor(urgent["noul"], dtype=torch.float64),
            torch.softmax(yes_no, dim=0)[0],
        )

        team = body["answers"]["team"]
        self.assertEqual(
            set(team), {"type", "choice", "confidence", "probabilities", "x_label_mass"}
        )
        self.assertEqual(list(team["probabilities"]), ["billing", "technical", "sales"])
        self.assertEqual(
            team["choice"], max(team["probabilities"], key=team["probabilities"].get)
        )
        self.assertAlmostEqual(
            team["confidence"], _choice_confidence(list(team["probabilities"].values()))
        )

        mood = body["answers"]["mood"]
        self.assertEqual(
            set(mood),
            {"type", "score", "confidence", "legend", "probabilities", "x_label_mass"},
        )
        self.assertEqual(mood["legend"], {"0": "Calm", "1": "Civil", "2": legend_level})
        self.assertEqual(list(mood["probabilities"]), ["0", "1", "2"])
        self.assertAlmostEqual(
            mood["score"], sum(int(k) * p for k, p in mood["probabilities"].items())
        )

    async def test_answers_equal_v1_decisions_bit_for_bit(self):
        legend_level = {"label": "Angry"}
        systemone = _systemone_request(
            {
                "urgent": {"type": "noul", "instructions": "Needs an answer today"},
                "team": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": None, "technical": "Bugs", "sales": None},
                },
                "mood": {
                    "type": "score",
                    "instructions": "Mood?",
                    "criteria": ["Calm", "Civil", legend_level],
                },
            }
        )
        decisions = DecisionRequest(
            input=systemone.state,
            questions=[
                {"id": "urgent", "type": "yes_no", "question": "Needs an answer today"},
                _question(
                    "choice",
                    {"billing": None, "technical": "Bugs", "sales": None},
                    "Which team?",
                )
                | {"id": "team"},
                _question("score", ["Calm", "Civil", legend_level], "Mood?")
                | {"id": "mood"},
            ],
        )
        managers = [ScoringManager(self.tokenizer) for _ in range(2)]
        bodies = []
        for manager, handler, request in (
            (managers[0], self._serving(managers[0]), systemone),
            (managers[1], _handler(managers[1]), decisions),
        ):
            response = await handler.handle_request(request, None)
            self.assertEqual(response.status_code, 200)
            bodies.append(json.loads(response.body)["answers"])
        self.assertEqual(
            managers[0].requests[0].input_ids, managers[1].requests[0].input_ids
        )
        self.assertEqual(
            managers[0].requests[0].token_ids_logprob,
            managers[1].requests[0].token_ids_logprob,
        )
        systemone_answers, decision_answers = bodies
        for question_id, answer in systemone_answers.items():
            with self.subTest(question_id):
                expected = decision_answers[question_id]
                self.assertEqual(answer["x_label_mass"], expected["label_mass"])
                if question_id == "urgent":
                    self.assertEqual(answer["noul"], expected["probabilities"]["yes"])
                    continue
                self.assertEqual(answer["probabilities"], expected["probabilities"])
                key = "choice" if question_id == "team" else "score"
                self.assertEqual(answer[key], expected[key])

    def test_answers_report_probabilities_as_scored(self):
        # A softmax can miss a sum of 1 by a few ulp. Only confidence normalizes.
        questions = _systemone_request(
            {
                "urgent": {"type": "noul", "instructions": "x"},
                "team": {
                    "type": "choice",
                    "criteria": {"a": None, "b": None, "c": None},
                },
                "mood": {"type": "score", "criteria": ["low", "mid", "high"]},
            }
        ).questions
        answers = {
            question_id: _answer(
                view=_view(question),
                probabilities=[0.3] * (2 if question_id == "urgent" else 3),
                mass=0.9,
                question_id=question_id,
            )
            for question_id, question in questions.items()
        }
        self.assertEqual(answers["urgent"].noul, 0.3)
        for question_id in ("team", "mood"):
            self.assertEqual(
                list(answers[question_id].probabilities.values()), [0.3] * 3
            )
        self.assertEqual(answers["mood"].score, math.fsum([0, 0.3, 0.6]))
        self.assertAlmostEqual(answers["team"].confidence, 0.0)
        self.assertAlmostEqual(
            answers["mood"].confidence, _score_confidence([1 / 3] * 3)
        )

    def test_confidence_matches_the_published_formulas(self):
        # Documented examples, a uniform choice, a split score, and one candidate.
        self.assertAlmostEqual(_score_confidence([0, 0.14, 0.86, 0, 0]), 1 - 0.14 / 1.2)
        self.assertAlmostEqual(_score_confidence([0, 0, 0.48, 0.52]), 0.52)
        self.assertEqual(_score_confidence([0.5, 0, 0, 0, 0.5]), 0.0)
        self.assertAlmostEqual(_choice_confidence([0.25] * 4), 0.0)
        self.assertAlmostEqual(_choice_confidence([0.1, 0.9]), 0.8)
        self.assertEqual(_choice_confidence([1.0]), 1.0)
        self.assertEqual(_score_confidence([1.0]), 1.0)

    async def test_options_beyond_26_get_two_letter_labels(self):
        manager = ScoringManager(self.tokenizer)
        criteria = {f"option {i}": None for i in range(30)}
        request = _systemone_request(
            {"q": {"type": "choice", "instructions": "Pick one", "criteria": criteria}}
        )
        encoded, _ = self._serving(manager)._convert_to_internal_request(request)
        prompts, label_ids = zip(*encoded)
        text = self.tokenizer.decode(prompts[0])
        self.assertIn("AA: option 0", text)
        self.assertNotIn("\nA: option 0", text)
        self.assertEqual(len(set(label_ids[0])), 30)
        # A tokenizer that reports no added tokens stays at 26 options.
        plain = self._serving(ScoringManager(PlainTokenizer(self.tokenizer)))
        response = await plain.handle_request(request, None)
        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "more than 26 options needs added tokens the server can read",
            json.loads(response.body)["message"],
        )
        # So does a template whose only added token comes before the message.
        original_template = self.tokenizer.chat_template
        self.addCleanup(setattr, self.tokenizer, "chat_template", original_template)
        self.tokenizer.chat_template = (
            "<|im_start|>{{ messages[0]['content'] }}\nassistant\n\n"
        )
        leading = self._serving()
        response = await leading.handle_request(request, None)
        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "more than 26 options needs an added token between the message",
            json.loads(response.body)["message"],
        )
        self.assertEqual(leading.tokenizer_manager.requests, [])
        few = _systemone_request(
            {"q": {"type": "choice", "criteria": {"a": None, "b": None}}}
        )
        response = await leading.handle_request(few, None)
        self.assertEqual(response.status_code, 200)

    def test_request_validation(self):
        levels = [str(i) for i in range(11)]
        refused = {
            "noul that asks nothing": {"q": {"type": "noul"}},
            "noul with blank criteria": {
                "q": {"type": "noul", "instructions": None, "criteria": {"true": " "}}
            },
            "unknown key in a question": {
                "q": {"type": "choice", "instruction": "x", "criteria": {"a": None}}
            },
            "unknown key in noul criteria": {
                "q": {"type": "noul", "criteria": {"yes": "x"}}
            },
            "no options": {"q": {"type": "choice", "criteria": {}}},
            "256 options": {
                "q": {"type": "choice", "criteria": {str(i): None for i in range(256)}}
            },
            "11 levels": {"q": {"type": "score", "criteria": levels}},
            "null level": {"q": {"type": "score", "criteria": ["low", None]}},
            "blank level": {"q": {"type": "score", "criteria": ["low", " "]}},
            "repeated option name": {
                "q": {"type": "choice", "criteria": {"a": None, "A ": None}}
            },
            "unknown type": {"q": {"type": "yes_no", "instructions": "x"}},
            "no questions": {},
        }
        for case, questions in refused.items():
            with self.subTest(case), self.assertRaises(ValidationError):
                _systemone_request(questions)
        for field, value in (
            ("temperature", 0.5),
            ("temperature", 1),
            ("prompt_format_version", 1),
            ("return_prompt_token_ids", True),
            ("return_prompt_token_ids", False),
        ):
            with (
                self.subTest(field=field, value=value),
                self.assertRaisesRegex(ValidationError, "/v1/decisions") as caught,
            ):
                _systemone_request(
                    {"q": {"type": "noul", "instructions": "x"}}, **{field: value}
                )
            self.assertEqual(caught.exception.errors()[0]["loc"], (field,))
        self.assertNotIn(
            "temperature", SystemOneRequest.model_json_schema()["properties"]
        )
        accepted = SystemOneRequest(
            state="",
            model="jev-latest",
            questions={
                "": {"type": "choice", "criteria": {"only": None}},
                " ": {"type": "score", "criteria": ["one"]},
            },
            temperature=None,
            future_field=True,
        )
        self.assertEqual(list(accepted.questions), ["", " "])

    async def test_validation_errors_are_422_on_this_route_only(self):
        from fastapi.exceptions import RequestValidationError

        from sglang.srt.entrypoints.http_server import validation_exception_handler

        # The input echo is left out, since JSON encoding can fail on it.
        nested = []
        for _ in range(300):
            nested = [nested]
        error = RequestValidationError(
            errors=[
                {
                    "loc": ("body", "questions"),
                    "msg": "Field required",
                    "type": "missing",
                    "input": {"state": nested},
                }
            ]
        )

        def request(root_path, path):
            # Behind --fastapi-root-path the URL path starts with the root path.
            return SimpleNamespace(
                url=SimpleNamespace(path=root_path + path),
                scope={"root_path": root_path},
            )

        for root_path in ["", "/prefix"]:
            with self.subTest(root_path=root_path):
                systemone = await validation_exception_handler(
                    request(root_path, "/v1/systemone"), error
                )
                self.assertEqual(systemone.status_code, 422)
                self.assertEqual(
                    json.loads(systemone.body)["detail"],
                    [
                        {
                            "loc": ["body", "questions"],
                            "msg": "Field required",
                            "type": "missing",
                        }
                    ],
                )
                decisions = await validation_exception_handler(
                    request(root_path, "/v1/decisions"), error
                )
                self.assertEqual(decisions.status_code, 400)

    async def test_levels_the_legend_cannot_return_are_refused_before_scoring(self):
        def nested(depth):
            level = "Calm"
            for _ in range(depth):
                level = [level]
            return level

        for case, level, status in (
            ("nested 200 deep", nested(200), 200),
            ("nested 300 deep", nested(300), 400),
        ):
            with self.subTest(case):
                manager = ScoringManager(self.tokenizer)
                request = _systemone_request(
                    {"mood": {"type": "score", "criteria": ["Fine", level]}}
                )
                response = await self._serving(manager).handle_request(request, None)
                self.assertEqual(response.status_code, status)
                if status == 400:
                    self.assertIn(
                        "question 'mood': a level cannot be returned in the legend",
                        json.loads(response.body)["message"],
                    )
                    self.assertEqual(manager.requests, [])

    async def test_reasoning_refusals_apply(self):
        request = _systemone_request(
            {"q": {"type": "noul", "instructions": "x"}},
            chat_template_kwargs={"enable_thinking": True},
        )
        plain = _systemone_request({"q": {"type": "noul", "instructions": "x"}})
        always = ReasoningToggleConfig(special_case="always")
        for message, serving, body in (
            ("sets 'enable_thinking' to True", self._serving(), request),
            ("always reason", self._serving(reasoning_config=always), plain),
        ):
            with self.subTest(message):
                response = await serving.handle_request(body, None)
                self.assertEqual(response.status_code, 400)
                self.assertIn(message, json.loads(response.body)["message"])
                self.assertEqual(serving.tokenizer_manager.requests, [])
        # The server checks of /v1/decisions apply too.
        serving = self._serving(
            ScoringManager(self.tokenizer, dllm_algorithm="LowConfidence")
        )
        response = await serving.handle_request(plain, None)
        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "/v1/systemone does not support diffusion",
            json.loads(response.body)["message"],
        )

    async def test_non_finite_scores_are_a_server_error(self):
        manager = ScoringManager(self.tokenizer)
        manager.logprobs[self.tokenizer.convert_tokens_to_ids(["yes", "no"])] = float(
            "nan"
        )
        request = _systemone_request({"q": {"type": "noul", "instructions": "x"}})
        response = await self._serving(manager).handle_request(request, None)
        self.assertEqual(response.status_code, 500)


if __name__ == "__main__":
    unittest.main()
