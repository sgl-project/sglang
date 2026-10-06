"""Unit tests for /v1/systemone on Clef checkpoints: the joint schema prompt and its
decision layout, and answers built from the joint schema head's option logits."""

import json
import math
import unittest
from types import SimpleNamespace

from transformers import AutoTokenizer

from sglang.srt.entrypoints.openai.protocol import DecisionRequest
from sglang.srt.entrypoints.openai.serving_decisions import OpenAIServingDecisions
from sglang.srt.entrypoints.systemone.joint_schema import encode_joint_schema
from sglang.srt.entrypoints.systemone.protocol import SystemOneRequest
from sglang.srt.entrypoints.systemone.serving import SystemOneServing
from sglang.srt.layers.joint_schema_head import parse_decision_layout
from sglang.srt.runtime_context import publish, restore_context, snapshot_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Tokenizer files only, with the Qwen3.5 vocabulary of Clef checkpoints.
TOKENIZER = "Qwen/Qwen3.5-35B-A3B"

REQUEST = {
    "model": "clef",
    "state": {"ticket": "Refund for order 1182", "amount": 42},
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team handles it?",
            "criteria": {
                "support": "Customer support",
                "billing": "Payments and refunds",
                "legal": None,
            },
        },
        "urgent": {
            "type": "noul",
            "instructions": "Is it urgent?",
            "criteria": {"false": None},
        },
        "severity": {"type": "score", "criteria": ["none", "low", "high"]},
    },
}


class HeadManager:
    """Replace model execution with option logits 0, 1, 2, ... per question, in head order."""

    def __init__(self, tokenizer):
        self.server_args = ServerArgs(model_path="dummy")
        publish(self.server_args, role="test")
        self.tokenizer = tokenizer
        self.model_config = SimpleNamespace(
            is_multimodal=True,
            hf_config=SimpleNamespace(
                architectures=["Qwen3_5ForConditionalGeneration"], model_type="qwen3_5"
            ),
            decision_config=None,
            joint_head_config={"hidden_size": 4096},
        )
        self.is_generation = False
        self.allow_auto_truncate = False
        self.context_len = 4096
        self.num_reserved_tokens = 0
        self.request_logger = SimpleNamespace(log_requests=False)
        self.served_model_name = "served-model"
        self.requests = []

    def config_value(self, name):
        return None

    async def generate_request(self, request, raw_request):
        self.requests.append(request)
        questions = parse_decision_layout(
            request.decision_layout, len(request.input_ids)
        )
        logits = [float(i) for q in questions for i in range(len(q.option_spans))]
        yield {
            "embedding": logits,
            "meta_info": {"prompt_tokens": len(request.input_ids)},
        }


def _handler(manager, serving_class=SystemOneServing):
    template_manager = SimpleNamespace(
        chat_template_name=None,
        jinja_template_content_format="openai",
        reasoning_config=None,
        suggested_reasoning_parser=None,
    )
    chat_serving = SimpleNamespace(
        tokenizer_manager=manager,
        template_manager=template_manager,
        default_chat_template_kwargs={},
        chat_encoding_spec=None,
        _prompt_text_round_trip_is_lossy=False,
        reasoning_parser=None,
    )
    return serving_class(chat_serving)


def _softmax(values):
    weights = [math.exp(value - max(values)) for value in values]
    return [weight / sum(weights) for weight in weights]


class TestJointSchemaPrompt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    def _encode(self, request=REQUEST, max_length=4096):
        input_ids, layout = encode_joint_schema(
            self.tokenizer,
            SystemOneRequest(**request),
            max_length=max_length,
            image_token_counts=[],
        )
        return input_ids, parse_decision_layout(layout, len(input_ids))

    def _text(self, input_ids, span):
        return self.tokenizer.decode(input_ids[span[0] : span[1]])

    def test_prompt_is_the_trained_joint_schema_prompt(self):
        input_ids, _ = self._encode()
        prompt = self.tokenizer.decode(input_ids)
        self.assertTrue(
            prompt.startswith(
                "<|im_start|>system\nRead the complete state and schema. Decide every "
                "field jointly. Each answer must be exactly one of that field's allowed "
                "options.<|im_end|>\n<|im_start|>user\nSTATE:\n"
                '{"amount":42,"ticket":"Refund for order 1182"}\n\nSCHEMA FIELDS:\n'
                "\nFIELD 1\nID: team\nTYPE: choice\nINSTRUCTION: Which team handles it?"
                "\nALLOWED OPTIONS:\n"
                'OPTION 1: {"description":"Payments and refunds","option_id":"billing"}\n'
            )
        )
        self.assertTrue(
            prompt.endswith(
                "END FIELD\n\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                "JOINT SCHEMA DECISIONS:"
            )
        )

    def test_layout_spans_hold_questions_and_options_in_head_order(self):
        input_ids, questions = self._encode()
        self.assertEqual([q.question_type for q in questions], [1, 0, 2])
        self.assertEqual(
            [self._text(input_ids, q.question_span) for q in questions],
            ["Which team handles it?", "Is it urgent?", "severity"],
        )
        options = [
            [json.loads(self._text(input_ids, span)) for span in q.option_spans]
            for q in questions
        ]
        self.assertEqual(
            options[0],
            [
                {"description": "Payments and refunds", "option_id": "billing"},
                {"option_id": "legal"},
                {"description": "Customer support", "option_id": "support"},
            ],
        )
        self.assertEqual(
            options[1],
            [
                {
                    "description": "The proposition is true or the answer is yes.",
                    "option_id": "true",
                },
                {"option_id": "false"},
            ],
        )
        self.assertEqual(
            options[2],
            [
                {"description": level, "option_id": str(index)}
                for index, level in enumerate(["none", "low", "high"])
            ],
        )

    def test_state_is_truncated_to_fit(self):
        input_ids, questions = self._encode()
        shorter_ids, shorter = self._encode(max_length=len(input_ids) - 3)
        self.assertEqual(len(shorter_ids), len(input_ids) - 3)
        self.assertEqual(
            [self._text(shorter_ids, q.question_span) for q in shorter],
            [self._text(input_ids, q.question_span) for q in questions],
        )
        with self.assertRaisesRegex(ValueError, "before the state"):
            self._encode(max_length=100)

    def test_image_placeholders_precede_the_state(self):
        request = {**REQUEST, "images": ["data:image/png;base64,AAAA"]}
        input_ids, layout = encode_joint_schema(
            self.tokenizer,
            SystemOneRequest(**request),
            max_length=4096,
            image_token_counts=[100],
        )
        self.assertIn(
            "STATE:\n<|vision_start|><|image_pad|><|vision_end|>\n{",
            self.tokenizer.decode(input_ids),
        )
        unexpanded = parse_decision_layout(layout, len(input_ids))
        expanded = parse_decision_layout(layout, len(input_ids) + 99)
        self.assertEqual(
            [q.option_spans[0][0] + 99 for q in unexpanded],
            [q.option_spans[0][0] for q in expanded],
        )


class TestJointSchemaAnswers(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    def setUp(self):
        self.addCleanup(restore_context, snapshot_context())

    async def test_answers_map_head_order_back_to_the_request(self):
        manager = HeadManager(self.tokenizer)
        response = await _handler(manager).handle_request(
            SystemOneRequest(**REQUEST), None
        )
        self.assertEqual(response.status_code, 200)
        body = json.loads(response.body)
        input_ids, layout = encode_joint_schema(
            self.tokenizer,
            SystemOneRequest(**REQUEST),
            max_length=4095,
            image_token_counts=[],
        )
        (sent,) = manager.requests
        self.assertEqual((sent.input_ids, sent.decision_layout), (input_ids, layout))
        self.assertIsNone(sent.image_data)
        self.assertEqual(body["model"], "served-model")
        self.assertEqual(body["usage"]["input_tokens"], len(input_ids))

        three, two = _softmax([0.0, 1.0, 2.0]), _softmax([0.0, 1.0])
        team, urgent, severity = (body["answers"][q] for q in REQUEST["questions"])
        # The head scores billing, legal, support; answers keep the request order.
        self.assertEqual(team["choice"], "support")
        self.assertEqual(list(team["probabilities"]), ["support", "billing", "legal"])
        for name, expected in zip(["billing", "legal", "support"], three):
            self.assertAlmostEqual(team["probabilities"][name], expected)
        self.assertAlmostEqual(urgent["noul"], two[0])
        self.assertAlmostEqual(severity["score"], three[1] + 2 * three[2])
        self.assertEqual(severity["legend"], {"0": "none", "1": "low", "2": "high"})
        for answer in (team, urgent, severity):
            self.assertNotIn("x_label_mass", answer)

    async def test_refusals(self):
        manager = HeadManager(self.tokenizer)
        cases = {
            "takes no chat_template_kwargs": (
                SystemOneServing,
                SystemOneRequest(**REQUEST, chat_template_kwargs={"x": 1}),
            ),
            "LoRA adapter": (
                SystemOneServing,
                SystemOneRequest(**{**REQUEST, "model": "clef:a"}),
            ),
            "requires a generation model": (
                OpenAIServingDecisions,
                DecisionRequest(
                    model="m",
                    input="x",
                    questions=[{"id": "q", "type": "yes_no", "question": "Q?"}],
                ),
            ),
        }
        for message, (serving_class, request) in cases.items():
            with self.subTest(message):
                response = await _handler(manager, serving_class).handle_request(
                    request, None
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn(message, json.loads(response.body)["message"])
        self.assertEqual(manager.requests, [])


if __name__ == "__main__":
    unittest.main()
