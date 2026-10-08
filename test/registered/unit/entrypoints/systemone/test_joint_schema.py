"""Unit tests for /v1/systemone on Clef checkpoints: the joint schema prompt and its
decision layout, and answers built from the joint schema head's option logits."""

import asyncio
import base64
import json
import math
import random
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest import mock

import pybase64
from PIL import Image
from transformers import AutoTokenizer

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.openai.protocol import DecisionRequest
from sglang.srt.entrypoints.openai.serving_decisions import OpenAIServingDecisions
from sglang.srt.entrypoints.systemone.joint_schema import encode_joint_schema
from sglang.srt.entrypoints.systemone.protocol import SystemOneRequest
from sglang.srt.entrypoints.systemone.serving import SystemOneServing, _read_image
from sglang.srt.layers.joint_schema_head import (
    LayoutQuestion,
    pack_decision_layout,
    parse_decision_layout,
)
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import (
    get_context,
    get_schedule,
    publish,
    restore_context,
    snapshot_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import ImageData
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

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


class PatchCounter:
    """Stands in for the multimodal processor: one image token per 28x28 patch."""

    @staticmethod
    def resolve_image_token_counts(images):
        return [(image.height // 28) * (image.width // 28) for image in images]


def _png_bytes(width, height):
    buffer = BytesIO()
    Image.new("RGB", (width, height), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def _png(width, height):
    return (
        "data:image/png;base64," + base64.b64encode(_png_bytes(width, height)).decode()
    )


def _expanded_length(request):
    images = [
        Image.open(BytesIO(base64.b64decode(image.url.split(",", 1)[1])))
        for image in request.image_data or []
    ]
    counts = PatchCounter.resolve_image_token_counts(images)
    return len(request.input_ids) + sum(count - 1 for count in counts)


class HeadManager:
    """Replace model execution with option logits 0, 1, 2, ... per question, in head order,
    after expanding images and refusing prompts that the tokenizer manager, the
    scheduler, or the memory reserved for one prefill cannot take."""

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
        # min(context_len - 1, kv_capacity - 1) - 5, as the scheduler reports it.
        self.max_req_input_len = 4090
        self.mm_processor = PatchCounter()
        self.request_logger = SimpleNamespace(log_requests=False)
        self.served_model_name = "served-model"
        self.requests = []

    def config_value(self, name):
        return None

    async def generate_request(self, request, raw_request):
        self.requests.append(request)
        num_tokens = _expanded_length(request)
        if num_tokens + self.num_reserved_tokens >= self.context_len:
            raise ValueError(f"The input ({num_tokens} tokens) is too long")
        if num_tokens >= self.max_req_input_len:
            raise ValueError(f"The scheduler refuses a prompt of {num_tokens} tokens")
        if num_tokens > get_schedule().max_prefill_tokens:
            raise ValueError(f"A prefill of {num_tokens} tokens outgrows its memory")
        questions = parse_decision_layout(request.decision_layout, num_tokens)
        logits = [float(i) for q in questions for i in range(len(q.option_spans))]
        yield {
            "embedding": logits,
            "meta_info": {"prompt_tokens": num_tokens},
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
            max_length=4089,
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
            # Its 5329 image tokens leave no room for the questions.
            "before the state": (
                SystemOneServing,
                SystemOneRequest(**{**REQUEST, "images": [_png(2048, 2048)]}),
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


def _admission_manager(joint_head_config):
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.model_config = SimpleNamespace(joint_head_config=joint_head_config)
    manager.is_generation = joint_head_config is None
    manager.context_len = 4096
    manager.num_reserved_tokens = 0
    manager.max_req_input_len = 4090
    # A layout must be refused before auto-truncation could cut its prompt.
    manager.allow_auto_truncate = True
    manager.validate_total_tokens = False
    manager._validate_token_ids_logprob = mock.Mock()
    return manager


class TestJointSchemaAdmission(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def test_tokenizer_refuses_generation_and_prompts_past_one_prefill(self):
        clef = _admission_manager({"hidden_size": 4096})
        layout = [LayoutQuestion(0, (10, 12), ((13, 14), (15, 16)))]

        def embedding(length):
            return EmbeddingReqInput(
                input_ids=[1] * length,
                decision_layout=pack_decision_layout(length, layout),
                sampling_params={},
            )

        generation = GenerateReqInput(input_ids=[1, 2, 3], sampling_params={})
        refused = {
            "generation": (generation, "/v1/systemone"),
            # 4095 tokens pass the context check, but the scheduler takes 4089.
            "past one prefill": (embedding(4095), "at most 4089"),
        }
        for name, (obj, message) in refused.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                clef._validate_one_request(obj, obj.input_ids)
        clef._validate_one_request(embedding(4089), [1] * 4089)
        # A server without a joint schema head keeps generating.
        _admission_manager(None)._validate_one_request(generation, [1, 2, 3])

    def test_client_decision_layouts_are_refused_at_encode_and_classify(self):
        manager = SimpleNamespace(requests=[])

        async def generate_request(obj, request):
            manager.requests.append(obj)
            yield {"embedding": [0.0]}

        manager.generate_request = generate_request
        self.addCleanup(
            setattr, http_server, "_global_state", http_server.get_global_state()
        )
        http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
        layout = pack_decision_layout(16, [LayoutQuestion(0, (0, 1), ((1, 2),))])
        responses = [
            asyncio.run(
                handler(
                    EmbeddingReqInput(
                        input_ids=list(range(16)), decision_layout=layout
                    ),
                    None,
                )
            )
            for handler in (http_server.encode_request, http_server.classify_request)
        ]
        self.assertEqual(manager.requests, [])
        self.assertEqual([response.status_code for response in responses], [400, 400])

    def test_inline_images_are_sized_from_a_header_prefix(self):
        """Sizing an inline image must not decode its whole payload, and an image
        whose header runs past the decoded prefix must still be sized."""
        noise = BytesIO()
        pixels = random.Random(0).randbytes(256 * 192 * 3)
        Image.frombytes("RGB", (256, 192), pixels).save(noise, format="PNG")
        late = BytesIO()
        # A 60 KB EXIF segment puts the JPEG frame header far into the payload.
        exif = b"Exif\x00\x00" + bytes(60000)
        Image.new("RGB", (64, 48)).save(late, format="JPEG", exif=exif)
        cases = {
            "png": ("image/png", noise.getvalue(), (256, 192)),
            "late jpeg header": ("image/jpeg", late.getvalue(), (64, 48)),
        }
        for name, (mime, data, size) in cases.items():
            with self.subTest(name):
                payload = base64.b64encode(data).decode()
                image = ImageData(url=f"data:{mime};base64,{payload}")
                with mock.patch("pybase64.b64decode", wraps=pybase64.b64decode) as read:
                    forwarded, header = _read_image(0, image)
                self.assertIs(forwarded, image)
                self.assertEqual(header.size, size)
                decoded = [len(call.args[0]) for call in read.call_args_list]
                if name == "png":
                    self.assertLess(max(decoded), len(payload))
                else:
                    self.assertEqual(decoded[-1], len(payload))


if __name__ == "__main__":
    unittest.main()
