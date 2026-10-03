import base64
import io
import math
import struct
import unittest
import zlib
from types import SimpleNamespace
from unittest import mock

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from PIL import Image
from transformers import AutoTokenizer

from sglang.srt.entrypoints.decision.families.intern import (
    ANSWER_SYMBOLS,
    InternDecisionFamily,
    compile_decision,
)
from sglang.srt.entrypoints.decision.request_id import (
    TypesafeRequestIdMiddleware,
    install_typesafe_request_id,
)
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.managers.tokenizer_manager import resolve_readout_anchor
from sglang.srt.managers.tokenizer_manager_score_mixin import TokenizerManagerScoreMixin
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.srt.parser.template_detection import detect_reasoning_pattern
from sglang.srt.runtime_context import (
    get_schedule,
    publish,
    restore_context,
    snapshot_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.auth import add_api_key_middleware
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

TOKENIZER = "internlm/Intern-Decision-0.8B"
# A chat model outside every decision model family.
GENERIC_TOKENIZER = "Qwen/Qwen3.5-35B-A3B"
ROUTES = ("/v1/decisions", "/v1/jev", "/v1/systemone")
NOUL = {"u": {"type": "noul"}}
IMAGE_TOKENS = 16

QUESTIONS = {
    "zeta": {"type": "noul", "instructions": "Urgent?"},
    "alpha": {"type": "noul", "criteria": {"TRUE": "Owed", "0": "Not owed"}},
    "team": {"type": "choice", "criteria": {"sales": "Pricing", "billing": None}},
    "mood": {"type": "score", "criteria": ["Calm", "Angry"]},
    "size": {"type": "score", "criteria": {"1": "Small", "2.5": "Large"}},
}
STATE = {"ticket": "Refund ünpaid", "tags": ["a", 1]}

# Rendered by compile_row of internlm/Intern-Decision for STATE and QUESTIONS.
OFFICIAL_MESSAGES = [
    "You are a careful decision assistant. Use the state and decision schema in "
    "the user message to make the requested decisions. For every field, choose "
    "exactly one answer symbol (e.g. A, B, C, ...) from its listed options and "
    "return one valid JSON object mapping each field name to its chosen symbol. "
    "Use the field names and symbols exactly as given. Do not include "
    "explanations, Markdown, or extra text.",
    "Return one answer for every field using the supplied answer symbols.\n\n"
    '## State\n{\n  "ticket": "Refund ünpaid",\n  "tags": [\n    "a",\n    1\n  ]\n}\n'
    "## Decision schema\n"
    "zeta: Urgent?\n"
    "    A = no: The answer is no (negative, or disagree with the claim).\n"
    "    B = yes: The answer is yes (affirmative, or align with the claim).\n"
    "alpha: \n    A = no: Not owed\n    B = yes: Owed\n"
    "team: \n    A = sales: Pricing\n    B = billing: None\n"
    "mood: \n    A = 0: Calm\n    B = 1: Angry\n"
    "size: \n    A = 1: Small\n    B = 2.5: Large",
    '{\n    "zeta": "<decision>",\n    "alpha": "<decision>",\n'
    '    "team": "<decision>",\n    "mood": "<decision>",\n    "size": "<decision>"\n}',
]
OFFICIAL_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


class ReadoutManager(TokenizerManagerScoreMixin):
    """Replace only model execution with fixed logprobs per readout position."""

    def __init__(self, tokenizer, rows, **server_args):
        self.server_args = ServerArgs(model_path="dummy", **server_args)
        publish(self.server_args, role="test")
        self.tokenizer = tokenizer
        self.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Qwen3_5ForConditionalGeneration"],
                model_type="qwen3_5",
            )
        )
        self.is_generation = True
        self.context_len = 8192
        self.num_reserved_tokens = 0
        self.request_logger = SimpleNamespace(log_requests=False)
        self.served_model_name = "served-model"
        self.rows = rows
        self.requests = []

    def config_value(self, name):
        return None

    async def generate_request(self, request, raw_request):
        if request.readout_anchor is None:
            self.requests.append((request, None))
            yield list(self._generic_results(request))
            return
        ids = request.input_ids
        if request.image_data:
            # Expand each image placeholder as the Qwen-VL processor does.
            pad = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            ids = []
            for token in self.tokenizer.encode(request.text, add_special_tokens=False):
                ids += [token] * (IMAGE_TOKENS if token == pad else 1)
        request.token_indices_to_pool = resolve_readout_anchor(
            input_ids=ids,
            anchor=request.readout_anchor,
            chunked_prefill_size=get_schedule().chunked_prefill_size,
        )
        self.requests.append((request, ids))
        labels = request.token_ids_logprob
        readouts = [[(lp, i, None) for lp, i in zip(row, labels)] for row in self.rows]
        meta = {"input_token_ids_logprobs": readouts, "prompt_tokens": len(ids)}
        yield {"meta_info": meta}

    @staticmethod
    def _generic_results(request):
        request.normalize_batch_and_arguments()
        for ids, labels in zip(request.input_ids, request.token_ids_logprob):
            logprobs = [(math.log(0.1), token, None) for token in labels]
            meta = {"prompt_tokens": len(ids), "output_token_ids_logprobs": [logprobs]}
            yield {"meta_info": meta}


def _chat_serving(manager):
    _, reasoning_config = detect_reasoning_pattern(manager.tokenizer.chat_template)
    template_manager = SimpleNamespace(
        chat_template_name=None,
        reasoning_config=reasoning_config,
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
    return chat_serving


def _logs(*ps):
    return [math.log(p) for p in ps]


def _encoded(image, image_format="PNG", **save):
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **save)
    return buffer.getvalue()


def _png(color, size=(8, 8)):
    return base64.b64encode(_encoded(Image.new("RGB", size, color))).decode()


def _b64(data):
    return base64.b64encode(data).decode()


def _png_header(width, height):
    def chunk(kind, data):
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\0"))
        + chunk(b"IEND", b"")
    )


class TestDecisionModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    def setUp(self):
        self.addCleanup(restore_context, snapshot_context())

    def _client(
        self, rows=(_logs(0.2, 0.6),), tokenizer=None, api_key=None, **server_args
    ):
        from sglang.srt.entrypoints import http_server as server

        app = FastAPI()
        routes = server.app.router.routes
        app.router.routes.extend(
            r for r in routes if isinstance(r, APIRoute) and r.path in ROUTES
        )
        # The server's middleware, with auth added at launch as the server does.
        app.user_middleware = list(server.app.user_middleware)
        install_typesafe_request_id(app)
        if api_key is not None:
            add_api_key_middleware(app, api_key=api_key, admin_api_key=None)
        app.add_exception_handler(
            RequestValidationError, server.validation_exception_handler
        )
        manager = ReadoutManager(tokenizer or self.tokenizer, rows, **server_args)
        (
            app.state.openai_serving_decisions,
            app.state.systemone_serving,
        ) = server.decision_route_servings(_chat_serving(manager))
        return TestClient(app), manager

    def test_prompt_matches_the_official_compiler(self):
        compiled = compile_decision(STATE, QUESTIONS)
        contents = [message["content"] for message in compiled.messages]
        self.assertEqual(contents, OFFICIAL_MESSAGES)
        self.assertEqual(ANSWER_SYMBOLS, OFFICIAL_SYMBOLS)

    def test_invalid_requests_are_422_at_the_offending_field(self):
        client, _ = self._client()
        cases = [
            ({f"f{i}": NOUL["u"] for i in range(17)}, ["body", "questions"]),
            ({"f": {"type": "score", "criteria": {"x": ""}}}, ["body", "questions"]),
        ]
        choices = {f"o{i}": "" for i in range(63)}
        family_errors = [
            (
                {
                    "state": {},
                    "questions": {"f": {"type": "choice", "criteria": choices}},
                },
                ["body", "questions", "f", "criteria"],
                "62 options",
            ),
            (
                {"state": "a <decision>", "questions": NOUL},
                ["body", "state"],
                "reserved decision marker",
            ),
            (
                {"state": {}, "questions": {"x<decision>": NOUL["u"]}},
                ["body", "questions", "x<decision>"],
                "reserved decision marker",
            ),
            (
                {
                    "state": {},
                    "questions": {"u": {"type": "noul", "instructions": "<decision>"}},
                },
                ["body", "questions", "u", "instructions"],
                "reserved decision marker",
            ),
            (
                {
                    "state": {},
                    "questions": {
                        "u": {"type": "choice", "criteria": {"a": "<decision>"}}
                    },
                },
                ["body", "questions", "u", "criteria"],
                "reserved decision marker",
            ),
        ]
        for route in ROUTES:
            for questions, loc in cases:
                with self.subTest(route=route, loc=loc):
                    body = {"state": {}, "questions": questions}
                    response = client.post(route, json=body)
                    self.assertEqual(response.status_code, 422, response.text)
                    detail = response.json()["detail"][0]
                    self.assertEqual(detail["loc"][: len(loc)], loc)
                    self.assertIn("x-typesafe-request-id", response.headers)
        for body, loc, message in family_errors:
            with self.subTest(family_loc=loc):
                response = client.post("/v1/jev", json=body)
                self.assertEqual(response.status_code, 422, response.text)
                detail = response.json()["detail"][0]
                self.assertIn(message, detail["msg"])
                self.assertEqual(detail["loc"], loc)

    def test_routes_dispatch_by_body_shape_and_checkpoint(self):
        client, _ = self._client()
        body = {"state": {}, "questions": NOUL, "model": "jev-latest"}
        for route in ROUTES:
            with self.subTest(route=route):
                sent = {"x-typesafe-request-id": "req-7"}
                response = client.post(route, json=body, headers=sent)
                self.assertAlmostEqual(response.json()["answers"]["u"]["noul"], 0.75)
                self.assertEqual(response.headers["x-typesafe-request-id"], "req-7")
                generated = client.post(route, json=body).headers
                self.assertEqual(len(generated["x-typesafe-request-id"]), 32)
        for model, status in [("served-model", 200), ("jev-preview", 200), ("x", 404)]:
            response = client.post("/v1/jev", json={**body, "model": model})
            self.assertEqual(response.status_code, status)
        question = {"id": "a", "type": "yes_no", "question": "q"}
        generic = {"input": "x", "questions": [question]}
        client, manager = self._client()
        response = client.post("/v1/decisions", json=generic)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["object"], "decisions")
        self.assertAlmostEqual(response.json()["answers"]["a"]["label_mass"], 0.2)
        self.assertEqual([ids for _, ids in manager.requests], [None])

        client, _ = self._client(
            tokenizer=AutoTokenizer.from_pretrained(GENERIC_TOKENIZER)
        )
        noul = {"u": {"type": "noul", "instructions": "q"}}
        systemone = {"state": "s", "model": "served-model", "questions": noul}
        response = client.post("/v1/systemone", json=systemone)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertAlmostEqual(response.json()["answers"]["u"]["x_label_mass"], 0.2)
        refused = client.post("/v1/jev", json=body)
        self.assertEqual(refused.status_code, 400)
        self.assertIn("Intern-Decision", refused.json()["message"])

    def test_readout_is_one_position_before_each_marker_in_field_order(self):
        client, manager = self._client(rows=[[0.0, 0.0]] * len(QUESTIONS))
        client.post("/v1/decisions", json={"state": STATE, "questions": QUESTIONS})
        ((request, ids),) = manager.requests
        marker = self.tokenizer.convert_tokens_to_ids("<decision>")
        markers = [i for i, token in enumerate(ids) if token == marker]
        self.assertEqual(request.token_indices_to_pool, [m - 1 for m in markers])
        keys = [self.tokenizer.decode(ids[:m]).rsplit("\n", 1)[-1] for m in markers]
        self.assertEqual(keys, [f'    "{name}": "' for name in QUESTIONS])
        symbols = self.tokenizer.convert_tokens_to_ids(["A", "B"])
        self.assertEqual(request.token_ids_logprob, symbols)
        self.assertEqual(request.sampling_params, {"max_new_tokens": 0})

    def test_image_readout_resolves_on_the_expanded_prompt(self):
        client, manager = self._client(rows=[[0.0, 0.0]] * 2)
        images = [_png("red"), "data:image/png;base64," + _png("blue")]
        body = {"state": {}, "questions": {"a": NOUL["u"], "b": NOUL["u"]}}
        response = client.post("/v1/jev", json={**body, "images": images})
        self.assertEqual(response.status_code, 200, response.text)
        ((request, ids),) = manager.requests
        self.assertIsNone(request.input_ids)
        self.assertEqual(len(request.image_data), 2)
        # Both images come, numbered, before the user text, as the official layout.
        user = request.text.split("<|im_start|>user\n", 1)[1]
        self.assertRegex(user, r"^Picture 1: <\|vision_start\|><\|image_pad\|>")
        self.assertIn("Picture 2: ", user.split("Return one answer")[0])
        marker = self.tokenizer.convert_tokens_to_ids("<decision>")
        markers = [i for i, token in enumerate(ids) if token == marker]
        self.assertEqual(request.token_indices_to_pool, [m - 1 for m in markers])
        text_only = self.tokenizer.encode(request.text, add_special_tokens=False)
        shift = 2 * (IMAGE_TOKENS - 1)
        self.assertEqual(markers[0] - text_only.index(marker), shift)
        self.assertEqual(response.json()["usage"]["input_tokens"], len(ids))

    def test_images_reach_the_loader_upright_in_rgb(self):
        # Orientation 6: stored 4x2, shown 2x4; the official service decodes it upright.
        exif = Image.Exif()
        exif[274] = 6
        rotated = _encoded(Image.new("RGBA", (4, 2), "red"), exif=exif)
        images = [
            {"type": "image/png", "data": _b64(rotated)},
            {"type": "ignored", "data": "data:image/png;base64," + _png("blue")},
            _png("green"),
        ]
        client, manager = self._client(rows=[[0.0, 0.0]])
        body = {"state": {}, "questions": NOUL, "images": images}
        response = client.post("/v1/jev", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        ((request, _),) = manager.requests
        forwarded = [
            Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
            for url in request.image_data
        ]
        self.assertEqual({(i.format, i.mode) for i in forwarded}, {("PNG", "RGB")})
        loaded = [
            QwenVLImageProcessor._load_single_item(url, Modality.IMAGE)
            for url in request.image_data
        ]
        self.assertEqual([(i.size, i.mode) for i in loaded][0], ((2, 4), "RGB"))
        self.assertEqual(
            [i.getpixel((0, 0)) for i in loaded[1:]], [(0, 0, 255), (0, 128, 0)]
        )

    def test_invalid_images_are_422(self):
        client, _ = self._client()
        request = {"state": {}, "questions": NOUL}
        frames = [Image.new("RGB", (2, 2), color) for color in ("red", "blue")]
        animated = _encoded(frames[0], "GIF", save_all=True, append_images=frames[1:])
        static_gif = _encoded(frames[0], "GIF")
        # A static GIF whose trailer is replaced by a truncated next block; counting
        # frames trips Pillow's parser with struct.error or IndexError.
        truncated_blocks = [
            b",\0",
            b",\0\0\0\0\x02\0\x02\0",
            b",\0\0\0\0\x02\0\x02\0\0",
            b"!",
            b"!\xf9\x01\x01\0",
        ]
        small = _encoded(Image.new("RGB", (8, 8)))
        padded = small + b"\0" * (11 * 1024 * 1024)
        over_limit = small + b"\0" * (12 * 1024 * 1024 + 1 - len(small))
        png = {"type": "image/png", "data": _png("red")}
        cases = [
            ([_png("red")] * 9, ["body", "images"]),
            (["data:text/plain;base64," + _png("red")], ["body", "images", 0]),
            ([_b64(b"not an image")], ["body", "images", 0]),
            # Lax base64 would drop the "!" and decode a valid PNG.
            ([_png("red")[:8] + "!" + _png("red")[8:]], ["body", "images", 0]),
            (["/etc/passwd"], ["body", "images", 0]),
            (
                [{"type": "image/png", "data": "https://example.com/a.png"}],
                ["body", "images", 0],
            ),
            ([{"data": _png("red")}], ["body", "images", 0]),
            ([{"type": "image/jpeg", "data": _png("red")}], ["body", "images", 0]),
            (
                [png, {"type": "image/gif", "data": _b64(animated)}],
                ["body", "images", 1],
            ),
            *(
                (
                    [png, {"type": "image/gif", "data": _b64(static_gif[:-1] + tail)}],
                    ["body", "images", 1],
                )
                for tail in truncated_blocks
            ),
            ([_b64(_png_header(20000, 20000))], ["body", "images", 0]),
            ([_png("red", size=(5000, 4000))], ["body", "images", 0]),
            ([_b64(over_limit)], ["body", "images", 0]),
            ([_b64(padded)] * 3, ["body", "images"]),
        ]
        for images, loc in cases:
            with self.subTest(loc=loc, last=str(images[-1])[-40:]):
                response = client.post("/v1/jev", json={**request, "images": images})
                self.assertEqual(response.status_code, 422, response.text)
                self.assertEqual(response.json()["detail"][0]["loc"], loc)
        # A literal image placeholder would take an attached image's slot.
        conflict = {
            "state": {},
            "questions": {"u": {"type": "noul", "instructions": "see <image>"}},
        }
        response = client.post("/v1/jev", json={**conflict, "images": [_png("red")]})
        self.assertEqual(response.status_code, 422)
        detail = response.json()["detail"][0]
        self.assertIn("image placeholder", detail["msg"])
        self.assertEqual(detail["loc"], ["body", "questions", "u", "instructions"])
        self.assertEqual(client.post("/v1/jev", json=conflict).status_code, 200)
        gif = {"type": "image/gif", "data": _b64(static_gif)}
        response = client.post("/v1/jev", json={**request, "images": [gif]})
        self.assertEqual(response.status_code, 200, response.text)

    def test_answers_follow_the_typesafe_shapes(self):
        questions = {
            "route": {"type": "choice", "criteria": {"s": "", "b": "", "l": ""}},
            "urgent": {"type": "noul"},
            "mood": {"type": "score", "criteria": ["Calm", "Tense", "Angry"]},
            "size": {"type": "score", "criteria": {"2.5": "", "1": ""}},
        }
        # Logprobs over the A, B, C union; two-option fields ignore C.
        rows = [_logs(0.2, 0.5, 0.3), _logs(0.2, 0.6, 1), _logs(0.1, 0.2, 0.7)]
        client, _ = self._client(rows=[*rows, _logs(0.5, 0.5, 1)])
        body = {"state": {}, "questions": questions}
        response = client.post("/v1/decisions", json=body).json()
        self.assertNotIn("calibration", response)
        route, urgent, mood, size = (response["answers"][name] for name in questions)
        self.assertEqual(list(route["probabilities"]), ["s", "b", "l"])
        self.assertEqual((route["decision"], route["choice"]), ("b", "b"))
        # TypeSafe choice confidence, (K * p_max - 1) / (K - 1).
        self.assertAlmostEqual(route["confidence"], (3 * 0.5 - 1) / 2)
        self.assertEqual(list(urgent["probabilities"]), ["no", "yes"])
        self.assertAlmostEqual(urgent["noul"], 0.75)
        self.assertAlmostEqual(urgent["confidence"], 0.75)
        self.assertAlmostEqual(mood["score"], 0.2 + 2 * 0.7)
        self.assertEqual(mood["legend"], {"0": "Calm", "1": "Tense", "2": "Angry"})
        # TypeSafe score confidence: 1 - E|level - mode| / (uniform spread 2/3).
        spread = 0.1 * 2 + 0.2 * 1
        self.assertAlmostEqual(mood["confidence"], 1 - spread / (2 / 3))
        # A tie goes to the smaller label, and the score is the expected key value.
        self.assertEqual((size["decision"], size["score"]), ("1", 1.75))

        body = client.post("/v1/jev", json={**body, "temperature": 2.0}).json()
        # softmax(log p / T), as the official temperature.py; argmax is unchanged.
        weights = [p**0.5 for p in (0.2, 0.5, 0.3)]
        calibrated = body["answers"]["route"]["probabilities"].values()
        for p, weight in zip(calibrated, weights):
            self.assertAlmostEqual(p, weight / sum(weights))
        self.assertEqual(body["answers"]["route"]["decision"], "b")
        self.assertEqual(body["calibration"]["temperature"], 2.0)

    def test_temperature_never_changes_the_decision(self):
        client, _ = self._client(rows=[_logs(0.25, 0.75)])
        body = {"state": {}, "questions": NOUL}
        calibrated = client.post("/v1/jev", json={**body, "temperature": 1e3}).json()
        self.assertEqual(calibrated["answers"]["u"]["decision"], "yes")
        # Both weights round to 1, which would tie and hand the decision to "no".
        response = client.post("/v1/jev", json={**body, "temperature": 1e20})
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(response.json()["detail"][0]["loc"], ["body", "temperature"])

    def test_systemone_openapi_documents_both_bodies(self):
        client, _ = self._client()
        body = client.app.openapi()["paths"]["/v1/systemone"]["post"]["requestBody"]
        self.assertTrue(body["required"])
        schema = body["content"]["application/json"]["schema"]
        refs = {option["$ref"].rsplit("/", 1)[-1] for option in schema["anyOf"]}
        self.assertEqual(refs, {"SystemOneRequest", "JevRequest"})
        self.assertIn("decision model", schema["description"])

    def test_request_id_tags_auth_rejections_and_server_errors(self):
        from sglang.srt.entrypoints import http_server as server

        stack = server.app.build_middleware_stack()
        self.assertIsInstance(stack, TypesafeRequestIdMiddleware)
        client, _ = self._client(api_key="secret")
        body = {"state": {}, "questions": NOUL}
        for route in ROUTES:
            with self.subTest(route=route):
                sent = {"x-typesafe-request-id": "req-9"}
                response = client.post(route, json=body, headers=sent)
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.headers["x-typesafe-request-id"], "req-9")
        authorized = {"Authorization": "Bearer secret"}
        response = client.post("/v1/jev", json=body, headers=authorized)
        self.assertEqual(response.status_code, 200)
        unhandled = TestClient(client.app, raise_server_exceptions=False)
        boom = mock.patch.object(
            InternDecisionFamily, "validate", side_effect=RuntimeError("boom")
        )
        with boom:
            response = unhandled.post("/v1/jev", json=body, headers=authorized)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(len(response.headers["x-typesafe-request-id"]), 32)

    def test_refusals(self):
        request = {"state": {}, "questions": NOUL}
        long_state = {"state": {"text": "word " * 400}}
        cases = [
            ({}, {"chat_template": "chatml"}, 400),
            ({}, {"enable_mis": True}, 400),
            (long_state, {"chunked_prefill_size": 256}, 400),
            ({"thinking": {"enabled": True}}, {}, 422),
        ]
        for extra, server_args, status in cases:
            with self.subTest(extra=list(extra), server_args=server_args):
                client, _ = self._client(**server_args)
                response = client.post("/v1/jev", json={**request, **extra})
                self.assertEqual(response.status_code, status)


if __name__ == "__main__":
    unittest.main()
