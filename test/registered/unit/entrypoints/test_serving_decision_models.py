"""Unit tests for decision model checkpoints on /v1/decisions, /v1/jev, and /v1/systemone."""

import base64
import io
import math
import unittest
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from PIL import Image
from transformers import AutoTokenizer

from sglang.srt.entrypoints.decision.families.intern import (
    ANSWER_SYMBOLS,
    compile_decision,
)
from sglang.srt.entrypoints.decision.serving import DecisionModelServing
from sglang.srt.managers.tokenizer_manager import resolve_readout_anchor
from sglang.srt.runtime_context import (
    get_schedule,
    publish,
    restore_context,
    snapshot_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

TOKENIZER = "internlm/Intern-Decision-0.8B"
ROUTES = ("/v1/decisions", "/v1/jev", "/v1/systemone")
NOUL = {"u": {"type": "noul"}}
# Image tokens one image expands to in the stand-in for the Qwen-VL processor.
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


def _manager(tokenizer, rows, **server_args):
    """A tokenizer manager whose model returns fixed logprobs per readout position."""
    server_args = ServerArgs(model_path="dummy", **server_args)
    publish(server_args, role="test")
    requests = []

    async def generate_request(request, raw_request):
        ids = request.input_ids
        if request.image_data:
            # Expand each image placeholder as the Qwen-VL processor does.
            pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            ids = []
            for token in tokenizer.encode(request.text, add_special_tokens=False):
                ids += [token] * (IMAGE_TOKENS if token == pad else 1)
        request.token_indices_to_pool = resolve_readout_anchor(
            input_ids=ids,
            anchor=request.readout_anchor,
            chunked_prefill_size=get_schedule().chunked_prefill_size,
        )
        requests.append((request, ids))
        labels = request.token_ids_logprob
        readouts = [[(lp, i, None) for lp, i in zip(row, labels)] for row in rows]
        meta = {"input_token_ids_logprobs": readouts, "prompt_tokens": len(ids)}
        yield {"meta_info": meta}

    return SimpleNamespace(
        server_args=server_args,
        tokenizer=tokenizer,
        is_generation=True,
        context_len=8192,
        num_reserved_tokens=0,
        request_logger=SimpleNamespace(log_requests=False),
        served_model_name="served-model",
        generate_request=generate_request,
        requests=requests,
    )


async def _handled_by(request, raw_request):
    return ORJSONResponse({"handled_by": type(request).__name__})


def _logs(*ps):
    return [math.log(p) for p in ps]


def _png(color):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class TestDecisionModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    def setUp(self):
        self.addCleanup(restore_context, snapshot_context())

    def _client(self, rows=(_logs(0.2, 0.6),), tokenizer=None, **server_args):
        from sglang.srt.entrypoints import http_server as server

        app = FastAPI()
        routes = server.app.router.routes
        app.router.routes.extend(
            r for r in routes if isinstance(r, APIRoute) and r.path in ROUTES
        )
        app.add_exception_handler(
            RequestValidationError, server.validation_exception_handler
        )
        manager = _manager(tokenizer or self.tokenizer, rows, **server_args)
        app.state.decision_model_serving = DecisionModelServing(manager)
        recorder = SimpleNamespace(handle_request=_handled_by)
        app.state.systemone_serving = app.state.openai_serving_decisions = recorder
        return TestClient(app), manager

    def test_prompt_matches_the_official_compiler(self):
        compiled = compile_decision(STATE, QUESTIONS)
        contents = [message["content"] for message in compiled.messages]
        self.assertEqual(contents, OFFICIAL_MESSAGES)
        self.assertEqual(ANSWER_SYMBOLS, OFFICIAL_SYMBOLS)

    def test_invalid_requests_are_422_at_the_offending_field(self):
        client, _ = self._client()
        choices = {f"o{i}": "" for i in range(63)}
        cases = [
            (
                {"f": {"type": "choice", "criteria": choices}},
                ["body", "questions", "f"],
            ),
            ({f"f{i}": NOUL["u"] for i in range(17)}, ["body", "questions"]),
            ({"f": {"type": "score", "criteria": {"x": ""}}}, ["body", "questions"]),
        ]
        marker = {"state": "a <decision>", "questions": NOUL}
        for route in ROUTES:
            for questions, loc in cases:
                with self.subTest(route=route, loc=loc):
                    body = {"state": {}, "questions": questions}
                    response = client.post(route, json=body)
                    self.assertEqual(response.status_code, 422, response.text)
                    detail = response.json()["detail"][0]
                    self.assertEqual(detail["loc"][: len(loc)], loc)
                    self.assertIn("x-typesafe-request-id", response.headers)
            detail = client.post(route, json=marker).json()["detail"][0]
            self.assertIn("reserved decision marker", detail["msg"])

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
        response = client.post("/v1/decisions", json=generic)
        self.assertEqual(response.json(), {"handled_by": "DecisionRequest"})

        client, _ = self._client(tokenizer=SimpleNamespace(get_added_vocab=dict))
        noul = {"u": {"type": "noul", "instructions": "q"}}
        systemone = {"state": "s", "model": "m", "questions": noul}
        response = client.post("/v1/systemone", json=systemone)
        self.assertEqual(response.json(), {"handled_by": "SystemOneRequest"})
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

    def test_invalid_images_are_422(self):
        client, _ = self._client()
        request = {"state": {}, "questions": NOUL}
        cases = [
            ([_png("red")] * 9, ["body", "images"]),
            ([_png("red"), "not-base64!"], ["body", "images", 1]),
            (["data:text/plain;base64," + _png("red")], ["body", "images", 0]),
            ([base64.b64encode(b"not an image").decode()], ["body", "images", 0]),
            (["/etc/passwd"], ["body", "images", 0]),
        ]
        for images, loc in cases:
            with self.subTest(loc=loc):
                response = client.post("/v1/jev", json={**request, "images": images})
                self.assertEqual(response.status_code, 422, response.text)
                self.assertEqual(response.json()["detail"][0]["loc"], loc)
        # A literal image placeholder would take an attached image's slot.
        conflict = {"state": {"note": "see <image>"}, "questions": NOUL}
        response = client.post("/v1/jev", json={**conflict, "images": [_png("red")]})
        self.assertEqual(response.status_code, 422)
        self.assertIn("image placeholder", response.json()["detail"][0]["msg"])
        self.assertEqual(client.post("/v1/jev", json=conflict).status_code, 200)

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

    def test_refusals(self):
        request = {"state": {}, "questions": NOUL}
        long_state = {"state": {"text": "word " * 400}}
        cases = [
            ({}, {"chat_template": "chatml"}, 400),
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
