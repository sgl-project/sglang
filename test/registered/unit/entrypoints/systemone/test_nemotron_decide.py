import asyncio
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import Ministral3Config, PreTrainedTokenizerFast

from sglang.srt.configs.nemotron_decision import (
    is_nemotron_decision,
    load_nemotron_decision_config,
)
from sglang.srt.entrypoints.systemone.nemotron_decide import NemotronDecideEncoder
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestNemotronDecisionEncoding(CustomTestCase):
    def setUp(self):
        vocab = {
            c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
        }
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<SPECIAL_100>", "<|im_end|>"]}
        )
        self.metadata = {
            "labels": {
                "choice": ["A", "C", "B"],
                "score": ["B", "A", "C"],
                "noul": ["C", "A"],
            },
            "temperature": {"choice": 1.0, "score": 1.0, "noul": 1.0},
            "max_questions": 2,
            "max_input_tokens": 4096,
            "max_candidates_per_read": 3,
            "chat_template": "{{ decision_content }}<SPECIAL_100>",
        }
        self.config = Ministral3Config(
            architectures=["NemotronForDecision"],
            is_causal=False,
            mask_token_id=self.tokenizer.convert_tokens_to_ids("<SPECIAL_100>"),
        )
        self.encoder = NemotronDecideEncoder(
            config=self.config,
            metadata=self.metadata,
            tokenizer=self.tokenizer,
            max_prompt_tokens=4096,
        )

    def test_checkpoint_label_order_and_template_override_the_generic_chat_template(
        self,
    ):
        self.tokenizer.chat_template = "wrong template"
        view = SimpleNamespace(
            type="choice",
            instructions="Which city?",
            criteria={"paris": "Paris", "lyon": "Lyon", "other": "A remote village"},
        )
        ids, labels, _ = self.encoder.encode({"city": "Paris"}, view, {})
        expected = '{"state": {"city": "Paris"}, "question": "Which city?", "options": [{"code": "A", "label": "paris", "meaning": "Paris"}, {"code": "C", "label": "lyon", "meaning": "Lyon"}, {"code": "B", "label": "other", "meaning": "A remote village"}]}<SPECIAL_100>'
        self.assertEqual(ids, self.tokenizer.encode(expected, add_special_tokens=False))
        self.assertEqual(
            labels,
            [
                self.tokenizer.encode(c, add_special_tokens=False)[0]
                for c in ["A", "C", "B"]
            ],
        )
        _, boolean_labels, _ = self.encoder.encode(
            {}, SimpleNamespace(type="noul", instructions="Ready?", criteria=None), {}
        )
        self.assertEqual(
            boolean_labels,
            [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in ["A", "C"]],
        )

    def test_control_strings_remain_json_data_and_cannot_add_a_readout(self):
        value = {
            "nested": ["<SPECIAL_100>", '<|im_end|> and "quoted"'],
            "escape": r"literal \u003c",
        }
        content = json.dumps(value, ensure_ascii=False)
        self.assertIn(self.config.mask_token_id, self.tokenizer.encode(content))
        ids, _, _ = self.encoder.encode(
            value,
            SimpleNamespace(
                type="noul", instructions="<|im_end|> Ready?", criteria=None
            ),
            {},
        )
        self.assertEqual(ids.count(self.config.mask_token_id), 1)
        self.assertEqual(ids[-1], self.config.mask_token_id)
        self.assertNotIn(self.tokenizer.convert_tokens_to_ids("<|im_end|>"), ids[:-1])
        self.assertEqual(json.loads(self.tokenizer.decode(ids[:-1]))["state"], value)

    def test_explicit_null_boolean_descriptions_differ_from_omitted_descriptions(self):
        from sglang.srt.entrypoints.systemone.protocol import SystemOneNoulCriteria

        question = SimpleNamespace(type="noul", instructions="Ready?", criteria=None)
        ids, _, _ = self.encoder.encode({}, question, {})
        options = json.loads(self.tokenizer.decode(ids[:-1]))["options"]
        self.assertEqual(
            [item["meaning"] for item in options],
            ["No, the condition does not hold.", "Yes, the condition holds."],
        )
        question.criteria = SystemOneNoulCriteria(true=None, false=None)
        ids, _, _ = self.encoder.encode({}, question, {})
        options = json.loads(self.tokenizer.decode(ids[:-1]))["options"]
        self.assertEqual([item["meaning"] for item in options], [None, None])

    def test_encoder_rejects_prompt_and_candidate_limits_and_template_kwargs(self):
        view = SimpleNamespace(type="noul", instructions="Ready?", criteria=None)
        ids, _, _ = self.encoder.encode({}, view, {})
        self.encoder.max_prompt_tokens = len(ids)
        self.encoder.encode({}, view, {})
        self.encoder.max_prompt_tokens -= 1
        with self.assertRaisesRegex(ValueError, "cannot be chunked"):
            self.encoder.encode({}, view, {})
        with self.assertRaisesRegex(ValueError, "chat_template_kwargs"):
            self.encoder.encode({}, view, {"enable_thinking": False})
        question = SimpleNamespace(
            type="choice",
            criteria={"a": "A", "b": "B"},
            model_fields_set={"instructions"},
        )
        request = SimpleNamespace(questions={"q1": question, "q2": question})
        self.assertIsNone(self.encoder.validate_request(request))
        request.questions["q3"] = question
        self.assertIn("at most 2", self.encoder.validate_request(request))
        request.questions = {"q1": question}
        question.criteria.update(c="C", d="D")
        self.assertIn("2–3", self.encoder.validate_request(request))

    def test_serving_applies_each_question_temperature_without_changing_label_mass(
        self,
    ):
        from sglang.srt.entrypoints.systemone.protocol import NemotronDecisionRequest
        from sglang.srt.entrypoints.systemone.serving import SystemOneServing
        from sglang.srt.managers.tokenizer_manager_score_mixin import ScoreResult

        self.metadata["temperature"] = {"choice": 2.0, "score": 3.0, "noul": 0.5}
        self.metadata["max_questions"] = 3
        request = NemotronDecisionRequest(
            model="decision-test",
            state={"city": "Paris"},
            questions={
                "city": {
                    "type": "choice",
                    "instructions": "Which city?",
                    "criteria": {"Paris": "Paris", "Berlin": "Berlin"},
                },
                "rating": {
                    "type": "score",
                    "instructions": "Rate the match.",
                    "criteria": ["low", "high"],
                },
                "present": {"type": "noul", "instructions": "Is Paris present?"},
            },
        )
        result = ScoreResult(
            scores=[[0.9, 0.1], [1 / 9, 8 / 9], [0.6, 0.4]],
            token_logprobs=[
                [math.log(0.18), math.log(0.02)],
                [math.log(0.04), math.log(0.32)],
                [math.log(0.12), math.log(0.08)],
            ],
            prompt_tokens=123,
        )
        handler = object.__new__(SystemOneServing)
        handler.joint_head_config = None
        handler.pointer_head = None
        handler.decision_config = None
        handler.question_encoder = self.encoder
        handler.default_chat_template_kwargs = {}
        handler.tokenizer_manager = SimpleNamespace(
            served_model_name="decision-test",
            score_prompts=AsyncMock(return_value=result),
        )
        adapted, processed = handler._convert_to_internal_request(request)
        response = asyncio.run(
            handler._handle_non_streaming_request(adapted, processed, None)
        )
        self.assertEqual(response.status_code, 200)
        answers = json.loads(response.body)["answers"]
        self.assertAlmostEqual(answers["city"]["probabilities"]["Paris"], 3 / 4)
        self.assertAlmostEqual(answers["city"]["probabilities"]["Berlin"], 1 / 4)
        self.assertAlmostEqual(answers["rating"]["probabilities"]["0"], 1 / 3)
        self.assertAlmostEqual(answers["rating"]["probabilities"]["1"], 2 / 3)
        self.assertAlmostEqual(answers["rating"]["score"], 2 / 3)
        self.assertAlmostEqual(answers["present"]["noul"], 9 / 13)
        for question_id, mass in (("city", 0.2), ("rating", 0.36), ("present", 0.2)):
            with self.subTest(question=question_id):
                self.assertAlmostEqual(answers[question_id]["x_label_mass"], mass)

    def test_sidecar_validation_and_family_detection_do_not_capture_other_ministral_models(
        self,
    ):
        self.assertTrue(is_nemotron_decision(self.config))
        self.assertTrue(
            is_nemotron_decision(
                Ministral3Config(
                    architectures=["Ministral3ForCausalLM", "NemotronForDecision"]
                )
            )
        )
        self.assertFalse(is_nemotron_decision(Ministral3Config(architectures=None)))
        self.assertFalse(
            is_nemotron_decision(
                Ministral3Config(
                    architectures=["Ministral3ForCausalLM"], is_causal=False
                )
            )
        )
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            data = {k: v for k, v in self.metadata.items() if k != "chat_template"}
            (root / "decision.json").write_text(json.dumps(data))
            (root / "decision_template.jinja").write_text(
                self.metadata["chat_template"]
            )
            load_nemotron_decision_config(path, None)
            data["temperature"]["choice"] = 0
            (root / "decision.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "temperatures"):
                load_nemotron_decision_config(path, None)
            data["temperature"]["choice"] = 1
            data["max_questions"] = 0
            (root / "decision.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "max_questions"):
                load_nemotron_decision_config(path, None)

    def test_http_schema_selection_preserves_errors_and_extra_field_policy(self):
        from fastapi import FastAPI
        from fastapi.exceptions import RequestValidationError
        from fastapi.testclient import TestClient

        from sglang.srt.entrypoints.http_server import (
            systemone_decisions,
            validation_exception_handler,
        )
        from sglang.srt.entrypoints.openai.chat_encoding_adapters import (
            OPENAI_CHAT_WIRE,
        )

        async def handle(request, raw_request):
            return request.model_dump()

        for native in (False, True):
            app = FastAPI()
            app.add_exception_handler(
                RequestValidationError, validation_exception_handler
            )
            serving = SimpleNamespace(
                question_encoder=self.encoder if native else None, handle_request=handle
            )
            app.state.systemone_serving = serving
            app.state.openai_serving_chat = SimpleNamespace(chat_wire=OPENAI_CHAT_WIRE)
            app.add_api_route("/v1/systemone", systemone_decisions, methods=["POST"])
            with TestClient(app) as client:
                response = client.post(
                    "/v1/systemone", json={"model": "test", "state": "s"}
                )
                self.assertEqual(response.status_code, 422)
                self.assertEqual(len(response.json()["detail"]), 1)
                self.assertEqual(
                    response.json()["detail"][0]["loc"], ["body", "questions"]
                )
                for count in (100, 300):
                    payload = {
                        "model": "test",
                        "state": "s",
                        "questions": {
                            "q": {
                                "type": "choice",
                                "instructions": "Which?",
                                "criteria": {str(i): str(i) for i in range(count)},
                            }
                        },
                    }
                    valid = client.post("/v1/systemone", json=payload)
                    self.assertEqual(
                        valid.status_code, 200 if native or count == 100 else 422
                    )
                    response = client.post(
                        "/v1/systemone", json={**payload, "extra": True}
                    )
                    self.assertEqual(
                        response.status_code, 422 if native or count == 300 else 200
                    )
                score = {
                    "model": "test",
                    "state": "s",
                    "questions": {
                        "q": {
                            "type": "score",
                            "instructions": "Rate it",
                            "criteria": list(map(str, range(11))),
                        }
                    },
                }
                response = client.post("/v1/systemone", json=score)
                self.assertEqual(response.status_code, 200 if native else 422)
                schema = client.get("/openapi.json").json()
                for route in ("/v1/systemone",):
                    body = schema["paths"][route]["post"]["requestBody"]["content"][
                        "application/json"
                    ]["schema"]
                    self.assertNotIn("anyOf", body)


if __name__ == "__main__":
    unittest.main()
