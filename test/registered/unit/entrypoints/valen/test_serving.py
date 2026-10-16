# SPDX-License-Identifier: Apache-2.0
"""Native Valen protocol, trained response semantics and prefill request shape."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from sglang.srt.entrypoints.valen.encoding import ValenQuestion, encode_valen_text
from sglang.srt.entrypoints.valen.protocol import ValenRequest
from sglang.srt.entrypoints.valen.responses import valen_answers
from sglang.srt.entrypoints.valen.serving import ValenServing
from sglang.srt.layers.valen_layout import parse_valen_layout
from sglang.srt.managers.io_struct import EmbeddingReqInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Tokenizer:
    all_special_tokens = ["<|im_start|>", "<|im_end|>"]

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def __call__(self, text, **kwargs):
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }

    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, return_dict
    ):
        assert tokenize and not add_generation_prompt and not return_dict
        return self.encode(
            "".join(
                f"<|im_start|>{m['role']}\n"
                + "".join(c["text"] for c in m["content"])
                + "<|im_end|>\n"
                for m in messages
            )
        )


class Manager:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.model_config = SimpleNamespace(
            is_valen=True, hf_config=SimpleNamespace(valen_max_length=32768)
        )
        self.context_len = self.max_req_input_len = 32768
        self.num_reserved_tokens = 0
        self.served_model_name = "Valen-test"
        self.requests = []
        self.closed_requests = []

    async def generate_request(self, obj, raw_request):
        self.requests.append(obj)
        layout = parse_valen_layout(obj.valen_layout, len(obj.input_ids))
        count = sum(len(q.candidate_spans) for q in layout.questions)
        try:
            yield {
                "embedding": [0.0] * count,
                "meta_info": {"prompt_tokens": len(obj.input_ids)},
            }
        finally:
            self.closed_requests.append(obj)


class TestValenServing(CustomTestCase):
    def test_batched_layouts_stay_with_their_prompt_after_normalization(self):
        from sglang.srt.entrypoints.valen.encoding import encode_valen_schema
        from sglang.srt.layers.valen_layout import pack_valen_layout

        first = encode_valen_schema(
            Tokenizer(), [1, 2], {"a": {"type": "noul", "instructions": "Is it true?"}}
        )
        second = encode_valen_schema(
            Tokenizer(),
            [3, 4, 5],
            {
                "b": {
                    "type": "score",
                    "instructions": "How many?",
                    "criteria": ["none", "one", "many"],
                }
            },
        )
        layouts = [pack_valen_layout(first.layout), pack_valen_layout(second.layout)]
        obj = EmbeddingReqInput(
            input_ids=[first.input_ids, second.input_ids], valen_layout=layouts
        )
        obj.normalize_batch_and_arguments()
        for i, prompt in enumerate((first, second)):
            item = obj[i]
            self.assertEqual(
                parse_valen_layout(item.valen_layout, len(item.input_ids)),
                prompt.layout,
            )
            self.assertEqual(item.sampling_params["max_new_tokens"], 0)
        invalid = EmbeddingReqInput(
            input_ids=[first.input_ids, second.input_ids], valen_layout=[layouts[0]]
        )
        with self.assertRaisesRegex(ValueError, "one packed layout"):
            invalid.normalize_batch_and_arguments()

    def test_native_score_accepts_255_levels_and_rejects_generation_fields(self):
        request = {
            "state": "Count.",
            "questions": {
                "n": {
                    "type": "score",
                    "instructions": "How many?",
                    "criteria": [str(i) for i in range(255)],
                }
            },
        }
        self.assertEqual(len(ValenRequest(**request).questions["n"].criteria), 255)
        with self.assertRaises(ValidationError):
            ValenRequest(**request, max_tokens=100)
        with self.assertRaises(ValidationError):
            ValenRequest(**request, temperature=float("nan"))

    def test_native_response_contract_and_score_confidence(self):
        questions = (
            ValenQuestion("team", "choice", ("z", "a"), ("Zebra", "Ant")),
            ValenQuestion("truth", "noul", ("true", "false"), ("yes", "no")),
            ValenQuestion(
                "rating", "score", ("0", "1", "2", "3"), ("zero", "one", "two", "three")
            ),
        )
        answers = valen_answers(
            questions, [0.0, 0.0, 0.0, 0.0, 0.0, -1000.0, -1000.0, 0.0]
        )
        self.assertEqual(answers["team"]["choice"], "z")
        self.assertEqual(answers["team"]["confidence"], 0.0)
        self.assertEqual(answers["truth"], {"type": "noul", "noul": 0.5})
        self.assertEqual(answers["rating"]["score"], 1.5)
        self.assertEqual(answers["rating"]["confidence"], 0.0)
        self.assertEqual(list(answers["team"]["probabilities"]), ["z", "a"])

    def test_state_rejects_control_tokens_and_unvalidated_media(self):
        with self.assertRaisesRegex(ValueError, "Reserved"):
            encode_valen_text(Tokenizer(), "hello<|im_start|>")
        for kind in ("image_url", "video_url"):
            with self.assertRaisesRegex(ValueError, "image/video parity"):
                encode_valen_text(
                    Tokenizer(),
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": kind, kind: {"url": "asset"}}],
                            }
                        ]
                    },
                )

    def test_request_uses_one_prefill_and_preserves_external_question_ids(self):
        manager = Manager()
        request = ValenRequest(
            state="A cat.",
            questions={
                "private-id": {
                    "type": "choice",
                    "instructions": "Which animal?",
                    "criteria": {"z": "Cat", "a": "Dog"},
                }
            },
        )
        with patch(
            "sglang.srt.entrypoints.valen.serving.get_schedule",
            return_value=SimpleNamespace(max_prefill_tokens=32768),
        ):
            response = asyncio.run(ValenServing(manager).handle_request(request))
        self.assertEqual(response.status_code, 200)
        body = json.loads(response.body)
        self.assertEqual(body["answers"]["private-id"]["choice"], "z")
        self.assertEqual(body["usage"]["output_tokens"], 0)
        self.assertEqual(len(manager.requests), 1)
        self.assertEqual(manager.closed_requests, manager.requests)
        text = "".join(map(chr, manager.requests[0].input_ids))
        self.assertNotIn("private-id", text)
        self.assertIn("Question 1 Decision:", text)

    def test_wrong_model_returns_client_error_without_scheduling(self):
        manager = Manager()
        manager.model_config.is_valen = False
        request = ValenRequest(
            state="A cat.",
            questions={"q": {"type": "noul", "instructions": "Is it a cat?"}},
        )
        response = asyncio.run(ValenServing(manager).handle_request(request))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(manager.requests, [])


if __name__ == "__main__":
    unittest.main()
