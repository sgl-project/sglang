# SPDX-License-Identifier: Apache-2.0
"""Native Valen protocol, trained response semantics and prefill request shape."""

import asyncio
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from pydantic import ValidationError

from sglang.srt.entrypoints.systemone.protocol import SystemOneRequest
from sglang.srt.entrypoints.valen.encoding import ValenQuestion, encode_valen_text
from sglang.srt.entrypoints.valen.media import (
    encode_valen_prompt,
    validate_valen_media_payload,
)
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
            is_valen=True,
            hf_config=SimpleNamespace(valen_max_length=32768, valen_media_kwargs={}),
        )
        self.context_len = self.max_req_input_len = 32768
        self.num_reserved_tokens = 0
        self.served_model_name = "Valen-test"
        self.requests = []
        self.closed_requests = []
        self.processor = None

    async def generate_request(self, obj, raw_request):
        self.requests.append(obj)
        validate_valen_media_payload(obj)
        layout = parse_valen_layout(obj.valen_layout, len(obj.input_ids))
        count = sum(len(q.candidate_spans) for q in layout.questions)
        try:
            yield {
                "embedding": [0.0] * count,
                "meta_info": {"prompt_tokens": len(obj.input_ids)},
            }
        finally:
            self.closed_requests.append(obj)


class Processor:
    """External HF boundary: its expanded context cannot be reconstructed from text."""

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.ids = [1000, 1001, 1001, 1002, 55, 1000, 1003, 1003, 1002]
        self.pixels = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        self.video_pixels = torch.arange(48, dtype=torch.float32).reshape(8, 6)
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        result = {
            "input_ids": torch.tensor([self.ids]),
            "pixel_values": self.pixels,
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "pixel_values_videos": self.video_pixels,
            "video_grid_thw": torch.tensor([[2, 2, 2]]),
            "video_metadata": [{"frames_indices": [0, 8], "fps": 4}],
            "second_per_grid_ts": torch.tensor([2.0]),
        }
        kinds = {c["type"] for m in messages for c in m["content"]}
        if "video" not in kinds:
            for key in (
                "pixel_values_videos",
                "video_grid_thw",
                "video_metadata",
                "second_per_grid_ts",
            ):
                result.pop(key)
        return result


def media_state():
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Observe."},
                    {"type": "image_url", "image_url": {"url": "image.png"}},
                    {"type": "video_url", "video_url": {"url": "clip.mp4"}},
                ],
            }
        ]
    }


class TestValenServing(CustomTestCase):
    def setUp(self):
        super().setUp()
        context = patch(
            "sglang.srt.entrypoints.valen.serving.get_mm",
            return_value=SimpleNamespace(limit_mm_data_per_request={}),
        )
        context.start()
        self.addCleanup(context.stop)

    def test_new_hf_video_expansion_preserves_trained_frame_wrapping(self):
        # HF 5.19 replaces only video_pad; the training processor also removed
        # its outer vision wrapper. An extra wrapper changes adjacent BPE tokens
        # and all downstream role positions (414 -> 417 on the real smoke video).
        processor = Processor()
        processor.vision_start_token = "<|vision_start|>"
        processor.video_token = "<|video_pad|>"
        processor.vision_end_token = "<|vision_end|>"
        template = "Observe.<|vision_start|><|video_pad|><|vision_end|>"
        frames = "<0.1 seconds><|vision_start|><|video_pad|><|vision_end|>"
        processor.chat_template = template
        processor.replace_video_token = lambda: frames
        original = processor.apply_chat_template

        def expand(messages, **kwargs):
            result = original(messages, **kwargs)
            text = kwargs.get("chat_template", template).replace(
                processor.video_token, frames
            )
            result["input_ids"] = torch.tensor([processor.tokenizer.encode(text)])
            return result

        processor.apply_chat_template = expand
        prompt = encode_valen_prompt(
            processor.tokenizer,
            processor,
            media_state(),
            {"q": {"type": "noul", "instructions": "Is it red?"}},
            media_kwargs={},
            max_length=32768,
        )
        expected = processor.tokenizer.encode("Observe." + frames)
        self.assertEqual(
            prompt.encoded.input_ids[: prompt.encoded.layout.context_span[1]],
            expected,
        )
        self.assertEqual(processor.chat_template, template)

    def test_remote_media_is_materialized_once_and_removed_after_success_or_error(self):
        state = media_state()
        state["messages"][0]["content"][1]["image_url"]["url"] = (
            "https://media.example/image.png"
        )
        state["messages"][0]["content"][2]["video_url"]["url"] = (
            "https://media.example/clip.mp4"
        )
        for fail in (False, True):
            processor = Processor()
            original = processor.apply_chat_template
            paths = []

            def consume(messages, **kwargs):
                sources = [
                    Path(c["url"])
                    for c in messages[0]["content"]
                    if c["type"] != "text"
                ]
                paths.extend(sources)
                self.assertEqual(
                    [p.read_bytes() for p in sources], [b"image-bytes", b"video-bytes"]
                )
                if fail:
                    raise OSError("bad video")
                return original(messages, **kwargs)

            processor.apply_chat_template = consume
            with (
                self.subTest(fail=fail),
                patch("sglang.srt.utils.get_image_bytes", return_value=b"image-bytes"),
                patch("sglang.srt.utils.get_video_bytes", return_value=b"video-bytes"),
            ):
                kwargs = dict(media_kwargs={}, max_length=32768)
                if fail:
                    with self.assertRaisesRegex(ValueError, "bad video"):
                        encode_valen_prompt(
                            processor.tokenizer,
                            processor,
                            state,
                            {"q": {"type": "noul", "instructions": "Is it red?"}},
                            **kwargs,
                        )
                else:
                    encode_valen_prompt(
                        processor.tokenizer,
                        processor,
                        state,
                        {"q": {"type": "noul", "instructions": "Is it red?"}},
                        **kwargs,
                    )
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(not p.exists() for p in paths))

    def test_media_url_policy_rejection_is_a_client_error_without_inference(self):
        manager = Manager()
        manager.processor = Processor()
        state = media_state()
        state["messages"][0]["content"][1]["image_url"]["url"] = (
            "https://denied.example/image.png"
        )
        request = ValenRequest(
            state=state, questions={"q": {"type": "noul", "instructions": "Is it red?"}}
        )
        with (
            patch(
                "sglang.srt.entrypoints.valen.serving.get_schedule",
                return_value=SimpleNamespace(max_prefill_tokens=32768),
            ),
            patch(
                "sglang.srt.utils.get_image_bytes",
                side_effect=ValueError("Media URL domain is not allowed"),
            ),
        ):
            response = asyncio.run(ValenServing(manager).handle_request(request))
        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "domain is not allowed", json.loads(response.body)["error"]["message"]
        )
        self.assertEqual(manager.requests, [])

    def test_systemone_images_use_the_native_media_path(self):
        manager = Manager()
        manager.processor = Processor()
        request = SystemOneRequest(
            model="Valen-test",
            state="Observe.",
            images=["image.png"],
            questions={"q": {"type": "noul", "instructions": "Is it green?"}},
        )
        with patch(
            "sglang.srt.entrypoints.valen.serving.get_schedule",
            return_value=SimpleNamespace(max_prefill_tokens=32768),
        ):
            response = asyncio.run(ValenServing(manager).handle_systemone(request))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body)["answers"]["q"]["noul"], 0.5)
        self.assertIsNone(manager.requests[0].video_data)
        self.assertEqual(
            manager.processor.calls[0][0][0]["content"][1],
            {"type": "image", "url": "image.png"},
        )

    def test_media_keeps_expanded_hf_context_and_does_not_mix_metadata_into_features(
        self,
    ):
        processor = Processor()
        state = media_state()
        settings = {"videos_kwargs": {"num_frames": 16, "fps": None}}
        questions = {"q": {"type": "noul", "instructions": "Is it red?"}}
        prompt = encode_valen_prompt(
            processor.tokenizer,
            processor,
            state,
            questions,
            media_kwargs=settings,
            max_length=32768,
        )
        self.assertEqual(prompt.encoded.input_ids[: len(processor.ids)], processor.ids)
        self.assertEqual(prompt.encoded.layout.context_span, (0, len(processor.ids)))
        self.assertGreaterEqual(
            prompt.encoded.layout.questions[0].instruction_span[0], len(processor.ids)
        )
        self.assertEqual(state, media_state())
        self.assertNotIn("return_metadata", settings)
        for data in (prompt.image_data, prompt.video_data):
            self.assertEqual(data[0]["input_ids"], prompt.encoded.input_ids)
            self.assertNotIn("video_metadata", data[0])
        torch.testing.assert_close(
            prompt.image_data[0]["pixel_values"], processor.pixels
        )
        torch.testing.assert_close(
            prompt.video_data[0]["pixel_values_videos"], processor.video_pixels
        )
        self.assertEqual(prompt.video_data[0]["second_per_grid_ts"].item(), 2.0)
        self.assertEqual(
            processor.calls[0][1]["processor_kwargs"]["videos_kwargs"],
            settings["videos_kwargs"],
        )
        with self.assertRaisesRegex(ValueError, "no truncation"):
            encode_valen_prompt(
                processor.tokenizer,
                processor,
                state,
                questions,
                media_kwargs=settings,
                max_length=len(prompt.encoded.input_ids) - 1,
            )
        with self.assertRaisesRegex(ValueError, "server limit is 0"):
            encode_valen_prompt(
                processor.tokenizer,
                processor,
                state,
                questions,
                media_kwargs=settings,
                max_length=32768,
                media_limits={"image": 0},
            )

    def test_media_admission_refuses_resampling_or_a_different_token_sequence(self):
        for payload in (
            "image.png",
            {"format": "processor_output", "input_ids": [2, 3]},
        ):
            obj = EmbeddingReqInput(input_ids=[1, 2], image_data=[payload])
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_valen_media_payload(obj)

    def test_media_route_transfers_both_modalities_in_one_prefill(self):
        manager = Manager()
        manager.processor = Processor()
        manager.tokenizer = manager.processor.tokenizer
        request = ValenRequest(
            state=media_state(),
            questions={"q": {"type": "noul", "instructions": "Is the light red?"}},
        )
        with patch(
            "sglang.srt.entrypoints.valen.serving.get_schedule",
            return_value=SimpleNamespace(max_prefill_tokens=32768),
        ):
            response = asyncio.run(ValenServing(manager).handle_request(request))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body)["answers"]["q"]["noul"], 0.5)
        self.assertEqual(manager.closed_requests, manager.requests)
        self.assertEqual(len(manager.requests), 1)
        self.assertIsNotNone(manager.requests[0].image_data)
        self.assertIsNotNone(manager.requests[0].video_data)

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

    def test_text_encoder_rejects_control_tokens_and_requires_processor_for_media(self):
        with self.assertRaisesRegex(ValueError, "Reserved"):
            encode_valen_text(Tokenizer(), "hello<|im_start|>")
        for kind in ("image_url", "video_url"):
            with self.assertRaisesRegex(ValueError, "multimodal processor"):
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
