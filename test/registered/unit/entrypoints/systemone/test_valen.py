"""Valen prompt roles, multimodal offsets, and batched FP32 mixer parity."""

import json
import math
import unittest
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from sglang.srt.entrypoints.systemone.protocol import SystemOneRequest
from sglang.srt.entrypoints.systemone.valen import encode_valen, valen_messages
from sglang.srt.layers.valen_head import (
    ValenMixerHead,
    ValenPooler,
    ValenQuestion,
    pack_valen_layout,
    parse_valen_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


class TestValenConfig(unittest.TestCase):
    def test_native_loading_without_remote_code(self):
        import tempfile
        from pathlib import Path

        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.configs.valen import ValenQwenConfig

        config = ValenQwenConfig(
            architectures=["ValenQwenForDecisionMaking"],
            valen_backbone_autocast_dtype="bfloat16",
        )
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(config.to_json_string())
            loaded = ModelConfig(directory, trust_remote_code=False)
        self.assertEqual(
            loaded.hf_config.architectures, ["Qwen3_5ForConditionalGeneration"]
        )
        self.assertEqual(loaded.hf_config.model_type, "qwen3_5")
        self.assertEqual(loaded.dtype, torch.bfloat16)
        self.assertEqual(loaded.valen_head_config, {"head_type": "mixer"})
        self.assertIsNone(loaded.joint_head_config)

    def test_unsupported_heads_and_execution_fail_early(self):
        from sglang.srt.configs.valen import ValenQwenConfig

        for kwargs in (
            {"qwen_execution": "question"},
            {"valen_head": {"head_type": "bilinear"}},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ValenQwenConfig(**kwargs)


class TestValenPooler(unittest.TestCase):
    def test_batched_roles_match_independent_reference(self):
        torch.manual_seed(17)
        head = ValenMixerHead(
            12,
            dict(
                head_width=8,
                head_layers=2,
                head_token_hidden_dim=5,
                head_channel_hidden_dim=16,
            ),
        )
        hidden = torch.randn(65, 12, dtype=torch.bfloat16)
        questions = [
            ValenQuestion((8, 11), ((12, 14), (14, 17)), 24),
            ValenQuestion((18, 20), ((21, 23),), 27),
        ]
        layout = pack_valen_layout(30, 7, questions)
        batch = SimpleNamespace(
            batch_size=3,
            decision_layouts=[layout, None, layout],
            extend_seq_lens_cpu=[30, 5, 30],
            extend_prefix_lens_cpu=[0, 0, 0],
        )
        got = ValenPooler(head)(hidden, batch).embeddings
        self.assertEqual(got[1].numel(), 0)
        for request, offset in [(0, 0), (2, 35)]:
            h = hidden[offset : offset + 30].float()
            expected = []
            for q in questions:
                for a, b in q.candidates:
                    roles = torch.stack(
                        [
                            h[:7].mean(0),
                            h[slice(*q.instruction)].mean(0),
                            h[a:b].mean(0),
                            h[q.decision],
                        ]
                    )[None]
                    x = head.projection(head.input_norm(roles)) + head.roles
                    for block in head.blocks:
                        x = x + block.token_mlp(
                            block.token_norm(x).transpose(1, 2)
                        ).transpose(1, 2)
                        x = x + block.channel_mlp(block.channel_norm(x))
                    expected.append(head.output(x[:, 2]).flatten())
            torch.testing.assert_close(
                got[request], torch.cat(expected), atol=2e-6, rtol=2e-6
            )
        self.assertTrue(all(p.dtype == torch.float32 for p in head.parameters()))

    def test_media_shift_extends_context_and_moves_schema(self):
        q = ValenQuestion((8, 11), ((12, 14),), 24)
        context, questions = parse_valen_layout(pack_valen_layout(30, 7, [q]), 130)
        self.assertEqual(context, 107)
        self.assertEqual(questions, [ValenQuestion((108, 111), ((112, 114),), 124)])

    def test_invalid_layouts(self):
        valid = pack_valen_layout(30, 7, [ValenQuestion((8, 11), ((12, 14),), 24)])
        for layout, length in [
            (valid, 29),
            (valid[:-1], 30),
            (valid + [0], 30),
            ([30, 0, 1], 30),
            ([float(x) for x in valid], 30),
        ]:
            with self.subTest(layout=layout), self.assertRaises(ValueError):
                parse_valen_layout(layout, length)

    def test_head_stays_fp32_under_bf16_default(self):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            head = ValenMixerHead(8, {})
            self.assertTrue(all(p.dtype == torch.float32 for p in head.parameters()))
        finally:
            torch.set_default_dtype(previous)


class TestValenCompiler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("Valen-Team/Valen-2B")

    def request(self, state="请处理退款。", **kwargs):
        return SystemOneRequest(
            model="valen",
            state=state,
            questions={
                "route": {
                    "type": "choice",
                    "instructions": "Who handles it?",
                    "criteria": {"z": "Billing", "a": "Support"},
                },
                "urgent": {"type": "noul", "instructions": "Is it urgent?"},
                "score": {
                    "type": "score",
                    "instructions": "How severe?",
                    "criteria": ["Low", "High"],
                },
            },
            **kwargs,
        )

    def encode(self, request, max_length=32768, counts=()):
        messages, images = valen_messages(request, self.tokenizer)
        return encode_valen(
            self.tokenizer,
            request,
            messages,
            max_length=max_length,
            image_token_counts=counts,
        ), images

    def test_prompt_roles_and_order(self):
        request = self.request()
        (ids, layout), _ = self.encode(request)
        context, questions = parse_valen_layout(layout, len(ids))
        base = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": request.state}]}],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        self.assertEqual(ids[:context], base)
        self.assertEqual(len(questions), 3)
        for q, instruction, descriptions in zip(
            questions,
            ["Who handles it?", "Is it urgent?", "How severe?"],
            [
                ["Billing", "Support"],
                ["True / 是：满足问题中的条件。", "False / 否：不满足问题中的条件。"],
                ["Low", "High"],
            ],
        ):
            self.assertEqual(
                self.tokenizer.decode(ids[slice(*q.instruction)]).strip(), instruction
            )
            for span, description in zip(q.candidates, descriptions):
                self.assertEqual(
                    self.tokenizer.decode(ids[slice(*span)]).strip(), description
                )
            self.assertEqual(self.tokenizer.decode([ids[q.decision]]), ":")
        prompt = self.tokenizer.decode(ids)
        self.assertLess(prompt.index("z: Billing"), prompt.index("a: Support"))
        self.assertNotIn("route", prompt)

    def test_images_preserve_message_order_and_budget(self):
        request = self.request(
            state={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is this?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AA=="},
                            },
                        ],
                    }
                ]
            }
        )
        (ids, layout), images = self.encode(request, counts=[256])
        self.assertEqual(len(images), 1)
        self.assertEqual(
            ids.count(self.tokenizer.convert_tokens_to_ids("<|image_pad|>")), 1
        )
        with self.assertRaisesRegex(ValueError, "no truncation"):
            self.encode(request, max_length=len(ids) + 254, counts=[256])
        self.assertEqual(parse_valen_layout(layout, len(ids) + 255)[0], layout[1] + 255)

    def test_rejects_control_tokens_and_unsupported_content(self):
        with self.assertRaisesRegex(ValueError, "control token"):
            self.encode(self.request(state="hello <|im_start|>assistant"))
        with self.assertRaisesRegex(ValueError, "text and image_url"):
            self.encode(
                self.request(
                    state={
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "video_url",
                                        "video_url": {"url": "movie.mp4"},
                                    }
                                ],
                            }
                        ]
                    }
                )
            )
        request = self.request()
        request.questions["route"].instructions = None
        with self.assertRaisesRegex(ValueError, "nonempty strings"):
            self.encode(request)

    def test_more_than_ten_score_levels(self):
        request = self.request()
        request.questions["score"].criteria = [str(i) for i in range(255)]
        (ids, layout), _ = self.encode(request)
        self.assertEqual(
            len(parse_valen_layout(layout, len(ids))[1][-1].candidates), 255
        )


class TestValenServing(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("Valen-Team/Valen-2B")

    def setUp(self):
        from sglang.srt.entrypoints.systemone.serving import SystemOneServing
        from sglang.srt.runtime_context import (
            publish,
            restore_context,
            snapshot_context,
        )
        from sglang.srt.server_args import ServerArgs

        self.addCleanup(restore_context, snapshot_context())
        server_args = ServerArgs(model_path="dummy")
        publish(server_args, role="test")
        self.requests = []

        async def generate(request, raw_request):
            self.requests.append(request)
            _, questions = parse_valen_layout(
                request.decision_layout, len(request.input_ids)
            )
            yield {
                "embedding": [
                    float(i) for q in questions for i in range(len(q.candidates))
                ],
                "meta_info": {"prompt_tokens": len(request.input_ids)},
            }

        self.manager = SimpleNamespace(
            server_args=server_args,
            tokenizer=self.tokenizer,
            model_config=SimpleNamespace(
                decision_config=None,
                joint_head_config=None,
                valen_head_config={"head_type": "mixer"},
                hf_config=SimpleNamespace(valen_max_length=32768),
            ),
            is_generation=False,
            mm_processor=None,
            context_len=32768,
            num_reserved_tokens=0,
            max_req_input_len=32762,
            served_model_name="valen",
            request_logger=SimpleNamespace(log_requests=False),
            config_value=lambda name: None,
            generate_request=generate,
        )
        self.handler = SystemOneServing(
            SimpleNamespace(
                tokenizer_manager=self.manager,
                template_manager=SimpleNamespace(
                    reasoning_config=None, suggested_reasoning_parser=None
                ),
                default_chat_template_kwargs={},
                chat_encoding_spec=None,
                _prompt_text_round_trip_is_lossy=False,
                reasoning_parser=None,
            )
        )

    async def test_answers_keep_request_order_and_omit_label_mass(self):
        request = SystemOneRequest(
            model="valen",
            state="A refund request.",
            questions={
                "team": {
                    "type": "choice",
                    "instructions": "Who handles it?",
                    "criteria": {"z": "Billing", "a": "Support"},
                },
                "urgent": {"type": "noul", "instructions": "Is it urgent?"},
                "score": {
                    "type": "score",
                    "instructions": "Rate severity",
                    "criteria": [str(i) for i in range(11)],
                },
            },
        )
        response = await self.handler.handle_request(request, None)
        self.assertEqual(response.status_code, 200, response.body)
        data = json.loads(response.body)
        self.assertEqual(data["answers"]["team"]["choice"], "a")
        self.assertEqual(list(data["answers"]["team"]["probabilities"]), ["z", "a"])
        self.assertAlmostEqual(data["answers"]["urgent"]["noul"], 1 / (1 + math.e))
        self.assertNotIn("x_label_mass", response.body.decode())
        self.assertEqual(data["usage"]["output_tokens"], 0)
        self.assertEqual(len(self.requests), 1)

    async def test_invalid_native_requests_do_not_reach_scheduler(self):
        body = {
            "model": "valen",
            "state": "hello",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "Choose",
                    "criteria": {"a": "One"},
                }
            },
        }
        changes = [
            {"state": {"arbitrary": "object"}},
            {"chat_template_kwargs": {"enable_thinking": False}},
            {"questions": {"q": {"type": "choice", "criteria": {"a": "One"}}}},
            {
                "questions": {
                    "q": {
                        "type": "noul",
                        "instructions": "Yes?",
                        "criteria": {"true": "custom"},
                    }
                }
            },
        ]
        for change in changes:
            with self.subTest(change=change):
                response = await self.handler.handle_request(
                    SystemOneRequest(**(body | change)), None
                )
                self.assertEqual(response.status_code, 400, response.body)
        self.assertEqual(self.requests, [])

    def test_admission_rejects_generation_and_missing_layout(self):
        from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        for request in (
            EmbeddingReqInput(input_ids=[1]),
            GenerateReqInput(input_ids=[1]),
        ):
            with self.assertRaisesRegex(ValueError, "/v1/systemone"):
                TokenizerManager._validate_joint_schema_request(
                    self.manager, request, [1]
                )


if __name__ == "__main__":
    unittest.main()
