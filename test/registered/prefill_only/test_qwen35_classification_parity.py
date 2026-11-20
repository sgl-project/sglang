"""Native Qwen3.5 scores agree with HF for a small, seeded hybrid checkpoint.

This exercises model loading and the embedding request path; it does not measure
the quality of a trained reward model. No checkpoint download is required.
"""

import tempfile
import unittest

import torch
from transformers import Qwen3_5Config, Qwen3_5ForSequenceClassification

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.io_struct import EmbeddingReqInput
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-small")


class TestQwen35ClassificationParity(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.checkpoint.cleanup)
        config = Qwen3_5Config(
            text_config=dict(
                vocab_size=128,
                hidden_size=512,
                intermediate_size=768,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=256,
                max_position_embeddings=512,
                layer_types=["linear_attention", "full_attention"],
                linear_key_head_dim=128,
                linear_value_head_dim=128,
                linear_num_key_heads=2,
                linear_num_value_heads=4,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                rope_parameters=dict(
                    rope_type="default",
                    rope_theta=10000000.0,
                    partial_rotary_factor=0.25,
                    mrope_section=[11, 11, 10],
                ),
            ),
            vision_config=dict(
                hidden_size=64,
                intermediate_size=128,
                depth=1,
                num_heads=4,
                patch_size=16,
                temporal_patch_size=1,
                spatial_merge_size=2,
                out_hidden_size=512,
                num_position_embeddings=64,
                deepstack_visual_indexes=[],
            ),
            image_token_id=120,
            video_token_id=121,
            vision_start_token_id=122,
            vision_end_token_id=123,
            tie_word_embeddings=False,
        )
        config.text_config.full_attention_interval = 2
        config.num_labels = 2
        config.pad_token_id = 0
        config.language_model_only = True
        config._attn_implementation = "eager"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(41110)
            model = Qwen3_5ForSequenceClassification(config).eval().bfloat16()
        model.save_pretrained(cls.checkpoint.name)
        # Different lengths catch pooling across request boundaries. The long
        # request spans more than one GDN chunk, with both layer types active.
        cls.requests = [
            [1, 11, 17],
            [1, 21, 24, 35, 16, 18, 45],
            [1] + [3 + (index * 7) % 100 for index in range(128)],
        ]
        with torch.no_grad():
            cls.expected = torch.stack(
                [
                    model(input_ids=torch.tensor([tokens]), use_cache=False).logits[0]
                    for tokens in cls.requests
                ]
            ).float()
        del model

        cls.engine = Engine(
            model_path=cls.checkpoint.name,
            skip_tokenizer_init=True,
            dtype="bfloat16",
            attention_backend="triton",
            linear_attn_prefill_backend="triton",
            disable_radix_cache=True,
            disable_cuda_graph=True,
            chunked_prefill_size=-1,
            max_total_tokens=1024,
            max_running_requests=8,
            mem_fraction_static=0.2,
            log_level="error",
        )
        cls.addClassCleanup(cls.engine.shutdown)

    def _encode(self, tokens):
        # The HTTP /encode handler uses the same request type. Passing IDs keeps
        # the fixture independent of a downloaded tokenizer or chat template.
        generator = self.engine.tokenizer_manager.generate_request(
            EmbeddingReqInput(input_ids=tokens), None
        )
        return self.engine.loop.run_until_complete(generator.__anext__())

    def test_batched_and_individual_scores_match_transformers(self):
        outputs = self._encode(self.requests)
        actual = torch.tensor([item["embedding"] for item in outputs])
        torch.testing.assert_close(actual, self.expected, atol=0.02, rtol=0.02)
        for index, tokens in enumerate(self.requests):
            with self.subTest(request=index):
                single = self._encode(tokens)
                torch.testing.assert_close(
                    torch.tensor(single["embedding"]),
                    self.expected[index],
                    atol=0.02,
                    rtol=0.02,
                )


if __name__ == "__main__":
    unittest.main()
