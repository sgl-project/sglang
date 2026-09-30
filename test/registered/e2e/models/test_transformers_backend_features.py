# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

import multiprocessing as mp
import os
import unittest
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=420, stage="base-b", runner_config="1-gpu-small")


def _assert_logprobs_close(
    reference, candidate, mean_tolerance, debug_text, max_tolerance=1.0
):
    """Correct bf16 implementations differ by ~0.05 nats on average per top logprob
    and by up to ~0.5 per entry; a wrong mask or norm moves the mean by an order
    of magnitude, so the criterion is the mean overall, the mean per prompt and
    a loose per-entry ceiling."""
    per_prompt = []
    for field in ("top_input_logprobs", "top_output_logprobs"):
        for index, (expected, actual) in enumerate(
            zip(getattr(reference, field), getattr(candidate, field), strict=True)
        ):
            expected, actual = torch.tensor(expected), torch.tensor(actual)
            assert expected.shape == actual.shape, f"{debug_text}: prompt {index}"
            assert torch.isfinite(actual).all(), f"{debug_text}: prompt {index}"
            per_prompt.append((expected - actual).abs().flatten())
    diffs = torch.cat(per_prompt)
    worst_prompt = max(chunk.mean().item() for chunk in per_prompt)
    print(
        f"{debug_text}: logprob mean abs diff {diffs.mean():.4f}, "
        f"worst prompt mean {worst_prompt:.4f}, max {diffs.max():.4f}"
    )
    assert diffs.mean() < mean_tolerance, (
        f"{debug_text}: mean abs logprob diff {diffs.mean():.4f} >= {mean_tolerance}"
    )
    assert worst_prompt < 2 * mean_tolerance, (
        f"{debug_text}: a prompt has mean abs logprob diff {worst_prompt:.4f} "
        f">= {2 * mean_tolerance}"
    )
    assert diffs.max() < max_tolerance, (
        f"{debug_text}: max abs logprob diff {diffs.max():.4f} >= {max_tolerance}"
    )


def test_logprob_criterion_rejects_corrupted_outputs():
    from sglang.test.runners import ModelOutput

    torch.manual_seed(0)
    inputs = [torch.randn(12, 5) for _ in range(6)]
    outputs = [torch.randn(4, 5).tolist() for _ in range(6)]
    strs = ["x"] * 6
    reference = ModelOutput(
        output_strs=strs,
        top_input_logprobs=[x.tolist() for x in inputs],
        top_output_logprobs=outputs,
    )

    def candidate(edit=None):
        noisy = [x + 0.05 * torch.randn(12, 5) for x in inputs]
        if edit is not None:
            edit(noisy)
        return ModelOutput(
            output_strs=strs,
            top_input_logprobs=[x.tolist() for x in noisy],
            top_output_logprobs=outputs,
        )

    def shift_prompt(noisy):
        noisy[1] += 0.5

    def corrupt_entry(noisy):
        noisy[2][3, 1] += 3.0

    def shift_all(noisy):
        for x in noisy:
            x -= 1.0

    _assert_logprobs_close(reference, candidate(), 0.15, "noise only")
    with pytest.raises(AssertionError, match="a prompt has mean"):
        _assert_logprobs_close(reference, candidate(shift_prompt), 0.15, "one prompt")
    with pytest.raises(AssertionError, match="max abs logprob diff"):
        _assert_logprobs_close(reference, candidate(corrupt_entry), 0.15, "one entry")
    with pytest.raises(AssertionError, match="mean abs logprob diff"):
        _assert_logprobs_close(reference, candidate(shift_all), 0.15, "all shifted")


@unittest.skipUnless(torch.cuda.is_available(), "Requires a CUDA GPU")
class TestTransformersBackendFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def test_qwen3_native_and_transformers_fusion_ablation(self):
        from sglang.test.runners import SRTRunner, check_close_model_outputs

        prompts = [
            "The capital of France is",
            "Write the next three numbers: 2, 4, 6,",
            "A compiler translates source code into",
        ]
        outputs = {}
        forced = {}
        for implementation, fusions in (
            ("sglang", True),
            ("transformers", False),
            ("transformers", True),
        ):
            with patch.dict(
                os.environ, {"SGLANG_ENABLE_TRANSFORMERS_FUSIONS": str(fusions)}
            ):
                with SRTRunner(
                    "Qwen/Qwen3-0.6B",
                    torch_dtype=torch.bfloat16,
                    model_type="generation",
                    model_impl=implementation,
                    context_length=1024,
                    max_total_tokens=4096,
                    chunked_prefill_size=128,
                    mem_fraction_static=0.5,
                ) as runner:
                    outputs[implementation, fusions] = runner.forward(
                        prompts, max_new_tokens=16
                    )
                    reference = outputs["sglang", True]
                    forced[implementation, fusions] = runner.forward(
                        [p + o for p, o in zip(prompts, reference.output_strs)],
                        max_new_tokens=1,
                    )
        for fusions in (False, True):
            debug_text = f"Qwen3-0.6B transformers fusions={fusions}"
            check_close_model_outputs(
                hf_outputs=reference,
                srt_outputs=outputs["transformers", fusions],
                prefill_tolerance=0.0,
                decode_tolerance=0.0,
                rouge_l_tolerance=0.2,
                check_logprobs=False,
                debug_text=debug_text,
            )
            _assert_logprobs_close(
                forced["sglang", True],
                forced["transformers", fusions],
                mean_tolerance=0.15,
                debug_text=debug_text,
            )
        _assert_logprobs_close(
            forced["transformers", False],
            forced["transformers", True],
            mean_tolerance=0.15,
            debug_text="Transformers fusion ablation",
        )

    def _check_embedding(self, model_path):
        from sglang.test.runners import HFRunner, SRTRunner

        prompts = [
            "search_document: Paris is the capital of France.",
            "search_document: Paris is known for museums and architecture.",
            "search_query: What is the capital of France?",
        ]
        with HFRunner(
            model_path, torch_dtype=torch.float32, model_type="embedding"
        ) as runner:
            reference = torch.tensor(runner.forward(prompts).embed_logits)
        with SRTRunner(
            model_path,
            torch_dtype=torch.float32,
            model_type="embedding",
            model_impl="transformers",
            attention_backend="torch_native",
            context_length=512,
            max_total_tokens=2048,
            mem_fraction_static=0.5,
            chunked_prefill_size=-1,
            disable_cuda_graph=True,
        ) as runner:
            batched = torch.tensor(runner.forward(prompts).embed_logits)
            single = torch.tensor(runner.forward(prompts[:1]).embed_logits)
            reversed_batch = torch.tensor(
                runner.forward(list(reversed(prompts))).embed_logits
            )
        self.assertEqual(batched.shape, reference.shape)
        self.assertTrue(torch.isfinite(batched).all())
        torch.testing.assert_close(batched, reference, rtol=0.01, atol=0.003)
        torch.testing.assert_close(single[0], batched[0], rtol=0.005, atol=0.001)
        torch.testing.assert_close(
            reversed_batch.flip(0), batched, rtol=0.005, atol=0.001
        )
        similarities = torch.nn.functional.cosine_similarity(batched, reference, dim=-1)
        self.assertTrue(torch.all(similarities > 0.999), similarities)

    def test_bge_cls_embedding(self):
        self._check_embedding("BAAI/bge-small-en-v1.5")

    def test_minilm_mean_embedding(self):
        self._check_embedding("sentence-transformers/all-MiniLM-L6-v2")

    def test_labse_trained_projection(self):
        self._check_embedding("sentence-transformers/LaBSE")

    def test_modernbert_mean_embedding(self):
        self._check_embedding("nomic-ai/modernbert-embed-base")

    def test_bert_cross_encoder_uses_trained_head_and_sentence_segments(self):
        from transformers import AutoConfig, AutoTokenizer

        from sglang.test.runners import HFRunner, SRTRunner

        model_path = "cross-encoder/ms-marco-MiniLM-L6-v2"
        pairs = [
            ["What is the capital of France?", "Paris is the capital of France."],
            ["What is the capital of France?", "Tokyo is the capital of Japan."],
            ["Where is Tokyo?", "Tokyo is the capital of Japan."],
        ]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        encoded = tokenizer(pairs, padding=True, return_tensors="pt")
        self.assertIn("token_type_ids", encoded)
        self.assertEqual(encoded.token_type_ids.max().item(), 1)
        with HFRunner(
            model_path, torch_dtype=torch.float32, model_type="cross_encoder"
        ) as runner:
            reference = torch.tensor(runner.forward(pairs).scores).reshape(-1)
        config = AutoConfig.from_pretrained(model_path)
        activation = (
            getattr(config, "sbert_ce_default_activation_function", None) or "Identity"
        ).rsplit(".", 1)[-1]
        if activation == "Sigmoid":
            reference = reference.sigmoid()
        elif activation == "Tanh":
            reference = reference.tanh()
        else:
            self.assertEqual(activation, "Identity")
        with SRTRunner(
            model_path,
            torch_dtype=torch.float32,
            model_type="cross_encoder",
            model_impl="transformers",
            attention_backend="torch_native",
            context_length=512,
            max_total_tokens=2048,
            mem_fraction_static=0.5,
            chunked_prefill_size=-1,
            disable_cuda_graph=True,
        ) as runner:
            scores = torch.tensor(runner.forward(pairs).scores).reshape(-1)
            singleton = torch.tensor(runner.forward(pairs[:1]).scores).reshape(-1)
        self.assertEqual(scores.numel(), len(pairs))
        torch.testing.assert_close(scores, reference, rtol=0.01, atol=0.01)
        torch.testing.assert_close(singleton[0], scores[0], rtol=0.005, atol=0.002)
        self.assertGreater(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
