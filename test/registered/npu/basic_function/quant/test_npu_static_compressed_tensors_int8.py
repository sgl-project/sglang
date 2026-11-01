"""Real static-checkpoint loading and eager/graph decoding on Ascend."""

import math
import os
import unittest
from unittest.mock import patch

import torch
from transformers import AutoTokenizer

import sglang
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=600, suite="base-b-test-1-npu-a3")
# Independently exported static activation checkpoints. Environment overrides
# allow the CI machines to point at a local cached copy of each model.
MODELS = {
    "channel": (
        "SGLANG_STATIC_INT8_CHANNEL_MODEL",
        "nm-testing/TinyLlama-1.1B-Chat-v1.0-W8A8_channel_weight_static_per_tensor-e2e",
    ),
    "tensor": (
        "SGLANG_STATIC_INT8_TENSOR_MODEL",
        "nm-testing/Qwen2.5-0.5B-W8A8_tensor_weight_static_per_tensor_act-e2e",
    ),
}
REFERENCE_PROMPTS = (
    "The capital of France is",
    "A triangle has three sides. A square has",
)
NEW_TOKENS = 8
MAX_LOGPROB_ERROR = 0.05


class TestStaticCompressedTensorsInt8(CustomTestCase):
    test_tp_size = 1

    @patch.dict(os.environ, {"SGLANG_EXTERNAL_MODEL_PACKAGE": ""})
    def test_checkpoint_eager_and_graph(self):
        for strategy, (env_key, model_id) in MODELS.items():
            for dtype in ("bfloat16", "float16"):
                with self.subTest(strategy=strategy, dtype=dtype):
                    self._run_checkpoint(os.environ.get(env_key, model_id), dtype)

    def _run_checkpoint(self, checkpoint, dtype):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        prompts = [
            tokenizer.encode(prompt, add_special_tokens=True)
            for prompt in REFERENCE_PROMPTS
        ]
        options = dict(
            dtype=dtype,
            context_length=512,
            max_total_tokens=2048,
            mem_fraction_static=0.45,
            chunked_prefill_size=-1,
            disable_overlap_schedule=True,
            disable_radix_cache=True,
            cuda_graph_max_bs_decode=4,
            max_running_requests=4,
        )
        if is_npu():
            options.update(
                device="npu",
                attention_backend="ascend",
                sampling_backend="pytorch",
            )
        target_tp = self.test_tp_size
        if target_tp > 1:
            self.assertTrue(is_npu())
            self.assertGreaterEqual(torch.npu.device_count(), target_tp)
        cases = [(1, True)]
        if target_tp > 1:
            cases.append((target_tp, True))
        cases.append((target_tp, False))
        references = None
        for tp_size, eager in cases:
            engine = sglang.Engine(
                model_path=str(checkpoint),
                tp_size=tp_size,
                disable_cuda_graph=eager,
                **options,
            )
            try:
                outputs = []
                for ids in (*prompts, prompts):
                    response = engine.generate(
                        input_ids=ids,
                        sampling_params=dict(
                            temperature=0,
                            max_new_tokens=NEW_TOKENS,
                            ignore_eos=True,
                        ),
                        return_logprob=True,
                    )
                    outputs.extend(
                        response if isinstance(response, list) else [response]
                    )
                self.assertEqual(len(outputs), 4)
                for output in outputs:
                    self.assertEqual(len(output["output_ids"]), NEW_TOKENS)
                    values = output["meta_info"]["output_token_logprobs"]
                    self.assertEqual(len(values), NEW_TOKENS)
                    self.assertTrue(all(math.isfinite(v[0]) for v in values))
                if references is None:
                    references = outputs
                else:
                    for actual, expected in zip(outputs, references):
                        self.assertEqual(actual["output_ids"], expected["output_ids"])
                        self.assertLessEqual(
                            max(
                                abs(a[0] - b[0])
                                for a, b in zip(
                                    actual["meta_info"]["output_token_logprobs"],
                                    expected["meta_info"]["output_token_logprobs"],
                                )
                            ),
                            MAX_LOGPROB_ERROR,
                        )
                if not eager:
                    states = engine.get_server_info()["internal_states"]
                    states = [states] if isinstance(states, dict) else states
                    self.assertTrue(
                        any(
                            float(
                                s.get("memory_usage", {})
                                .get("graph", {})
                                .get("decode", 0)
                            )
                            > 0
                            for s in states
                        )
                    )
            finally:
                engine.shutdown()


if __name__ == "__main__":
    unittest.main()
