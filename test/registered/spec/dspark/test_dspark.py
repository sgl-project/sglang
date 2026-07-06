"""GPU server-integration test for DSpark speculative decoding on a
Qwen3-family target (extends the DeepSeek-V4-only baseline from #29538).
Mirrors test/registered/spec/dflash/test_dflash.py's structure: a single
shared server, CustomTestCase + MatchedStopMixin + GSM8KMixin.

DSpark always forces synchronous (non-overlap) scheduling internally
(``_handle_dspark`` in arg_groups/speculative_hook.py sets
``disable_overlap_schedule=True`` unconditionally), so unlike DFlash there is
no overlap/no-overlap variant to parametrize here.
"""

import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_DSPARK_QWEN3,
    DEFAULT_TARGET_MODEL_DSPARK_QWEN3,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")


class TestDSparkServerBase(CustomTestCase, MatchedStopMixin, GSM8KMixin):
    model = DEFAULT_TARGET_MODEL_DSPARK_QWEN3
    draft_model = DEFAULT_DRAFT_MODEL_DSPARK_QWEN3
    attention_backend = "triton"
    dspark_block_size = 7
    mem_fraction_static = 0.85
    # Measured on a single RTX A6000 48GB SM86, bf16, --attention-backend
    # triton: GSM8K-200 5-shot greedy accuracy 0.875-0.885 across runs
    # (non-speculative baseline 0.880; binomial noise at n=200 ~= 0.023),
    # accept length 3.94-4.84 during GSM8K. Thresholds below sit at a safety
    # margin under the measured floor, matching the dflash precedent's
    # below-measured-floor convention (test_dflash.py: 0.75 acc / 2.8 accept
    # against a "roughly 3.2-3.6" documented accept length).
    gsm8k_accuracy_thres = 0.84
    gsm8k_accept_length_thres = 3.0

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                cls.attention_backend,
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                cls.draft_model,
                "--speculative-dspark-block-size",
                str(cls.dspark_block_size),
                "--mem-fraction-static",
                str(cls.mem_fraction_static),
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_early_stop(self):
        client = openai.Client(base_url=self.base_url + "/v1", api_key="EMPTY")
        for i in range(8):
            max_tokens = (i % 3) + 1
            response = client.completions.create(
                model=self.model,
                prompt=f"There are {i} apples on the table. How to divide them equally?",
                max_tokens=max_tokens,
                temperature=0,
            )
            text = response.choices[0].text
            print(f"early_stop: max_tokens={max_tokens}, text={text!r}")
        assert self.process.poll() is None

    def test_finish_stop_eos(self):
        # Overrides MatchedStopMixin.test_finish_stop_eos: the mixin's version
        # hardcodes a Llama-3 chat prompt and Llama-3 special-token ids
        # (128000/128009/2), inherited from the dflash precedent's Llama-3.1
        # target. Qwen3's chat-template special tokens and EOS ids differ
        # entirely, so the base version fails on any Qwen3 target regardless
        # of DSpark. Verified directly against this server: both the raw
        # completions endpoint (manually-templated prompt) and the chat
        # endpoint stop with matched_stop=151645 (``<|im_end|>``) well inside
        # 1000 tokens even with Qwen3's default thinking mode on;
        # generation_config.json also lists 151643 (``<|endoftext|>``) as a
        # valid EOS id, so both are accepted here.
        qwen_format_prompt = """\
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2 + 2?<|im_end|>
<|im_start|>assistant
"""
        eos_token_ids = [151643, 151645]
        self._run_completions_generation(
            prompt=qwen_format_prompt,
            max_tokens=1000,
            finish_reason="stop",
            matched_stop=eos_token_ids,
        )
        self._run_chat_completions_generation(
            prompt="What is 2 + 2?",
            max_tokens=1000,
            finish_reason="stop",
            matched_stop=eos_token_ids,
        )

    def test_eos_handling(self):
        client = openai.Client(base_url=self.base_url + "/v1", api_key="EMPTY")
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "Today is a sunny day and I like"}],
            max_tokens=256,
            temperature=0.1,
        )
        text = response.choices[0].message.content
        print(f"eos_handling: text={text!r}")
        self.assertNotIn("<|im_end|>", text)
        self.assertNotIn("<|endoftext|>", text)
        assert self.process.poll() is None

    def test_greedy_determinism(self):
        client = openai.Client(base_url=self.base_url + "/v1", api_key="EMPTY")
        prompt = "The capital of France is"
        outputs = []
        for _ in range(2):
            response = client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=32,
                temperature=0,
            )
            outputs.append(response.choices[0].text)
        print(f"determinism: {outputs=}")
        self.assertEqual(outputs[0], outputs[1])
        assert self.process.poll() is None


if __name__ == "__main__":
    unittest.main()
