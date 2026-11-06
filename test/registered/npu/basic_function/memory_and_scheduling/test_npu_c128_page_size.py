import os
import unittest
from threading import Lock
from unittest.mock import patch

from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.simple_eval_common import ChatCompletionSampler
from sglang.test.simple_eval_mixed_prefix_gsm8k import GSM8KEval, get_one_example
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(suite="full-8-npu-a3")

# DeepSeek-V4-Flash W8A8 requires the same NPU runtime environment as the
# existing accuracy/performance DSV4 cases.
DEEPSEEK_V4_FLASH_W8A8_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "DEEPEP_HCCL_BUFFSIZE": "2048",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "35",
    "DEEPEP_HYBRID_DEPLOYMENT": "1",
    "SGLANG_ENABLE_WAR_BARRIER": "1",
    "SGLANG_FORCE_COARSE_WAR_BARRIER": "1",
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "False",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
}

# A C128 physical page holds 16 compressed tokens, or 2048 original tokens.
C128_PAGE_SIZE = 16
PHYSICAL_PAGE_BOUNDARY_TOKENS = 128 * (C128_PAGE_SIZE + 1)
TARGET_PREFIX_TOKENS = 3072
PADDING_EXAMPLE = (
    "Question: Mia has 4 apples and buys 3 more. How many apples does she have?\n"
    "Answer: 4 + 3 = 7. #### 7\n\n"
)

DEEPSEEK_V4_FLASH_W8A8_OTHER_ARGS = [
    "--page-size",
    "128",
    "--tp-size",
    "8",
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "dsv4",
    "--mem-fraction-static",
    "0.70",
    "--max-running-requests",
    "16",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
    "--kv-cache-dtype",
    "bfloat16",
    "--c128-page-size",
    str(C128_PAGE_SIZE),
    "--context-length",
    "8192",
    "--disable-radix-cache",
    "--skip-server-warmup",
]


class TestNpuC128PageSize(CustomTestCase):
    """Testcase: Verify the NPU DSV4 C128 KV cache physical page size parameter
    (--c128-page-size) supports inference across a C128 physical page boundary.

    [Test Category] Parameter
    [Test Target] --c128-page-size
    """

    model = DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        env = os.environ.copy()
        env.update(DEEPSEEK_V4_FLASH_W8A8_ENVS)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            other_args=DEEPSEEK_V4_FLASH_W8A8_OTHER_ARGS,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_c128_page_size_server_and_inference(self):
        eval_obj = GSM8KEval(num_examples=100, num_threads=16, num_shots=20)
        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        # Keep the shared prefix before sending requests so every question still crosses
        # the 2176-token bound.
        padding_examples = 0
        prefix_tokens = len(
            tokenizer.encode(eval_obj._few_shot_prompt, add_special_tokens=False)
        )
        while prefix_tokens < TARGET_PREFIX_TOKENS:
            eval_obj._few_shot_prompt += PADDING_EXAMPLE
            padding_examples += 1
            next_prefix_tokens = len(
                tokenizer.encode(eval_obj._few_shot_prompt, add_special_tokens=False)
            )
            prefix_tokens = next_prefix_tokens

        min_encoded_prompt = min(
            len(
                tokenizer.encode(
                    eval_obj._few_shot_prompt
                    + get_one_example(eval_obj._lines, i, include_answer=False),
                    add_special_tokens=False,
                )
            )
            for i in range(100)
        )
        self.assertGreater(min_encoded_prompt, PHYSICAL_PAGE_BOUNDARY_TOKENS)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "EMPTY"}):
            sampler = ChatCompletionSampler(
                base_url=f"{self.base_url}/v1",
                model=self.model,
                max_tokens=4096,
                temperature=0.0,
                extra_body={
                    "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}
                },
            )

        # Record the server's token count for each request. A C128 physical page
        # holds 16 compressed tokens, each formed from 128 original tokens; the
        # 17th compressed token (at 2176 tokens) must use a second page.
        prompt_tokens = []
        lock = Lock()
        create = sampler.client.chat.completions.create

        def record_prompt_tokens(*args, **kwargs):
            result = create(*args, **kwargs)
            if result.usage is not None and result.usage.prompt_tokens is not None:
                with lock:
                    prompt_tokens.append(result.usage.prompt_tokens)
            return result

        sampler.client.chat.completions.create = record_prompt_tokens
        result = eval_obj(sampler)
        self.assertEqual(len(prompt_tokens), 100)
        self.assertGreater(min(prompt_tokens), PHYSICAL_PAGE_BOUNDARY_TOKENS)
        self.assertGreaterEqual(result.score, 0.93)


if __name__ == "__main__":
    unittest.main()
