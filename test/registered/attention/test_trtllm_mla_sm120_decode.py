"""TRTLLM MLA decode on SM120/SM121, where FlashInfer runs it on the XQA kernel.

Decode must not pass the trtllm-gen multi-CTA KV counter, which XQA rejects
during CUDA graph capture; TRTLLM MLA prefill must be refused at startup.
"""

import json
import unittest

import requests

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=130, stage="base-b", runner_config="1-gpu-small")

# XQA MLA only supports 128 query heads per rank; two layers keep startup short.
MODEL_OVERRIDE_ARGS = json.dumps({"num_hidden_layers": 2, "num_attention_heads": 128})
MAX_NEW_TOKENS = 8


@unittest.skipUnless(is_sm120_supported(), "requires an SM120/SM121 GPU")
class TestTrtllmMlaSm120Decode(CustomTestCase):
    def test_decode_with_cuda_graph(self):
        for kv_cache_dtype in ("bf16", "fp8_e4m3"):
            with self.subTest(kv_cache_dtype=kv_cache_dtype):
                process = popen_launch_server(
                    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
                    DEFAULT_URL_FOR_TEST,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=[
                        "--load-format",
                        "dummy",
                        "--json-model-override-args",
                        MODEL_OVERRIDE_ARGS,
                        "--prefill-attention-backend",
                        "flashinfer",
                        "--decode-attention-backend",
                        "trtllm_mla",
                        "--kv-cache-dtype",
                        kv_cache_dtype,
                        "--cuda-graph-backend-decode",
                        "full",
                        "--cuda-graph-max-bs-decode",
                        "4",
                        "--max-total-tokens",
                        "4096",
                    ],
                )
                try:
                    response = requests.post(
                        DEFAULT_URL_FOR_TEST + "/generate",
                        json={
                            "text": ["The capital of France is", "1 2 3 4 5 6 7"],
                            "sampling_params": {
                                "max_new_tokens": MAX_NEW_TOKENS,
                                "temperature": 0,
                                "ignore_eos": True,
                            },
                        },
                        timeout=120,
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    outputs = response.json()
                    self.assertEqual(len(outputs), 2)
                    for output in outputs:
                        self.assertEqual(
                            output["meta_info"]["completion_tokens"], MAX_NEW_TOKENS
                        )
                finally:
                    terminate_and_kill_process_tree(process)

    def test_trtllm_mla_prefill_rejected(self):
        server_args = prepare_server_args(
            [
                "--model-path",
                DEFAULT_MLA_MODEL_NAME_FOR_TEST,
                "--json-model-override-args",
                MODEL_OVERRIDE_ARGS,
                "--attention-backend",
                "trtllm_mla",
            ]
        )
        with self.assertRaisesRegex(ValueError, "TRTLLM MLA prefill requires SM10x"):
            server_args.resolve_once()


if __name__ == "__main__":
    unittest.main()
