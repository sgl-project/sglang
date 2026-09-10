"""Real-checkpoint parity through both HTTP frontends, tokenization and inference.

Run with: python test/registered/openai_server/basic/test_deepseek_v4_rust.py
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=900, stage="base-c", runner_config="8-gpu-h200")


def capture_prompt(config):
    """Observe actual model inputs through the existing forward-hook interface."""
    import torch.distributed as dist

    def hook(module, args, output):
        batch = args[3]
        if dist.get_rank() == 0 and batch.forward_mode.is_extend():
            count = sum(batch.extend_seq_lens_cpu)
            with open(config["path"], "a") as output_file:
                output_file.write(json.dumps(args[0][:count].tolist()) + "\n")

    return hook


def chat_cases():
    user = {"role": "user", "content": "What is the capital of France?"}
    yield "default", {"messages": [user]}
    yield "system", {"messages": [{"role": "system", "content": "Be concise."}, user]}
    yield "text_parts", {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the capital"},
                    {"type": "text", "text": "of France?"},
                ],
            }
        ]
    }
    for thinking in (False, True):
        for effort in (None, "low", "high", "max", "medium"):
            yield f"thinking={thinking},effort={effort}", {
                "messages": [user],
                "chat_template_kwargs": {"thinking": thinking},
                "reasoning_effort": effort,
            }
    yield "reasoning_history", {
        "messages": [
            {"role": "user", "content": "What is 1+1?"},
            {
                "role": "assistant",
                "content": "2",
                "reasoning_content": "Add one and one.",
            },
            user,
        ],
        "chat_template_kwargs": {"thinking": True},
    }
    yield "effort_kwarg_overrides_top_level", {
        "messages": [user],
        "reasoning_effort": "low",
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"},
    }
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
    yield "tools", {"messages": [user], "tools": [tool], "tool_choice": "none"}
    yield "tool_history", {
        "tools": [tool],
        "tool_choice": "none",
        "chat_template_kwargs": {"thinking": True},
        "messages": [
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Look up the weather.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny."},
            {"role": "user", "content": "Summarize the result."},
        ],
    }


class TestDeepSeekV4RustParity(CustomTestCase):
    def collect(self, model, rust, prompt_file):
        process = popen_launch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={
                "PYTHONPATH": os.pathsep.join(
                    [str(Path(__file__).parent), os.environ.get("PYTHONPATH", "")]
                ),
                "SGLANG_RUST_SERVER": str(int(rust)),
                "SGLANG_DEFAULT_THINKING": "0",
                "SGLANG_DSV4_REASONING_EFFORT": "",
                "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
            },
            other_args=[
                "--forward-hooks",
                json.dumps(
                    [
                        {
                            "target_modules": ["logits_processor"],
                            "hook_factory": f"{Path(__file__).stem}:capture_prompt",
                            "config": {"path": str(prompt_file)},
                        }
                    ]
                ),
                "--tp",
                "2",
                "--moe-runner-backend",
                "marlin",
                "--context-length",
                "8192",
                "--chunked-prefill-size",
                "512",
                "--max-running-requests",
                "1",
                "--mem-fraction-static",
                "0.8",
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--disable-radix-cache",
                "--random-seed",
                "42",
            ],
        )
        try:
            results = {}
            for name, body in chat_cases():
                prompt_file.write_text("")
                response = requests.post(
                    DEFAULT_URL_FOR_TEST + "/v1/chat/completions",
                    json={
                        "model": model,
                        "temperature": 0,
                        "max_tokens": 8,
                        **body,
                    },
                    timeout=180,
                )
                self.assertEqual(response.status_code, 200, (name, response.text))
                result = response.json()
                # Chunked prefill can contribute multiple model-forward inputs.
                input_ids = [
                    token
                    for line in prompt_file.read_text().splitlines()
                    for token in json.loads(line)
                ]
                self.assertTrue(input_ids, name)
                self.assertEqual(len(input_ids), result["usage"]["prompt_tokens"], name)
                self.assertGreater(result["usage"]["completion_tokens"], 0, name)
                choice = result["choices"][0]
                self.assertTrue(choice["message"]["content"], name)
                self.assertIn(choice["finish_reason"], ("stop", "length"), name)
                results[name] = input_ids
            return results
        finally:
            kill_process_tree(process.pid)
            process.wait()

    def test_chat_parity(self):
        # Quantized inference can differ even for repeated identical requests.
        # Assert exact token IDs entering the model, and successful generation
        # through both complete HTTP paths, without comparing unstable logits.
        with tempfile.TemporaryDirectory() as directory:
            prompt_file = Path(directory) / "prompts.jsonl"
            for model in (
                "deepseek-ai/DeepSeek-V4-Flash",
                "deepseek-ai/DeepSeek-V4-Flash-0731",
            ):
                python = self.collect(model, rust=False, prompt_file=prompt_file)
                rust = self.collect(model, rust=True, prompt_file=prompt_file)
                for name in python:
                    with self.subTest(model=model, case=name):
                        self.assertEqual(rust[name], python[name])


if __name__ == "__main__":
    unittest.main()
