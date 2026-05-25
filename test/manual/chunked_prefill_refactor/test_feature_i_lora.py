"""Feature (i): LoRA + chunked prefill.

Manual fixture for the chunked-prefill refactor accuracy net. The
LoRA-drainer × chunked-resume deadlock fixed in commit 5ed4faf0ab lives
here — when an adapter enters the drainer mid-chunk, the chunked-resume
admission gate must skip the LoRA check or the request stalls forever
while holding KV. Mixed-prefix gsm8k provides varied chunk patterns;
mode 2's per-question unique prefix is the most likely to trigger
mid-chunk admission decisions.

LoRA setup borrowed from
``python/sglang/test/lora_utils.py::CI_MULTI_LORA_MODELS``.

GPU requirement: 1 GPU (large; Llama-2-7b-hf + 2 LoRA adapters).

Not registered with CI. Run by hand from
``test/manual/chunked_prefill_refactor/``.
"""

import unittest

from test.manual.chunked_prefill_refactor.common import ChunkedRefactorTestBase


class TestChunkedFeatureI_LoRA(ChunkedRefactorTestBase):
    model = "meta-llama/Llama-2-7b-hf"
    feature_args = [
        "--enable-lora",
        "--lora-paths",
        "winddude/wizardLM-LlaMA-LoRA-7B",
        "RuterNorway/Llama-2-7b-chat-norwegian-LoRa",
        "--max-loras-per-batch",
        "2",
        "--max-loaded-loras",
        "4",
        "--mem-fraction-static",
        "0.75",
    ]


if __name__ == "__main__":
    unittest.main()
