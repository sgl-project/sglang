"""LoRA overlap loading × chunked: naive ScriptedRuntime smoke.

Adapter H2D copy runs concurrently with the chunked prefill loop. We
want the chunked-resume path to make progress even while LoRA H2D is
in flight, and not double-fire when an abort hits during H2D (issue
#25413).

The naive smoke just verifies the engine starts and completes a
chunked LoRA request with overlap loading enabled.
"""

import unittest
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until_finished,
)

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.test_utils import CustomTestCase

_LORA_BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
_LORA_ADAPTER = "philschmid/llama-3-2-1b-instruct-finetuning-lora-cookbook-test"


def _script_naive_lora_overlap_chunked(t: ScriptedRuntime):
    r = t.start_req(
        prompt_len=VERY_LONG_PROMPT_LEN,
        max_new_tokens=4,
        lora_path=_LORA_ADAPTER,
    )
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done >= 2


class TestFeatureLoRAOverlapChunked(CustomTestCase):
    def test_naive_lora_overlap_chunked(self):
        execute_scripted_runtime(
            _script_naive_lora_overlap_chunked,
            **base_engine_kwargs(
                model_path=_LORA_BASE_MODEL,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_lora=True,
                lora_paths=[_LORA_ADAPTER],
                enable_lora_overlap_loading=True,
            ),
        )


if __name__ == "__main__":
    unittest.main()
