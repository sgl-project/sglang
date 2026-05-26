"""Feature i — LoRA × chunked: naive ScriptedRuntime smoke.

Reproduces the path that 5ed4faf0ab "Bypass LoRA scheduling gate for
chunked-resume reqs" had to fix: a LoRA request must be able to
chunk-prefill across multiple iterations without the LoRA drainer
deadlock that previously kept it stuck in waiting_queue.

This is a *smoke* — it doesn't deliberately trigger the drainer. The
deliberate drainer-deadlock regression lives in
``test_regression_309b6dc.py`` (B-60).
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.test_utils import CustomTestCase

from test.manual.scripted_runtime.common import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until_finished,
)


_LORA_BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
_LORA_ADAPTER = "philschmid/llama-3-2-1b-instruct-finetuning-lora-cookbook-test"


def _script_naive_lora_chunked(t: ScriptedRuntime):
    # When ``start_req`` learns ``lora_path=`` (wishlist §4 P2 (10)
    # adjacent — same kwarg-routing pattern as ``priority``), this
    # script will route to the adapter; until then it exercises the
    # chunked path with LoRA enabled at engine level.
    r = t.start_req(
        prompt_len=VERY_LONG_PROMPT_LEN,
        max_new_tokens=4,
        lora_path=_LORA_ADAPTER,
    )
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done >= 2


class TestFeatureILoRAChunked(CustomTestCase):
    def test_naive_lora_chunked(self):
        execute_scripted_runtime(
            _script_naive_lora_chunked,
            **base_engine_kwargs(
                model_path=_LORA_BASE_MODEL,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_lora=True,
                lora_paths=[_LORA_ADAPTER],
            ),
        )


if __name__ == "__main__":
    unittest.main()
