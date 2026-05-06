"""End-to-end smoke test for Quest WITHOUT hisparse offloading (Mode 2).

  --hisparse-config '{"algorithm": "quest", ...}'   (NO --enable-hisparse)
  --attention-backend flashinfer_quest
  --disable-radix-cache
  --kv-cache-dtype bfloat16

Verifies:
  * server launches with the new ``flashinfer_quest`` backend,
  * generation completes,
  * output is non-empty.
"""

import os
import unittest
from pathlib import Path

import torch

import sglang as sgl
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu


SMALL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _model_in_hf_cache(model_id: str) -> bool:
    cache = Path(
        os.environ.get(
            "HF_HOME",
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
        )
    ) / "hub"
    if not cache.exists():
        return False
    safe = "models--" + model_id.replace("/", "--")
    return (cache / safe).exists()


@unittest.skipUnless(
    torch.cuda.is_available()
    and not is_npu()
    and not is_xpu()
    and (is_cuda() or is_hip()),
    "Quest e2e requires CUDA/ROCm.",
)
@unittest.skipUnless(
    _model_in_hf_cache(SMALL_MODEL),
    f"{SMALL_MODEL} not in local HF cache.",
)
class TestQuestOnlyEngineE2E(unittest.TestCase):
    """Mode 2 (quest-only, no hisparse offloading) launches + generates."""

    def test_quest_only_engine_generates(self):
        hisparse_config = (
            '{"algorithm": "quest", "top_k": 256, "quest_page_size": 64}'
        )
        engine = sgl.Engine(
            model_path=SMALL_MODEL,
            # Mode 2 does NOT set enable_hisparse.
            hisparse_config=hisparse_config,
            attention_backend="flashinfer_quest",
            disable_radix_cache=True,
            kv_cache_dtype="bfloat16",
            disable_cuda_graph=True,
            max_total_tokens=8192,
            max_running_requests=4,
            mem_fraction_static=0.5,
            log_level="warning",
        )
        try:
            output = engine.generate(
                prompt="The capital of France is",
                sampling_params={"max_new_tokens": 8, "temperature": 0.0},
            )
            self.assertIsInstance(output, dict)
            text = output.get("text", "")
            self.assertIsInstance(text, str)
            self.assertGreater(
                len(text), 0,
                f"flashinfer_quest produced empty text; output={output}",
            )
            meta = output.get("meta_info", {})
            self.assertGreater(meta.get("completion_tokens", 0), 0)
        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
