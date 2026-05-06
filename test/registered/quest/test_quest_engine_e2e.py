"""End-to-end smoke test for Quest + HiSparse + FlashInfer.

Loads a small MHA model into ``sgl.Engine`` with the production wiring:

  --enable-hisparse
  --hisparse-config '{"algorithm":"quest", ...}'
  --disable-radix-cache
  --kv-cache-dtype bfloat16
  --prefill-attention-backend flashinfer
  --decode-attention-backend flashinfer_hisparse

Then runs a small generation and verifies the engine doesn't crash and
produces a non-empty output.

Skipped if the small model isn't already in the local HuggingFace cache —
this test isn't supposed to download from the internet.
"""

import os
import unittest
from pathlib import Path

import torch

import sglang as sgl
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu


SMALL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _model_in_hf_cache(model_id: str) -> bool:
    """Return True iff ``model_id`` appears in the local HF hub cache."""
    cache = Path(
        os.environ.get(
            "HF_HOME",
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
        )
    ) / "hub"
    if not cache.exists():
        return False
    # HF format: models--{org}--{name}
    safe = "models--" + model_id.replace("/", "--")
    return (cache / safe).exists()


@unittest.skipUnless(
    torch.cuda.is_available()
    and not is_npu()
    and not is_xpu()
    and (is_cuda() or is_hip()),
    "Quest e2e test requires CUDA/ROCm.",
)
@unittest.skipUnless(
    _model_in_hf_cache(SMALL_MODEL),
    f"{SMALL_MODEL} not in local HF cache; skip to avoid network download.",
)
class TestQuestEngineE2E(unittest.TestCase):
    """End-to-end smoke: load + generate + shutdown without errors."""

    def test_quest_engine_generates_text(self):
        # Quest config kept small to fit on test GPUs.  top_k must divide
        # cleanly into quest_page_size (= 64 default).
        hisparse_config = (
            '{"algorithm": "quest", "top_k": 256, '
            '"device_buffer_size": 1024, "host_to_device_ratio": 2}'
        )
        engine = sgl.Engine(
            model_path=SMALL_MODEL,
            enable_hisparse=True,
            hisparse_config=hisparse_config,
            disable_radix_cache=True,
            kv_cache_dtype="bfloat16",
            prefill_attention_backend="flashinfer",
            decode_attention_backend="flashinfer_hisparse",
            # Skip cuda graph capture for the v1 e2e — keeps the test simpler
            # while still exercising the full quest+hisparse+flashinfer path
            # in eager mode.
            disable_cuda_graph=True,
            # The MHA hisparse pool inflates physical storage to
            # ``size * (host_to_device_ratio + 1)`` (hot buffer + logical
            # rows), so we tightly bound max_total_tokens to keep the e2e
            # smoke within reasonable GPU memory.
            max_total_tokens=8192,
            max_running_requests=4,
            mem_fraction_static=0.3,
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
                f"Quest engine produced empty text for the prompt; output={output}",
            )
            # Sanity: also check meta_info has token counts.
            meta = output.get("meta_info", {})
            self.assertGreater(meta.get("completion_tokens", 0), 0)
        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
