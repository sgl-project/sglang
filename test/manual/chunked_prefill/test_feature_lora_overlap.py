"""LoRA overlap loading + chunked prefill.

LoRA overlap loading enables async adapter weight transfer. The
interaction surface with chunked prefill is the in-flight loading state
when a chunked-resume request needs an adapter that's still being loaded
— this fixture stresses that with multiple adapters available and
``--enable-lora-overlap-loading``.

Reference config sources:
  - LoRA base + adapter set: ``python/sglang/test/lora_utils.py``
    (``CI_MULTI_LORA_MODELS``), same as ``test_feature_lora.py``
  - Overlap-loading flag and unit-level coverage:
    ``test/registered/lora/test_lora_overlap_loading.py``
  - Cancellation-path bug surface: PR #25413
    ("[lora] Fix overlap loading for cancelled requests")

GPU requirement: 1 GPU (large).

Not registered with CI. Run by hand from
``test/manual/chunked_prefill/``.
"""

import unittest
from test.manual.chunked_prefill.common import ChunkedRefactorTestBase


class TestChunkedFeatureLoRAOverlap(ChunkedRefactorTestBase):
    model = "meta-llama/Llama-2-7b-hf"
    feature_args = [
        "--enable-lora",
        "--enable-lora-overlap-loading",
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
