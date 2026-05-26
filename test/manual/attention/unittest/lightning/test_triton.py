import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.attention_methods.lightning_attention import (
    LightningAttentionCase,
    make_lightning_cases,
    run_lightning_attention_case,
)
from common.runner_modes.cuda_graph_decode_runner import (
    run_lightning_cuda_graph_decode_case,
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTritonLightningBackendCorrectness(CustomTestCase):
    CASES = make_lightning_cases("triton")
    # Lightning installs `LightningAttentionBackend` directly via
    # `ForwardContext` (not through `HybridLinearAttnBackend`), but the
    # `MambaAttnBackendBase` capture/replay contract still applies. See
    # lightning/README.md.
    CUDA_GRAPH_CASES = (
        LightningAttentionCase(
            name="runner_cuda_graph_lightning_decode_page_boundary",
            backend="triton",
            forward_mode=ForwardMode.DECODE,
            num_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )

    def test_projected_lightning_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_lightning_attention_case(self, case)

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_lightning_cuda_graph_decode_case(self, case)


if __name__ == "__main__":
    unittest.main()
