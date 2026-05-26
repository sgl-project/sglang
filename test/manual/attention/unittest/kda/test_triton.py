import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.attention_methods.kda_attention import (
    KDAAttentionCase,
    make_kda_cases,
    run_kda_attention_case,
)
from common.runner_modes.cuda_graph_decode_runner import (
    run_kda_cuda_graph_decode_case,
)
from common.runner_modes.speculative_target_verify_runner import (
    run_kda_eagle_verify_case,
    run_kda_eagle_verify_cuda_graph_case,
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTritonKDABackendCorrectness(CustomTestCase):
    CASES = make_kda_cases("triton")
    # KDA inherits the same `MambaAttnBackendBase` capture/replay path as GDN
    # through `HybridLinearAttnBackend`. See kda/README.md.
    CUDA_GRAPH_CASES = (
        KDAAttentionCase(
            name="runner_cuda_graph_kda_decode_page_boundary",
            backend="triton",
            forward_mode=ForwardMode.DECODE,
            num_k_heads=2,
            num_v_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )
    EAGLE_VERIFY_CASES = (
        (
            KDAAttentionCase(
                name="runner_eagle_verify_kda_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
        ),
        (
            KDAAttentionCase(
                name="runner_eagle_verify_kda_tree",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(5, 6),
                extend_lens=(3, 3),
            ),
            2,
        ),
    )
    EAGLE_VERIFY_CUDA_GRAPH_CASES = (
        (
            KDAAttentionCase(
                name="runner_cuda_graph_eagle_verify_kda_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
        ),
        (
            KDAAttentionCase(
                name="runner_cuda_graph_eagle_verify_kda_tree",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(5, 6),
                extend_lens=(3, 3),
            ),
            2,
        ),
    )

    def test_projected_kda_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_kda_attention_case(self, case)

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_kda_cuda_graph_decode_case(self, case)

    def test_runner_mode_eagle_verify_cases(self):
        for case, topk in self.EAGLE_VERIFY_CASES:
            with self.subTest(case=case.name, backend=case.backend, topk=topk):
                run_kda_eagle_verify_case(self, case, topk=topk)

    def test_runner_mode_eagle_verify_cuda_graph_cases(self):
        for case, topk in self.EAGLE_VERIFY_CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend, topk=topk):
                run_kda_eagle_verify_cuda_graph_case(self, case, topk=topk)


if __name__ == "__main__":
    unittest.main()
