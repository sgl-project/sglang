import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.lightning_attention import (
    LightningAttentionCase,
    make_lightning_cases,
    run_lightning_attention_case,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_lightning_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    run_lightning_eagle_verify_case,
    run_lightning_eagle_verify_cuda_graph_case,
)

register_cuda_ci(est_time=11, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-large-amd")


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
    # Lightning's `seg_la` kernel processes draft tokens as a chain — it
    # has no parent-indices / retrieve-index plumbing for tree-shaped
    # drafts (see `linear/seg_la.py`). Tree verify (topk>1) is therefore
    # structurally unsupported and intentionally omitted; only the
    # chain (topk=1) shape is covered. The non-EAGLE chain spec kinds
    # (frozen_kv_mtp, dflash, ngram) match the chain-only contract and
    # pass against the seg_la recurrence reference.
    EAGLE_VERIFY_CASES = (
        (
            LightningAttentionCase(
                name="runner_eagle_verify_lightning_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            LightningAttentionCase(
                name="runner_frozen_kv_mtp_verify_lightning_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "frozen_kv_mtp",
        ),
        (
            LightningAttentionCase(
                name="runner_dflash_verify_lightning_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "dflash",
        ),
        (
            LightningAttentionCase(
                name="runner_ngram_verify_lightning_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "ngram",
        ),
    )
    EAGLE_VERIFY_CUDA_GRAPH_CASES = (
        (
            LightningAttentionCase(
                name="runner_cuda_graph_eagle_verify_lightning_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
    )

    def test_projected_lightning_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_lightning_attention_case(self, case)

    # Layout-robustness. See dense/test_triton.py for the rationale.
    LAYOUT_ROBUSTNESS_CASES = (
        LightningAttentionCase(
            name="layout_lightning_extend_two_request",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_heads=2,
            page_size=16,
            prefix_lens=(0, 0),
            extend_lens=(16, 16),
        ),
        LightningAttentionCase(
            name="layout_lightning_decode_page_boundary",
            backend="triton",
            forward_mode=ForwardMode.DECODE,
            num_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )

    def test_layout_robustness_cases(self):
        for case in self.LAYOUT_ROBUSTNESS_CASES:
            for layout in ("interleaved_pages", "non_monotonic_extend"):
                if layout == "non_monotonic_extend" and case.forward_mode.is_decode():
                    continue
                with self.subTest(case=case.name, layout=layout):
                    run_lightning_attention_case(self, case, loc_layout=layout)

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_lightning_cuda_graph_decode_case(self, case)

    def test_runner_mode_eagle_verify_cases(self):
        for case, topk, spec_kind in self.EAGLE_VERIFY_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_lightning_eagle_verify_case(
                    self, case, topk=topk, spec_kind=spec_kind
                )

    def test_runner_mode_eagle_verify_cuda_graph_cases(self):
        for case, topk, spec_kind in self.EAGLE_VERIFY_CUDA_GRAPH_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_lightning_eagle_verify_cuda_graph_case(self, case, topk=topk)

    # PCG/BCG split-op extend is deliberately NOT covered. Lightning's
    # backend `forward_extend` flattens the output via `o.view(-1,
    # tp_q_head_num * v_head_dim)` (`lightning_backend.py:335`), so eager
    # forward returns flat `[T, num_heads * head_dim]`. But under
    # piecewise CG (the split-op path), `RadixAttention.forward` writes
    # through `output = torch.empty_like(q)` of per-head shape
    # `[T, num_heads, head_dim]`, ignoring the backend's intended
    # flatten. The split-op runner compares eager_actual to the
    # piecewise actual, which then trips a shape mismatch. KDA and GDN
    # avoid this because their backends keep the per-head shape on the
    # return path. Fixing requires either a Lightning-specific split-op
    # runner that reshapes actual to flat, or a Lightning backend
    # change to keep per-head shape under piecewise CG.


if __name__ == "__main__":
    unittest.main()
