"""Unit tests for LoRA batch-info preparation under TARGET_VERIFY.

TARGET_VERIFY reports is_extend() True but ForwardBatch.init_new leaves
extend_seq_lens / extend_seq_lens_cpu as None (verify is routed through the
decode-style positions branch), while every request carries a uniform
spec_info.draft_token_num-token segment. Regression guards for the paths that
assumed extend fields exist on every is_extend() mode:

- triton eager path: ``max(None)`` TypeError, and 1-token/req segments for a
  draft_token_num-token layout (silent adapter mis-segmentation).
- MoE _add_moe_lora_info: ``sum(None)`` TypeError.
- static cuda-graph path: seg_lens are pre-filled at the captured width; a
  verify batch of another width must fail loudly, not mis-segment.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


def _verify_batch(bs: int, draft_token_num: int) -> SimpleNamespace:
    return SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=bs,
        spec_info=SimpleNamespace(draft_token_num=draft_token_num),
        extend_seq_lens=None,
        extend_seq_lens_cpu=None,
    )


class TestLoRASpecVerifyBatchInfo(CustomTestCase):
    def _backend(self, max_loras_per_batch: int = 2) -> TritonLoRABackend:
        return TritonLoRABackend(
            max_loras_per_batch=max_loras_per_batch, device=torch.device("cuda")
        )

    def _prepare(self, backend, forward_batch, use_cuda_graph: bool):
        backend.prepare_lora_batch(
            forward_batch,
            weight_indices=[0, 1],
            lora_ranks=[8, 8],
            scalings=[1.0, 1.0],
            use_cuda_graph=use_cuda_graph,
        )
        return backend.batch_info

    def test_eager_target_verify_builds_uniform_draft_width_segments(self):
        backend = self._backend()
        batch_info = self._prepare(
            backend, _verify_batch(bs=2, draft_token_num=4), use_cuda_graph=False
        )
        self.assertEqual(batch_info.max_len, 4)
        self.assertEqual(batch_info.seg_lens[:2].tolist(), [4, 4])
        self.assertEqual(batch_info.seg_indptr[:3].tolist(), [0, 4, 8])
        self.assertEqual(batch_info.num_segments, 2)

    def test_graph_path_serves_the_captured_width_and_rejects_others(self):
        """prepare_lora_batch predicts graph use before can_run_graph decides,
        so a mismatched width used to silently apply the captured segment
        layout to a differently-shaped batch (wrong adapter on wrong rows).
        The matching case also pins that bs is rebound per batch, which is
        what makes a mixed-adapter batch index the right slots."""
        backend = self._backend()
        backend.init_cuda_graph_batch_info(max_bs_in_cuda_graph=4, num_tokens_per_req=4)

        batch_info = self._prepare(
            backend, _verify_batch(bs=2, draft_token_num=4), use_cuda_graph=True
        )
        self.assertIs(batch_info, backend.cuda_graph_batch_info)
        self.assertEqual(batch_info.seg_lens.tolist(), [4, 4, 4, 4])
        self.assertEqual(batch_info.bs, 2)

        with self.assertRaisesRegex(AssertionError, "width"):
            self._prepare(
                backend, _verify_batch(bs=2, draft_token_num=8), use_cuda_graph=True
            )

    def test_moe_lora_info_uses_draft_width_token_counts_for_verify(self):
        backend = self._backend()
        backend.is_moe_lora = True
        captured = {}

        def _capture(num_tokens, seg_indptr, lora_ranks, req_to_lora, *args, **kwargs):
            captured["num_tokens"] = num_tokens
            captured["max_len"] = kwargs["max_len"]
            return None, None

        with patch(
            "sglang.srt.lora.backend.base_backend._compute_moe_lora_info",
            side_effect=_capture,
        ):
            self._prepare(
                backend, _verify_batch(bs=2, draft_token_num=4), use_cuda_graph=False
            )
        self.assertEqual(captured["num_tokens"], 8)
        self.assertEqual(captured["max_len"], 4)


if __name__ == "__main__":
    unittest.main()
