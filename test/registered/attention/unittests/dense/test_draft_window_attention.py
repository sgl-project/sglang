"""Multi-step EAGLE draft decode and draft extend on a sliding-window layer.

Each eager draft step's attention is checked against an independent oracle over
the keys at [pos - W, pos] (W = sliding_window_size), and the captured graph's
replay against the eager outputs. Cases straddle the window boundary, replay a
padded batch, run several draft steps, and store the layer in a plain pool or
in a static hybrid-SWA pool whose full->swa mapping moves every page, where
each step's window writes and mapped reads must land in their own slots.
Triton keeps W keys instead of W + 1 and is not covered here; the attention kit
builds no fa3 backend for a window model (it reads one as chunked local
attention).
"""

import unittest

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.common import (
    is_sm90_supported,
    is_sm100_supported,
    is_sm120_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_draft_window_runner import (
    run_dense_eagle_draft_extend_window_case,
    run_dense_eagle_draft_window_case,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=170, stage="base-b", runner_config="1-gpu-large")

WINDOW = 8


def _decode_case(backend, name, prefix_lens, *, page_size):
    return DenseAttentionCase(
        name=f"draft_window_{backend}_{name}",
        backend=backend,
        forward_mode=ForwardMode.DECODE,
        num_heads=4,
        num_kv_heads=2,
        page_size=page_size,
        prefix_lens=prefix_lens,
        sliding_window_size=WINDOW,
    )


class _DraftWindowCases:
    BACKEND = ""
    PAGE_SIZE = 16
    TREE = False
    HEAD_DIM = 64
    HIDDEN_SIZE = 256

    def _shapes(self):
        return dict(head_dim=self.HEAD_DIM, hidden_size=self.HIDDEN_SIZE)

    def test_draft_steps_read_and_write_their_window(self):
        # Below, at and past the window over three draft steps; the first
        # batch replays padded (3 of the captured 4).
        for name, prefix_lens in (
            ("padded", (3, 8, 13)),
            ("full_batch", (1, 6, 9, 21)),
        ):
            case = _decode_case(
                self.BACKEND, name, prefix_lens, page_size=self.PAGE_SIZE
            )
            for swa_pool in (False, True):
                with self.subTest(case=name, swa_pool=swa_pool):
                    run_dense_eagle_draft_window_case(
                        self,
                        case,
                        speculative_num_steps=4,
                        swa_pool=swa_pool,
                        **self._shapes(),
                    )

    def test_tree_draft_steps_read_their_branch_window(self):
        if not self.TREE:
            self.skipTest(f"{self.BACKEND} draft decode is chain-only here")
        case = _decode_case(self.BACKEND, "tree", (3, 9, 14), page_size=1)
        for swa_pool in (False, True):
            with self.subTest(swa_pool=swa_pool):
                run_dense_eagle_draft_window_case(
                    self,
                    case,
                    topk=2,
                    speculative_num_steps=3,
                    speculative_num_draft_tokens=4,
                    swa_pool=swa_pool,
                    **self._shapes(),
                )

    def test_draft_extend_rows_read_their_window(self):
        # A constant-width block of 4 whose windows reach into the prefix.
        case = DenseAttentionCase(
            name=f"draft_extend_window_{self.BACKEND}",
            backend=self.BACKEND,
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            num_heads=4,
            num_kv_heads=2,
            page_size=self.PAGE_SIZE,
            prefix_lens=(2, 9, 20),
            extend_lens=(4, 4, 4),
            sliding_window_size=WINDOW,
        )
        for swa_pool in (False, True):
            with self.subTest(swa_pool=swa_pool):
                run_dense_eagle_draft_extend_window_case(
                    self, case, swa_pool=swa_pool, **self._shapes()
                )


@unittest.skipIf(
    not torch.cuda.is_available() or not is_flashinfer_available(),
    "CUDA + flashinfer are required",
)
class TestFlashInferDraftWindowAttention(_DraftWindowCases, CustomTestCase):
    BACKEND = "flashinfer"
    TREE = True


@unittest.skipIf(
    not torch.cuda.is_available()
    or not is_flashinfer_available()
    or not (is_sm90_supported() or is_sm100_supported() or is_sm120_supported()),
    "CUDA + FlashInfer TRT-LLM MHA decode support are required",
)
class TestTRTLLMMHADraftWindowAttention(_DraftWindowCases, CustomTestCase):
    BACKEND = "trtllm_mha"


if __name__ == "__main__":
    unittest.main()
