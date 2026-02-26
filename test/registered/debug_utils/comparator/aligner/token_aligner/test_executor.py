from __future__ import annotations

import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.token_aligner.executor import (
    execute_token_aligner,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.planner import (
    compute_token_aligner_plan,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.seq_info_builder import (
    build_seqs_info,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    SGLangSeqId,
    TokenAlignerGlobalAux,
    TokenAlignerPlan,
    TokenAlignerStepAux,
    TokenLocator,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="default", nightly=True)


class TestExecuteAlignment:
    """Tests for token alignment execution (single-step)."""

    def test_thd_vs_thd_identity(self):
        """Two identical thd sides produce element-wise equal aligned tensors."""
        torch.manual_seed(42)
        hidden = torch.randn(5, 8)  # 5 tokens, hidden_dim=8

        aux = TokenAlignerStepAux(
            input_ids=[10, 20, 30, 40, 50],
            positions=[0, 1, 2, 0, 1],
            seq_lens=[3, 2],
            seq_ids=[SGLangSeqId(rid="A"), SGLangSeqId(rid="B")],
        )

        side_aux = TokenAlignerGlobalAux(
            step_auxs={0: aux},
            framework="sglang",
            layout="thd",
        )

        index = build_seqs_info(side_aux)
        plan = compute_token_aligner_plan(seqs_info_pair=Pair(x=index, y=index))

        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan, tensor_pair=Pair(x=hidden, y=hidden)
        )

        assert torch.equal(aligned.x, aligned.y)
        assert aligned.x.shape[0] == len(plan.locators.x.token_index_in_step)

    def test_zero_matched_tokens(self):
        """Empty TokenAlignerPlan (no matched tokens) returns shape[0]==0 without crash."""
        torch.manual_seed(42)

        plan = TokenAlignerPlan(
            locators=Pair(
                x=TokenLocator(token_index_in_step=[]),
                y=TokenLocator(token_index_in_step=[]),
            ),
        )

        tensor = torch.randn(5, 8)
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan, tensor_pair=Pair(x=tensor, y=tensor)
        )

        assert aligned.x.shape[0] == 0
        assert aligned.y.shape[0] == 0
        assert aligned.x.shape[1:] == (8,)
        assert aligned.y.shape[1:] == (8,)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
