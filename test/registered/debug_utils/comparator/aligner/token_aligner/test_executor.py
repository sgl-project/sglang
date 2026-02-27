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
from sglang.srt.debug_utils.comparator.dims import TokenLayout
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="default", nightly=True)


def _named(tensor: torch.Tensor, names: list[str]) -> torch.Tensor:
    return tensor.refine_names(*names)


class TestExecuteAlignment:
    """Tests for token alignment execution."""

    def test_thd_vs_thd_identity(self):
        """Two identical thd sides produce element-wise equal aligned tensors."""
        torch.manual_seed(42)
        hidden_step0 = torch.randn(5, 8).refine_names("t", "h")
        hidden_step1 = torch.randn(2, 8).refine_names("t", "h")

        aux = TokenAlignerStepAux(
            input_ids=[10, 20, 30, 40, 50],
            positions=[0, 1, 2, 0, 1],
            seq_lens=[3, 2],
            seq_ids=[SGLangSeqId(rid="A"), SGLangSeqId(rid="B")],
        )
        aux_step1 = TokenAlignerStepAux(
            input_ids=[31, 51],
            positions=[3, 2],
            seq_lens=[1, 1],
            seq_ids=[SGLangSeqId(rid="A"), SGLangSeqId(rid="B")],
        )

        side_aux = TokenAlignerGlobalAux(
            step_auxs={0: aux, 1: aux_step1},
            framework="sglang",
            layout=TokenLayout.T,
        )

        index = build_seqs_info(side_aux)
        plan = compute_token_aligner_plan(seqs_info_pair=Pair(x=index, y=index))

        tensors = {0: hidden_step0, 1: hidden_step1}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan, tensor_of_step_pair=Pair(x=tensors, y=tensors)
        )

        assert torch.equal(aligned.x, aligned.y)
        assert aligned.x.shape[0] == len(plan.locators.x.steps)

    def test_zero_matched_tokens(self):
        """Empty TokenAlignerPlan (no matched tokens) returns shape[0]==0 without crash."""
        torch.manual_seed(42)

        plan = TokenAlignerPlan(
            locators=Pair(
                x=TokenLocator(steps=[], token_index_in_step=[]),
                y=TokenLocator(steps=[], token_index_in_step=[]),
            ),
            layouts=Pair(x=TokenLayout.T, y=TokenLayout.T),
        )

        tensors = {0: torch.randn(5, 8).refine_names("t", "h")}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan, tensor_of_step_pair=Pair(x=tensors, y=tensors)
        )

        assert aligned.x.shape[0] == 0
        assert aligned.y.shape[0] == 0
        assert aligned.x.shape[1:] == (8,)
        assert aligned.y.shape[1:] == (8,)


class TestTokenDim:
    """Tests for non-zero token_dim support."""

    def _make_simple_plan(self, *, num_tokens: int) -> TokenAlignerPlan:
        locator = TokenLocator(
            steps=[0] * num_tokens,
            token_index_in_step=list(range(num_tokens)),
        )
        return TokenAlignerPlan(
            locators=Pair(x=locator, y=locator),
            layouts=Pair(x=TokenLayout.T, y=TokenLayout.T),
        )

    def test_token_dim_nonzero(self) -> None:
        """tensor shape [3, 5, 8], token_dim=1 -> token dim stays at dim 1."""
        torch.manual_seed(42)
        tensor: torch.Tensor = _named(torch.randn(3, 5, 8), ["a", "t", "h"])
        plan: TokenAlignerPlan = self._make_simple_plan(num_tokens=5)

        tensors: dict[int, torch.Tensor] = {0: tensor}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
        )

        assert aligned.x.shape == (3, 5, 8)
        assert torch.equal(aligned.x, aligned.y)
        plain: torch.Tensor = tensor.rename(None)
        for i in range(5):
            assert torch.equal(
                aligned.x.select(dim=1, index=i), plain.select(dim=1, index=i)
            )

    def test_token_dim_last(self) -> None:
        """tensor shape [3, 8, 5], token_dim=2 -> token dim stays at dim 2."""
        torch.manual_seed(42)
        tensor: torch.Tensor = _named(torch.randn(3, 8, 5), ["a", "h", "t"])
        plan: TokenAlignerPlan = self._make_simple_plan(num_tokens=5)

        tensors: dict[int, torch.Tensor] = {0: tensor}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
        )

        assert aligned.x.shape == (3, 8, 5)
        plain: torch.Tensor = tensor.rename(None)
        for i in range(5):
            assert torch.equal(
                aligned.x.select(dim=2, index=i), plain.select(dim=2, index=i)
            )

    def test_token_dim_zero(self) -> None:
        """token_dim=0 selects along first dimension (standard t-h-d layout)."""
        torch.manual_seed(42)
        tensor: torch.Tensor = _named(torch.randn(5, 8), ["t", "h"])
        plan: TokenAlignerPlan = self._make_simple_plan(num_tokens=5)

        tensors: dict[int, torch.Tensor] = {0: tensor}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
        )

        assert aligned.x.shape == (5, 8)
        plain: torch.Tensor = tensor.rename(None)
        for i in range(5):
            assert torch.equal(aligned.x[i], plain.select(dim=0, index=i))

    def test_zero_matched_tokens_nonzero_token_dim(self) -> None:
        """Empty plan with token_dim=1 produces correct empty shape."""
        torch.manual_seed(42)

        plan = TokenAlignerPlan(
            locators=Pair(
                x=TokenLocator(steps=[], token_index_in_step=[]),
                y=TokenLocator(steps=[], token_index_in_step=[]),
            ),
            layouts=Pair(x=TokenLayout.T, y=TokenLayout.T),
        )

        tensors: dict[int, torch.Tensor] = {
            0: _named(torch.randn(3, 5, 8), ["a", "t", "h"])
        }
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
        )

        # token dim (dim 1) set to 0, other dims preserved -> [3, 0, 8]
        assert aligned.x.shape == (3, 0, 8)
        assert aligned.y.shape == (3, 0, 8)

    def test_high_rank_tensor(self) -> None:
        """tensor shape [2, 3, 5, 4, 8] (a b t c d), token_dim=2 -> stays at dim 2."""
        torch.manual_seed(42)
        tensor: torch.Tensor = _named(
            torch.randn(2, 3, 5, 4, 8), ["a", "x", "t", "c", "h"]
        )
        plan: TokenAlignerPlan = self._make_simple_plan(num_tokens=5)

        tensors: dict[int, torch.Tensor] = {0: tensor}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
        )

        assert aligned.x.shape == (2, 3, 5, 4, 8)
        plain: torch.Tensor = tensor.rename(None)
        for i in range(5):
            assert torch.equal(
                aligned.x.select(dim=2, index=i), plain.select(dim=2, index=i)
            )


class TestBSHDExecutor:
    """BSHD tensor collapse: B+S dims -> flat token dim for alignment."""

    def test_bshd_standard_bs_at_front(self):
        """Standard "b s h d": B=dim0, S=dim1. [2, 3, 4, 5] -> collapse -> [6, 4, 5]."""
        torch.manual_seed(42)
        tensor: torch.Tensor = _named(torch.randn(2, 3, 4, 5), ["b", "s", "h", "d"])
        flat: torch.Tensor = tensor.rename(None).reshape(6, 4, 5)

        locator = TokenLocator(
            steps=[0, 0, 0],
            token_index_in_step=[0, 3, 5],
        )
        plan = TokenAlignerPlan(
            locators=Pair(x=locator, y=locator),
            layouts=Pair(x=TokenLayout.BS, y=TokenLayout.BS),
        )

        tensors: dict[int, torch.Tensor] = {0: tensor}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
        )

        assert aligned.x.shape == (3, 4, 5)
        assert torch.equal(aligned.x[0], flat[0])
        assert torch.equal(aligned.x[1], flat[3])
        assert torch.equal(aligned.x[2], flat[5])

    def test_bshd_3d_bs_at_front(self):
        """Minimal 3D "b s h": B=dim0, S=dim1. [2, 3, 4] -> collapse -> [6, 4]."""
        torch.manual_seed(42)
        tensor: torch.Tensor = _named(torch.randn(2, 3, 4), ["b", "s", "h"])
        flat: torch.Tensor = tensor.rename(None).reshape(6, 4)

        locator = TokenLocator(
            steps=[0, 0, 0, 0],
            token_index_in_step=[0, 2, 3, 5],
        )
        plan = TokenAlignerPlan(
            locators=Pair(x=locator, y=locator),
            layouts=Pair(x=TokenLayout.BS, y=TokenLayout.BS),
        )

        tensors: dict[int, torch.Tensor] = {0: tensor}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
        )

        assert aligned.x.shape == (4, 4)
        assert torch.equal(aligned.x[0], flat[0])
        assert torch.equal(aligned.x[1], flat[2])
        assert torch.equal(aligned.x[2], flat[3])
        assert torch.equal(aligned.x[3], flat[5])

    def test_bshd_bs_not_at_front(self):
        """Non-leading "h b s d": B=dim1, S=dim2. [4, 2, 3, 5] -> collapse -> [4, 6, 5]."""
        torch.manual_seed(42)
        tensor: torch.Tensor = _named(torch.randn(4, 2, 3, 5), ["h", "b", "s", "d"])
        flat: torch.Tensor = tensor.rename(None).reshape(4, 6, 5)

        locator = TokenLocator(
            steps=[0, 0, 0],
            token_index_in_step=[0, 3, 5],
        )
        plan = TokenAlignerPlan(
            locators=Pair(x=locator, y=locator),
            layouts=Pair(x=TokenLayout.BS, y=TokenLayout.BS),
        )

        tensors: dict[int, torch.Tensor] = {0: tensor}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
        )

        assert aligned.x.shape == (4, 3, 5)
        for idx, flat_idx in enumerate([0, 3, 5]):
            assert torch.equal(
                aligned.x.select(dim=1, index=idx),
                flat.select(dim=1, index=flat_idx),
            )

    def test_bshd_expert_before_bs(self):
        """Expert dim before B: "e b s h d". [2, 3, 4, 5, 6] -> collapse -> [2, 12, 5, 6]."""
        torch.manual_seed(42)
        tensor: torch.Tensor = _named(
            torch.randn(2, 3, 4, 5, 6), ["e", "b", "s", "h", "d"]
        )
        flat: torch.Tensor = tensor.rename(None).reshape(2, 12, 5, 6)

        locator = TokenLocator(
            steps=[0, 0, 0],
            token_index_in_step=[0, 5, 11],
        )
        plan = TokenAlignerPlan(
            locators=Pair(x=locator, y=locator),
            layouts=Pair(x=TokenLayout.BS, y=TokenLayout.BS),
        )

        tensors: dict[int, torch.Tensor] = {0: tensor}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
        )

        assert aligned.x.shape == (2, 3, 5, 6)
        for idx, flat_idx in enumerate([0, 5, 11]):
            assert torch.equal(
                aligned.x.select(dim=1, index=idx),
                flat.select(dim=1, index=flat_idx),
            )

    def test_bshd_bs_at_end(self):
        """B and S at end: "h d b s". [4, 5, 2, 3] -> collapse -> [4, 5, 6]."""
        torch.manual_seed(42)
        tensor: torch.Tensor = _named(torch.randn(4, 5, 2, 3), ["h", "d", "b", "s"])
        flat: torch.Tensor = tensor.rename(None).reshape(4, 5, 6)

        locator = TokenLocator(
            steps=[0, 0, 0],
            token_index_in_step=[1, 3, 5],
        )
        plan = TokenAlignerPlan(
            locators=Pair(x=locator, y=locator),
            layouts=Pair(x=TokenLayout.BS, y=TokenLayout.BS),
        )

        tensors: dict[int, torch.Tensor] = {0: tensor}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
        )

        assert aligned.x.shape == (4, 5, 3)
        for idx, flat_idx in enumerate([1, 3, 5]):
            assert torch.equal(
                aligned.x.select(dim=2, index=idx),
                flat.select(dim=2, index=flat_idx),
            )

    def test_cross_layout_thd_vs_bshd(self):
        """Cross-layout: x=THD [6, 8], y=BSHD [2, 3, 8] -> y collapse -> [6, 8]."""
        torch.manual_seed(42)
        tensor_thd: torch.Tensor = _named(torch.randn(6, 8), ["t", "h"])
        tensor_bshd: torch.Tensor = _named(torch.randn(2, 3, 8), ["b", "s", "h"])
        flat_bshd: torch.Tensor = tensor_bshd.rename(None).reshape(6, 8)

        locator = TokenLocator(
            steps=[0, 0, 0],
            token_index_in_step=[0, 2, 5],
        )
        plan = TokenAlignerPlan(
            locators=Pair(x=locator, y=locator),
            layouts=Pair(x=TokenLayout.T, y=TokenLayout.BS),
        )

        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x={0: tensor_thd}, y={0: tensor_bshd}),
        )

        assert aligned.x.shape == (3, 8)
        assert aligned.y.shape == (3, 8)
        assert torch.equal(aligned.x[0], tensor_thd.rename(None)[0])
        assert torch.equal(aligned.y[0], flat_bshd[0])
        assert torch.equal(aligned.y[2], flat_bshd[5])

    def test_bshd_reversed_sb_order(self):
        """Reversed "s b h": S=dim0, B=dim1. Collapse is batch-major: (b s)."""
        torch.manual_seed(42)
        tensor: torch.Tensor = _named(torch.randn(3, 2, 4), ["s", "b", "h"])
        # batch-major flatten: rearrange("s b h -> (b s) h")
        from einops import rearrange

        flat: torch.Tensor = rearrange(tensor.rename(None), "s b h -> (b s) h")

        locator = TokenLocator(
            steps=[0, 0, 0],
            token_index_in_step=[0, 2, 5],
        )
        plan = TokenAlignerPlan(
            locators=Pair(x=locator, y=locator),
            layouts=Pair(x=TokenLayout.BS, y=TokenLayout.BS),
        )

        tensors: dict[int, torch.Tensor] = {0: tensor}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
        )

        assert aligned.x.shape == (3, 4)
        assert torch.equal(aligned.x[0], flat[0])
        assert torch.equal(aligned.x[1], flat[2])
        assert torch.equal(aligned.x[2], flat[5])

    def test_bshd_empty_plan_bs_not_at_front(self):
        """Empty plan with non-leading B,S: "h b s d". [4, 2, 3, 5] -> collapse -> [4, 0, 5]."""
        plan = TokenAlignerPlan(
            locators=Pair(
                x=TokenLocator(steps=[], token_index_in_step=[]),
                y=TokenLocator(steps=[], token_index_in_step=[]),
            ),
            layouts=Pair(x=TokenLayout.BS, y=TokenLayout.BS),
        )

        tensors: dict[int, torch.Tensor] = {
            0: _named(torch.randn(4, 2, 3, 5), ["h", "b", "s", "d"])
        }
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
        )

        assert aligned.x.shape == (4, 0, 5)
        assert aligned.y.shape == (4, 0, 5)

    def test_bshd_empty_plan_bs_at_front(self):
        """Empty plan with standard BSHD: "b s h". [2, 3, 4] -> collapse -> [0, 4]."""
        plan = TokenAlignerPlan(
            locators=Pair(
                x=TokenLocator(steps=[], token_index_in_step=[]),
                y=TokenLocator(steps=[], token_index_in_step=[]),
            ),
            layouts=Pair(x=TokenLayout.BS, y=TokenLayout.BS),
        )

        tensors: dict[int, torch.Tensor] = {
            0: _named(torch.randn(2, 3, 4), ["b", "s", "h"])
        }
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
        )

        assert aligned.x.shape == (0, 4)
        assert aligned.y.shape == (0, 4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
