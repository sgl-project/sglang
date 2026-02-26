import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.token_aligner.aux_plugins import (
    _infer_positions,
    _MegatronPlugin,
    _SGLangPlugin,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    PositionalSeqId,
    SGLangSeqId,
    TokenAlignerStepAux,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="default", nightly=True)

_sglang_plugin = _SGLangPlugin()
_megatron_plugin = _MegatronPlugin()


class TestNormalizeSGLang:
    """Tests for SGLang aux tensor normalization."""

    def test_with_rids(self):
        """SGLang tensors with rids produce string seq_ids."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20, 30]),
            "positions": torch.tensor([0, 1, 2]),
            "seq_lens": torch.tensor([3]),
            "rids": ["A"],
        }

        result: TokenAlignerStepAux = _sglang_plugin.compute_step_aux(
            step_data, layout="thd", step=0
        )

        assert result.input_ids == [10, 20, 30]
        assert result.positions == [0, 1, 2]
        assert result.seq_lens == [3]
        assert result.seq_ids == [SGLangSeqId(rid="A")]

    def test_rids_none_fallback(self):
        """Missing rids results in (step, index) fallback seq_ids."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20]),
            "positions": torch.tensor([0, 1]),
            "seq_lens": torch.tensor([2]),
        }

        result: TokenAlignerStepAux = _sglang_plugin.compute_step_aux(
            step_data, layout="thd", step=3
        )
        assert result.seq_ids == [PositionalSeqId(step=3, seq_index=0)]

    def test_multiple_seqs_with_rids(self):
        """Multiple sequences with rids."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20, 30, 40, 50]),
            "positions": torch.tensor([0, 1, 2, 0, 1]),
            "seq_lens": torch.tensor([3, 2]),
            "rids": ["A", "B"],
        }

        result: TokenAlignerStepAux = _sglang_plugin.compute_step_aux(
            step_data, layout="thd", step=0
        )
        assert result.seq_ids == [SGLangSeqId(rid="A"), SGLangSeqId(rid="B")]


class TestNormalizeMegatron:
    """Tests for Megatron aux tensor normalization."""

    def test_cu_seqlens_to_seq_lens(self):
        """cu_seqlens_q is converted to seq_lens via differencing."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20, 30, 40, 50]),
            "cu_seqlens_q": torch.tensor([0, 3, 5]),
        }

        result: TokenAlignerStepAux = _megatron_plugin.compute_step_aux(
            step_data, layout="thd", step=0
        )

        assert result.seq_lens == [3, 2]

    def test_positions_inferred_thd(self):
        """Positions inferred from seq_lens in thd layout."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20, 30, 40, 50]),
            "cu_seqlens_q": torch.tensor([0, 3, 5]),
        }

        result: TokenAlignerStepAux = _megatron_plugin.compute_step_aux(
            step_data, layout="thd", step=0
        )

        assert result.positions == [0, 1, 2, 0, 1]

    def test_position_ids_passthrough(self):
        """Explicit position_ids used directly instead of inference."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20, 30, 40, 50]),
            "position_ids": torch.tensor([5, 6, 7, 8, 9]),
            "cu_seqlens_q": torch.tensor([0, 5]),
        }

        result: TokenAlignerStepAux = _megatron_plugin.compute_step_aux(
            step_data, layout="thd", step=0
        )

        assert result.positions == [5, 6, 7, 8, 9]

    def test_seq_ids_are_step_index_tuples(self):
        """Megatron seq_ids are (step, seq_index) tuples."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20, 30, 40, 50]),
            "cu_seqlens_q": torch.tensor([0, 3, 5]),
        }

        result: TokenAlignerStepAux = _megatron_plugin.compute_step_aux(
            step_data, layout="thd", step=5
        )
        assert result.seq_ids == [
            PositionalSeqId(step=5, seq_index=0),
            PositionalSeqId(step=5, seq_index=1),
        ]


class TestInferPositions:
    """Tests for position inference helper."""

    def test_thd_multiple_sequences(self):
        """thd: positions reset to 0 for each sequence."""
        result = _infer_positions(
            seq_lens=torch.tensor([2, 3]),
        )
        assert torch.equal(result, torch.tensor([0, 1, 0, 1, 2]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
