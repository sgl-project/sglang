import sys
import types
from pathlib import Path


repo_root = Path(__file__).resolve().parents[5]
sglang_dir = repo_root / "python" / "sglang"
sglang_module = types.ModuleType("sglang")
sglang_module.__path__ = [str(sglang_dir)]
sglang_module.__package__ = "sglang"
sys.modules.setdefault("sglang", sglang_module)

from sglang.srt.layers.attention.dsv4.sparse_prefill_gate import (
    can_use_sparse_prefill,
)


def test_sparse_prefill_gate_rejects_missing_extend_lens():
    assert not can_use_sparse_prefill(
        q_num_rows=8,
        batch_size=1,
        extend_seq_lens_cpu=None,
        is_cp_round_robin=False,
    )


def test_sparse_prefill_gate_requires_dense_rows_to_match_extend_lens():
    assert can_use_sparse_prefill(
        q_num_rows=6,
        batch_size=2,
        extend_seq_lens_cpu=[2, 4],
        is_cp_round_robin=False,
    )
    assert not can_use_sparse_prefill(
        q_num_rows=8,
        batch_size=2,
        extend_seq_lens_cpu=[2, 4],
        is_cp_round_robin=False,
    )


def test_sparse_prefill_gate_requires_cp_local_rows_and_single_request():
    assert can_use_sparse_prefill(
        q_num_rows=8,
        batch_size=1,
        extend_seq_lens_cpu=[8],
        is_cp_round_robin=True,
        cp_num_rows=8,
    )
    assert not can_use_sparse_prefill(
        q_num_rows=8,
        batch_size=1,
        extend_seq_lens_cpu=[6],
        is_cp_round_robin=True,
        cp_num_rows=6,
    )
    assert not can_use_sparse_prefill(
        q_num_rows=8,
        batch_size=2,
        extend_seq_lens_cpu=[4, 4],
        is_cp_round_robin=True,
        cp_num_rows=8,
    )
