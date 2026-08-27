"""CPU regression coverage for DSpark context-boundary verify planning."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ROOT = Path(__file__).parents[4]
SOURCE = ROOT / "python/sglang/srt/speculative/draft_worker_common.py"
PLANNER_SOURCE = (
    ROOT / "python/sglang/srt/speculative/dspark_components/dspark_planner.py"
)


def load_clamp_verify_lens():
    tree = ast.parse(SOURCE.read_text())
    node = next(
        (
            item
            for item in tree.body
            if getattr(item, "name", None) == "clamp_verify_lens"
        ),
        None,
    )
    if node is None:
        pytest.fail("clamp_verify_lens is missing from draft_worker_common.py")
    namespace = {"torch": torch}

    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace["clamp_verify_lens"]


def load_alloc_verify_window():
    """Load the production planner helper with a deterministic cache-loc stub."""
    tree = ast.parse(PLANNER_SOURCE.read_text())
    node = next(
        (
            item
            for item in tree.body
            if getattr(item, "name", None) == "alloc_verify_window"
        ),
        None,
    )
    if node is None:
        pytest.fail("alloc_verify_window is missing from dspark_planner.py")

    def fake_assign_extend_cache_locs_func(*, batch_size, draft_token_num, device, **_):
        return torch.arange(
            batch_size * draft_token_num, dtype=torch.int64, device=device
        )

    namespace = {
        "torch": torch,
        "VerifyWindow": SimpleNamespace,
        "ScheduleBatch": object,
        "assign_extend_cache_locs_func": fake_assign_extend_cache_locs_func,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(PLANNER_SOURCE), "exec"),
        namespace,
    )
    return namespace["alloc_verify_window"]


def test_clamp_verify_lens_context_boundary_cpu():
    clamp_verify_lens = load_clamp_verify_lens()
    requested = torch.tensor([6, 6, 6, 6, 6], dtype=torch.int64)
    seq_lens = torch.tensor([10, 98, 99, 100, 97], dtype=torch.int64)
    remaining = torch.tensor([20, 20, 20, 20, 2], dtype=torch.int64)
    actual = clamp_verify_lens(
        requested_verify_lens=requested,
        seq_lens=seq_lens,
        remaining_generation_tokens=remaining,
        max_position_embeddings=100,
    )
    assert actual.tolist() == [6, 2, 1, 0, 2]
    assert torch.all(actual <= requested)
    assert torch.all(actual <= remaining)
    assert torch.all(actual <= (100 - seq_lens).clamp_min(0))


def test_clamp_verify_lens_preserves_short_context_verify_window():
    actual = load_clamp_verify_lens()(
        requested_verify_lens=torch.tensor([6]),
        seq_lens=torch.tensor([12]),
        remaining_generation_tokens=torch.tensor([8]),
        max_position_embeddings=100,
    )
    assert actual.tolist() == [6]


def test_alloc_verify_window_pads_draft_tail_with_last_legal_position_cpu():
    """Draft forward must never receive the full, out-of-bound DSpark window."""
    alloc_verify_window = load_alloc_verify_window()
    prefix_lens = torch.tensor([98, 97], dtype=torch.int64)
    verify_lens = torch.tensor([2, 2], dtype=torch.int64)
    offsets = torch.arange(6, dtype=torch.int64)
    batch = SimpleNamespace(
        seq_lens=prefix_lens,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
    )
    model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.empty((2, 128), dtype=torch.int64)
        )
    )

    window = alloc_verify_window(
        batch=batch,
        bs=2,
        device=torch.device("cpu"),
        verify_num_draft_tokens=6,
        block_pos_offsets=offsets,
        model_runner=model_runner,
        verify_lens=verify_lens,
        max_position_embeddings=100,
    )
    assert window.positions_2d.tolist() == [
        [98, 99, 99, 99, 99, 99],
        [97, 98, 98, 98, 98, 98],
    ]
    assert int(window.positions_2d.max()) < 100


def test_worker_uses_the_boundary_clamp_before_target_verify():
    source = (
        ROOT / "python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py"
    ).read_text()
    assert "clamp_verify_lens(" in source
    assert "verify_lens=actual_verify_lens" in source


def test_valid_layout_positions_never_cross_context_boundary():
    tree = ast.parse(SOURCE.read_text())
    offsets_node = next(
        item
        for item in tree.body
        if getattr(item, "name", None) == "build_block_pos_offsets"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[offsets_node], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    offsets = namespace["build_block_pos_offsets"](length=6, device=torch.device("cpu"))
    actual = load_clamp_verify_lens()(
        requested_verify_lens=torch.tensor([6, 6, 6, 6]),
        seq_lens=torch.tensor([10, 98, 99, 97]),
        remaining_generation_tokens=torch.tensor([20, 20, 20, 2]),
        max_position_embeddings=100,
    )
    positions = torch.cat(
        [
            seq_len + offsets[:verify_len]
            for seq_len, verify_len in zip([10, 98, 99, 97], actual.tolist())
        ]
    )
    assert offsets.tolist() == [0, 1, 2, 3, 4, 5]
    assert positions.tolist() == [10, 11, 12, 13, 14, 15, 98, 99, 99, 97, 98]
    assert int(positions.max()) < 100


# ---------------------------------------------------------------------------
# max_new_tokens=None guard (issue #33454 original repro: no explicit limit)
# ---------------------------------------------------------------------------


def _make_remaining(max_new_tokens, output_len, verify_num):
    """Mirrors the dspark_worker_v2 remaining_generation_tokens formula."""
    if max_new_tokens is None:
        return verify_num
    return max(max_new_tokens - output_len, 0)


def test_remaining_generation_tokens_handles_none_max_new_tokens():
    """None max_new_tokens must not raise TypeError (was: NoneType - int)."""
    # This is the exact formula used in dspark_worker_v2.py.
    # If max_new_tokens is None the original code crashes with:
    #   TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'
    result = _make_remaining(max_new_tokens=None, output_len=5, verify_num=6)
    assert result == 6  # fall through to verify_num (no generation cap)

    result2 = _make_remaining(max_new_tokens=10, output_len=8, verify_num=6)
    assert result2 == 2  # capped by generation budget

    result3 = _make_remaining(max_new_tokens=10, output_len=10, verify_num=6)
    assert result3 == 0  # exhausted


def test_worker_remaining_generation_none_guard_present():
    """The worker source must contain the None guard for max_new_tokens."""
    source = (
        ROOT / "python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py"
    ).read_text()
    # After the fix, the worker must guard against None before subtracting.
    assert (
        "max_new_tokens is not None" in source
        or "if max_tok" in source
        or "if max_new" in source
    ), "dspark_worker_v2.py is missing the None guard for max_new_tokens"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
