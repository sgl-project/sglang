import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _batch(
    reuse: bool, carried, *, is_extend: bool = False, seed_buf=None, seed_select=None
) -> SimpleNamespace:
    return SimpleNamespace(
        reuse_dsa_topk_indices=reuse,
        forward_mode=SimpleNamespace(
            is_extend=lambda include_draft_extend_v2: is_extend
        ),
        spec_info=SimpleNamespace(
            dsa_topk_indices=carried,
            dsa_seed_topk_capture=seed_buf,
            dsa_seed_topk_select=seed_select,
        ),
    )


def test_mtp_carry_is_empty_when_reuse_disabled():
    batch = _batch(reuse=False, carried="old")
    state = IndexTopKShareState.from_mtp_carry(batch)

    assert state.topk_indices is None
    state.update("new")
    state.publish()

    assert state.topk_indices == "new"
    assert batch.spec_info.dsa_topk_indices == "old"


def test_mtp_carry_reads_and_publishes_when_reuse_enabled():
    batch = _batch(reuse=True, carried="old")
    state = IndexTopKShareState.from_mtp_carry(batch)

    assert state.topk_indices == "old"
    state.update("new")
    state.publish()

    assert state.topk_indices == "new"
    assert batch.spec_info.dsa_topk_indices == "new"


def test_target_carry_stays_local_without_publish():
    batch = _batch(reuse=False, carried="batch")
    state = IndexTopKShareState(batch, "layer")

    assert state.topk_indices == "layer"
    assert batch.spec_info.dsa_topk_indices == "batch"

    state.update(None)

    assert state.topk_indices is None
    assert batch.spec_info.dsa_topk_indices == "batch"


def test_target_none_does_not_fall_back_to_mtp_carry():
    batch = _batch(reuse=True, carried="batch")
    state = IndexTopKShareState(batch, None)

    assert state.topk_indices is None

    state.update("layer")

    assert state.topk_indices == "layer"
    assert batch.spec_info.dsa_topk_indices == "batch"


def test_publish_captures_draft_extend_seed():
    seed_buf = torch.zeros(2, 3, dtype=torch.int64)
    batch = _batch(reuse=False, carried=None, is_extend=True, seed_buf=seed_buf)
    state = IndexTopKShareState.from_mtp_carry(batch)

    assert state.should_publish
    state.update(torch.arange(12, dtype=torch.int64).view(4, 3))
    state.publish()

    assert torch.equal(seed_buf, torch.arange(6, dtype=torch.int64).view(2, 3))
    assert batch.spec_info.dsa_topk_indices is None


def test_seed_buffer_is_ignored_outside_extend():
    seed_buf = torch.zeros(2, 3, dtype=torch.int64)
    batch = _batch(reuse=False, carried=None, is_extend=False, seed_buf=seed_buf)
    state = IndexTopKShareState.from_mtp_carry(batch)

    assert not state.should_publish
    state.update(torch.ones(4, 3, dtype=torch.int64))
    state.publish()

    assert torch.equal(seed_buf, torch.zeros(2, 3, dtype=torch.int64))


def test_mtp_iteration_clears_batch_state():
    batch = _batch(reuse=False, carried="stale")

    with IndexTopKShareState.mtp_iteration(batch) as state:
        assert state is not None
        assert batch.reuse_dsa_topk_indices
        assert batch.spec_info.dsa_topk_indices is None
        batch.spec_info.dsa_topk_indices = "draft-topk"

    assert not batch.reuse_dsa_topk_indices
    assert batch.spec_info.dsa_topk_indices is None


def test_mtp_iteration_clears_batch_state_on_exception():
    batch = _batch(reuse=False, carried=None)

    with pytest.raises(RuntimeError, match="draft step blew up"):
        with IndexTopKShareState.mtp_iteration(batch):
            batch.spec_info.dsa_topk_indices = "draft-topk"
            raise RuntimeError("draft step blew up")

    assert not batch.reuse_dsa_topk_indices
    assert batch.spec_info.dsa_topk_indices is None


def test_disabled_mtp_iteration_is_passthrough():
    batch = _batch(reuse=False, carried="untouched")

    with IndexTopKShareState.mtp_iteration(batch, enabled=False) as state:
        assert state is None
        assert not batch.reuse_dsa_topk_indices
        assert batch.spec_info.dsa_topk_indices == "untouched"

    assert not batch.reuse_dsa_topk_indices
    assert batch.spec_info.dsa_topk_indices == "untouched"


def test_mtp_iteration_preserves_draft_extend_seed():
    batch = _batch(reuse=False, carried="extend-seed")

    with IndexTopKShareState.mtp_iteration(batch, keep_carry_seed=True) as state:
        assert state is not None
        assert batch.spec_info.dsa_topk_indices == "extend-seed"
        assert state.topk_indices == "extend-seed"

    assert not batch.reuse_dsa_topk_indices
    assert batch.spec_info.dsa_topk_indices is None


def test_mtp_iteration_clears_missing_draft_extend_seed():
    batch = _batch(reuse=False, carried=None)

    with IndexTopKShareState.mtp_iteration(batch, keep_carry_seed=True):
        assert batch.spec_info.dsa_topk_indices is None

    assert not batch.reuse_dsa_topk_indices
    assert batch.spec_info.dsa_topk_indices is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
