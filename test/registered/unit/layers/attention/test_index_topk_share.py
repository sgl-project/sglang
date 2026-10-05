import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mha
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _batch(
    reuse: bool, carried, *, is_extend: bool = False, seed_buf=None, seed_select=None
) -> SimpleNamespace:
    return SimpleNamespace(
        reuse_dsa_topk_indices=reuse,
        forward_mode=ForwardMode.DRAFT_EXTEND_V2 if is_extend else ForwardMode.DECODE,
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


def _capture(indices, *, fused=True, ragged=True, flattened=True, select=None):
    # Two requests share their first two physical slots. Logical index 5 is
    # outside the four-slot physical pool, but maps to the valid physical slot 2.
    table = torch.tensor([[3, 1, 0], [3, 1, 2]], dtype=torch.int32)
    metadata = SimpleNamespace(
        page_table_1=table,
        page_table_1_flattened=table.flatten() if flattened else None,
        indexer_seq_lens_cpu=torch.tensor([3, 3]),
    )
    backend = SimpleNamespace(
        use_fused_topk=fused,
        get_topk_transform_method=Mock(
            return_value=(
                TopkTransformMethod.RAGGED if ragged else TopkTransformMethod.PAGED
            )
        ),
        forward_metadata=metadata,
    )
    rows = len(indices) if select is None else len(select)
    capture = torch.full((rows + 1, indices.shape[1]), -99, dtype=torch.int32)
    batch = SimpleNamespace(
        forward_mode=object(),
        spec_info=SimpleNamespace(
            dsa_seed_topk_capture=capture,
            dsa_seed_topk_select=select,
        ),
    )
    indexer = Mock(return_value=indices)
    with patch.object(forward_mha, "resolve_attn_backend", return_value=backend):
        forward_mha.forward_dsa_indexer_for_mha(
            indexer,
            hidden_states=None,
            q_lora=None,
            positions=None,
            forward_batch=batch,
            layer_id=0,
        )
    assert indexer.call_args.kwargs["return_indices"]
    assert torch.all(capture[-1] == -99)
    return capture[:-1]


@pytest.mark.parametrize("flattened", [True, False])
@pytest.mark.parametrize("select", [None, torch.tensor([2, 0])])
def test_ragged_seed_maps_shared_prefix_and_preserves_padding(flattened, select):
    indices = torch.tensor([[0, 2, -1], [3, 4, -1], [5, 1, -1]], dtype=torch.int32)
    original = indices.clone()
    expected = torch.tensor([[3, 0, -1], [3, 1, -1], [2, 1, -1]], dtype=torch.int32)
    if select is not None:
        expected = expected[select]
    actual = _capture(indices, flattened=flattened, select=select)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(indices, original)


@pytest.mark.parametrize("fused,ragged", [(True, False), (False, True), (False, False)])
def test_paged_and_unfused_seed_contracts_are_preserved(fused, ragged):
    indices = torch.tensor([[2, 0, -1], [1, 2, -1]], dtype=torch.int32)
    torch.testing.assert_close(_capture(indices, fused=fused, ragged=ragged), indices)


@pytest.mark.parametrize("invalid", [-2, 6])
def test_invalid_ragged_indices_are_not_clamped(invalid):
    with pytest.raises((IndexError, RuntimeError)):
        _capture(torch.tensor([[invalid, -1]], dtype=torch.int32))


def test_no_capture_does_not_resolve_backend():
    indexer = Mock(return_value=None)
    batch = SimpleNamespace(spec_info=None)
    with patch.object(forward_mha, "resolve_attn_backend") as resolve:
        forward_mha.forward_dsa_indexer_for_mha(
            indexer,
            hidden_states=None,
            q_lora=None,
            positions=None,
            forward_batch=batch,
            layer_id=0,
        )
    assert not indexer.call_args.kwargs["return_indices"]
    resolve.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
