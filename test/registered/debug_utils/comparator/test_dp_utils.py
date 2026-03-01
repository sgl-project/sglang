import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.dp_utils import (
    _extract_dp_info,
    _group_has_data,
    filter_to_non_empty_dp_rank,
)
from sglang.srt.debug_utils.dump_loader import ValueWithMeta
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="default", nightly=True)


def _make_sglang_meta(
    *, tp_rank: int = 0, tp_size: int = 1, dp_rank: int = 0, dp_size: int = 1
) -> dict:
    return {
        "sglang_parallel_info": {
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "dp_rank": dp_rank,
            "dp_size": dp_size,
        }
    }


def _make_megatron_meta(
    *, tp_rank: int = 0, tp_size: int = 1, dp_rank: int = 0, dp_size: int = 1
) -> dict:
    return {
        "megatron_parallel_info": {
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "dp_rank": dp_rank,
            "dp_size": dp_size,
        }
    }


def _make_item(value: object, meta: dict) -> ValueWithMeta:
    return ValueWithMeta(value=value, meta=meta)


# ---------------------------------------------------------------------------
# _extract_dp_info
# ---------------------------------------------------------------------------


class TestExtractDpInfo:
    def test_sglang_dp(self) -> None:
        meta: dict = _make_sglang_meta(dp_rank=1, dp_size=4)
        assert _extract_dp_info(meta) == (1, 4)

    def test_megatron_dp(self) -> None:
        meta: dict = _make_megatron_meta(dp_rank=2, dp_size=8)
        assert _extract_dp_info(meta) == (2, 8)

    def test_no_parallel_info(self) -> None:
        assert _extract_dp_info({}) is None

    def test_no_dp_fields(self) -> None:
        meta: dict = {"sglang_parallel_info": {"tp_rank": 0, "tp_size": 2}}
        assert _extract_dp_info(meta) is None


# ---------------------------------------------------------------------------
# _group_has_data
# ---------------------------------------------------------------------------


class TestGroupHasData:
    def test_non_empty_tensor(self) -> None:
        item: ValueWithMeta = _make_item(value=torch.tensor([1, 2, 3]), meta={})
        assert _group_has_data([item]) is True

    def test_empty_tensor(self) -> None:
        item: ValueWithMeta = _make_item(value=torch.tensor([]), meta={})
        assert _group_has_data([item]) is False

    def test_non_tensor_value(self) -> None:
        item: ValueWithMeta = _make_item(value="hello", meta={})
        assert _group_has_data([item]) is False

    def test_empty_group(self) -> None:
        assert _group_has_data([]) is False


# ---------------------------------------------------------------------------
# filter_to_non_empty_dp_rank
# ---------------------------------------------------------------------------


class TestFilterToNonEmptyDpRank:
    def test_dp_size_1_returns_unchanged(self) -> None:
        items: list[ValueWithMeta] = [
            _make_item(
                value=torch.tensor([1.0]),
                meta=_make_sglang_meta(dp_size=1),
            ),
        ]
        result: list[ValueWithMeta] = filter_to_non_empty_dp_rank(items)
        assert result is items

    def test_no_parallel_info_returns_unchanged(self) -> None:
        items: list[ValueWithMeta] = [
            _make_item(value=torch.tensor([1.0]), meta={}),
        ]
        result: list[ValueWithMeta] = filter_to_non_empty_dp_rank(items)
        assert result is items

    def test_empty_list_returns_empty(self) -> None:
        result: list[ValueWithMeta] = filter_to_non_empty_dp_rank([])
        assert result == []

    def test_dp2_all_non_tensor_returns_unchanged(self) -> None:
        """DP=2 with non-tensor values: skip filtering, return unchanged."""
        items: list[ValueWithMeta] = [
            _make_item(
                value=["req_A"],
                meta=_make_sglang_meta(dp_rank=0, dp_size=2),
            ),
            _make_item(
                value=["req_A"],
                meta=_make_sglang_meta(dp_rank=1, dp_size=2),
            ),
        ]

        result: list[ValueWithMeta] = filter_to_non_empty_dp_rank(items)

        assert result is items

    def test_dp2_one_empty_one_nonempty_sglang(self) -> None:
        """DP=2, rank 0 has data, rank 1 has empty tensor."""
        items: list[ValueWithMeta] = [
            _make_item(
                value=torch.tensor([1.0, 2.0]),
                meta=_make_sglang_meta(dp_rank=0, dp_size=2),
            ),
            _make_item(
                value=torch.tensor([]),
                meta=_make_sglang_meta(dp_rank=1, dp_size=2),
            ),
        ]

        result: list[ValueWithMeta] = filter_to_non_empty_dp_rank(items)

        assert len(result) == 1
        assert torch.equal(result[0].value, torch.tensor([1.0, 2.0]))

    def test_dp2_one_empty_one_nonempty_megatron(self) -> None:
        """DP=2 megatron, rank 1 has data, rank 0 has empty tensor."""
        items: list[ValueWithMeta] = [
            _make_item(
                value=torch.tensor([]),
                meta=_make_megatron_meta(dp_rank=0, dp_size=2),
            ),
            _make_item(
                value=torch.tensor([3.0, 4.0]),
                meta=_make_megatron_meta(dp_rank=1, dp_size=2),
            ),
        ]

        result: list[ValueWithMeta] = filter_to_non_empty_dp_rank(items)

        assert len(result) == 1
        assert torch.equal(result[0].value, torch.tensor([3.0, 4.0]))

    def test_dp2_both_nonempty_raises(self) -> None:
        """DP=2, both ranks have data: assertion error."""
        items: list[ValueWithMeta] = [
            _make_item(
                value=torch.tensor([1.0]),
                meta=_make_sglang_meta(dp_rank=0, dp_size=2),
            ),
            _make_item(
                value=torch.tensor([2.0]),
                meta=_make_sglang_meta(dp_rank=1, dp_size=2),
            ),
        ]

        with pytest.raises(
            AssertionError, match="Expected exactly 1 non-empty dp_rank"
        ):
            filter_to_non_empty_dp_rank(items)

    def test_dp2_with_tp2_filters_correctly(self) -> None:
        """DP=2 x TP=2: 4 items total, 2 non-empty from dp_rank=0."""
        items: list[ValueWithMeta] = [
            _make_item(
                value=torch.tensor([1.0]),
                meta=_make_sglang_meta(tp_rank=0, tp_size=2, dp_rank=0, dp_size=2),
            ),
            _make_item(
                value=torch.tensor([2.0]),
                meta=_make_sglang_meta(tp_rank=1, tp_size=2, dp_rank=0, dp_size=2),
            ),
            _make_item(
                value=torch.tensor([]),
                meta=_make_sglang_meta(tp_rank=0, tp_size=2, dp_rank=1, dp_size=2),
            ),
            _make_item(
                value=torch.tensor([]),
                meta=_make_sglang_meta(tp_rank=1, tp_size=2, dp_rank=1, dp_size=2),
            ),
        ]

        result: list[ValueWithMeta] = filter_to_non_empty_dp_rank(items)

        assert len(result) == 2
        assert torch.equal(result[0].value, torch.tensor([1.0]))
        assert torch.equal(result[1].value, torch.tensor([2.0]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
