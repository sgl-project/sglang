import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from sglang.srt.debug_utils.comparator.bundle_comparator import (
    _build_skip_from_one_empty_side,
    _load_all_values,
)
from sglang.srt.debug_utils.comparator.log_sink import LogSink
from sglang.srt.debug_utils.comparator.output_types import ErrorLog
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.srt.debug_utils.dump_loader import ValueWithMeta
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu-only", nightly=True)


def _save_tensor(
    dump_path: Path,
    *,
    name: str,
    step: int = 0,
    rank: int = 0,
) -> str:
    filename: str = f"step={step}___rank={rank}___dump_index=0___name={name}.pt"
    tensor: torch.Tensor = torch.randn(4)
    torch.save({"value": tensor, "meta": {}}, dump_path / filename)
    return filename


class TestLoadAllValues:
    def test_all_success(self, tmp_path: Path) -> None:
        """All files load successfully — no warnings emitted."""
        fn0: str = _save_tensor(tmp_path, name="a", rank=0)
        fn1: str = _save_tensor(tmp_path, name="a", rank=1)

        sink = LogSink()
        with sink.context() as warnings:
            with patch(
                "sglang.srt.debug_utils.comparator.bundle_comparator.log_sink",
                sink,
            ):
                result = _load_all_values(filenames=[fn0, fn1], base_path=tmp_path)

        assert len(result) == 2
        assert len(warnings) == 0

    def test_one_corrupted_emits_warning(self, tmp_path: Path) -> None:
        """One corrupted file is filtered out and emits a load_failed warning."""
        fn_good: str = _save_tensor(tmp_path, name="a", rank=0)

        fn_bad: str = "step=0___rank=1___dump_index=0___name=a.pt"
        (tmp_path / fn_bad).write_text("not a valid pt file")

        sink = LogSink()
        with sink.context() as warnings:
            with patch(
                "sglang.srt.debug_utils.comparator.bundle_comparator.log_sink",
                sink,
            ):
                result = _load_all_values(
                    filenames=[fn_good, fn_bad], base_path=tmp_path
                )

        assert len(result) == 1
        assert len(warnings) == 1
        assert isinstance(warnings[0], ErrorLog)
        assert warnings[0].category == "load_failed"
        assert fn_bad in warnings[0].message

    def test_all_corrupted_emits_warnings_returns_empty(self, tmp_path: Path) -> None:
        """All files corrupted — returns empty list and emits one warning per file."""
        fn0: str = "step=0___rank=0___dump_index=0___name=a.pt"
        fn1: str = "step=0___rank=1___dump_index=0___name=a.pt"
        (tmp_path / fn0).write_text("corrupt")
        (tmp_path / fn1).write_text("corrupt")

        sink = LogSink()
        with sink.context() as warnings:
            with patch(
                "sglang.srt.debug_utils.comparator.bundle_comparator.log_sink",
                sink,
            ):
                result = _load_all_values(filenames=[fn0, fn1], base_path=tmp_path)

        assert len(result) == 0
        assert len(warnings) == 2
        assert all(w.category == "load_failed" for w in warnings)


def _tensor_item(value: torch.Tensor, rank: int = 0) -> ValueWithMeta:
    return ValueWithMeta(
        value=value,
        meta={
            "rank": rank,
            "dims": "b s",
            "sglang_parallel_info": {},
            "megatron_parallel_info": {},
            "filename": f"rank_{rank}.pt",
        },
    )


class TestBuildSkipFromOneEmptySide:
    def test_baseline_empty_sets_reason_and_side(self) -> None:
        """Empty baseline → reason='baseline_load_failed', available_side='target'."""
        item = _tensor_item(torch.randn(2, 3))
        record = _build_skip_from_one_empty_side(
            name="test_tensor",
            pair=Pair(x=[], y=[item]),
        )
        assert record.reason == "baseline_load_failed"
        assert record.available_side == "target"
        assert record.available_tensor_info is not None

    def test_target_empty_sets_reason_and_side(self) -> None:
        """Empty target → reason='target_load_failed', available_side='baseline'."""
        item = _tensor_item(torch.randn(2, 3))
        record = _build_skip_from_one_empty_side(
            name="test_tensor",
            pair=Pair(x=[item], y=[]),
        )
        assert record.reason == "target_load_failed"
        assert record.available_side == "baseline"
        assert record.available_tensor_info is not None

    def test_no_tensor_items_returns_minimal_skip(self) -> None:
        """All items are non-tensor → skip record with no tensor info."""
        non_tensor_item = ValueWithMeta(value="not_a_tensor", meta={"rank": 0})
        record = _build_skip_from_one_empty_side(
            name="test_tensor",
            pair=Pair(x=[], y=[non_tensor_item]),
        )
        assert record.reason == "baseline_load_failed"
        assert record.available_tensor_info is None
        assert record.available_bundle_info is None

    def test_with_tensor_items_populates_info(self) -> None:
        """Tensor items present → tensor_info and bundle_info are populated."""
        item = _tensor_item(torch.randn(2, 3))
        record = _build_skip_from_one_empty_side(
            name="test_tensor",
            pair=Pair(x=[], y=[item]),
        )
        assert record.available_tensor_info is not None
        assert record.available_tensor_info.shape == [2, 3]
        assert record.available_bundle_info is not None
        assert record.available_bundle_info.num_files >= 1

    def test_multiple_tensor_items_uses_first_for_info(self) -> None:
        """When multiple tensor items exist, tensor_info comes from the first."""
        item1 = _tensor_item(torch.randn(2, 3), rank=0)
        item2 = _tensor_item(torch.randn(4, 5), rank=1)
        record = _build_skip_from_one_empty_side(
            name="multi",
            pair=Pair(x=[], y=[item1, item2]),
        )
        assert record.available_tensor_info is not None
        assert record.available_tensor_info.shape == [2, 3]
        assert record.available_bundle_info is not None
        assert record.available_bundle_info.num_files == 2

    def test_mixed_tensor_and_non_tensor_filters_non_tensor(self) -> None:
        """Non-tensor items are filtered; tensor_info comes from tensor items only."""
        non_tensor = ValueWithMeta(value="string_value", meta={"rank": 0})
        tensor_item = _tensor_item(torch.randn(5, 6), rank=1)
        record = _build_skip_from_one_empty_side(
            name="mixed",
            pair=Pair(x=[], y=[non_tensor, tensor_item]),
        )
        assert record.available_tensor_info is not None
        assert record.available_tensor_info.shape == [5, 6]
        assert record.available_bundle_info is not None
        assert record.available_bundle_info.num_files == 1

    def test_tensor_info_includes_sample(self) -> None:
        """Tensor info should include a sample string for skip records."""
        item = _tensor_item(torch.tensor([1.0, 2.0, 3.0]))
        record = _build_skip_from_one_empty_side(
            name="sample_check",
            pair=Pair(x=[item], y=[]),
        )
        assert record.available_tensor_info is not None
        assert record.available_tensor_info.sample is not None

    def test_name_preserved_in_record(self) -> None:
        """The tensor name is preserved in the skip record."""
        item = _tensor_item(torch.randn(2, 3))
        record = _build_skip_from_one_empty_side(
            name="my_layer.weight",
            pair=Pair(x=[], y=[item]),
        )
        assert record.name == "my_layer.weight"

    def test_bundle_info_has_dims_from_meta(self) -> None:
        """Bundle info dims field should come from the meta."""
        item = _tensor_item(torch.randn(2, 3))
        record = _build_skip_from_one_empty_side(
            name="dims_check",
            pair=Pair(x=[], y=[item]),
        )
        assert record.available_bundle_info is not None
        assert record.available_bundle_info.dims == "b s"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
