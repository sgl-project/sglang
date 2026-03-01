import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from sglang.srt.debug_utils.comparator.bundle_comparator import _load_all_values
from sglang.srt.debug_utils.comparator.output_types import GeneralWarning
from sglang.srt.debug_utils.comparator.warning_sink import WarningSink
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="default", nightly=True)


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

        sink = WarningSink()
        with sink.context() as warnings:
            with patch(
                "sglang.srt.debug_utils.comparator.bundle_comparator.warning_sink",
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

        sink = WarningSink()
        with sink.context() as warnings:
            with patch(
                "sglang.srt.debug_utils.comparator.bundle_comparator.warning_sink",
                sink,
            ):
                result = _load_all_values(
                    filenames=[fn_good, fn_bad], base_path=tmp_path
                )

        assert len(result) == 1
        assert len(warnings) == 1
        assert isinstance(warnings[0], GeneralWarning)
        assert warnings[0].category == "load_failed"
        assert fn_bad in warnings[0].message

    def test_all_corrupted_emits_warnings_returns_empty(self, tmp_path: Path) -> None:
        """All files corrupted — returns empty list and emits one warning per file."""
        fn0: str = "step=0___rank=0___dump_index=0___name=a.pt"
        fn1: str = "step=0___rank=1___dump_index=0___name=a.pt"
        (tmp_path / fn0).write_text("corrupt")
        (tmp_path / fn1).write_text("corrupt")

        sink = WarningSink()
        with sink.context() as warnings:
            with patch(
                "sglang.srt.debug_utils.comparator.bundle_comparator.warning_sink",
                sink,
            ):
                result = _load_all_values(filenames=[fn0, fn1], base_path=tmp_path)

        assert len(result) == 0
        assert len(warnings) == 2
        assert all(w.category == "load_failed" for w in warnings)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
