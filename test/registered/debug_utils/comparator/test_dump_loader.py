import sys
from pathlib import Path

import pytest
import torch

from sglang.srt.debug_utils.dump_loader import read_tokenizer_path
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


def _save_pt(
    directory: Path, filename: str, *, value: torch.Tensor, meta: dict
) -> None:
    torch.save({"value": value, "meta": meta}, directory / filename)


class TestReadTokenizerPath:
    def test_finds_tokenizer_path(self, tmp_path: Path) -> None:
        _save_pt(
            tmp_path,
            "name=x___step=0___rank=0___dump_index=0.pt",
            value=torch.tensor([1.0]),
            meta={"tokenizer_path": "/models/llama-3"},
        )
        result = read_tokenizer_path(tmp_path)
        assert result == "/models/llama-3"

    def test_returns_none_when_no_tokenizer_path(self, tmp_path: Path) -> None:
        _save_pt(
            tmp_path,
            "name=x___step=0___rank=0___dump_index=0.pt",
            value=torch.tensor([1.0]),
            meta={},
        )
        result = read_tokenizer_path(tmp_path)
        assert result is None

    def test_returns_none_for_empty_directory(self, tmp_path: Path) -> None:
        result = read_tokenizer_path(tmp_path)
        assert result is None

    def test_skips_files_without_tokenizer_path(self, tmp_path: Path) -> None:
        _save_pt(
            tmp_path,
            "name=a___step=0___rank=0___dump_index=0.pt",
            value=torch.tensor([1.0]),
            meta={},
        )
        _save_pt(
            tmp_path,
            "name=b___step=0___rank=0___dump_index=1.pt",
            value=torch.tensor([2.0]),
            meta={"tokenizer_path": "/models/deepseek"},
        )
        result = read_tokenizer_path(tmp_path)
        assert result == "/models/deepseek"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
