import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from sglang.srt.debug_utils.comparator.entrypoint import run
from sglang.srt.debug_utils.comparator.output_types import (
    AnyRecord,
    ComparisonRecord,
    ConfigRecord,
    SkipRecord,
    SummaryRecord,
    _OutputRecord,
    parse_record_json,
)
from sglang.srt.debug_utils.dumper import DumperConfig, _Dumper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="default", nightly=True)


def _make_dumper(directory: Path) -> _Dumper:
    return _Dumper(
        config=DumperConfig(enable=True, dir=str(directory), enable_http_server=False)
    )


def _create_dumps(
    tmp_path: Path,
    tensor_names: list[str],
    *,
    baseline_names: list[str] | None = None,
    num_steps: int = 1,
) -> tuple[Path, Path]:
    """Create baseline and target dump directories with given tensor names.

    If baseline_names is None, uses the same names as tensor_names.
    Each step dumps all names with the same tensor (different per baseline/target).
    """
    if baseline_names is None:
        baseline_names = tensor_names

    d_baseline = tmp_path / "baseline"
    d_target = tmp_path / "target"
    d_baseline.mkdir()
    d_target.mkdir()

    torch.manual_seed(42)
    baseline_tensor = torch.randn(10, 10)
    target_tensor = baseline_tensor + torch.randn(10, 10) * 0.01

    exp_paths: list[Path] = []
    for d, names, tensor in [
        (d_baseline, baseline_names, baseline_tensor),
        (d_target, tensor_names, target_tensor),
    ]:
        dumper = _make_dumper(d)
        for _ in range(num_steps):
            for name in names:
                dumper.dump(name, tensor)
            dumper.step()
        exp_paths.append(d / dumper._config.exp_name)

    return exp_paths[0], exp_paths[1]


def _make_args(baseline_path: Path, target_path: Path, **overrides) -> Namespace:
    defaults = dict(
        baseline_path=str(baseline_path),
        target_path=str(target_path),
        start_step=0,
        end_step=1000000,
        diff_threshold=1e-3,
        filter=None,
        output_format="text",
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _parse_jsonl(output: str) -> list[AnyRecord]:
    return [parse_record_json(line) for line in output.strip().splitlines()]


class TestEntrypoint:
    def test_run_basic(self, tmp_path, capsys):
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a", "tensor_b"])
        args = _make_args(baseline_path, target_path)
        capsys.readouterr()

        run(args)

        output = capsys.readouterr().out
        assert "Config:" in output
        assert "rel_diff" in output
        assert "Summary:" in output
        assert "Skip" not in output

    def test_filter(self, tmp_path, capsys):
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a", "tensor_b"])
        args = _make_args(baseline_path, target_path, filter="tensor_a")
        capsys.readouterr()

        run(args)

        output = capsys.readouterr().out
        assert "rel_diff" in output

    def test_no_baseline_skip(self, tmp_path, capsys):
        baseline_path, target_path = _create_dumps(
            tmp_path,
            tensor_names=["tensor_a", "tensor_extra"],
            baseline_names=["tensor_a"],
        )
        args = _make_args(baseline_path, target_path)
        capsys.readouterr()

        run(args)

        output = capsys.readouterr().out
        assert "Skip:" in output
        assert "no_baseline" in output

    def test_step_range(self, tmp_path, capsys):
        baseline_path, target_path = _create_dumps(tmp_path, ["t"], num_steps=3)
        args = _make_args(baseline_path, target_path, start_step=1, end_step=1)
        capsys.readouterr()

        run(args)

        output = capsys.readouterr().out
        assert "Summary:" in output


class TestEntrypointJsonl:
    def test_jsonl_basic(self, tmp_path, capsys):
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a", "tensor_b"])
        args = _make_args(baseline_path, target_path, output_format="json")
        capsys.readouterr()

        run(args)

        records = _parse_jsonl(capsys.readouterr().out)
        assert isinstance(records[0], ConfigRecord)

        comparisons = [r for r in records if isinstance(r, ComparisonRecord)]
        assert len(comparisons) == 2

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.total == 2
        assert summary.skipped == 0

    def test_jsonl_skip(self, tmp_path, capsys):
        baseline_path, target_path = _create_dumps(
            tmp_path,
            tensor_names=["tensor_a", "tensor_extra"],
            baseline_names=["tensor_a"],
        )
        args = _make_args(baseline_path, target_path, output_format="json")
        capsys.readouterr()

        run(args)

        records = _parse_jsonl(capsys.readouterr().out)
        skips = [r for r in records if isinstance(r, SkipRecord)]
        assert len(skips) == 1
        assert skips[0].reason == "no_baseline"

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.skipped == 1

    def test_jsonl_all_valid_records(self, tmp_path, capsys):
        baseline_path, target_path = _create_dumps(tmp_path, ["t"], num_steps=2)
        args = _make_args(baseline_path, target_path, output_format="json")
        capsys.readouterr()

        run(args)

        records = _parse_jsonl(capsys.readouterr().out)
        assert all(isinstance(r, _OutputRecord) for r in records)


class TestValueWithMeta:
    def test_load_dict_format(self, tmp_path) -> None:
        path = tmp_path / "step=0___rank=0___dump_index=1___name=hidden.pt"
        tensor = torch.randn(4, 8)
        torch.save({"value": tensor, "meta": {"custom": "field"}}, path)

        from sglang.srt.debug_utils.dump_loader import ValueWithMeta

        loaded = ValueWithMeta.load(path)
        assert torch.allclose(loaded.value, tensor)
        assert loaded.meta["custom"] == "field"
        assert loaded.meta["name"] == "hidden"
        assert loaded.meta["rank"] == 0

    def test_load_bare_tensor(self, tmp_path) -> None:
        path = tmp_path / "step=0___rank=0___dump_index=1___name=bare.pt"
        tensor = torch.randn(3, 3)
        torch.save(tensor, path)

        from sglang.srt.debug_utils.dump_loader import ValueWithMeta

        loaded = ValueWithMeta.load(path)
        assert torch.allclose(loaded.value, tensor)
        assert loaded.meta["name"] == "bare"

    def test_load_corrupted_file(self, tmp_path) -> None:
        path = tmp_path / "step=0___rank=0___dump_index=1___name=bad.pt"
        path.write_text("not a valid pt file")

        from sglang.srt.debug_utils.dump_loader import ValueWithMeta

        loaded = ValueWithMeta.load(path)
        assert loaded.value is None
        assert loaded.meta["name"] == "bad"


class TestDumperDims:
    def test_dims_in_meta_not_filename(self, tmp_path) -> None:
        dumper = _make_dumper(tmp_path)
        tensor = torch.randn(4, 8)
        dumper.dump("hidden", tensor, dims="b h(tp)")
        dumper.step()

        exp_dir = tmp_path / dumper._config.exp_name
        pt_files = list(exp_dir.glob("*.pt"))
        assert len(pt_files) == 1

        assert "dims" not in pt_files[0].stem

        data = torch.load(pt_files[0], weights_only=False)
        assert "dims" in data["meta"]
        assert data["meta"]["dims"] == "b h(tp)"

    def test_dims_grad_override(self, tmp_path) -> None:
        dumper = _Dumper(
            config=DumperConfig(
                enable=True,
                dir=str(tmp_path),
                enable_http_server=False,
                enable_grad=True,
            )
        )

        tensor = torch.randn(4, 8, requires_grad=True)
        dumper.dump("hidden", tensor, dims="b h(tp)", dims_grad="b h(tp,partial)")
        dumper.step()

        tensor.backward(torch.ones_like(tensor))

        exp_dir = tmp_path / dumper._config.exp_name
        pt_files = sorted(exp_dir.glob("*.pt"))
        assert len(pt_files) == 2

        value_file = [f for f in pt_files if "grad__" not in f.stem][0]
        grad_file = [f for f in pt_files if "grad__" in f.stem][0]

        value_data = torch.load(value_file, weights_only=False)
        assert value_data["meta"]["dims"] == "b h(tp)"
        assert value_data["meta"]["dims_grad"] == "b h(tp,partial)"

        grad_data = torch.load(grad_file, weights_only=False)
        assert grad_data["meta"]["dims"] == "b h(tp,partial)"

    def test_dims_grad_inherits(self, tmp_path) -> None:
        dumper = _Dumper(
            config=DumperConfig(
                enable=True,
                dir=str(tmp_path),
                enable_http_server=False,
                enable_grad=True,
            )
        )

        tensor = torch.randn(4, 8, requires_grad=True)
        dumper.dump("hidden", tensor, dims="b h(tp)")
        dumper.step()

        tensor.backward(torch.ones_like(tensor))

        exp_dir = tmp_path / dumper._config.exp_name
        grad_file = [f for f in exp_dir.glob("*.pt") if "grad__" in f.stem][0]
        grad_data = torch.load(grad_file, weights_only=False)
        assert grad_data["meta"]["dims"] == "b h(tp)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
