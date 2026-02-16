import sys
import time
from pathlib import Path

import pytest
import requests
import torch
import torch.distributed as dist

from sglang.srt.debug_utils.dumper import (
    _Dumper,
    _obj_to_dict,
    _torch_save,
    get_tensor_info,
    get_truncated_value,
)
from sglang.srt.environ import temp_set_env
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import run_distributed_test

register_cuda_ci(est_time=30, suite="nightly-2-gpu", nightly=True)
register_amd_ci(est_time=60, suite="nightly-amd", nightly=True)


class TestDumperPureFunctions:
    def test_get_truncated_value(self):
        assert get_truncated_value(None) is None
        assert get_truncated_value(42) == 42
        assert len(get_truncated_value((torch.randn(10), torch.randn(20)))) == 2
        assert get_truncated_value(torch.randn(10, 10)).shape == (10, 10)
        assert get_truncated_value(torch.randn(100, 100)).shape == (5, 5)

    def test_obj_to_dict(self):
        assert _obj_to_dict({"a": 1}) == {"a": 1}

        class Obj:
            x, y = 10, 20

            def method(self):
                pass

        result = _obj_to_dict(Obj())
        assert result["x"] == 10
        assert "method" not in result

    def test_get_tensor_info(self):
        info = get_tensor_info(torch.randn(10, 10))
        for key in ["shape=", "dtype=", "min=", "max=", "mean="]:
            assert key in info

        assert "value=42" in get_tensor_info(42)
        assert "min=None" in get_tensor_info(torch.tensor([]))


class TestTorchSave:
    def test_normal(self, tmp_path):
        path = str(tmp_path / "a.pt")
        tensor = torch.randn(3, 3)

        _torch_save(tensor, path)

        assert torch.equal(torch.load(path, weights_only=True), tensor)

    def test_parameter_fallback(self, tmp_path):
        class BadParam(torch.nn.Parameter):
            def __reduce_ex__(self, protocol):
                raise RuntimeError("not pickleable")

        path = str(tmp_path / "b.pt")
        param = BadParam(torch.randn(4))

        _torch_save(param, path)

        assert torch.equal(torch.load(path, weights_only=True), param.data)

    def test_silent_skip(self, tmp_path, capsys):
        path = str(tmp_path / "c.pt")

        _torch_save({"fn": lambda: None}, path)

        captured = capsys.readouterr()
        assert "[Dumper] Observe error=" in captured.out
        assert "skip the tensor" in captured.out


class TestDumperDistributed:
    def test_basic(self, tmp_path):
        with temp_set_env(
            allow_sglang=True,
            SGLANG_DUMPER_ENABLE="1",
            SGLANG_DUMPER_DIR=str(tmp_path),
        ):
            run_distributed_test(self._test_basic_func, tmpdir=str(tmp_path))

    @staticmethod
    def _test_basic_func(rank, tmpdir):
        from sglang.srt.debug_utils.dumper import dumper

        tensor = torch.randn(10, 10, device=f"cuda:{rank}")

        dumper.on_forward_pass_start()
        dumper.dump("tensor_a", tensor, arg=100)

        dumper.on_forward_pass_start()
        dumper.set_ctx(ctx_arg=200)
        dumper.dump("tensor_b", tensor)
        dumper.set_ctx(ctx_arg=None)

        dumper.on_forward_pass_start()
        dumper.override_enable(False)
        dumper.dump("tensor_skip", tensor)
        dumper.override_enable(True)

        dumper.on_forward_pass_start()
        dumper.dump_dict("obj", {"a": torch.randn(3, device=f"cuda:{rank}"), "b": 42})

        dist.barrier()
        filenames = _get_filenames(tmpdir)
        _assert_files(
            filenames,
            exist=["tensor_a", "tensor_b", "arg=100", "ctx_arg=200", "obj_a", "obj_b"],
            not_exist=["tensor_skip"],
        )

    def test_http_enable(self):
        with temp_set_env(allow_sglang=True, SGLANG_DUMPER_ENABLE="0"):
            run_distributed_test(self._test_http_func)

    @staticmethod
    def _test_http_func(rank):
        from sglang.srt.debug_utils.dumper import dumper

        assert not dumper._enable
        dumper.on_forward_pass_start()

        for enable in [True, False]:
            dist.barrier()
            if rank == 0:
                time.sleep(0.1)
                requests.post(
                    "http://localhost:40000/dumper", json={"enable": enable}
                ).raise_for_status()
            dist.barrier()
            assert dumper._enable == enable

    def test_file_content_correctness(self, tmp_path):
        with temp_set_env(
            allow_sglang=True,
            SGLANG_DUMPER_ENABLE="1",
            SGLANG_DUMPER_DIR=str(tmp_path),
        ):
            run_distributed_test(self._test_file_content_func, tmpdir=str(tmp_path))

    @staticmethod
    def _test_file_content_func(rank, tmpdir):
        from sglang.srt.debug_utils.dumper import dumper

        tensor = torch.arange(12, device=f"cuda:{rank}").reshape(3, 4).float()

        dumper.on_forward_pass_start()
        dumper.dump("content_check", tensor)

        dist.barrier()
        path = _find_dump_file(tmpdir, rank=rank, name="content_check")
        raw = _load_dump(path)
        assert isinstance(raw, dict), f"Expected dict, got {type(raw)}"
        assert "value" in raw and "meta" in raw
        assert torch.equal(raw["value"], tensor.cpu())
        assert raw["meta"]["name"] == "content_check"
        assert raw["meta"]["rank"] == rank


class TestDumperFileWriteControl:
    def test_filter(self, tmp_path):
        with temp_set_env(
            allow_sglang=True,
            SGLANG_DUMPER_ENABLE="1",
            SGLANG_DUMPER_DIR=str(tmp_path),
            SGLANG_DUMPER_FILTER="^keep",
        ):
            run_distributed_test(self._test_filter_func, tmpdir=str(tmp_path))

    @staticmethod
    def _test_filter_func(rank, tmpdir):
        from sglang.srt.debug_utils.dumper import dumper

        dumper.on_forward_pass_start()
        dumper.dump("keep_this", torch.randn(5, device=f"cuda:{rank}"))
        dumper.dump("skip_this", torch.randn(5, device=f"cuda:{rank}"))
        dumper.dump("not_keep_this", torch.randn(5, device=f"cuda:{rank}"))

        dist.barrier()
        filenames = _get_filenames(tmpdir)
        _assert_files(
            filenames,
            exist=["keep_this"],
            not_exist=["skip_this", "not_keep_this"],
        )

    def test_write_disabled(self, tmp_path):
        with temp_set_env(
            allow_sglang=True,
            SGLANG_DUMPER_ENABLE="1",
            SGLANG_DUMPER_DIR=str(tmp_path),
            SGLANG_DUMPER_WRITE_FILE="0",
        ):
            run_distributed_test(self._test_write_disabled_func, tmpdir=str(tmp_path))

    @staticmethod
    def _test_write_disabled_func(rank, tmpdir):
        from sglang.srt.debug_utils.dumper import dumper

        dumper.on_forward_pass_start()
        dumper.dump("no_write", torch.randn(5, device=f"cuda:{rank}"))

        dist.barrier()
        assert len(_get_filenames(tmpdir)) == 0

    def test_save_false(self, tmp_path):
        with temp_set_env(
            allow_sglang=True,
            SGLANG_DUMPER_ENABLE="1",
            SGLANG_DUMPER_DIR=str(tmp_path),
        ):
            run_distributed_test(self._test_save_false_func, tmpdir=str(tmp_path))

    @staticmethod
    def _test_save_false_func(rank, tmpdir):
        from sglang.srt.debug_utils.dumper import dumper

        dumper.on_forward_pass_start()
        dumper.dump("no_save_tensor", torch.randn(5, device=f"cuda:{rank}"), save=False)

        dist.barrier()
        assert len(_get_filenames(tmpdir)) == 0


class TestDumpDictFormat:
    """Verify that dump files use the dict output format: {"value": ..., "meta": {...}}."""

    def test_dict_format_structure(self, tmp_path):
        dumper = _make_test_dumper(tmp_path)
        tensor = torch.randn(4, 4)
        dumper.dump("fmt_test", tensor, custom_key="hello")

        path = _find_dump_file(str(tmp_path), rank=0, name="fmt_test")
        raw = _load_dump(path)

        assert isinstance(raw, dict)
        assert set(raw.keys()) == {"value", "meta"}
        assert torch.equal(raw["value"], tensor)

        meta = raw["meta"]
        assert meta["name"] == "fmt_test"
        assert meta["custom_key"] == "hello"
        assert "forward_pass_id" in meta
        assert "rank" in meta
        assert "dump_index" in meta

    def test_dict_format_with_context(self, tmp_path):
        dumper = _make_test_dumper(tmp_path)
        dumper.set_ctx(ctx_val=42)
        tensor = torch.randn(2, 2)
        dumper.dump("ctx_fmt", tensor)

        path = _find_dump_file(str(tmp_path), rank=0, name="ctx_fmt")
        raw = _load_dump(path)

        assert raw["meta"]["ctx_val"] == 42
        assert torch.equal(raw["value"], tensor)


def _make_test_dumper(tmp_path: Path, **overrides) -> _Dumper:
    """Create a _Dumper for CPU testing without HTTP server or distributed."""
    defaults: dict = dict(
        enable=True,
        base_dir=tmp_path,
        partial_name="test",
        enable_http_server=False,
    )
    d = _Dumper(**{**defaults, **overrides})
    d.on_forward_pass_start()
    return d


def _get_filenames(tmpdir):
    return {f.name for f in Path(tmpdir).glob("sglang_dump_*/*.pt")}


def _assert_files(filenames, *, exist=(), not_exist=()):
    for p in exist:
        assert any(p in f for f in filenames), f"{p} not found in {filenames}"
    for p in not_exist:
        assert not any(
            p in f for f in filenames
        ), f"{p} should not exist in {filenames}"


def _load_dump(path: Path) -> dict:
    """Load a dump file and return the raw dict (with 'value' and 'meta' keys)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def _find_dump_file(tmpdir, *, rank: int = 0, name: str) -> Path:
    matches = [
        f
        for f in Path(tmpdir).glob("sglang_dump_*/*.pt")
        if f"rank={rank}" in f.name and name in f.name
    ]
    assert (
        len(matches) == 1
    ), f"Expected 1 file matching rank={rank} name={name}, got {matches}"
    return matches[0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
