import sys
import time
from pathlib import Path

import pytest
import requests
import torch
import torch.distributed as dist

from sglang.srt.debug_utils.dumper import (
    _collect_megatron_parallel_info,
    _collect_sglang_parallel_info,
    _Dumper,
    _materialize_value,
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


class TestMaterializeValue:
    def test_materialize_value_callable(self):
        tensor = torch.randn(3, 3)
        result = _materialize_value(lambda: tensor)
        assert torch.equal(result, tensor)

    def test_materialize_value_passthrough(self):
        tensor = torch.randn(3, 3)
        result = _materialize_value(tensor)
        assert result is tensor

    def test_dump_with_callable_value(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        tensor = torch.randn(4, 4)
        d.dump("lazy_tensor", lambda: tensor)

        _assert_files(_get_filenames(tmp_path), exist=["name=lazy_tensor"])

        path = _find_dump_file(tmp_path, rank=0, name="lazy_tensor")
        assert torch.equal(_load_dump(path)["value"], tensor)


class TestSaveValue:
    def test_dump_output_format(self, tmp_path):
        dumper = _make_test_dumper(tmp_path)
        tensor = torch.randn(4, 4)

        dumper.dump("dict_test", tensor)

        path = _find_dump_file(tmp_path, rank=0, name="dict_test")
        loaded = _load_dump(path)
        assert torch.equal(loaded["value"], tensor)
        assert loaded["meta"]["name"] == "dict_test"
        assert loaded["meta"]["rank"] == 0


class TestStaticMetadata:
    def test_static_meta_contains_world_info(self):
        dumper = _make_test_dumper(Path("/tmp"))
        meta = dumper._static_meta
        assert "world_rank" in meta
        assert "world_size" in meta
        assert meta["world_rank"] == 0
        assert meta["world_size"] == 1

    def test_static_meta_caching(self):
        dumper = _make_test_dumper(Path("/tmp"))
        meta1 = dumper._static_meta
        meta2 = dumper._static_meta
        assert meta1 is meta2

    def test_parallel_info_graceful_fallback(self):
        sglang_info = _collect_sglang_parallel_info()
        assert isinstance(sglang_info, dict)

        megatron_info = _collect_megatron_parallel_info()
        assert isinstance(megatron_info, dict)

    def test_dump_includes_static_meta(self, tmp_path):
        dumper = _make_test_dumper(tmp_path)
        tensor = torch.randn(2, 2)

        dumper.dump("meta_test", tensor)

        path = _find_dump_file(tmp_path, rank=0, name="meta_test")
        loaded = _load_dump(path)
        meta = loaded["meta"]
        assert "world_rank" in meta
        assert "world_size" in meta


class TestDumpGrad:
    def test_dump_grad_basic(self, tmp_path):
        d = _make_test_dumper(tmp_path, enable_grad=True)
        x = torch.randn(3, 3, requires_grad=True)
        y = (x * 2).sum()

        d.dump("test_tensor", x)
        y.backward()

        filenames = _get_filenames(tmp_path)
        assert any("name=test_tensor" in f and "grad__" not in f for f in filenames)
        _assert_files(filenames, exist=["grad__test_tensor"])

    def test_dump_grad_non_tensor_skipped(self, tmp_path):
        d = _make_test_dumper(tmp_path, enable_grad=True)
        d.dump("not_tensor", 42)

        _assert_files(_get_filenames(tmp_path), not_exist=["grad__"])

    def test_dump_grad_no_requires_grad_skipped(self, tmp_path):
        d = _make_test_dumper(tmp_path, enable_grad=True)
        x = torch.randn(3, 3, requires_grad=False)
        d.dump("no_grad_tensor", x)

        _assert_files(
            _get_filenames(tmp_path),
            exist=["name=no_grad_tensor"],
            not_exist=["grad__"],
        )

    def test_dump_grad_captures_forward_pass_id(self, tmp_path):
        d = _make_test_dumper(tmp_path, enable_grad=True)
        d._forward_pass_id = 42
        x = torch.randn(3, 3, requires_grad=True)
        y = (x * 2).sum()

        d.dump("id_test", x)
        d._forward_pass_id = 999
        y.backward()

        grad_file = _find_dump_file(tmp_path, name="grad__id_test")
        assert "forward_pass_id=42" in grad_file.name

    def test_dump_grad_file_content(self, tmp_path):
        d = _make_test_dumper(tmp_path, enable_grad=True)
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = (x * 3).sum()

        d.dump("content_check", x)
        y.backward()

        grad_path = _find_dump_file(tmp_path, name="grad__content_check")
        expected_grad = torch.full((2, 2), 3.0)
        assert torch.equal(_load_dump(grad_path)["value"], expected_grad)

    def test_disable_value(self, tmp_path):
        d = _make_test_dumper(tmp_path, enable_value=False, enable_grad=True)
        x = torch.randn(3, 3, requires_grad=True)
        y = (x * 2).sum()

        d.dump("fwd_disabled", x)
        y.backward()

        filenames = _get_filenames(tmp_path)
        assert not any(
            "name=fwd_disabled" in f and "grad__" not in f for f in filenames
        )
        _assert_files(filenames, exist=["grad__fwd_disabled"])

    def test_disable_grad(self, tmp_path):
        d = _make_test_dumper(tmp_path, enable_grad=False)
        x = torch.randn(3, 3, requires_grad=True)
        y = (x * 2).sum()

        d.dump("grad_disabled", x)
        y.backward()

        _assert_files(
            _get_filenames(tmp_path),
            exist=["name=grad_disabled"],
            not_exist=["grad__"],
        )


class TestDumpModel:
    def test_grad_basic(self, tmp_path):
        d = _make_test_dumper(tmp_path, enable_model_value=False)
        model = torch.nn.Linear(4, 2)
        x = torch.randn(3, 4)
        y = model(x).sum()
        y.backward()

        d.dump_model(model, name_prefix="model")

        _assert_files(
            _get_filenames(tmp_path),
            exist=["grad__model__weight", "grad__model__bias"],
        )

    def test_value_basic(self, tmp_path):
        d = _make_test_dumper(tmp_path, enable_model_grad=False)
        model = torch.nn.Linear(4, 2, bias=False)

        d.dump_model(model, name_prefix="model")

        _assert_files(
            _get_filenames(tmp_path),
            exist=["model__weight"],
        )

    def test_no_grad_skipped(self, tmp_path):
        d = _make_test_dumper(tmp_path, enable_model_value=False)
        model = torch.nn.Linear(4, 2)

        d.dump_model(model, name_prefix="model")

        filenames = _get_filenames(tmp_path)
        assert len(filenames) == 0

    def test_filter(self, tmp_path):
        d = _make_test_dumper(tmp_path, filter="weight")
        model = torch.nn.Linear(4, 2)
        x = torch.randn(3, 4)
        y = model(x).sum()
        y.backward()

        d.dump_model(model, name_prefix="model")

        _assert_files(
            _get_filenames(tmp_path),
            exist=["model__weight", "grad__model__weight"],
            not_exist=["model__bias", "grad__model__bias"],
        )

    def test_grad_file_content(self, tmp_path):
        d = _make_test_dumper(tmp_path, enable_model_value=False)
        model = torch.nn.Linear(4, 2, bias=False)
        x = torch.ones(1, 4)
        y = model(x).sum()
        y.backward()

        d.dump_model(model, name_prefix="p")

        path = _find_dump_file(tmp_path, name="grad__p__weight")
        assert torch.equal(_load_dump(path)["value"], model.weight.grad)

    def test_disable_model_grad(self, tmp_path):
        d = _make_test_dumper(tmp_path, enable_model_grad=False)
        model = torch.nn.Linear(4, 2)
        x = torch.randn(3, 4)
        y = model(x).sum()
        y.backward()

        d.dump_model(model, name_prefix="model")

        filenames = _get_filenames(tmp_path)
        assert all("grad" not in f for f in filenames)

    def test_disable_model_value(self, tmp_path):
        d = _make_test_dumper(tmp_path, enable_model_value=False)
        model = torch.nn.Linear(4, 2, bias=False)
        x = torch.ones(1, 4)
        y = model(x).sum()
        y.backward()

        d.dump_model(model, name_prefix="model")

        filenames = _get_filenames(tmp_path)
        assert all("grad" in f for f in filenames)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
