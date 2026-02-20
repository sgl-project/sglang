import io
import multiprocessing
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import requests
import torch
import torch.distributed as dist

from sglang.srt.debug_utils.dumper import (
    _collect_megatron_parallel_info,
    _collect_sglang_parallel_info,
    _collective_with_timeout,
    _Dumper,
    _DumperConfig,
    _format_tags,
    _materialize_value,
    _obj_to_dict,
    _torch_save,
    dumper,
    get_tensor_info,
    get_truncated_value,
)
from sglang.srt.environ import temp_set_env
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    find_available_port,
    popen_launch_server,
    run_distributed_test,
)

register_cuda_ci(est_time=30, suite="nightly-2-gpu", nightly=True)
register_amd_ci(est_time=60, suite="nightly-amd", nightly=True)


@contextmanager
def _capture_stdout():
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        yield captured
    finally:
        sys.stdout = old_stdout


class TestDumperConfig:
    def test_from_env_defaults_match_dataclass_defaults(self):
        assert _DumperConfig.from_env() == _DumperConfig()

    def test_from_env_bool(self):
        with temp_set_env(allow_sglang=True, SGLANG_DUMPER_ENABLE="1"):
            assert _DumperConfig.from_env().enable is True
        with temp_set_env(allow_sglang=True, SGLANG_DUMPER_ENABLE="false"):
            assert _DumperConfig.from_env().enable is False

    def test_from_env_str(self):
        with temp_set_env(allow_sglang=True, SGLANG_DUMPER_FILTER="layer_id=0"):
            assert _DumperConfig.from_env().filter == "layer_id=0"

    def test_from_env_dir(self):
        with temp_set_env(allow_sglang=True, SGLANG_DUMPER_DIR="/my/dir"):
            assert _DumperConfig.from_env().dir == "/my/dir"

    def test_from_env_int(self):
        with temp_set_env(allow_sglang=True, SGLANG_DUMPER_COLLECTIVE_TIMEOUT="120"):
            assert _DumperConfig.from_env().collective_timeout == 120

    def test_configure_overrides(self):
        d = _make_test_dumper("/tmp")
        d.configure(enable=False)
        assert d._config.enable is False
        d.configure(enable=True)
        assert d._config.enable is True

    def test_type_validation(self):
        with pytest.raises(TypeError, match="enable.*expected bool.*got str"):
            _DumperConfig(enable="yes")
        with pytest.raises(
            TypeError, match="collective_timeout.*expected int.*got str"
        ):
            _DumperConfig(collective_timeout="abc")
        with pytest.raises(TypeError, match="filter.*expected str.*got int"):
            _DumperConfig(filter=123)

    def test_configure_default_skips_when_env_set(self):
        with temp_set_env(allow_sglang=True, SGLANG_DUMPER_FILTER="from_env"):
            d = _Dumper(config=_DumperConfig.from_env())
            d.configure_default(filter="from_code")
            assert d._config.filter == "from_env"

    def test_configure_default_applies_when_no_env(self):
        d = _Dumper(config=_DumperConfig.from_env())
        d.configure_default(filter="from_code")
        assert d._config.filter == "from_code"


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


class TestCollectiveTimeout:
    def test_watchdog_fires_on_timeout(self):
        block_event = threading.Event()
        output = ""

        def run_with_timeout():
            nonlocal output
            with _capture_stdout() as captured:
                _collective_with_timeout(
                    lambda: block_event.wait(),
                    operation_name="test_blocked_op",
                    timeout_seconds=2,
                )
            output = captured.getvalue()

        worker = threading.Thread(target=run_with_timeout)
        worker.start()

        time.sleep(4)
        block_event.set()
        worker.join(timeout=5)

        print(f"Captured output: {output!r}")
        assert "WARNING" in output
        assert "test_blocked_op" in output
        assert "2s" in output


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
        tensor = torch.randn(10, 10, device=f"cuda:{rank}")

        dumper.on_forward_pass_start()
        dumper.dump("tensor_a", tensor, arg=100)

        dumper.on_forward_pass_start()
        dumper.set_ctx(ctx_arg=200)
        dumper.dump("tensor_b", tensor)
        dumper.set_ctx(ctx_arg=None)

        dumper.on_forward_pass_start()
        dumper.configure(filter=r"^$")
        dumper.dump("tensor_skip", tensor)
        dumper.configure(filter=None)

        dumper.on_forward_pass_start()
        dumper.dump_dict("obj", {"a": torch.randn(3, device=f"cuda:{rank}"), "b": 42})

        dist.barrier()
        filenames = _get_filenames(tmpdir)
        _assert_files(
            filenames,
            exist=["tensor_a", "tensor_b", "arg=100", "ctx_arg=200", "obj_a", "obj_b"],
            not_exist=["tensor_skip"],
        )

    def test_collective_timeout(self):
        with temp_set_env(allow_sglang=True, SGLANG_DUMPER_ENABLE="1"):
            run_distributed_test(self._test_collective_timeout_func)

    @staticmethod
    def _test_collective_timeout_func(rank):
        dumper = _Dumper(
            config=_DumperConfig(
                enable=True,
                collective_timeout=3,
                enable_http_server=False,
            ),
        )

        with _capture_stdout() as captured:
            if rank != 0:
                time.sleep(6)
            dumper.on_forward_pass_start()

        output = captured.getvalue()
        print(f"Rank {rank} captured output: {output!r}")

        if rank == 0:
            assert "WARNING" in output, f"Expected WARNING in rank 0 output: {output}"
            assert "has not completed after 3s" in output

    def test_file_content_correctness(self, tmp_path):
        with temp_set_env(
            allow_sglang=True,
            SGLANG_DUMPER_ENABLE="1",
            SGLANG_DUMPER_DIR=str(tmp_path),
        ):
            run_distributed_test(self._test_file_content_func, tmpdir=str(tmp_path))

    @staticmethod
    def _test_file_content_func(rank, tmpdir):
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
            SGLANG_DUMPER_FILTER="name=keep",
        ):
            run_distributed_test(self._test_filter_func, tmpdir=str(tmp_path))

    @staticmethod
    def _test_filter_func(rank, tmpdir):
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

    def test_save_false(self, tmp_path):
        with temp_set_env(
            allow_sglang=True,
            SGLANG_DUMPER_ENABLE="1",
            SGLANG_DUMPER_DIR=str(tmp_path),
        ):
            run_distributed_test(self._test_save_false_func, tmpdir=str(tmp_path))

    @staticmethod
    def _test_save_false_func(rank, tmpdir):
        dumper.on_forward_pass_start()
        dumper.dump("no_save_tensor", torch.randn(5, device=f"cuda:{rank}"), save=False)

        dist.barrier()
        assert len(_get_filenames(tmpdir)) == 0


class TestOutputControl:
    def test_file_enabled_by_default(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        d.dump("file_on", torch.randn(3, 3))

        _assert_files(_get_filenames(tmp_path), exist=["file_on"])

    def test_file_disabled(self, tmp_path, capsys):
        d = _make_test_dumper(tmp_path, enable_output_file=False)
        d.dump("file_off", torch.randn(3, 3))

        assert len(_get_filenames(tmp_path)) == 0
        assert "file_off" in capsys.readouterr().out

    def test_console_enabled_by_default(self, tmp_path, capsys):
        d = _make_test_dumper(tmp_path)
        d.dump("console_on", torch.randn(3, 3))

        captured = capsys.readouterr()
        assert "[Dumper.Value]" in captured.out
        assert "console_on" in captured.out

    def test_console_disabled(self, tmp_path, capsys):
        d = _make_test_dumper(tmp_path, enable_output_console=False)
        d.dump("console_off", torch.randn(3, 3))

        assert "console_off" not in capsys.readouterr().out
        _assert_files(_get_filenames(tmp_path), exist=["console_off"])

    def test_capture_output_basic(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        tensor = torch.randn(4, 4)

        with d.capture_output() as captured:
            d.dump("cap_basic", tensor)

        assert "cap_basic" in captured
        assert set(captured["cap_basic"].keys()) == {"value", "meta"}
        assert torch.equal(captured["cap_basic"]["value"], tensor)
        assert captured["cap_basic"]["meta"]["name"] == "cap_basic"

    def test_capture_output_no_file(self, tmp_path):
        d = _make_test_dumper(tmp_path)

        with d.capture_output() as captured:
            d.dump("cap_no_file", torch.randn(3, 3))

        assert "cap_no_file" in captured
        assert len(_get_filenames(tmp_path)) == 0

    def test_capture_output_multiple(self, tmp_path):
        d = _make_test_dumper(tmp_path)

        with d.capture_output() as captured:
            d.dump("first", torch.randn(2, 2))
            d.dump("second", torch.randn(3, 3))

        assert set(captured.keys()) == {"first", "second"}
        assert captured["first"]["value"].shape == (2, 2)
        assert captured["second"]["value"].shape == (3, 3)

    def test_capture_output_value_cloned(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        tensor = torch.zeros(3, 3)

        with d.capture_output() as captured:
            d.dump("clone_check", tensor)

        tensor.fill_(999.0)
        assert torch.equal(captured["clone_check"]["value"], torch.zeros(3, 3))

    def test_capture_output_respects_filter(self, tmp_path):
        d = _make_test_dumper(tmp_path, filter="name=keep")

        with d.capture_output() as captured:
            d.dump("keep_this", torch.randn(3, 3))
            d.dump("skip_this", torch.randn(3, 3))

        assert "keep_this" in captured
        assert "skip_this" not in captured


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


def _make_test_dumper(tmp_path, **overrides) -> _Dumper:
    """Create a _Dumper for CPU testing without HTTP server or distributed."""
    config = _DumperConfig(
        enable=True,
        dir=str(tmp_path),
        partial_name="test",
        enable_http_server=False,
        **overrides,
    )
    d = _Dumper(config=config)
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
        dumper = _make_test_dumper("/tmp")
        meta = dumper._static_meta
        assert "world_rank" in meta
        assert "world_size" in meta
        assert meta["world_rank"] == 0
        assert meta["world_size"] == 1

    def test_static_meta_caching(self):
        dumper = _make_test_dumper("/tmp")
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


class TestKvFilter:
    def test_format_tags(self):
        assert _format_tags({"a": 1, "b": "hello"}) == "a=1___b=hello"
        assert _format_tags({}) == ""

    def test_filter_matches_extra_kwargs(self, tmp_path):
        d = _make_test_dumper(tmp_path, filter="layer_id=0")
        d.dump("tensor_a", torch.randn(3), layer_id=0)
        d.dump("tensor_b", torch.randn(3), layer_id=1)

        filenames = _get_filenames(tmp_path)
        _assert_files(filenames, exist=["tensor_a"], not_exist=["tensor_b"])

    def test_filter_matches_global_ctx(self, tmp_path):
        d = _make_test_dumper(tmp_path, filter="ctx_arg=200")
        d.set_ctx(ctx_arg=200)
        d.dump("tensor_a", torch.randn(3))
        d.set_ctx(ctx_arg=None)
        d.dump("tensor_b", torch.randn(3))

        filenames = _get_filenames(tmp_path)
        _assert_files(filenames, exist=["tensor_a"], not_exist=["tensor_b"])

    def test_filter_matches_name(self, tmp_path):
        d = _make_test_dumper(tmp_path, filter="name=keep")
        d.dump("keep_this", torch.randn(3))
        d.dump("skip_this", torch.randn(3))

        filenames = _get_filenames(tmp_path)
        _assert_files(filenames, exist=["keep_this"], not_exist=["skip_this"])

    def test_filter_regex(self, tmp_path):
        d = _make_test_dumper(tmp_path, filter=r"layer_id=[0-2]")
        d.dump("t0", torch.randn(3), layer_id=0)
        d.dump("t1", torch.randn(3), layer_id=1)
        d.dump("t5", torch.randn(3), layer_id=5)

        filenames = _get_filenames(tmp_path)
        _assert_files(filenames, exist=["name=t0", "name=t1"], not_exist=["name=t5"])

    def test_no_filter_dumps_all(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        d.dump("a", torch.randn(3))
        d.dump("b", torch.randn(3))

        filenames = _get_filenames(tmp_path)
        _assert_files(filenames, exist=["name=a", "name=b"])


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


class TestCleanup:
    def test_cleanup_removes_old_dumps(self, tmp_path):
        old_dir = tmp_path / "sglang_dump_old"
        old_dir.mkdir()
        (old_dir / "dummy.pt").touch()

        dumper = _make_test_dumper(tmp_path, cleanup_previous=True)
        dumper.dump("new_tensor", torch.randn(3, 3))

        assert not old_dir.exists()
        _assert_files(_get_filenames(tmp_path), exist=["new_tensor"])

    def test_no_cleanup_by_default(self, tmp_path):
        old_dir = tmp_path / "sglang_dump_old"
        old_dir.mkdir()
        (old_dir / "dummy.pt").touch()

        dumper = _make_test_dumper(tmp_path)
        dumper.dump("new_tensor", torch.randn(3, 3))

        assert old_dir.exists()
        _assert_files(_get_filenames(tmp_path), exist=["new_tensor"])


class TestReset:
    def test_reset_clears_state(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        d.set_ctx(layer_id=1)
        d.dump("before_reset", torch.randn(3, 3))

        d.reset()

        assert d._dump_index == 0
        assert d._forward_pass_id == 0
        assert d._global_ctx == {}

    def test_dump_works_after_reset(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        d.dump("pre", torch.randn(3, 3))

        d.reset()
        d.on_forward_pass_start()
        d.dump("post", torch.randn(3, 3))

        filenames = _get_filenames(tmp_path)
        _assert_files(filenames, exist=["pre", "post"])
        post_file = _find_dump_file(tmp_path, name="post")
        assert "dump_index=1" in post_file.name


class TestDumperHttp:
    """Test /dumper/* HTTP control — parametrized over standalone vs sglang server."""

    @pytest.fixture(scope="class", params=["standalone", "sglang"])
    def dumper_http_url(self, request):
        if request.param == "standalone":
            http_port = find_available_port(40000)
            base_url = f"http://127.0.0.1:{http_port}"
            stop_event = multiprocessing.get_context("spawn").Event()
            thread = threading.Thread(
                target=run_distributed_test,
                args=(TestDumperHttp._standalone_mode_worker,),
                kwargs={"http_port": http_port, "stop_event": stop_event},
            )
            thread.start()
            try:
                TestDumperHttp._wait_for_http(base_url)
                yield base_url
            finally:
                stop_event.set()
                thread.join(timeout=10)
        else:
            base_url = DEFAULT_URL_FOR_TEST
            env = {**os.environ, "SGLANG_DUMPER_SERVER_PORT": "reuse"}
            proc = popen_launch_server(
                "Qwen/Qwen3-0.6B",
                base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--max-total-tokens", "128"],
                env=env,
            )
            try:
                yield base_url
            finally:
                kill_process_tree(proc.pid)

    @staticmethod
    def _standalone_mode_worker(rank, http_port: int, stop_event):
        dumper.configure(enable=False, server_port=str(http_port))
        dumper.on_forward_pass_start()
        stop_event.wait()

    @staticmethod
    def _wait_for_http(url: str, timeout: float = 30) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                requests.post(f"{url}/dumper/configure", json={}, timeout=2)
                return
            except requests.ConnectionError:
                time.sleep(0.5)
        raise TimeoutError(f"Standalone dumper HTTP server not reachable at {url}")

    @staticmethod
    def _post(base_url: str, method: str, **kwargs) -> list[dict]:
        resp = requests.post(f"{base_url}/dumper/{method}", json=kwargs or None)
        resp.raise_for_status()
        states = resp.json()
        assert isinstance(states, list) and len(states) >= 1
        return states

    @staticmethod
    def _assert_all_ranks(states: list[dict], path: str, expected):
        """Assert that ``state[path]`` equals ``expected`` on every rank."""
        keys = path.split(".")
        for rank, state in enumerate(states):
            val = state
            for k in keys:
                val = val[k]
            assert (
                val == expected
            ), f"rank {rank}: {path}={val!r}, expected {expected!r}"

    def test_configure_enable_toggle(self, dumper_http_url: str):
        for enable in [True, False]:
            self._post(dumper_http_url, "configure", enable=enable)
            states = self._post(dumper_http_url, "get_state")
            self._assert_all_ranks(states, "config.enable", enable)

    def test_configure_multi_field(self, dumper_http_url: str):
        self._post(
            dumper_http_url,
            "configure",
            enable=True,
            filter="layer_id=0",
            dir="/tmp/test_http",
        )
        states = self._post(dumper_http_url, "get_state")
        self._assert_all_ranks(states, "config.enable", True)
        self._assert_all_ranks(states, "config.filter", "layer_id=0")
        self._assert_all_ranks(states, "config.dir", "/tmp/test_http")

    def test_configure_clear_optional(self, dumper_http_url: str):
        self._post(dumper_http_url, "configure", filter="layer_id=0")
        self._post(dumper_http_url, "configure", filter=None)
        states = self._post(dumper_http_url, "get_state")
        self._assert_all_ranks(states, "config.filter", None)

    def test_reset(self, dumper_http_url: str):
        self._post(dumper_http_url, "configure", enable=True)
        self._post(dumper_http_url, "reset")
        states = self._post(dumper_http_url, "get_state")
        self._assert_all_ranks(states, "dump_index", 0)
        self._assert_all_ranks(states, "forward_pass_id", 0)

    def test_get_state(self, dumper_http_url: str):
        self._post(dumper_http_url, "configure", enable=True, filter="layer_id=[0-3]")
        states = self._post(dumper_http_url, "get_state")
        self._assert_all_ranks(states, "config.enable", True)
        self._assert_all_ranks(states, "config.filter", "layer_id=[0-3]")
        for state in states:
            assert "dump_index" in state
            assert "forward_pass_id" in state

    def test_all_ranks_consistent(self, dumper_http_url: str):
        self._post(dumper_http_url, "configure", enable=True, dir="/tmp/multi")
        states = self._post(dumper_http_url, "get_state")
        configs = [s["config"] for s in states]
        for rank_config in configs[1:]:
            assert rank_config == configs[0], f"rank configs diverged: {configs}"

    def test_error_unknown_field(self, dumper_http_url: str):
        resp = requests.post(
            f"{dumper_http_url}/dumper/configure",
            json={"nonexistent_field": 123},
        )
        assert resp.status_code == 400

    def test_error_wrong_type(self, dumper_http_url: str):
        resp = requests.post(
            f"{dumper_http_url}/dumper/configure",
            json={"enable": "not_a_bool"},
        )
        assert resp.status_code == 400


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
