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
    DumperConfig,
    _collective_with_timeout,
    _deepcopy_or_clone,
    _detect_recompute_status,
    _Dumper,
    _format_tags,
    _get_default_exp_name,
    _map_tensor,
    _materialize_value,
    _MegatronPlugin,
    _obj_to_dict,
    _RecomputeStatus,
    _register_forward_hook_or_replace_fn,
    _SGLangPlugin,
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
        assert DumperConfig.from_env() == DumperConfig()

    def test_from_env_bool(self):
        with temp_set_env(DUMPER_ENABLE="1"):
            assert DumperConfig.from_env().enable is True
        with temp_set_env(DUMPER_ENABLE="false"):
            assert DumperConfig.from_env().enable is False

    def test_from_env_str(self):
        with temp_set_env(DUMPER_FILTER="layer_id=0"):
            assert DumperConfig.from_env().filter == "layer_id=0"

    def test_from_env_dir(self):
        with temp_set_env(DUMPER_DIR="/my/dir"):
            assert DumperConfig.from_env().dir == "/my/dir"

    def test_from_env_int(self):
        with temp_set_env(DUMPER_COLLECTIVE_TIMEOUT="120"):
            assert DumperConfig.from_env().collective_timeout == 120

    def test_configure_overrides(self):
        d = _make_test_dumper("/tmp")
        d.configure(enable=False)
        assert d._config.enable is False
        d.configure(enable=True)
        assert d._config.enable is True

    def test_type_validation(self):
        with pytest.raises(TypeError, match="enable.*expected bool.*got str"):
            DumperConfig(enable="yes")
        with pytest.raises(
            TypeError, match="collective_timeout.*expected int.*got str"
        ):
            DumperConfig(collective_timeout="abc")
        with pytest.raises(TypeError, match="filter.*expected str.*got int"):
            DumperConfig(filter=123)

    def test_configure_default_skips_when_env_set(self):
        with temp_set_env(DUMPER_FILTER="from_env"):
            d = _Dumper(config=DumperConfig.from_env())
            d.configure_default(filter="from_code")
            assert d._config.filter == "from_env"

    def test_configure_default_applies_when_no_env(self):
        d = _Dumper(config=DumperConfig.from_env())
        d.configure_default(filter="from_code")
        assert d._config.filter == "from_code"

    def test_from_env_whitespace_treated_as_unset(self):
        with temp_set_env(DUMPER_FILTER="   "):
            assert DumperConfig.from_env().filter is None

    def test_may_enable_default_false(self):
        d = _Dumper(config=DumperConfig())
        assert d.may_enable is False

    def test_may_enable_true_when_enabled(self):
        d = _Dumper(config=DumperConfig(enable=True))
        assert d.may_enable is True

    def test_may_enable_true_when_server_port_set(self):
        d = _Dumper(config=DumperConfig(server_port="40000"))
        assert d.may_enable is True

        d2 = _Dumper(config=DumperConfig(server_port="reuse"))
        assert d2.may_enable is True


class TestServerPortParsed:
    def test_negative_returns_none(self):
        assert DumperConfig(server_port="-1").server_port_parsed is None

    def test_zero_returns_none(self):
        assert DumperConfig(server_port="0").server_port_parsed is None

    def test_positive_returns_int(self):
        result = DumperConfig(server_port="40000").server_port_parsed
        assert result == 40000
        assert isinstance(result, int)

    def test_reuse_returns_string(self):
        assert DumperConfig(server_port="reuse").server_port_parsed == "reuse"


class TestDefaultExpName:
    def test_starts_with_prefix(self):
        name = _get_default_exp_name(timeout_seconds=5)
        assert name.startswith("dump_")

    def test_suffix_format(self):
        name = _get_default_exp_name(timeout_seconds=5)
        suffix = name[len("dump_") :]
        assert len(suffix) == 22
        assert suffix[8] == "_"


class TestKvPairsParsing:
    def test_from_kv_pairs_none_returns_defaults(self):
        assert DumperConfig.from_kv_pairs(None) == DumperConfig()

    def test_from_kv_pairs_empty_returns_defaults(self):
        assert DumperConfig.from_kv_pairs([]) == DumperConfig()

    def test_from_kv_pairs_bool_field(self):
        cfg = DumperConfig.from_kv_pairs(["enable=true"])
        assert cfg.enable is True
        assert cfg.dir == "/tmp/dumper"

    def test_from_kv_pairs_bool_numeric(self):
        assert DumperConfig.from_kv_pairs(["enable=1"]).enable is True
        assert DumperConfig.from_kv_pairs(["enable=0"]).enable is False

    def test_from_kv_pairs_int_field(self):
        cfg = DumperConfig.from_kv_pairs(["collective_timeout=120"])
        assert cfg.collective_timeout == 120
        assert type(cfg.collective_timeout) is int

    def test_from_kv_pairs_int_field_zero_stays_int(self):
        cfg = DumperConfig.from_kv_pairs(["collective_timeout=0"])
        assert cfg.collective_timeout == 0
        assert type(cfg.collective_timeout) is int

    def test_from_kv_pairs_str_field_not_coerced(self):
        cfg = DumperConfig.from_kv_pairs(["server_port=0"])
        assert cfg.server_port == "0"
        assert type(cfg.server_port) is str

    def test_from_kv_pairs_str_field_one_stays_str(self):
        cfg = DumperConfig.from_kv_pairs(["server_port=1"])
        assert cfg.server_port == "1"
        assert type(cfg.server_port) is str

    def test_from_kv_pairs_optional_str_field(self):
        cfg = DumperConfig.from_kv_pairs(
            ["filter=layer_id is not None and layer_id < 3"]
        )
        assert cfg.filter == "layer_id is not None and layer_id < 3"

    def test_from_kv_pairs_optional_str_exp_name(self):
        cfg = DumperConfig.from_kv_pairs(["exp_name=my_experiment"])
        assert cfg.exp_name == "my_experiment"

    def test_from_kv_pairs_multiple_fields(self):
        cfg = DumperConfig.from_kv_pairs(
            [
                "enable=true",
                "dir=/my/dir",
                "filter=name == 'foo'",
                "collective_timeout=30",
                "enable_grad=1",
            ]
        )
        assert cfg.enable is True
        assert cfg.dir == "/my/dir"
        assert cfg.filter == "name == 'foo'"
        assert cfg.collective_timeout == 30
        assert cfg.enable_grad is True

    def test_from_kv_pairs_missing_equals_raises(self):
        with pytest.raises(ValueError, match="missing '='"):
            DumperConfig.from_kv_pairs(["enable"])

    def test_from_kv_pairs_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown config key"):
            DumperConfig.from_kv_pairs(["nonexistent=true"])

    def test_kv_pairs_to_dict_returns_only_explicit(self):
        d = DumperConfig._kv_pairs_to_dict(["enable=true", "dir=/x"])
        assert d == {"enable": True, "dir": "/x"}
        assert "filter" not in d
        assert "collective_timeout" not in d

    def test_kv_pairs_to_dict_none_returns_empty(self):
        assert DumperConfig._kv_pairs_to_dict(None) == {}

    def test_kv_pairs_to_dict_empty_returns_empty(self):
        assert DumperConfig._kv_pairs_to_dict([]) == {}

    def test_from_kv_pairs_value_with_equals_in_value(self):
        cfg = DumperConfig.from_kv_pairs(["filter=name == 'foo'"])
        assert cfg.filter == "name == 'foo'"

    def test_from_kv_pairs_type_validation_still_works(self):
        with pytest.raises(TypeError, match="collective_timeout.*expected int"):
            DumperConfig.from_kv_pairs(["collective_timeout=not_a_number"])


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

    def test_deepcopy_or_clone_tensor(self):
        original = torch.randn(3, 3)
        cloned = _deepcopy_or_clone(original)
        assert torch.equal(cloned, original)
        original.fill_(999.0)
        assert not torch.equal(cloned, original)

    def test_deepcopy_or_clone_non_tensor(self):
        original = {"a": [1, 2, 3]}
        cloned = _deepcopy_or_clone(original)
        assert cloned == original
        assert cloned is not original
        original["a"].append(4)
        assert len(cloned["a"]) == 3

    def test_get_tensor_info(self):
        info = get_tensor_info(torch.randn(10, 10))
        for key in ["shape=", "dtype=", "min=", "max=", "mean="]:
            assert key in info

        assert "value=42" in get_tensor_info(42)
        assert "min=None" in get_tensor_info(torch.tensor([]))


class TestMapTensor:
    def test_bare_tensor(self):
        t = torch.randn(4)
        result = _map_tensor(t, lambda x: x * 2)
        assert torch.equal(result, t * 2)

    def test_bare_tensor_no_change(self):
        t = torch.randn(4)
        result = _map_tensor(t, lambda x: x)
        assert result is t

    def test_dict_with_tensor_values(self):
        t1 = torch.randn(3)
        t2 = torch.randn(5)
        value = {"a": t1, "b": t2, "meta": "not a tensor"}
        result = _map_tensor(value, lambda x: x.clone())
        assert torch.equal(result["a"], t1)
        assert torch.equal(result["b"], t2)
        assert result["a"] is not t1
        assert result["b"] is not t2
        assert result["meta"] == "not a tensor"

    def test_dict_no_tensors(self):
        value = {"a": 1, "b": "hello"}
        result = _map_tensor(value, lambda x: x.clone())
        assert result == value

    def test_nested_dict(self):
        inner_t = torch.randn(3)
        value = {"outer": {"inner": inner_t, "label": "ok"}, "top": torch.randn(2)}
        result = _map_tensor(value, lambda x: x.clone())
        assert torch.equal(result["outer"]["inner"], inner_t)
        assert result["outer"]["inner"] is not inner_t
        assert result["outer"]["label"] == "ok"
        assert result is not value
        assert result["outer"] is not value["outer"]

    def test_non_tensor_non_dict(self):
        result = _map_tensor(42, lambda x: x.clone())
        assert result == 42


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

    def test_shared_storage_not_bloated(self, tmp_path):
        big = torch.randn(1000, 1000)
        view = big[0]
        path = str(tmp_path / "view.pt")

        _torch_save({"value": view, "meta": {}}, path)

        file_size = Path(path).stat().st_size
        expected_max = view.nelement() * view.element_size() * 10
        assert file_size < expected_max, (
            f"File {file_size} bytes but view is only "
            f"{view.nelement() * view.element_size()} bytes — "
            f"torch.save likely serialized the full "
            f"{big.nelement() * big.element_size()} byte storage"
        )

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
            DUMPER_ENABLE="1",
            DUMPER_DIR=str(tmp_path),
        ):
            run_distributed_test(self._test_basic_func, tmpdir=str(tmp_path))

    @staticmethod
    def _test_basic_func(rank, tmpdir):
        tensor = torch.randn(10, 10, device=f"cuda:{rank}")

        dumper.dump("tensor_a", tensor, arg=100)
        dumper.step()

        dumper.set_ctx(ctx_arg=200)
        dumper.dump("tensor_b", tensor)
        dumper.set_ctx(ctx_arg=None)
        dumper.step()

        dumper.configure(filter="False")
        dumper.dump("tensor_skip", tensor)
        dumper.configure(filter=None)
        dumper.step()

        dumper.dump_dict("obj", {"a": torch.randn(3, device=f"cuda:{rank}"), "b": 42})
        dumper.step()

        dist.barrier()
        filenames = _get_filenames(tmpdir)
        _assert_files(
            filenames,
            exist=["tensor_a", "tensor_b", "arg=100", "ctx_arg=200", "obj_a", "obj_b"],
            not_exist=["tensor_skip"],
        )

    def test_collective_timeout(self):
        with temp_set_env(DUMPER_ENABLE="1"):
            run_distributed_test(self._test_collective_timeout_func)

    @staticmethod
    def _test_collective_timeout_func(rank):
        dumper = _Dumper(
            config=DumperConfig(
                enable=True,
                collective_timeout=3,
            ),
        )

        with _capture_stdout() as captured:
            if rank != 0:
                time.sleep(6)
            dumper.step()

        output = captured.getvalue()
        print(f"Rank {rank} captured output: {output!r}")

        if rank == 0:
            assert "WARNING" in output, f"Expected WARNING in rank 0 output: {output}"
            assert "has not completed after 3s" in output

    def test_file_content_correctness(self, tmp_path):
        with temp_set_env(
            DUMPER_ENABLE="1",
            DUMPER_DIR=str(tmp_path),
        ):
            run_distributed_test(self._test_file_content_func, tmpdir=str(tmp_path))

    @staticmethod
    def _test_file_content_func(rank, tmpdir):
        tensor = torch.arange(12, device=f"cuda:{rank}").reshape(3, 4).float()

        dumper.dump("content_check", tensor)
        dumper.step()

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
            DUMPER_ENABLE="1",
            DUMPER_DIR=str(tmp_path),
            DUMPER_FILTER="name.startswith('keep')",
        ):
            run_distributed_test(self._test_filter_func, tmpdir=str(tmp_path))

    @staticmethod
    def _test_filter_func(rank, tmpdir):
        dumper.dump("keep_this", torch.randn(5, device=f"cuda:{rank}"))
        dumper.dump("skip_this", torch.randn(5, device=f"cuda:{rank}"))
        dumper.dump("not_keep_this", torch.randn(5, device=f"cuda:{rank}"))
        dumper.step()

        dist.barrier()
        filenames = _get_filenames(tmpdir)
        _assert_files(
            filenames,
            exist=["keep_this"],
            not_exist=["skip_this", "not_keep_this"],
        )

    def test_save_false(self, tmp_path):
        with temp_set_env(
            DUMPER_ENABLE="1",
            DUMPER_DIR=str(tmp_path),
        ):
            run_distributed_test(self._test_save_false_func, tmpdir=str(tmp_path))

    @staticmethod
    def _test_save_false_func(rank, tmpdir):
        dumper.dump("no_save_tensor", torch.randn(5, device=f"cuda:{rank}"), save=False)
        dumper.step()

        dist.barrier()
        assert len(_get_filenames(tmpdir)) == 0


class TestDumpEnableFlags:
    def test_all_enables_false_no_output(self, tmp_path):
        d = _make_test_dumper(tmp_path, enable_value=False, enable_grad=False)
        d.dump("should_skip", torch.randn(3, 3))
        assert len(_get_filenames(tmp_path)) == 0


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

    def test_capture_output_nested_raises(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        with d.capture_output():
            with pytest.raises(AssertionError):
                with d.capture_output():
                    pass

    def test_capture_output_respects_filter(self, tmp_path):
        d = _make_test_dumper(tmp_path, filter="'keep' in name")

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
        assert "step" in meta
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
    """Create a _Dumper for CPU testing without distributed."""
    defaults = dict(
        enable=True,
        dir=str(tmp_path),
        exp_name="test",
    )
    defaults.update(overrides)
    config = DumperConfig(**defaults)
    return _Dumper(config=config)


def _get_filenames(tmpdir):
    return {f.name for f in Path(tmpdir).glob("*/*.pt")}


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
        for f in Path(tmpdir).glob("*/*.pt")
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
        sglang_info = _SGLangPlugin().collect_parallel_info()
        assert isinstance(sglang_info, dict)

        megatron_info = _MegatronPlugin().collect_parallel_info()
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

    def test_dump_grad_captures_step(self, tmp_path):
        d = _make_test_dumper(tmp_path, enable_grad=True)
        d._state.step = 42
        x = torch.randn(3, 3, requires_grad=True)
        y = (x * 2).sum()

        d.dump("id_test", x)
        d._state.step = 999
        y.backward()

        grad_file = _find_dump_file(tmp_path, name="grad__id_test")
        assert "step=42" in grad_file.name

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
        d = _make_test_dumper(tmp_path, filter="layer_id == 0")
        d.dump("tensor_a", torch.randn(3), layer_id=0)
        d.dump("tensor_b", torch.randn(3), layer_id=1)

        filenames = _get_filenames(tmp_path)
        _assert_files(filenames, exist=["tensor_a"], not_exist=["tensor_b"])

    def test_filter_matches_global_ctx(self, tmp_path):
        d = _make_test_dumper(tmp_path, filter="ctx_arg == 200")
        d.set_ctx(ctx_arg=200)
        d.dump("tensor_a", torch.randn(3))
        d.set_ctx(ctx_arg=None)
        d.dump("tensor_b", torch.randn(3))

        filenames = _get_filenames(tmp_path)
        _assert_files(filenames, exist=["tensor_a"], not_exist=["tensor_b"])

    def test_filter_matches_name(self, tmp_path):
        d = _make_test_dumper(tmp_path, filter="'keep' in name")
        d.dump("keep_this", torch.randn(3))
        d.dump("skip_this", torch.randn(3))

        filenames = _get_filenames(tmp_path)
        _assert_files(filenames, exist=["keep_this"], not_exist=["skip_this"])

    def test_filter_expr_range(self, tmp_path):
        d = _make_test_dumper(tmp_path, filter="layer_id is not None and layer_id < 3")
        d.dump("t0", torch.randn(3), layer_id=0)
        d.dump("t1", torch.randn(3), layer_id=1)
        d.dump("t5", torch.randn(3), layer_id=5)

        filenames = _get_filenames(tmp_path)
        _assert_files(filenames, exist=["name=t0", "name=t1"], not_exist=["name=t5"])

    def test_filter_expr_with_none(self, tmp_path):
        d = _make_test_dumper(tmp_path, filter="layer_id is None or layer_id < 3")
        d.dump("no_layer", torch.randn(3))
        d.dump("layer0", torch.randn(3), layer_id=0)
        d.dump("layer5", torch.randn(3), layer_id=5)

        filenames = _get_filenames(tmp_path)
        _assert_files(
            filenames,
            exist=["no_layer", "layer0"],
            not_exist=["layer5"],
        )

    def test_filter_expr_with_re_search(self, tmp_path):
        d = _make_test_dumper(tmp_path, filter="search(r'attn|mlp', name)")
        d.dump("self_attn", torch.randn(3))
        d.dump("mlp_proj", torch.randn(3))
        d.dump("layernorm", torch.randn(3))

        filenames = _get_filenames(tmp_path)
        _assert_files(
            filenames,
            exist=["self_attn", "mlp_proj"],
            not_exist=["layernorm"],
        )

    def test_filter_expr_syntax_error(self, tmp_path):
        d = _make_test_dumper(tmp_path, filter="layer_id ===")
        with pytest.raises(SyntaxError):
            d.dump("tensor", torch.randn(3))

    def test_no_filter_dumps_all(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        d.dump("a", torch.randn(3))
        d.dump("b", torch.randn(3))

        filenames = _get_filenames(tmp_path)
        _assert_files(filenames, exist=["name=a", "name=b"])


class TestDumpModel:
    def test_grad_basic(self, tmp_path):
        d = _make_test_dumper(
            tmp_path, enable_model_grad=True, enable_model_value=False
        )
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
        d = _make_test_dumper(
            tmp_path, enable_model_value=True, enable_model_grad=False
        )
        model = torch.nn.Linear(4, 2, bias=False)

        d.dump_model(model, name_prefix="model")

        _assert_files(
            _get_filenames(tmp_path),
            exist=["model__weight"],
        )

    def test_no_grad_skipped(self, tmp_path):
        d = _make_test_dumper(
            tmp_path, enable_model_grad=True, enable_model_value=False
        )
        model = torch.nn.Linear(4, 2)

        d.dump_model(model, name_prefix="model")

        filenames = _get_filenames(tmp_path)
        assert len(filenames) == 0

    def test_filter(self, tmp_path):
        d = _make_test_dumper(
            tmp_path,
            enable_model_value=True,
            enable_model_grad=True,
            filter="'weight' in name",
        )
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
        d = _make_test_dumper(
            tmp_path, enable_model_grad=True, enable_model_value=False
        )
        model = torch.nn.Linear(4, 2, bias=False)
        x = torch.ones(1, 4)
        y = model(x).sum()
        y.backward()

        d.dump_model(model, name_prefix="p")

        path = _find_dump_file(tmp_path, name="grad__p__weight")
        assert torch.equal(_load_dump(path)["value"], model.weight.grad)

    def test_disable_model_grad(self, tmp_path):
        d = _make_test_dumper(
            tmp_path, enable_model_value=True, enable_model_grad=False
        )
        model = torch.nn.Linear(4, 2)
        x = torch.randn(3, 4)
        y = model(x).sum()
        y.backward()

        d.dump_model(model, name_prefix="model")

        filenames = _get_filenames(tmp_path)
        assert all("grad" not in f for f in filenames)

    def test_parameter_saved_as_parameter(self, tmp_path):
        d = _make_test_dumper(
            tmp_path, enable_model_value=True, enable_model_grad=False
        )
        model = torch.nn.Linear(4, 2, bias=False)

        d.dump_model(model, name_prefix="p")

        path = _find_dump_file(tmp_path, name="p__weight")
        loaded = _load_dump(path)
        assert isinstance(loaded["value"], torch.nn.Parameter)
        assert torch.equal(loaded["value"], model.weight)

    def test_unpicklable_parameter_falls_back_to_data(self, tmp_path):
        class BadParam(torch.nn.Parameter):
            def __reduce_ex__(self, protocol):
                raise RuntimeError("not pickleable")

        d = _make_test_dumper(
            tmp_path, enable_model_value=True, enable_model_grad=False
        )
        model = torch.nn.Linear(4, 2, bias=False)
        model.weight = BadParam(model.weight.data)

        d.dump_model(model, name_prefix="p")

        path = _find_dump_file(tmp_path, name="p__weight")
        loaded = _load_dump(path)
        assert isinstance(loaded["value"], torch.Tensor)
        assert not isinstance(loaded["value"], torch.nn.Parameter)
        assert torch.equal(loaded["value"], model.weight.data)

    def test_disable_model_value(self, tmp_path):
        d = _make_test_dumper(
            tmp_path, enable_model_grad=True, enable_model_value=False
        )
        model = torch.nn.Linear(4, 2, bias=False)
        x = torch.ones(1, 4)
        y = model(x).sum()
        y.backward()

        d.dump_model(model, name_prefix="model")

        filenames = _get_filenames(tmp_path)
        assert all("grad" in f for f in filenames)


class TestCleanup:
    def test_cleanup_removes_old_dumps(self, tmp_path):
        old_dir = tmp_path / "dump_old"
        old_dir.mkdir()
        (old_dir / "dummy.pt").touch()

        dumper = _make_test_dumper(tmp_path, cleanup_previous=True)
        dumper.dump("new_tensor", torch.randn(3, 3))

        assert not old_dir.exists()
        _assert_files(_get_filenames(tmp_path), exist=["new_tensor"])

    def test_cleanup_removes_exp_name_dir(self, tmp_path):
        exp_name = "my_custom_exp"
        old_exp_dir = tmp_path / exp_name
        old_exp_dir.mkdir()
        (old_exp_dir / "old_data.pt").touch()

        dumper = _make_test_dumper(tmp_path, exp_name=exp_name, cleanup_previous=True)
        dumper.dump("new_tensor", torch.randn(3, 3))

        assert not (tmp_path / exp_name / "old_data.pt").exists()
        _assert_files(_get_filenames(tmp_path), exist=["new_tensor"])

    def test_cleanup_removes_both_dump_prefix_and_exp_name(self, tmp_path):
        old_dump = tmp_path / "dump_old"
        old_dump.mkdir()
        (old_dump / "dummy.pt").touch()

        exp_name = "custom_run"
        old_exp = tmp_path / exp_name
        old_exp.mkdir()
        (old_exp / "stale.pt").touch()

        dumper = _make_test_dumper(tmp_path, exp_name=exp_name, cleanup_previous=True)
        dumper.dump("new_tensor", torch.randn(3, 3))

        assert not old_dump.exists()
        assert not (tmp_path / exp_name / "stale.pt").exists()
        _assert_files(_get_filenames(tmp_path), exist=["new_tensor"])

    def test_no_cleanup_by_default(self, tmp_path):
        old_dir = tmp_path / "dump_old"
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

        assert d._state.dump_index == 0
        assert d._state.step == 0
        assert d._state.global_ctx == {}

    def test_dump_works_after_reset(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        d.dump("pre", torch.randn(3, 3))

        d.reset()
        d.dump("post", torch.randn(3, 3))

        filenames = _get_filenames(tmp_path)
        _assert_files(filenames, exist=["pre", "post"])
        post_file = _find_dump_file(tmp_path, name="post")
        assert "dump_index=1" in post_file.name

    def test_cleanup_previous_re_triggers_after_reset(self, tmp_path):
        """Miles pattern: reset() + configure(cleanup_previous=True) should re-clean."""
        exp_alpha = "exp_alpha"
        exp_beta = "exp_beta"

        (tmp_path / exp_alpha).mkdir()
        (tmp_path / exp_alpha / "stale.pt").touch()
        (tmp_path / exp_beta).mkdir()
        (tmp_path / exp_beta / "stale.pt").touch()

        d = _make_test_dumper(tmp_path, exp_name=exp_alpha, cleanup_previous=True)
        d.dump("phase1", torch.randn(2, 2))

        d.reset()
        d.configure(exp_name=exp_beta, cleanup_previous=True)
        d.dump("phase2", torch.randn(2, 2))

        assert not (tmp_path / exp_alpha / "stale.pt").exists()
        assert not (tmp_path / exp_beta / "stale.pt").exists()
        filenames = _get_filenames(tmp_path)
        _assert_files(filenames, exist=["phase1", "phase2"])

    def test_no_cleanup_when_config_false(self, tmp_path):
        """cleanup_previous=False: handled stays False but no cleanup runs."""
        old_dir = tmp_path / "dump_old"
        old_dir.mkdir()
        (old_dir / "dummy.pt").touch()

        d = _make_test_dumper(tmp_path, cleanup_previous=False)
        d.dump("tensor", torch.randn(2, 2))

        assert old_dir.exists()
        assert d._state.cleanup_previous_handled is False

    def test_multi_phase_switch(self, tmp_path):
        """Simulate Miles multi-phase: configure → dump → reset → configure new phase → dump."""
        d = _make_test_dumper(tmp_path, cleanup_previous=True)

        d.configure(exp_name="fwd_only")
        d.dump("weight", torch.randn(2, 2))
        d.step()
        d.configure(enable=False)

        d.reset()
        d.configure(exp_name="fwd_bwd", enable=True, cleanup_previous=True)
        d.dump("weight", torch.randn(2, 2))
        d.step()

        fwd_only_files = list(Path(tmp_path).glob("fwd_only/*.pt"))
        fwd_bwd_files = list(Path(tmp_path).glob("fwd_bwd/*.pt"))
        assert len(fwd_only_files) > 0
        assert len(fwd_bwd_files) > 0
        assert d._state.step == 1
        assert d._state.dump_index == 1

    def test_reset_removes_non_intrusive_hooks(self, tmp_path):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 4),
        )
        d = _make_test_dumper(tmp_path, non_intrusive_mode="all")
        d.register_non_intrusive_dumper(model)

        x = torch.randn(2, 4)
        with d.capture_output() as captured:
            model(x)
        assert len(captured) > 0

        d.reset()
        d.configure(enable=True, dir=str(tmp_path), non_intrusive_mode="all")

        with d.capture_output() as captured_after:
            model(x)
        assert len(captured_after) == 0

    def test_reset_removes_non_intrusive_hooks_multiple_models(self, tmp_path):
        model_a = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.ReLU(),
        )
        model_b = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.ReLU(),
        )
        d = _make_test_dumper(tmp_path, non_intrusive_mode="all")
        d.register_non_intrusive_dumper(model_a)
        d.register_non_intrusive_dumper(model_b)

        x = torch.randn(2, 4)
        with d.capture_output() as captured:
            model_a(x)
            model_b(x)
        assert len(captured) > 0

        d.reset()
        d.configure(enable=True, dir=str(tmp_path), non_intrusive_mode="all")

        with d.capture_output() as captured_a:
            model_a(x)
        assert len(captured_a) == 0

        with d.capture_output() as captured_b:
            model_b(x)
        assert len(captured_b) == 0


def _dumper_worker(rank, http_port: int, stop_event):
    """Minimal distributed dumper worker: configure, step (triggers ZMQ setup), then wait."""
    dumper.configure(enable=False, server_port=str(http_port))
    dumper.step()
    stop_event.wait()


def _wait_for_dumper_http(url: str, timeout: float = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.post(f"{url}/dumper/configure", json={}, timeout=2)
            return
        except requests.ConnectionError:
            time.sleep(0.5)
    raise TimeoutError(f"Dumper HTTP server not reachable at {url}")


class TestZmqPortIsolation:
    """Multiple independent dumper instances (each with 2 ranks) must not conflict on ZMQ ports."""

    NUM_INSTANCES = 3

    def test_concurrent_instances_no_port_conflict(self):
        ports = [
            find_available_port(40000 + i * 1000) for i in range(self.NUM_INSTANCES)
        ]
        stop_events = []
        threads = []
        ctx = multiprocessing.get_context("spawn")

        for port in ports:
            stop_event = ctx.Event()
            stop_events.append(stop_event)
            thread = threading.Thread(
                target=run_distributed_test,
                args=(_dumper_worker,),
                kwargs={"http_port": port, "stop_event": stop_event},
            )
            thread.start()
            threads.append(thread)

        try:
            for port in ports:
                _wait_for_dumper_http(f"http://127.0.0.1:{port}")

            for i, port in enumerate(ports):
                resp = requests.post(
                    f"http://127.0.0.1:{port}/dumper/get_state", json={}
                )
                resp.raise_for_status()
                states = resp.json()
                assert (
                    len(states) == 2
                ), f"Instance {i} (port {port}): expected 2 ranks, got {len(states)}"
        finally:
            for event in stop_events:
                event.set()
            for thread in threads:
                thread.join(timeout=10)


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
                args=(_dumper_worker,),
                kwargs={"http_port": http_port, "stop_event": stop_event},
            )
            thread.start()
            try:
                _wait_for_dumper_http(base_url)
                yield base_url
            finally:
                stop_event.set()
                thread.join(timeout=10)
        else:
            base_url = DEFAULT_URL_FOR_TEST
            env = {**os.environ, "DUMPER_SERVER_PORT": "reuse"}
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
            filter="layer_id == 0",
            dir="/tmp/test_http",
        )
        states = self._post(dumper_http_url, "get_state")
        self._assert_all_ranks(states, "config.enable", True)
        self._assert_all_ranks(states, "config.filter", "layer_id == 0")
        self._assert_all_ranks(states, "config.dir", "/tmp/test_http")

    def test_configure_clear_optional(self, dumper_http_url: str):
        self._post(dumper_http_url, "configure", filter="layer_id == 0")
        self._post(dumper_http_url, "configure", filter=None)
        states = self._post(dumper_http_url, "get_state")
        self._assert_all_ranks(states, "config.filter", None)

    def test_reset(self, dumper_http_url: str):
        self._post(dumper_http_url, "configure", enable=True)
        self._post(dumper_http_url, "reset")
        states = self._post(dumper_http_url, "get_state")
        self._assert_all_ranks(states, "dump_index", 0)
        self._assert_all_ranks(states, "step", 0)

    def test_get_state(self, dumper_http_url: str):
        self._post(
            dumper_http_url,
            "configure",
            enable=True,
            filter="layer_id is not None and layer_id < 3",
        )
        states = self._post(dumper_http_url, "get_state")
        self._assert_all_ranks(states, "config.enable", True)
        self._assert_all_ranks(
            states, "config.filter", "layer_id is not None and layer_id < 3"
        )
        for state in states:
            assert "dump_index" in state
            assert "step" in state

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

    def test_error_unknown_method(self, dumper_http_url: str):
        resp = requests.post(
            f"{dumper_http_url}/dumper/nonexistent",
            json={},
        )
        assert resp.status_code == 400

    def test_error_wrong_type(self, dumper_http_url: str):
        resp = requests.post(
            f"{dumper_http_url}/dumper/configure",
            json={"enable": "not_a_bool"},
        )
        assert resp.status_code == 400


class TestRegisterForwardHookOrReplaceFn:
    def test_unknown_mode_raises(self):
        module = torch.nn.Linear(4, 4)
        with pytest.raises(ValueError, match="Unknown mode"):
            _register_forward_hook_or_replace_fn(
                module,
                pre_hook=lambda _mod, _input: None,
                hook=lambda _mod, _input, _output: None,
                mode="bad",
            )


class _NonIntrusiveTestBase:
    _PREFIX = "non_intrusive__"

    @staticmethod
    def _assert_captured_contains(
        captured: dict, expected: list[str], prefix: str = "non_intrusive__"
    ) -> None:
        for suffix in expected:
            key = f"{prefix}{suffix}"
            assert key in captured, f"missing {key}"

    @staticmethod
    def _wrap_as_outer(inner_cls: type) -> torch.nn.Module:
        """Wrap an inner module class as OuterModel.model, mimicking typical model nesting."""

        class OuterModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = inner_cls()

            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)

        return OuterModel()

    @staticmethod
    def _make_dumper(tmp_path, **overrides) -> "_Dumper":
        return _make_test_dumper(tmp_path, non_intrusive_mode="all", **overrides)

    def _run(self, tmp_path, inner_cls, **dumper_overrides):
        d = self._make_dumper(tmp_path, **dumper_overrides)
        model = self._wrap_as_outer(inner_cls)
        d.register_non_intrusive_dumper(model)
        x = torch.randn(2, 4)
        with d.capture_output() as captured:
            output = model(x)
        return captured, x, output


class TestNonIntrusiveDumper(_NonIntrusiveTestBase):
    """Tests for mode='all' — hooks on every module, non_intrusive__ prefix."""

    def test_basic_inputs_and_outputs(self, tmp_path):
        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.linear(x))

        captured, x, output = self._run(tmp_path, Inner)

        self._assert_captured_contains(
            captured,
            [
                "output",
                "inputs.0",
                "model.output",
                "model.inputs.0",
                "model.linear.output",
                "model.linear.inputs.0",
                "model.relu.output",
                "model.relu.inputs.0",
            ],
        )
        P = self._PREFIX
        assert torch.allclose(captured[f"{P}output"]["value"], output)

    def test_inputs_dumped_before_forward(self, tmp_path):
        """Inputs are captured *before* forward(); in-place mutation must not affect them."""

        class Mutator(torch.nn.Module):
            def forward(self, x):
                x.fill_(999.0)
                return x

        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mutator = Mutator()

            def forward(self, x):
                return self.mutator(x)

        d = self._make_dumper(tmp_path)
        model = self._wrap_as_outer(Inner)
        d.register_non_intrusive_dumper(model)

        x = torch.randn(2, 4)
        original_x = x.clone()
        with d.capture_output() as captured:
            model(x)

        P = self._PREFIX
        dumped_input = captured[f"{P}model.mutator.inputs.0"]["value"]
        assert torch.allclose(dumped_input, original_x), (
            f"pre-hook should capture inputs before forward mutates them; "
            f"got {dumped_input} but expected {original_x}"
        )

        dumped_output = captured[f"{P}model.mutator.output"]["value"]
        assert (
            dumped_output == 999.0
        ).all(), "post-hook should capture outputs after forward"

    def test_hooks_all_module_levels(self, tmp_path):
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv_proj = torch.nn.Linear(4, 12)
                self.o_proj = torch.nn.Linear(4, 4)

            def forward(self, x):
                _qkv = self.qkv_proj(x)
                return self.o_proj(x)

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attention()
                self.mlp = torch.nn.Linear(4, 4)

            def forward(self, x):
                x = self.self_attn(x)
                return self.mlp(x)

        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer()])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        captured, x, output = self._run(tmp_path, Inner)

        self._assert_captured_contains(
            captured,
            [
                "output",
                "model.output",
                "model.layers.0.output",
                "model.layers.0.self_attn.output",
                "model.layers.0.self_attn.qkv_proj.output",
                "model.layers.0.self_attn.o_proj.output",
                "model.layers.0.mlp.output",
                "model.layers.0.self_attn.qkv_proj.inputs.0",
                "model.layers.0.self_attn.o_proj.inputs.0",
                "model.layers.0.mlp.inputs.0",
            ],
        )
        P = self._PREFIX
        assert f"{P}model.layers.output" not in captured

    def test_multi_tensor_tuple_output(self, tmp_path):
        class TupleModule(torch.nn.Module):
            def forward(self, x):
                return x, x * 2

        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.split = TupleModule()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                a, b = self.split(x)
                return self.linear(a + b)

        captured, x, output = self._run(tmp_path, Inner)

        assert "non_intrusive__model.split.output.0" in captured
        assert "non_intrusive__model.split.output.1" in captured
        assert torch.allclose(
            captured["non_intrusive__model.split.output.0"]["value"], x
        )

    def test_single_tensor_tuple_collapses(self, tmp_path):
        class SingleTupleModule(torch.nn.Module):
            def forward(self, x):
                return (x * 3,)

        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.wrap = SingleTupleModule()

            def forward(self, x):
                return self.wrap(x)[0]

        captured, x, output = self._run(tmp_path, Inner)

        assert "non_intrusive__model.wrap.output" in captured
        assert "non_intrusive__model.wrap.output.0" not in captured

    def test_multiple_forward_inputs(self, tmp_path):
        class TwoInputModule(torch.nn.Module):
            def forward(self, x, mask):
                return x * mask

        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mul = TwoInputModule()

            def forward(self, x):
                mask = torch.ones_like(x)
                return self.mul(x, mask)

        captured, x, output = self._run(tmp_path, Inner)

        assert "non_intrusive__model.mul.inputs.0" in captured
        assert "non_intrusive__model.mul.inputs.1" in captured

    def test_none_output_only_dumps_inputs(self, tmp_path):
        class NoneModule(torch.nn.Module):
            def forward(self, x):
                return None

        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sink = NoneModule()

            def forward(self, x):
                self.sink(x)
                return x

        captured, x, output = self._run(tmp_path, Inner)

        assert "non_intrusive__model.sink.inputs.0" in captured
        assert not any(
            k.startswith("non_intrusive__model.sink.output") for k in captured
        )

    def test_non_tensor_value_silently_skipped(self, tmp_path):
        class IntModule(torch.nn.Module):
            def forward(self, x):
                return 42

        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = IntModule()

            def forward(self, x):
                self.const(x)
                return x

        captured, x, output = self._run(tmp_path, Inner)

        assert "non_intrusive__model.const.inputs.0" in captured
        assert not any(
            k.startswith("non_intrusive__model.const.output") for k in captured
        )

    def test_root_module_name_no_malformed_dots(self, tmp_path):
        d = self._make_dumper(tmp_path)
        model = torch.nn.Linear(4, 4)
        d.register_non_intrusive_dumper(model)

        x = torch.randn(2, 4)
        with d.capture_output() as captured:
            model(x)

        for key in captured:
            assert not key.startswith("non_intrusive__."), f"malformed key: {key}"
            assert ".." not in key, f"double dot in key: {key}"

        assert "non_intrusive__output" in captured
        assert "non_intrusive__inputs.0" in captured

    def test_respects_dumper_filter(self, tmp_path):
        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.linear(x))

        captured, x, output = self._run(
            tmp_path, Inner, filter="name == 'non_intrusive__model.linear.output'"
        )

        assert "non_intrusive__model.linear.output" in captured
        assert "non_intrusive__model.relu.output" not in captured
        assert "non_intrusive__model.linear.inputs.0" not in captured

    def test_disabled_dumper_no_output(self, tmp_path):
        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        d = self._make_dumper(tmp_path)
        d.configure(enable=False)
        model = self._wrap_as_outer(Inner)
        d.register_non_intrusive_dumper(model)

        x = torch.randn(2, 4)
        with d.capture_output() as captured:
            model(x)

        assert len(captured) == 0


def _make_forward_batch():
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=2,
        input_ids=torch.tensor([10, 20]),
        req_pool_indices=torch.zeros(2, dtype=torch.long),
        seq_lens=torch.tensor([5, 6]),
        out_cache_loc=torch.zeros(2, dtype=torch.long),
        seq_lens_sum=11,
        positions=torch.tensor([0, 1]),
    )


class TestNonIntrusiveDumperConfigMode(_NonIntrusiveTestBase):
    @staticmethod
    def _build_model() -> torch.nn.Module:
        class SubLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, forward_batch):
                return self.linear(
                    forward_batch.input_ids.float().unsqueeze(-1).expand(-1, 4)
                )

        class Root(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = SubLayer()

            def forward(self, forward_batch):
                return self.layer(forward_batch)

        return Root()

    def _run(self, tmp_path, mode: str) -> tuple:
        d = _make_test_dumper(tmp_path, non_intrusive_mode=mode)
        model = self._build_model()
        d.register_non_intrusive_dumper(model)
        forward_batch = _make_forward_batch()
        with d.capture_output() as captured:
            model(forward_batch)
        return captured, forward_batch

    def test_off_mode(self, tmp_path):
        captured, _ = self._run(tmp_path, "off")
        assert len(captured) == 0

    def test_core_mode(self, tmp_path):
        captured, fb = self._run(tmp_path, "core")

        # core fields dumped with clean names
        assert "input_ids" in captured
        assert "positions" in captured
        assert "seq_lens" in captured
        assert torch.equal(captured["input_ids"]["value"], fb.input_ids)
        assert torch.equal(captured["positions"]["value"], fb.positions)
        assert torch.equal(captured["seq_lens"]["value"], fb.seq_lens)

        # nothing with non_intrusive__ prefix
        assert not any(k.startswith("non_intrusive__") for k in captured)

    def test_all_mode(self, tmp_path):
        captured, fb = self._run(tmp_path, "all")

        # core fields dumped with clean names
        assert "input_ids" in captured
        assert "positions" in captured
        assert "seq_lens" in captured
        assert torch.equal(captured["input_ids"]["value"], fb.input_ids)
        assert torch.equal(captured["positions"]["value"], fb.positions)
        assert torch.equal(captured["seq_lens"]["value"], fb.seq_lens)

        # core fields NOT duplicated with prefix
        for field in ("input_ids", "positions", "seq_lens"):
            assert not any(
                k.startswith("non_intrusive__") and k.endswith(field) for k in captured
            )

        # ForwardBatch skipped on sub-modules (no duplication)
        assert not any(
            k.startswith("non_intrusive__layer.inputs.") and "seq_lens" in k
            for k in captured
        ), f"ForwardBatch skipped on sub-module, got: {list(captured.keys())}"

        # regular tensor outputs on sub-modules still dumped
        assert "non_intrusive__layer.linear.output" in captured
        assert "non_intrusive__layer.output" in captured


class _LayerWithNumber(torch.nn.Module):
    """Test helper: module with a ``layer_number`` attribute (Megatron style)."""

    def __init__(self, layer_number: int):
        super().__init__()
        self.layer_number = layer_number
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


class TestNonIntrusiveLayerIdCtx(_NonIntrusiveTestBase):
    """Tests for automatic layer_id context injection via set_ctx."""

    def test_layer_id_from_layer_number(self, tmp_path):
        """Megatron PP: layer_number (1-based global) -> layer_id = layer_number - 1."""

        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [_LayerWithNumber(10), _LayerWithNumber(11)]
                )

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        captured, x, output = self._run(tmp_path, Inner)

        layer0_key = "non_intrusive__model.layers.0.linear.output"
        layer1_key = "non_intrusive__model.layers.1.linear.output"
        assert layer0_key in captured
        assert layer1_key in captured
        assert captured[layer0_key]["meta"]["layer_id"] == 9
        assert captured[layer1_key]["meta"]["layer_id"] == 10

        root_key = "non_intrusive__output"
        assert root_key in captured
        assert "layer_id" not in captured[root_key]["meta"]

    def test_layer_id_from_layer_id_attr(self, tmp_path):
        """SGLang style: module has layer_id attribute directly."""

        class Layer(torch.nn.Module):
            def __init__(self, layer_id: int):
                super().__init__()
                self.layer_id = layer_id
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer(5)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        captured, x, output = self._run(tmp_path, Inner)

        layer_key = "non_intrusive__model.layers.0.linear.output"
        assert layer_key in captured
        assert captured[layer_key]["meta"]["layer_id"] == 5

    def test_layer_id_fallback_from_module_name(self, tmp_path):
        """layers.N modules without layer_number/layer_id -> layer_id from module name."""

        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)]
                )

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        captured, x, output = self._run(tmp_path, Inner)

        assert len(captured) > 0
        input_keys: list[str] = [
            k for k in captured if "model.layers." in k and "inputs" in k
        ]
        assert len(input_keys) > 0
        for key in input_keys:
            meta = captured[key]["meta"]
            assert "layer_id" in meta, f"{key} missing layer_id"
            if "layers.0" in key:
                assert meta["layer_id"] == 0
            elif "layers.1" in key:
                assert meta["layer_id"] == 1

    def test_filter_by_layer_id(self, tmp_path):
        """filter='layer_id == 0' keeps only layer 0 dumps."""

        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [_LayerWithNumber(1), _LayerWithNumber(2)]
                )

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        captured, x, output = self._run(tmp_path, Inner, filter="layer_id == 0")

        layer0_keys = [k for k in captured if "layers.0" in k]
        layer1_keys = [k for k in captured if "layers.1" in k]
        assert len(layer0_keys) > 0, "layer 0 dumps should be kept"
        assert len(layer1_keys) == 0, f"layer 1 dumps should be filtered: {layer1_keys}"


class TestDumperE2E:
    def test_step_and_non_intrusive_hooks(self, tmp_path):
        base_url = DEFAULT_URL_FOR_TEST
        dump_dir = str(tmp_path)
        env = {
            **os.environ,
            "DUMPER_SERVER_PORT": "reuse",
        }
        proc = popen_launch_server(
            "Qwen/Qwen3-0.6B",
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--tp", "2", "--max-total-tokens", "128"],
            env=env,
        )
        try:
            states = requests.post(f"{base_url}/dumper/get_state", json={}).json()
            assert len(states) == 2, f"Expected 2 ranks (tp=2), got {len(states)}"
            for state in states:
                assert state["config"]["enable"] is False
                assert state["step"] == 0

            requests.post(
                f"{base_url}/dumper/configure",
                json={"enable": True, "dir": dump_dir},
            ).raise_for_status()

            states = requests.post(f"{base_url}/dumper/get_state", json={}).json()
            assert len(states) == 2
            for rank, state in enumerate(states):
                assert (
                    state["config"]["enable"] is True
                ), f"rank {rank}: enable should be True after configure"
                assert state["config"]["dir"] == dump_dir

            resp = requests.post(
                f"{base_url}/generate",
                json={"text": "Hello", "sampling_params": {"max_new_tokens": 8}},
            )
            assert resp.status_code == 200, f"Generate failed: {resp.text}"

            states = requests.post(f"{base_url}/dumper/get_state", json={}).json()
            assert len(states) == 2
            steps = [s["step"] for s in states]
            for rank, step in enumerate(steps):
                assert step > 0, f"rank {rank}: step should be > 0, got {step}"
            assert steps[0] == steps[1], f"step mismatch across ranks: {steps}"

            dump_files = list(Path(dump_dir).glob("dump_*/*.pt"))
            assert len(dump_files) > 0, f"No dump files in {dump_dir}"
            filenames = {f.name for f in dump_files}

            for field in ("input_ids", "positions", "rids"):
                assert any(f"name={field}" in f for f in filenames), (
                    f"Missing {field} dump from non-intrusive hooks, "
                    f"got: {sorted(filenames)[:10]}"
                )

            for rank in range(2):
                assert any(
                    f"rank={rank}" in f for f in filenames
                ), f"No dump files for rank {rank}"

            sample_file = dump_files[0]
            loaded = torch.load(sample_file, map_location="cpu", weights_only=False)
            assert isinstance(loaded, dict), f"Expected dict, got {type(loaded)}"
            assert (
                "value" in loaded and "meta" in loaded
            ), f"Missing value/meta keys: {loaded.keys()}"
            assert "name" in loaded["meta"]
            assert "rank" in loaded["meta"]
            assert "step" in loaded["meta"]

            par = loaded["meta"].get("sglang_parallel_info", {})
            expected_keys = [
                "tp_rank",
                "tp_size",
                "pp_rank",
                "pp_size",
                "moe_ep_rank",
                "moe_ep_size",
                "moe_tp_rank",
                "moe_tp_size",
                "moe_dp_rank",
                "moe_dp_size",
                "enable_dp_attention",
                "attn_tp_rank",
                "attn_tp_size",
                "attn_dp_rank",
                "attn_dp_size",
                "local_attn_dp_rank",
                "local_attn_dp_size",
                "attn_cp_rank",
                "attn_cp_size",
            ]
            for key in expected_keys:
                assert (
                    key in par
                ), f"Missing {key} in sglang_parallel_info, got: {sorted(par)}"

            rids_files = [f for f in dump_files if "name=rids" in f.name]
            rids_loaded = torch.load(
                rids_files[0], map_location="cpu", weights_only=False
            )
            rids_value = rids_loaded["value"]
            assert isinstance(
                rids_value, list
            ), f"rids should be a list, got {type(rids_value)}"
            assert len(rids_value) > 0, "rids should be non-empty"
            assert all(
                isinstance(r, str) for r in rids_value
            ), f"each rid should be a str, got {[type(r) for r in rids_value]}"
        finally:
            kill_process_tree(proc.pid)


class TestRegisterForwardHook:
    @pytest.mark.parametrize("mode", ["hook", "replace_fn"])
    def test_handles_removable(self, mode):
        call_log: list[str] = []

        def pre_hook(_module, _args, _kwargs):
            call_log.append("pre")

        def hook(_module, _input, _output):
            call_log.append("post")

        module = torch.nn.Linear(4, 4)
        handles = _register_forward_hook_or_replace_fn(
            module,
            pre_hook=pre_hook,
            hook=hook,
            mode=mode,
        )

        x = torch.randn(2, 4)
        if mode == "hook":
            module(x)
        else:
            module.forward(x)
        assert call_log == ["pre", "post"]

        call_log.clear()
        for h in handles:
            h.remove()

        if mode == "hook":
            module(x)
        else:
            module.forward(x)
        assert call_log == []

    @pytest.mark.parametrize("mode", ["hook", "replace_fn"])
    def test_kwargs_passed_to_pre_hook(self, mode):
        received: list[tuple] = []

        class KwargsModule(torch.nn.Module):
            def forward(self, x, *, scale=1.0):
                return x * scale

        def pre_hook(_module, _args, _kwargs):
            received.append((_args, _kwargs))

        def hook(_module, _input, _output):
            pass

        module = KwargsModule()
        _register_forward_hook_or_replace_fn(
            module,
            pre_hook=pre_hook,
            hook=hook,
            mode=mode,
        )

        x = torch.randn(2, 4)
        if mode == "hook":
            module(x, scale=2.0)
        else:
            module.forward(x, scale=2.0)

        assert len(received) == 1
        args, kwargs = received[0]
        assert len(args) == 1
        assert torch.equal(args[0], x)
        assert kwargs == {"scale": 2.0}

    def test_replace_fn_remove_asserts_on_rewrap(self):
        module = torch.nn.Linear(4, 4)
        handles = _register_forward_hook_or_replace_fn(
            module,
            pre_hook=lambda _m, _a, _kw: None,
            hook=lambda _m, _i, _o: None,
            mode="replace_fn",
        )

        module.forward = lambda *a, **kw: None

        with pytest.raises(AssertionError):
            handles[0].remove()


class TestPluginCoreFields:
    def test_sglang_core_fields(self):
        plugin = _SGLangPlugin()
        assert plugin.core_fields() == frozenset(
            {"input_ids", "positions", "seq_lens", "req_pool_indices", "rids"}
        )

    def test_megatron_core_fields(self):
        plugin = _MegatronPlugin()
        assert plugin.core_fields() == frozenset(
            {"input_ids", "position_ids", "cu_seqlens_q", "cu_seqlens_kv", "qkv_format"}
        )


class TestMegatronConvertValue:
    @pytest.fixture(autouse=True)
    def _patch_megatron(self, monkeypatch):
        class FakePackedSeqParams:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        monkeypatch.setattr(_MegatronPlugin, "_available", True)
        monkeypatch.setattr(
            _MegatronPlugin, "PackedSeqParams", FakePackedSeqParams, raising=False
        )
        self._FakePackedSeqParams = FakePackedSeqParams

    def test_extracts_packed_seq_params(self):
        plugin = _MegatronPlugin()
        cu_q = torch.tensor([0, 3, 7])
        cu_kv = torch.tensor([0, 3, 7])
        value = self._FakePackedSeqParams(
            cu_seqlens_q=cu_q, cu_seqlens_kv=cu_kv, qkv_format="thd"
        )

        result = plugin.convert_value(value, skip_forward_batch=False)
        assert set(result.keys()) == {"cu_seqlens_q", "cu_seqlens_kv", "qkv_format"}
        assert torch.equal(result["cu_seqlens_q"], cu_q)
        assert torch.equal(result["cu_seqlens_kv"], cu_kv)
        assert result["qkv_format"] == "thd"

    def test_non_packed_returns_none(self):
        plugin = _MegatronPlugin()
        assert plugin.convert_value(torch.randn(4), skip_forward_batch=False) is None
        assert plugin.convert_value("hello", skip_forward_batch=False) is None


class TestNonIntrusiveKwargsModel(_NonIntrusiveTestBase):
    def test_kwargs_core_fields(self, tmp_path):
        class KwargsModel(torch.nn.Module):
            def forward(self, *, input_ids, position_ids):
                return input_ids + position_ids

        model = KwargsModel()
        d = _make_test_dumper(tmp_path, non_intrusive_mode="core")
        d.register_non_intrusive_dumper(model)

        ids = torch.randn(4)
        pos = torch.randn(4)
        with d.capture_output() as captured:
            model(input_ids=ids, position_ids=pos)

        assert "input_ids" in captured
        assert "position_ids" in captured
        assert torch.equal(captured["input_ids"]["value"], ids)
        assert torch.equal(captured["position_ids"]["value"], pos)

    def test_kwargs_all_mode(self, tmp_path):
        class KwargsModel(torch.nn.Module):
            def forward(self, *, input_ids, position_ids, custom_value):
                return input_ids + position_ids + custom_value

        model = KwargsModel()
        d = _make_test_dumper(tmp_path, non_intrusive_mode="all")
        d.register_non_intrusive_dumper(model)

        ids = torch.randn(4)
        pos = torch.randn(4)
        custom = torch.randn(4)
        with d.capture_output() as captured:
            model(input_ids=ids, position_ids=pos, custom_value=custom)

        assert "input_ids" in captured
        assert "position_ids" in captured

        P = self._PREFIX
        assert f"{P}inputs.custom_value" in captured

    def test_mixed_args_and_kwargs(self, tmp_path):
        class MixedModel(torch.nn.Module):
            def forward(self, x, *, input_ids):
                return x + input_ids

        model = MixedModel()
        d = _make_test_dumper(tmp_path, non_intrusive_mode="all")
        d.register_non_intrusive_dumper(model)

        x = torch.randn(4)
        ids = torch.randn(4)
        with d.capture_output() as captured:
            model(x, input_ids=ids)

        assert "input_ids" in captured

        P = self._PREFIX
        assert f"{P}inputs.0" in captured

    def test_packed_seq_params_core_fields(self, tmp_path, monkeypatch):
        class FakePackedSeqParams:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        monkeypatch.setattr(_MegatronPlugin, "_available", True)
        monkeypatch.setattr(
            _MegatronPlugin, "PackedSeqParams", FakePackedSeqParams, raising=False
        )

        class MegatronLikeModel(torch.nn.Module):
            def forward(self, *, input_ids, packed_seq_params):
                return input_ids

        model = MegatronLikeModel()
        d = _make_test_dumper(tmp_path, non_intrusive_mode="core")
        d.register_non_intrusive_dumper(model)

        ids = torch.randn(4)
        cu_q = torch.tensor([0, 3, 7])
        cu_kv = torch.tensor([0, 3, 7])
        psp = FakePackedSeqParams(
            cu_seqlens_q=cu_q, cu_seqlens_kv=cu_kv, qkv_format="thd"
        )
        with d.capture_output() as captured:
            model(input_ids=ids, packed_seq_params=psp)

        assert "input_ids" in captured
        assert torch.equal(captured["input_ids"]["value"], ids)
        assert "cu_seqlens_q" in captured
        assert torch.equal(captured["cu_seqlens_q"]["value"], cu_q)
        assert "cu_seqlens_kv" in captured
        assert torch.equal(captured["cu_seqlens_kv"]["value"], cu_kv)
        assert "qkv_format" in captured
        assert captured["qkv_format"]["value"] == "thd"


class TestDumperDims:
    def test_dims_in_meta_not_filename(self, tmp_path) -> None:
        dumper = _make_test_dumper(tmp_path)
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
                enable_grad=True,
            )
        )

        tensor = torch.randn(4, 8, requires_grad=True)
        dumper.dump("hidden", tensor, dims="b h(tp)", dims_grad="b h(tp:partial)")
        dumper.step()

        tensor.backward(torch.ones_like(tensor))

        exp_dir = tmp_path / dumper._config.exp_name
        pt_files = sorted(exp_dir.glob("*.pt"))
        assert len(pt_files) == 2

        value_file = [f for f in pt_files if "grad__" not in f.stem][0]
        grad_file = [f for f in pt_files if "grad__" in f.stem][0]

        value_data = torch.load(value_file, weights_only=False)
        assert value_data["meta"]["dims"] == "b h(tp)"
        assert value_data["meta"]["dims_grad"] == "b h(tp:partial)"

        grad_data = torch.load(grad_file, weights_only=False)
        assert grad_data["meta"]["dims"] == "b h(tp:partial)"

    def test_dims_grad_inherits(self, tmp_path) -> None:
        dumper = _Dumper(
            config=DumperConfig(
                enable=True,
                dir=str(tmp_path),
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


class TestCtxDecorator:
    def test_ctx_dynamic_lambda(self, tmp_path: Path) -> None:
        d = _make_test_dumper(tmp_path)

        class FakeLayer:
            def __init__(self, layer_id: int) -> None:
                self.layer_id = layer_id

            @d.ctx(lambda self: dict(layer_id=self.layer_id))
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                d.dump("hidden", x)
                return x

        layer = FakeLayer(layer_id=42)
        layer.forward(torch.randn(3))

        filenames = _get_filenames(tmp_path)
        _assert_files(filenames, exist=["layer_id=42"])

    def test_ctx_static_kwargs(self, tmp_path: Path) -> None:
        d = _make_test_dumper(tmp_path)

        @d.ctx(phase="decode")
        def decode_step(x: torch.Tensor) -> torch.Tensor:
            d.dump("step_out", x)
            return x

        decode_step(torch.randn(3))

        filenames = _get_filenames(tmp_path)
        _assert_files(filenames, exist=["phase=decode"])

    def test_ctx_clears_on_exception(self, tmp_path: Path) -> None:
        d = _make_test_dumper(tmp_path)

        @d.ctx(phase="train")
        def buggy_fn() -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            buggy_fn()

        assert d._state.global_ctx == {}

    def test_ctx_rejects_mixed_args(self) -> None:
        d = _make_test_dumper("/tmp")

        with pytest.raises(ValueError, match="cannot mix"):
            d.ctx(lambda self: dict(a=1), phase="x")

    def test_ctx_rejects_empty_args(self) -> None:
        d = _make_test_dumper("/tmp")

        with pytest.raises(ValueError, match="must provide"):
            d.ctx()


class TestRecomputeStatus:
    def test_disabled_by_default(self, tmp_path: Path) -> None:
        d = _make_test_dumper(tmp_path)
        tensor = torch.randn(3, 3)
        d.dump("test_tensor", tensor)

        filenames = _get_filenames(tmp_path)
        _assert_files(filenames, exist=["recompute_status=disabled"])

    def test_recompute_status_in_embedded_meta(self, tmp_path: Path) -> None:
        d = _make_test_dumper(tmp_path)
        tensor = torch.randn(3, 3)
        d.dump("test_tensor", tensor)

        path = _find_dump_file(tmp_path, rank=0, name="test_tensor")
        raw = _load_dump(path)
        assert raw["meta"]["recompute_status"] == "disabled"

    def test_recompute_status_recompute(self, tmp_path: Path, monkeypatch) -> None:
        import sglang.srt.debug_utils.dumper as dumper_mod

        monkeypatch.setattr(
            dumper_mod, "_detect_recompute_status", lambda: _RecomputeStatus.RECOMPUTE
        )

        d = _make_test_dumper(tmp_path)
        tensor = torch.randn(3, 3)
        d.dump("test_tensor", tensor)

        filenames = _get_filenames(tmp_path)
        _assert_files(filenames, exist=["recompute_status=recompute"])

        path = _find_dump_file(tmp_path, rank=0, name="test_tensor")
        raw = _load_dump(path)
        assert raw["meta"]["recompute_status"] == "recompute"
        assert raw["meta"]["recompute_pseudo_rank"] == 1
        assert raw["meta"]["recompute_pseudo_size"] == 2

    def test_recompute_status_original(self, tmp_path: Path, monkeypatch) -> None:
        import sglang.srt.debug_utils.dumper as dumper_mod

        monkeypatch.setattr(
            dumper_mod,
            "_detect_recompute_status",
            lambda: _RecomputeStatus.ORIGINAL,
        )

        d = _make_test_dumper(tmp_path)
        tensor = torch.randn(3, 3)
        d.dump("test_tensor", tensor)

        filenames = _get_filenames(tmp_path)
        _assert_files(filenames, exist=["recompute_status=original"])

        path = _find_dump_file(tmp_path, rank=0, name="test_tensor")
        raw = _load_dump(path)
        assert raw["meta"]["recompute_status"] == "original"
        assert raw["meta"]["recompute_pseudo_rank"] == 0
        assert raw["meta"]["recompute_pseudo_size"] == 2

    def test_disabled_no_recompute_pseudo_fields(self, tmp_path: Path) -> None:
        d = _make_test_dumper(tmp_path)
        tensor = torch.randn(3, 3)
        d.dump("test_tensor", tensor)

        path = _find_dump_file(tmp_path, rank=0, name="test_tensor")
        raw = _load_dump(path)
        assert "recompute_pseudo_rank" not in raw["meta"]
        assert "recompute_pseudo_size" not in raw["meta"]

    def test_grad_hook_has_no_recompute_status(self, tmp_path: Path) -> None:
        d = _make_test_dumper(tmp_path, enable_grad=True)
        x = torch.randn(3, 3, requires_grad=True)
        y = (x * 2).sum()

        d.dump("test_tensor", x)
        y.backward()

        grad_files = [f for f in _get_filenames(tmp_path) if "grad__test_tensor" in f]
        assert len(grad_files) == 1
        assert "recompute_status" not in grad_files[0]

    def test_non_intrusive_hooks_have_recompute_status(self, tmp_path: Path) -> None:
        class Simple(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        model = Simple()
        d = _make_test_dumper(tmp_path, non_intrusive_mode="all")
        d.register_non_intrusive_dumper(model)

        with d.capture_output() as captured:
            model(torch.randn(2, 4))

        for key, data in captured.items():
            assert (
                "recompute_status" in data["meta"]
            ), f"missing recompute_status in {key}"
            assert data["meta"]["recompute_status"] == "disabled"

    def test_detect_recompute_status_default(self) -> None:
        assert _detect_recompute_status() == _RecomputeStatus.DISABLED


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
