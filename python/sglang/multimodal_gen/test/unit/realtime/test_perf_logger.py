import importlib.util
import sys
import types
from pathlib import Path


def _stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    if name == "sglang" or ".multimodal_gen" in name:
        module.__path__ = []
    sys.modules[name] = module
    return module


def _load_perf_logger_with_namespace_sglang(monkeypatch):
    sglang = _stub_module("sglang", __file__=None)
    _stub_module("sglang.multimodal_gen")
    _stub_module("sglang.multimodal_gen.envs")
    _stub_module("sglang.multimodal_gen.runtime")
    _stub_module("sglang.multimodal_gen.runtime.utils")
    _stub_module(
        "sglang.multimodal_gen.runtime.utils.logging_utils",
        CYAN="",
        RESET="",
        _SGLDiffusionLogger=object,
        get_is_main_process=lambda: True,
        init_logger=lambda name: None,
    )
    _stub_module(
        "sglang.multimodal_gen.runtime.platforms",
        current_platform=types.SimpleNamespace(),
    )
    _stub_module(
        "torch",
        get_device_module=lambda: types.SimpleNamespace(
            is_available=lambda: False,
            memory_allocated=lambda: 0,
            memory_reserved=lambda: 0,
            max_memory_allocated=lambda: 0,
            max_memory_reserved=lambda: 0,
        ),
    )

    module_name = "perf_logger_under_test"
    module_path = (
        Path(__file__).resolve().parents[3]
        / "runtime"
        / "utils"
        / "perf_logger.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.sglang is sglang
    return module


def test_diffusion_perf_log_dir_falls_back_when_sglang_file_is_missing(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SGLANG_PERF_LOG_DIR", raising=False)

    module = _load_perf_logger_with_namespace_sglang(monkeypatch)

    assert module.get_diffusion_perf_log_dir() == str(tmp_path / ".cache" / "logs")
