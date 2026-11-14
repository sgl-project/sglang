# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""CPU-only checks for the private TokenSpeed 2.0.3 launch contract."""

from __future__ import annotations

import builtins
import importlib.util
import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.kernel_plugins.nvidia.kimi_k3 import compat
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _not_called(*args, **kwargs):
    raise AssertionError("compatibility validation must not invoke a runtime API")


def _api(parameters):
    def function(*args, **kwargs):
        return _not_called(*args, **kwargs)

    function.__signature__ = inspect.Signature(parameters)
    return function


def _positional(*names):
    return [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in names
    ]


@pytest.fixture
def runtime(monkeypatch):
    # Names/kinds from the published 2.0.3 wheel, including the required
    # has_local_visible_lens argument and keyword-only partial_fp16 argument.
    compiler = _api(
        _positional(
            "torch_dtype",
            "page_size",
            "kv_lora_rank",
            "qk_rope_head_dim",
            "is_persistent",
            "is_var_seq",
            "is_var_split_kv",
            "compute_capability",
            "has_local_visible_lens",
        )
        + [
            inspect.Parameter(
                name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
            )
            for name in (
                "skip_correction_threshold",
                "is_workspace_size_zero",
                "fold_sq_factor",
                "causal_mask",
                "num_heads",
                "seq_len_q",
                "window_left",
                "cp_world",
                "cp_rank",
                "cp_interleave_size",
                "use_pdl",
                "return_lse",
                "reducer_d_tiles",
                "reducer_max_splits",
                "pack_q",
            )
        ]
        + [inspect.Parameter("partial_fp16", inspect.Parameter.KEYWORD_ONLY)]
    )
    md = SimpleNamespace(
        select_mla_decode_tilers=_api(
            _positional("num_heads", "seq_len_q")
            + [
                inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
                for name in ("is_fp8", "compute_capability")
            ]
        ),
        get_mla_decode_fold_sq_factor=_api(
            _positional("num_heads", "seq_len_q", "mma_m_tile")
        ),
        compute_q_tile_layout=_api(_positional("num_heads", "seq_len_q", "m_tile")),
        get_num_sm=_api(_positional("device")),
        _get_split_kv_and_workspace_size=_api(
            _positional(
                "B",
                "q_len",
                "H",
                "kv_lora_rank",
                "max_active_blocks",
                "max_seq_len",
                "torch_dtype",
                "mma_qk_tiler_mn",
                "min_split_kv",
            )
        ),
        _get_compiled_mla_kernel=compiler,
        _get_reducer_d_tiles=_api(
            _positional("batch_size", "seq_len_q", "num_heads", "num_sms", "split_kv")
        ),
        _get_reducer_max_splits=_api(_positional("split_kv")),
    )
    modules = {
        "tokenspeed_mla": SimpleNamespace(get_num_sm=md.get_num_sm),
        "tokenspeed_mla.mla_decode": md,
        "tvm_ffi": SimpleNamespace(
            use_torch_stream=_api(
                [
                    inspect.Parameter(
                        "context", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
                    )
                ]
            )
        ),
    }
    versions = {"tokenspeed-mla": "2.0.3", "apache-tvm-ffi": "0.1.11"}
    version = Mock(side_effect=versions.__getitem__)
    importer = Mock(side_effect=modules.__getitem__)
    monkeypatch.setattr(compat.importlib.metadata, "version", version)
    monkeypatch.setattr(compat.importlib, "import_module", importer)
    compat.validate_tokenspeed_compatibility.cache_clear()
    yield SimpleNamespace(
        modules=modules, md=md, versions=versions, version=version, importer=importer
    )
    compat.validate_tokenspeed_compatibility.cache_clear()


def test_module_import_does_not_load_gpu_dependencies(monkeypatch):
    original = builtins.__import__

    def cpu_import(name, *args, **kwargs):
        assert name.split(".", 1)[0] not in {
            "torch",
            "cutlass",
            "flashinfer",
            "tokenspeed_mla",
            "tvm_ffi",
        }
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", cpu_import)
    spec = importlib.util.spec_from_file_location(
        "k3_compat_cpu_import", compat.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_pinned_apis_validate_without_execution_and_are_cached(runtime):
    first = compat.validate_tokenspeed_compatibility()
    assert first == compat.validate_tokenspeed_compatibility()
    assert len(first) == 64
    assert runtime.version.call_count == 2
    assert runtime.importer.call_count == 3


@pytest.mark.parametrize("distribution", ["tokenspeed-mla", "apache-tvm-ffi"])
def test_missing_wheel_fails_before_runtime_imports(runtime, distribution):
    def version(name):
        if name == distribution:
            raise compat.importlib.metadata.PackageNotFoundError(name)
        return runtime.versions[name]

    runtime.version.side_effect = version
    with pytest.raises(RuntimeError, match=rf"{distribution}==.*not installed"):
        compat.validate_tokenspeed_compatibility()
    runtime.importer.assert_not_called()


@pytest.mark.parametrize(
    ("distribution", "wrong"),
    [
        ("tokenspeed-mla", "2.0.4"),
        ("tokenspeed-mla", "2.0.3+custom"),
        ("apache-tvm-ffi", "0.1.12"),
    ],
)
def test_non_pinned_version_fails_before_runtime_imports(runtime, distribution, wrong):
    runtime.versions[distribution] = wrong
    with pytest.raises(RuntimeError, match=rf"{distribution}==.*found"):
        compat.validate_tokenspeed_compatibility()
    runtime.importer.assert_not_called()


def test_broken_runtime_import_reports_dependency_and_cause(runtime):
    def importer(name):
        if name == "tokenspeed_mla.mla_decode":
            raise ImportError("missing CUDA runtime dependency")
        return runtime.modules[name]

    runtime.importer.side_effect = importer
    with pytest.raises(
        RuntimeError,
        match="cannot import tokenspeed_mla.mla_decode.*CUDA runtime dependency",
    ):
        compat.validate_tokenspeed_compatibility()


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("tokenspeed_mla", "get_num_sm"),
        ("tokenspeed_mla.mla_decode", "_get_compiled_mla_kernel"),
        ("tvm_ffi", "use_torch_stream"),
    ],
)
def test_missing_required_callable_is_actionable(runtime, module, name):
    delattr(runtime.modules[module], name)
    with pytest.raises(RuntimeError, match=rf"requires callable .*{name}"):
        compat.validate_tokenspeed_compatibility()


def test_old_compiler_without_local_visibility_argument_is_rejected(runtime):
    function = runtime.md._get_compiled_mla_kernel
    function.__signature__ = inspect.Signature(
        parameter
        for name, parameter in inspect.signature(function).parameters.items()
        if name != "has_local_visible_lens"
    )
    with pytest.raises(
        RuntimeError,
        match="_get_compiled_mla_kernel.*missing.*has_local_visible_lens",
    ):
        compat.validate_tokenspeed_compatibility()


def test_new_required_compiler_argument_is_rejected(runtime):
    function = runtime.md._get_compiled_mla_kernel
    function.__signature__ = inspect.Signature(
        [
            *inspect.signature(function).parameters.values(),
            inspect.Parameter("new_required_option", inspect.Parameter.KEYWORD_ONLY),
        ]
    )
    with pytest.raises(
        RuntimeError, match="_get_compiled_mla_kernel.*new_required_option"
    ):
        compat.validate_tokenspeed_compatibility()


def test_positional_workspace_geometry_reordering_is_rejected(runtime):
    function = runtime.md._get_split_kv_and_workspace_size
    parameters = list(inspect.signature(function).parameters.values())
    parameters[0], parameters[1] = parameters[1], parameters[0]
    function.__signature__ = inspect.Signature(parameters)
    with pytest.raises(
        RuntimeError,
        match="_get_split_kv_and_workspace_size signature is incompatible",
    ):
        compat.validate_tokenspeed_compatibility()


def test_unavailable_package_fallback_is_rejected(runtime):
    runtime.modules["tokenspeed_mla"].get_num_sm = _not_called
    with pytest.raises(
        RuntimeError, match="tokenspeed_mla.get_num_sm signature is incompatible"
    ):
        compat.validate_tokenspeed_compatibility()


def test_tvm_stream_context_must_accept_no_arguments(runtime):
    runtime.modules["tvm_ffi"].use_torch_stream = _api(_positional("context"))
    with pytest.raises(RuntimeError, match="tvm_ffi.use_torch_stream.*context"):
        compat.validate_tokenspeed_compatibility()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
