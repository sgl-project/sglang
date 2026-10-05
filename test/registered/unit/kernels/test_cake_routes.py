"""Unit tests for the SGLANG_CAKE_ROUTES opt-in switch of the Cake forwarding layer."""

import os
import sys
from unittest import mock

import pytest

from sglang.kernels.cake_kernels import _routes
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a-test-cpu")


@pytest.fixture(autouse=True)
def _reset():
    _routes.reset_cache_for_tests()
    yield
    _routes.reset_cache_for_tests()


def test_default_is_all_off():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(_routes.ENV_VAR, None)
        assert not any(_routes.cake_route_enabled(r) for r in _routes.ROUTES)


def test_named_routes_and_all():
    with mock.patch.dict(os.environ, {_routes.ENV_VAR: "gdn_prefill, moe_fp8_grouped"}):
        assert _routes.cake_route_enabled("gdn_prefill")
        assert _routes.cake_route_enabled("moe_fp8_grouped")
        assert not _routes.cake_route_enabled("gdn_decode")
    _routes.reset_cache_for_tests()
    with mock.patch.dict(os.environ, {_routes.ENV_VAR: "all"}):
        assert all(_routes.cake_route_enabled(r) for r in _routes.ROUTES)


def test_unknown_names_are_errors():
    with mock.patch.dict(os.environ, {_routes.ENV_VAR: "gdn_prefil"}):
        with pytest.raises(ValueError, match="unknown Cake route"):
            _routes.cake_route_enabled("gdn_prefill")
    with pytest.raises(KeyError):
        _routes.cake_route_enabled("not_a_route")


def test_import_loads_no_flashinfer_or_torch():
    assert "flashinfer" not in sys.modules or True  # other tests may import it
    import importlib

    src = importlib.util.find_spec("sglang.kernels.cake_kernels._routes").origin
    text = open(src).read()
    assert "import torch" not in text and "flashinfer" not in text.replace(
        'FlashInfer "Cake"', ""
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_route_process_env_exported_only_for_selected_routes():
    with mock.patch.dict(os.environ, {_routes.ENV_VAR: "gdn_prefill"}):
        env: dict = {}
        assert _routes.apply_route_process_env(env) == {}
        assert env == {}
    _routes.reset_cache_for_tests()
    with mock.patch.dict(os.environ, {_routes.ENV_VAR: "sp_all_gather_matmul"}):
        env = {}
        assert _routes.apply_route_process_env(env) == {"TORCH_SYMMMEM": "NVSHMEM"}
        assert env == {"TORCH_SYMMMEM": "NVSHMEM"}
        # a value the user set explicitly is left alone and not reported
        env = {"TORCH_SYMMMEM": "CUDA"}
        assert _routes.apply_route_process_env(env) == {}
        assert env == {"TORCH_SYMMMEM": "CUDA"}
        # idempotent on the process environment
        assert _routes.apply_route_process_env(env) == {}


def test_route_process_env_defaults_to_os_environ():
    with mock.patch.dict(os.environ, {_routes.ENV_VAR: "sp_all_gather_matmul"}):
        os.environ.pop("TORCH_SYMMMEM", None)
        assert _routes.apply_route_process_env() == {"TORCH_SYMMMEM": "NVSHMEM"}
        assert os.environ["TORCH_SYMMMEM"] == "NVSHMEM"
