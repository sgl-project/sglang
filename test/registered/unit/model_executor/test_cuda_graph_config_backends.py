"""Validate backend retirement without importing accelerator backends."""

import argparse
import sys
from unittest.mock import patch

import pytest

from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    default_prefill_backend,
    parse_cuda_graph_backend_arg,
    parse_cuda_graph_config_arg,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("backend", [Backend.FULL, Backend.BREAKABLE, Backend.DISABLED])
def test_supported_prefill_backends_round_trip(backend):
    raw = parse_cuda_graph_config_arg('{"prefill":{"backend":"' + backend + '"}}')
    config = CudaGraphConfig.from_dict(raw)
    assert config.prefill.backend == backend


def test_removed_backend_is_rejected():
    with pytest.raises(ValueError, match="tc_piecewise was removed"):
        CudaGraphConfig.from_dict({"prefill": {"backend": "tc_piecewise"}})


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("json_config", [False, True])
def test_cli_reports_backend_removal(phase, json_config, capsys):
    parser = argparse.ArgumentParser()
    flag = f"--cuda-graph-backend-{phase}"
    parser.add_argument(flag, type=parse_cuda_graph_backend_arg, choices=Backend.ALL)
    parser.add_argument("--cuda-graph-config", type=parse_cuda_graph_config_arg)
    argv = (
        ["--cuda-graph-config", '{"' + phase + '":{"backend":"tc_piecewise"}}']
        if json_config
        else [flag, "tc_piecewise"]
    )
    with pytest.raises(SystemExit) as error:
        parser.parse_args(argv)
    assert error.value.code == 2
    assert (
        "tc_piecewise was removed; select breakable, full, or disabled"
        in capsys.readouterr().err
    )


def test_removed_compiler_option_is_rejected():
    with pytest.raises(argparse.ArgumentTypeError, match="tc_compiler"):
        parse_cuda_graph_config_arg('{"prefill":{"tc_compiler":"eager"}}')
    with pytest.raises(ValueError, match="tc_compiler was removed"):
        CudaGraphConfig.from_dict({"prefill": {"tc_compiler": "inductor"}})


@pytest.mark.parametrize(
    "cuda, expected", [(True, Backend.BREAKABLE), (False, Backend.DISABLED)]
)
def test_prefill_defaults_do_not_enable_unvalidated_platforms(cuda, expected):
    with patch("sglang.srt.utils.is_cuda", return_value=cuda):
        assert default_prefill_backend() == expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
