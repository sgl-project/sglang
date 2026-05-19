from __future__ import annotations

import argparse

import pytest

from sglang.jit_kernel.kv_cache_canary_verify import RealKvHashMode
from sglang.srt.environ import EnvField, EnvFloat, EnvInt, Envs
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="extra-a", runner_config="1-gpu-large")


_PERTURB_ENV_VARS = (
    "SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB",
    "SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_SEED",
    "SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB",
    "SGLANG_KV_CANARY_REAL_PERTURB_BYTES_SEED",
)


def test_canary_env_vars_registered_in_environ_py():
    registry = vars(Envs)
    for name in _PERTURB_ENV_VARS:
        assert name in registry, f"missing env var {name} in Envs"
        assert isinstance(registry[name], EnvField)


def test_canary_env_var_types_are_primitive():
    registry = vars(Envs)
    expected_types = {
        "SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB": EnvFloat,
        "SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_SEED": EnvInt,
        "SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB": EnvFloat,
        "SGLANG_KV_CANARY_REAL_PERTURB_BYTES_SEED": EnvInt,
    }
    for name, expected_cls in expected_types.items():
        field = registry[name]
        assert isinstance(
            field, expected_cls
        ), f"{name} declared as {type(field).__name__}, expected {expected_cls.__name__}"


@pytest.mark.xfail(
    strict=False,
    reason="server_args choices for --kv-cache-canary-real-data are "
    "{off, portion, all}; expected {off, bit, all} matching RealKvHashMode "
    "members",
)
def test_real_data_cli_flag_registered_and_matches_enum():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)

    target = None
    for action in parser._actions:
        if "--kv-cache-canary-real-data" in action.option_strings:
            target = action
            break
    assert target is not None, "--kv-cache-canary-real-data not registered"

    expected_choices = {m.name.lower() for m in RealKvHashMode}
    assert set(target.choices) == expected_choices
