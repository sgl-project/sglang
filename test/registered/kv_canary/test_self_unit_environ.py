from __future__ import annotations

import argparse
import unittest

from sglang.jit_kernel.kv_canary.verify import RealKvHashMode
from sglang.srt.environ import EnvField, EnvFloat, EnvInt, Envs
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="extra-a", runner_config="1-gpu-large")


_PERTURB_ENV_VARS = (
    "SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB",
    "SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_SEED",
    "SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB",
    "SGLANG_KV_CANARY_REAL_PERTURB_BYTES_SEED",
)


class TestSelfUnitEnviron(CustomTestCase):
    def test_canary_env_vars_registered_in_environ_py(self):
        registry = vars(Envs)
        for name in _PERTURB_ENV_VARS:
            self.assertIn(name, registry, f"missing env var {name} in Envs")
            self.assertIsInstance(registry[name], EnvField)

    def test_canary_env_var_types_are_primitive(self):
        registry = vars(Envs)
        expected_types = {
            "SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB": EnvFloat,
            "SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_SEED": EnvInt,
            "SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB": EnvFloat,
            "SGLANG_KV_CANARY_REAL_PERTURB_BYTES_SEED": EnvInt,
        }
        for name, expected_cls in expected_types.items():
            field = registry[name]
            self.assertIsInstance(
                field,
                expected_cls,
                f"{name} declared as {type(field).__name__}, expected {expected_cls.__name__}",
            )

    @unittest.expectedFailure
    def test_real_data_cli_flag_registered_and_matches_enum(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        target = None
        for action in parser._actions:
            if "--kv-canary-real-data" in action.option_strings:
                target = action
                break
        self.assertIsNotNone(target, "--kv-canary-real-data not registered")

        expected_choices = {m.name.lower() for m in RealKvHashMode}
        self.assertEqual(set(target.choices), expected_choices)


if __name__ == "__main__":
    unittest.main()
