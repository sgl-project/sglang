"""CPU bootstrap preserves native quantization validation and ordinary imports."""

import importlib.util
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("sglang") is None,
    reason="requires the SGLang CPU test environment",
)


def run_python(code, *, bootstrap=False, cpu=False):
    env = os.environ.copy()
    for key in ("SGLANG_SIMULATOR_BOOTSTRAP", "SGLANG_USE_CPU_ENGINE"):
        env.pop(key, None)
    if bootstrap:
        env["SGLANG_SIMULATOR_BOOTSTRAP"] = "1"
    if cpu:
        env["SGLANG_USE_CPU_ENGINE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "bootstrap,cpu", [(False, False), (True, False), (False, True)]
)
def test_bootstrap_requires_both_flags(bootstrap, cpu):
    run_python(
        """
import sys, torch
original = torch.cuda.is_available
hip = torch.version.hip
import usercustomize
usercustomize.apply_cpu_simulation_compat()
assert torch.cuda.is_available is original
assert torch.version.hip == hip
assert 'sglang.srt.layers.quantization' not in sys.modules
""",
        bootstrap=bootstrap,
        cpu=cpu,
    )


def test_cpu_bootstrap_preserves_native_quantization():
    run_python(
        """
import sys
import usercustomize
usercustomize.apply_cpu_simulation_compat()
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.quantization import QUANTIZATION_METHODS
from sglang.srt.layers.quantization.fp8 import Fp8Config
assert QUANTIZATION_METHODS['fp8'] is Fp8Config
assert 'aiter' not in sys.modules

def validate(method, hf=None):
    config = ModelConfig.__new__(ModelConfig)
    config.quantization = method
    config.is_draft_model = False
    config._parse_quant_hf_config = lambda: hf
    config._find_quant_modelslim_config = lambda: None
    config._verify_quantization()
    return config.quantization

for method in (None, 'fp8', 'awq', 'compressed-tensors', 'w8a8_fp8', 'mxfp8'):
    assert validate(method) == method
assert validate('AWQ') == 'awq'
assert validate(None, {'quant_method': 'fp8'}) == 'fp8'
assert validate('w8a8_fp8', {'quant_method': 'compressed-tensors'}) == 'w8a8_fp8'
for method, hf in [('unknown', None), ('awq', {'quant_method': 'fp8'})]:
    try:
        validate(method, hf)
    except ValueError:
        pass
    else:
        raise AssertionError((method, hf))
""",
        bootstrap=True,
        cpu=True,
    )
