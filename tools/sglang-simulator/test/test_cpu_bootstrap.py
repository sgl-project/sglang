"""CPU bootstrap preserves native quantization validation and ordinary imports."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

requires_cpu_runtime = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("sglang") is None,
    reason="requires the SGLang CPU test environment",
)


def environment_defaults_owner(monkeypatch):
    for key in (
        "SGLANG_SIMULATOR_BOOTSTRAP",
        "SGLANG_USE_CPU_ENGINE",
        "SGLANG_TIMEOUT_KEEP_ALIVE",
    ):
        monkeypatch.delenv(key, raising=False)
    path = Path(__file__).resolve().parents[1] / "src/usercustomize.py"
    spec = importlib.util.spec_from_file_location("cpu_environment_defaults_test", path)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    return owner


@pytest.mark.parametrize(
    "bootstrap,cpu", [(False, False), (True, False), (False, True), (True, True)]
)
def test_http_lifetime_default_requires_explicit_cpu_simulation(
    monkeypatch, bootstrap, cpu
):
    owner = environment_defaults_owner(monkeypatch)
    monkeypatch.setenv("SGLANG_SIMULATOR_BOOTSTRAP", "1" if bootstrap else "0")
    monkeypatch.setenv("SGLANG_USE_CPU_ENGINE", "1" if cpu else "0")
    enabled = bootstrap and cpu
    assert owner.configure_cpu_simulation_environment() is enabled
    assert os.environ.get("SGLANG_TIMEOUT_KEEP_ALIVE") == ("3600" if enabled else None)


@pytest.mark.parametrize("seconds", ["5", "120"])
def test_explicit_cpu_http_lifetime_is_preserved(monkeypatch, seconds):
    owner = environment_defaults_owner(monkeypatch)
    monkeypatch.setenv("SGLANG_SIMULATOR_BOOTSTRAP", "1")
    monkeypatch.setenv("SGLANG_USE_CPU_ENGINE", "1")
    monkeypatch.setenv("SGLANG_TIMEOUT_KEEP_ALIVE", seconds)
    assert owner.configure_cpu_simulation_environment() is True
    assert os.environ["SGLANG_TIMEOUT_KEEP_ALIVE"] == seconds


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
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "bootstrap,cpu", [(False, False), (True, False), (False, True)]
)
@requires_cpu_runtime
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


@requires_cpu_runtime
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


@requires_cpu_runtime
def test_text_scheduler_defers_optional_multimodal_hashing():
    run_python(
        """
import importlib.abc
import sys

class BlockMultimodalHash(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'blake3':
            raise ModuleNotFoundError('multimodal hash dependency blocked', name=fullname)

sys.meta_path.insert(0, BlockMultimodalHash())
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.multimodal.cache.digest import structured_digest
assert ScheduleBatch is not None and 'blake3' not in sys.modules
try:
    structured_digest(None)
except ModuleNotFoundError as error:
    assert error.name == 'blake3'
else:
    raise AssertionError('selected multimodal hashing must require its real dependency')
""",
        bootstrap=True,
        cpu=True,
    )
