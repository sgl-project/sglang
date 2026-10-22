"""Regression coverage for cold diffusion imports and quantization dispatch."""

import os
import subprocess
import sys

import pytest

from sglang.multimodal_gen.runtime.layers import quantization


@pytest.mark.parametrize(
    "module",
    [
        "runtime.layers.linear",
        "runtime.utils.hf_diffusers_utils",
        "test.single_test_file.test_ar_models",
        "runtime.layers.quantization.configs.convrot_int8_config",
    ],
)
def test_cold_import(module):
    # A new interpreter is necessary: pytest's conftest imports the registry first.
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib; importlib.import_module('sglang.multimodal_gen.{module}')",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "method,name",
    [
        ("auto-round", "AutoRoundConfig"),
        ("bitsandbytes", "BitsAndBytesConfig"),
        ("modelopt", "ModelOptFp8DiffusionConfig"),
        ("modelopt_fp8", "ModelOptFp8Config"),
        ("modelopt_fp4", "ModelOptFp4Config"),
        ("modelslim", "ModelSlimConfig"),
        ("fp8", "Fp8Config"),
        ("mxfp4", "Mxfp4Config"),
        ("mxfp8", "MXFP8Config"),
        ("mxfp4_npu", "NPUMXFP4Config"),
        ("convrot_int8", "ConvRotInt8Config"),
    ],
)
def test_builtin_config_and_package_export(method, name):
    config = quantization.get_quantization_config(method)
    assert issubclass(config, quantization.QuantizationConfig)
    assert config is getattr(quantization, name)
    assert quantization.get_quantization_config(method) is config


def test_invalid_method_and_package_attribute():
    for name in ("unknown", "kitchen_int8"):
        with pytest.raises(ValueError, match="Invalid quantization method"):
            quantization.get_quantization_config(name)
    with pytest.raises(AttributeError, match="no attribute"):
        _ = quantization.UnknownConfig


def test_lookup_loads_only_requested_backend():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from sglang.multimodal_gen.runtime.layers import quantization as q; "
                "seen = []; sentinel = type('Module', (), {'Fp8Config': object()})(); "
                "q.import_module = lambda name: (seen.append(name) or sentinel); "
                "assert q.get_quantization_config('fp8') is sentinel.Fp8Config; "
                "assert q.Fp8Config is sentinel.Fp8Config; "
                "assert seen == [q.__name__ + '.fp8']"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
