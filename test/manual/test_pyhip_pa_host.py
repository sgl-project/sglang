"""Host checks for the vendored attention module; no GPU kernel is launched."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sglang.srt.layers.attention import pyhip_pa


def _factory(**overrides):
    args = dict(
        num_qo_heads=16,
        num_kv_heads=1,
        head_dim_qk=192,
        head_dim_v=128,
        page_size=64,
        is_causal=True,
        quant_query_mode="per-tensor",
    )
    args.update(overrides)
    return pyhip_pa.PagedAttention(**args)


@pytest.mark.parametrize("dq,dv", [(128, 128), (128, 192), (192, 128), (192, 192)])
@pytest.mark.parametrize("page", [32, 64, 128])
def test_paged_factory_preserves_supported_dimensions(dq, dv, page):
    first = _factory(head_dim_qk=dq, head_dim_v=dv, page_size=page)
    assert first is _factory(head_dim_qk=dq, head_dim_v=dv, page_size=page)
    assert (first.dq, first.dv, first.page_size) == (dq, dv, page)
    assert first.persistent is True
    assert first._launch is pyhip_pa._launch_attention


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"head_dim_qk": 256, "head_dim_v": 256}, NotImplementedError),
        ({"head_dim_qk": 64}, NotImplementedError),
        ({"page_size": 16}, NotImplementedError),
        ({"window_left": 128}, NotImplementedError),
        ({"has_sink": True}, NotImplementedError),
        ({"memory_mode": "vmem"}, NotImplementedError),
        ({"key_layout": "linear"}, NotImplementedError),
        ({"num_qo_heads": 0}, ValueError),
        ({"num_qo_heads": True}, ValueError),
        ({"num_qo_heads": 15, "num_kv_heads": 2}, ValueError),
        ({"quant_query_mode": "invalid"}, ValueError),
        ({"persistent": 1}, ValueError),
        ({"is_causal": 1}, ValueError),
    ],
)
def test_factory_rejects_unsupported_contracts(overrides, error):
    with pytest.raises(error):
        _factory(**overrides)


@pytest.mark.parametrize(
    "valid,invalid",
    [
        ({"is_causal": True}, {"is_causal": 1}),
        ({"num_kv_heads": 1}, {"num_kv_heads": 1.0}),
        ({"head_dim_qk": 192}, {"head_dim_qk": 192.0}),
        ({"has_sink": False}, {"has_sink": 0}),
        ({"persistent": False}, {"persistent": 0}),
    ],
)
def test_cached_factory_does_not_bypass_parameter_type_checks(valid, invalid):
    first = _factory(**valid)
    assert first is _factory(**valid)
    with pytest.raises(ValueError):
        _factory(**invalid)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), 1e-200, 1e200])
def test_softmax_scale_rejects_invalid_fp32_range(scale):
    with pytest.raises(ValueError):
        pyhip_pa._softmax_scale(scale, 192**-0.5)


@pytest.mark.parametrize("scale", [True, "0.5", torch.tensor(0.5)])
def test_softmax_scale_requires_host_real_scalar(scale):
    with pytest.raises(TypeError):
        pyhip_pa._softmax_scale(scale, 192**-0.5)


def test_valid_scale_and_explicit_nonpersistent_factory():
    assert pyhip_pa._softmax_scale(None, 192**-0.5) == 192**-0.5
    assert pyhip_pa._softmax_scale(0.125, 192**-0.5) == 0.125
    ordinary = _factory(persistent=False)
    assert ordinary.persistent is False
    assert ordinary is not _factory()


def test_cpu_input_is_rejected_before_device_query(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("CPU rejection must happen before a GPU device query")

    monkeypatch.setattr(torch.cuda, "get_device_properties", unexpected)
    with pytest.raises(ValueError, match="same gfx942 GPU"):
        _factory()(
            torch.empty(1, 16, 192, dtype=torch.bfloat16),
            None,
            None,
            None,
            None,
            None,
            None,
            1,
            1,
            True,
            None,
            None,
            None,
            None,
        )


def test_import_without_external_pyhip_or_gpu_initialization():
    # A fresh interpreter catches accidental imports hidden by sys.modules.
    script = """
import importlib.abc
import sys
from pathlib import Path

class BlockPyhip(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyhip" or fullname.startswith("pyhip."):
            raise AssertionError("Unexpected external dependency: " + fullname)

sys.meta_path.insert(0, BlockPyhip())
import torch
assert not torch.cuda.is_initialized()
from sglang.srt.layers.attention import pyhip_pa
assert Path(pyhip_pa.__file__).resolve() == Path(sys.argv[1]).resolve()
pyhip_pa.PagedAttention(16, 1, 192, 128, 64, True, "per-tensor")
assert not torch.cuda.is_initialized()
assert not any(n == "pyhip" or n.startswith("pyhip.") for n in sys.modules)
"""
    env = os.environ.copy()
    for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        env[key] = ""
    source_root = Path(__file__).resolve().parents[2] / "python"
    env["PYTHONPATH"] = str(source_root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script, pyhip_pa.__file__],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
