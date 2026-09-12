"""CPU wrapper contracts; numerical GPU coverage lives in the gfx950 test."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
DSA_DIR = (
    Path(__file__).resolve().parents[4] / "python/sglang/kernels/ops/attention/dsa"
)


def _load_function(module, name, **dependencies):
    # Avoid importing GPU-only Triton on CPU. Execute the real wrapper, mocking
    # only the GPU kernel it dispatches, without replacing global modules.
    path = DSA_DIR / module
    tree = ast.parse(path.read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    namespace = {"torch": torch, **dependencies}
    exec(compile(ast.Module([function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("width,expected", [(264, (256, 0)), (656, (512, 64))])
def test_infer_packed_geometry(width, expected):
    infer = _load_function("dequant_k_cache.py", "_infer_dsa_dims")
    assert infer(width) == expected


@pytest.mark.parametrize("width", [0, 1024])
def test_unknown_packed_geometry_fails(width):
    infer = _load_function("dequant_k_cache.py", "_infer_dsa_dims")
    with pytest.raises(ValueError):
        infer(width)


def test_dequant_rejects_mismatched_explicit_dv():
    kernel = Mock()
    dequant = _load_function(
        "dequant_k_cache.py",
        "_dequantize_k_cache_fast_wrapped",
        _infer_dsa_dims=_load_function("dequant_k_cache.py", "_infer_dsa_dims"),
        _dequantize_k_cache_fast=kernel,
    )
    with pytest.raises(ValueError, match="does not match"):
        dequant(torch.empty(1, 1, 1, 264), dv=512)
    kernel.assert_not_called()


@pytest.mark.parametrize("nope,rope", [(256, 0), (512, 64)])
def test_quant_dispatches_inferred_geometry(nope, rope):
    kernel = Mock(return_value=torch.empty(2, nope + nope // 128 * 4 + rope * 2))
    quant = _load_function(
        "quant_k_cache.py",
        "_quantize_k_cache_fast_wrapped",
        _quantize_k_cache_fast=kernel,
    )
    quant(torch.empty(1, 2, 1, nope + rope, dtype=torch.bfloat16))
    assert kernel.call_args.kwargs["k_nope"].shape == (2, nope)
    assert kernel.call_args.kwargs["k_rope"].shape == (2, rope)


@pytest.mark.parametrize("missing_rope", [True, False])
def test_rocm_mha_concat_accepts_zero_rope_cache_reads(missing_rope):
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha_rocm.py"
    )
    tree = ast.parse(path.read_text())
    mixin = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    method = next(n for n in mixin.body if n.name == "_concat_and_cast_mha_k_rocm")
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    kernel = Mock()
    namespace = {"torch": torch, "concat_and_cast_mha_k_triton": kernel}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module([future, method], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    layer = SimpleNamespace(
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        qk_head_dim=256,
        num_local_heads=4,
        current_attention_backend="aiter",
    )
    # kv_b_proj stores K and V together; K is a non-contiguous view.
    kv = torch.arange(3 * 4 * 512, dtype=torch.float32).reshape(3, 4, 512)
    k_nope = kv[..., :256]
    k_pe = None if missing_rope else torch.empty(3, 1, 0)
    result = namespace[method.name](layer, k_nope, k_pe)
    assert result.is_contiguous()
    assert torch.equal(result, k_nope)
    kernel.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
