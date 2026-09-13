"""CPU dispatch checks; real attention parity is a separate GPU test."""

import ast
import inspect
import logging
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "python/sglang/multimodal_gen/runtime/layers/attention/backends/aiter.py"
)


def namespace():
    ns = dict(inspect=inspect, cache=cache, logger=logging.getLogger(__name__))
    helper = next(
        n
        for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_supports_int32_varlen_strides"
    )
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


def test_capability_detects_old_and_new_apis():
    ns = namespace()
    assert ns["_supports_int32_varlen_strides"](lambda q, prefer_int32_strides=False: q)
    assert not ns["_supports_int32_varlen_strides"](lambda q: q)


class TensorStub:
    device = "cuda"

    def contiguous(self):
        return self

    def to(self, **kwargs):
        return self


@pytest.mark.parametrize(
    "gfx942,enabled,supported",
    [(True, True, True), (True, True, False), (True, False, True), (False, True, True)],
)
def test_only_supported_opted_in_gfx942_calls_receive_keyword(
    gfx942, enabled, supported
):
    def attention_func(**kwargs):
        return kwargs

    ns = namespace()
    ns.update(
        USE_AITER_GFX942=gfx942,
        _use_int32_varlen_strides=enabled,
        _supports_int32_varlen_strides=lambda fn: supported,
        importlib=SimpleNamespace(
            import_module=lambda name: SimpleNamespace(
                flash_attn_varlen_func=attention_func
            )
        ),
        aiter=SimpleNamespace(flash_attn_varlen_func=attention_func),
        torch=SimpleNamespace(int32="int32"),
    )
    method = next(
        n
        for n in ast.walk(ast.parse(SOURCE.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "forward_varlen"
    )
    method.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), ns)
    tensor = TensorStub()
    result = ns["forward_varlen"](
        SimpleNamespace(softmax_scale=0.125, causal=False),
        tensor,
        tensor,
        tensor,
        cu_seqlens=tensor,
        max_seqlen=65536,
    )
    assert result.get("prefer_int32_strides", False) == (
        gfx942 and enabled and supported
    )
    assert result["max_seqlen_q"] == result["max_seqlen_k"] == 65536


def test_no_global_stride_mutation():
    assert "mha_set_use_int64_strides" not in SOURCE.read_text()
