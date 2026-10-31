from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.models.dits import base
from sglang.multimodal_gen.runtime.models.dits.base import cached_per_request


def _run(forward_batch, name, inputs, calls):
    ctx = SimpleNamespace(forward_batch=forward_batch) if forward_batch else None
    with patch.object(base, "get_forward_context_or_none", lambda: ctx):
        return cached_per_request(
            name, inputs, lambda: (calls.append(1), inputs[0] * 2)[1]
        )


def test_reuse_within_request_and_isolation_across_requests():
    calls = []
    req, other = SimpleNamespace(extra={}), SimpleNamespace(extra={})
    a, b = torch.randn(1, 3, 4), torch.randn(1, 3, 4)
    with torch.no_grad():
        for fb in (req, req, other):
            for t in (a, b):
                torch.testing.assert_close(_run(fb, "x", (t,), calls), t * 2)
    assert len(calls) == 4


def test_no_cache_without_context_or_with_grad():
    calls = []
    a = torch.randn(2, 2)
    with torch.no_grad():
        _run(None, "x", (a,), calls)
        _run(None, "x", (a,), calls)
    assert len(calls) == 2
    req = SimpleNamespace(extra={})
    _run(req, "x", (a,), calls)
    _run(req, "x", (a,), calls)
    assert len(calls) == 4


def test_cache_is_bounded_when_inputs_change_every_step():
    calls = []
    req = SimpleNamespace(extra={})
    with torch.no_grad():
        for _ in range(20):
            _run(req, "x", (torch.randn(2, 2),), calls)
    assert len(req.extra["x"]) <= base._REQUEST_CACHE_MAX


def test_sana_context_kvs_projected_once_per_request():
    from sglang.multimodal_gen.runtime.models.dits.sana import SanaTransformer2DModel

    calls = []

    def project_kv(text):
        calls.append(1)
        return text * 2, text * 3

    model = SimpleNamespace(
        layerwise_offload_managers=[],
        transformer_blocks=[
            SimpleNamespace(attn2=SimpleNamespace(project_kv=project_kv))
        ]
        * 2,
        _cached_per_request=lambda *a: cached_per_request(*a),
    )
    text = torch.randn(1, 3, 16)
    ctx = SimpleNamespace(forward_batch=SimpleNamespace(extra={}))
    with torch.no_grad():
        with patch.object(base, "get_forward_context_or_none", lambda: ctx):
            for _ in range(3):
                kvs = SanaTransformer2DModel._context_kvs(model, text)
        # one projection per block for the whole request
        assert len(calls) == 2
        torch.testing.assert_close(kvs[0][0], text * 2)
        torch.testing.assert_close(kvs[1][1], text * 3)
        # Offloaded blocks project inline.
        model.layerwise_offload_managers = [object()]
        assert SanaTransformer2DModel._context_kvs(model, text) == [None, None]
