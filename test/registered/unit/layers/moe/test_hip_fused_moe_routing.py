import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe import hip_fused_moe_routing as routing
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeRouter:
    def __init__(self):
        self.calls = []

    def __call__(self, *, logits, bias_f32, x, rsf, shared_w, num_token_non_padded):
        self.calls.append(dict(logits=logits, rsf=rsf, shared_w=shared_w))
        m = x.shape[0]
        return dict(
            topk_w=torch.zeros((m, 7)),
            topk_ids=torch.zeros((m, 7), dtype=torch.int32),
            sorted_ids="sid",
            sorted_w="sw",
            sorted_eids="seid",
            num_valid="nv",
            a1="a1",
            a1_scale="a1s",
            block_m=32,
        )


@pytest.fixture
def router(monkeypatch):
    r = _FakeRouter()
    monkeypatch.setattr(routing, "_ENABLED", True)
    monkeypatch.setitem(routing._state, "router", r)
    monkeypatch.setitem(routing._state, "bias", {})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    yield r
    routing.clear()
    routing.set_decode(False)


def _call(x, logits, *, rsf=2.5):
    return routing.try_route(
        hidden_states=x,
        router_logits=logits,
        correction_bias=torch.zeros(384, dtype=torch.bfloat16),
        renormalize=True,
        routed_scaling_factor=rsf,
        shared_w=1.0,
        num_token_non_padded=None,
    )


def _inputs(m):
    return torch.zeros((m, 5120), dtype=torch.bfloat16), torch.zeros((m, 384))


def test_routes_staged_verify_call(router):
    x, logits = _inputs(384)
    routing.set_decode(True)
    routing.stage(x)
    out = _call(x, logits.bfloat16())
    assert out is not None and out[1].shape == (384, 7)
    assert router.calls[0]["logits"].dtype == torch.float32
    assert router.calls[0]["rsf"] == 2.5


@pytest.mark.parametrize(
    "rows, decode, staged",
    [(256, True, True), (2049, True, True), (384, False, True), (384, True, False)],
)
def test_falls_back(router, rows, decode, staged):
    x, logits = _inputs(rows)
    routing.set_decode(decode)
    routing.stage(x if staged else x.clone())
    assert _call(x, logits) is None
    assert not router.calls


def test_failed_caller_condition_never_routes(router):
    x, logits = _inputs(384)
    routing.set_decode(True)
    routing.stage(x)
    out = routing.route_or_skip(
        skip_reasons=(("scoring", True), ("packed ids", False)),
        hidden_states=x,
        router_logits=logits,
        correction_bias=torch.zeros(384),
        renormalize=True,
        routed_scaling_factor=1.0,
        shared_w=1.0,
        num_token_non_padded=None,
    )
    assert out is None and not router.calls


def test_disabled_never_routes(monkeypatch):
    monkeypatch.setattr(routing, "_ENABLED", False)
    x, logits = _inputs(384)
    routing.stage(x)
    assert _call(x, logits) is None
    routing.clear()


def test_aiter_hook_takes_kernel_outputs_once(router, monkeypatch):
    orig = SimpleNamespace(enabled=0, sort_quant=0)

    def orig_enabled(*a, **kw):
        orig.enabled += 1
        return False

    def orig_sort_quant(*a, **kw):
        orig.sort_quant += 1
        return "aiter"

    fm = ModuleType("aiter.fused_moe")
    fm._is_fused_decode_sort_quant_enabled = orig_enabled
    fm._fused_decode_sort_quant = orig_sort_quant
    aiter = ModuleType("aiter")
    aiter.fused_moe = fm
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.fused_moe", fm)
    routing._install_aiter_hook()

    x, logits = _inputs(384)
    routing.set_decode(True)
    routing.stage(x)
    w, ids = _call(x, logits)
    kw = dict(model_dim=5120, global_experts=385, block_m=32)
    sq = dict(kw, moebuf_dtype=torch.bfloat16, accumulate=False)
    assert fm._is_fused_decode_sort_quant_enabled(x, ids, w, **kw)
    out = fm._fused_decode_sort_quant(x, ids, w, **sq, num_local_tokens=None)
    assert out[0] == "sid" and out[5] == "a1" and out[6] == "a1s"
    # consumed: the next call goes to AITER's own path
    assert fm._fused_decode_sort_quant(x, ids, w, **sq) == "aiter"
    other = torch.zeros((384, 7), dtype=torch.int32)
    assert not fm._is_fused_decode_sort_quant_enabled(x, other, w, **kw)
    assert orig.enabled == 1 and orig.sort_quant == 1


def test_bias_cached_from_eager_calls_only(router, monkeypatch):
    """A bias copy made inside a CUDA-graph capture is only written when that graph replays;
    other graphs must not reuse it."""
    x, logits = _inputs(384)
    routing.set_decode(True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    routing.stage(x)
    _call(x, logits)
    assert not routing._state["bias"]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    routing.stage(x)
    _call(x, logits)
    assert len(routing._state["bias"]) == 1
