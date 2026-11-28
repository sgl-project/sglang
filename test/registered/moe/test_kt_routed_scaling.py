from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
from sglang.srt.models import deepseek_v2
from sglang.srt.models.deepseek_v2 import _post_moe_needs_routed_scaling
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu-intel")


def _device_flags(monkeypatch, *, cuda=False, musa=False, xpu=False, aiter=False):
    monkeypatch.setattr(deepseek_v2, "_is_cuda", cuda)
    monkeypatch.setattr(deepseek_v2, "_is_musa", musa)
    monkeypatch.setattr(deepseek_v2, "_is_xpu", xpu)
    monkeypatch.setattr(deepseek_v2, "_use_aiter", aiter)


def _kt_method() -> KTEPWrapperMethod:
    # The predicate only reads the `applies_routing_weights_as_given` class
    # attribute; wrapper state is never touched and no init runs.
    return object.__new__(KTEPWrapperMethod)


def test_predicate_duck_types_the_wrapper_flag(monkeypatch):
    # The decision follows the declared attribute, not the wrapper type: any
    # method claiming it consumes the routing weights as given gets KT rules.
    _device_flags(monkeypatch, cuda=True)

    class _StubMethod:
        applies_routing_weights_as_given = True

    assert not _post_moe_needs_routed_scaling(_StubMethod(), fused_in_topk=True)
    assert _post_moe_needs_routed_scaling(_StubMethod(), fused_in_topk=False)


def test_kt_skips_the_scale_when_topk_folded_it(monkeypatch):
    # MXFP4 methods report fuse_routed_scaling_factor_in_topk, so the routing
    # weights already carry the factor and scaling again would square it.
    _device_flags(monkeypatch, cuda=True)
    assert not _post_moe_needs_routed_scaling(_kt_method(), fused_in_topk=True)


def test_kt_scales_once_when_topk_did_not_fold_it(monkeypatch):
    # e.g. KT over FP8: the top-k leaves the factor out, so the epilogue owes it.
    _device_flags(monkeypatch, cuda=True)
    assert _post_moe_needs_routed_scaling(_kt_method(), fused_in_topk=False)


def test_kt_ignores_device_flags(monkeypatch):
    _device_flags(monkeypatch)
    assert not _post_moe_needs_routed_scaling(_kt_method(), fused_in_topk=True)
    assert _post_moe_needs_routed_scaling(_kt_method(), fused_in_topk=False)


def test_non_kt_device_semantics_unchanged(monkeypatch):
    # One guard for the paths this refactor must not move: CUDA/MUSA/XPU keep
    # the runner-owned scale, plain CPU keeps the epilogue-owned scale, and
    # aiter stays unconditional-fold-aware. Every non-KT quant method hits
    # these lines exactly as on main.
    _device_flags(monkeypatch, cuda=True)
    assert not _post_moe_needs_routed_scaling(object(), fused_in_topk=True)
    assert not _post_moe_needs_routed_scaling(object(), fused_in_topk=False)
    _device_flags(monkeypatch, musa=True)
    assert not _post_moe_needs_routed_scaling(object(), fused_in_topk=False)
    _device_flags(monkeypatch, xpu=True)
    assert not _post_moe_needs_routed_scaling(object(), fused_in_topk=False)
    _device_flags(monkeypatch)
    assert _post_moe_needs_routed_scaling(object(), fused_in_topk=False)
    assert _post_moe_needs_routed_scaling(object(), fused_in_topk=True)
    _device_flags(monkeypatch, aiter=True)
    assert not _post_moe_needs_routed_scaling(object(), fused_in_topk=True)
    assert _post_moe_needs_routed_scaling(object(), fused_in_topk=False)
