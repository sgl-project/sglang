"""Numerical equivalence and eligibility tests for the Path B fused swiglu kernel.

Two groups:
  * Model-based equivalence (``@requires_model``): loads a small MoE model, runs
    the fused gate_qmv + silu + ×x_up kernel against the unfused reference
    (``mx.gather_qmm`` + ``nn.silu(gate) * x_up``) on both the unsorted and
    sorted paths. Gated by SGLANG_MLX_TEST_MODEL so CI hosts without a model
    cache skip them.
  * Synthetic eligibility (no model, MLX only): the learned-bias fallback. The
    fused kernel recomputes the gate matmul and has no slot for the per-expert
    learned bias QuantizedSwitchLinear adds after the matmul, so ``can_fuse``
    must exclude a gate carrying one, and the patch must leave such a layer
    unfused. These run whenever MLX is importable.
"""

import os

import pytest

mx = pytest.importorskip("mlx.core")


# Model-based tests need a real checkpoint; synthetic tests below do not.
requires_model = pytest.mark.skipif(
    not os.environ.get("SGLANG_MLX_TEST_MODEL"),
    reason="Set SGLANG_MLX_TEST_MODEL to a HuggingFace model id to enable",
)


def _max_rel_diff(a, b):
    diff = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    max_abs = diff.max().item()
    ref_max = mx.abs(a.astype(mx.float32)).max().item()
    return max_abs, max_abs / max(ref_max, 1e-9)


@requires_model
def test_fused_gate_qmv_silu_mul_matches_unfused():
    """Kernel output matches ``nn.silu(gate_qmv) * x_up`` within bf16 ULP."""
    import mlx.nn as nn
    from mlx_lm import load

    from sglang.srt.hardware_backend.mlx.moe.fused_swiglu import (
        can_fuse,
        fused_gate_qmv_silu_mul,
    )

    model, _ = load(os.environ["SGLANG_MLX_TEST_MODEL"])
    sw = model.model.layers[0].mlp.switch_mlp
    assert can_fuse(sw), "layer 0 not eligible for fused swiglu"

    up = sw.up_proj
    gate = sw.gate_proj
    in_dim = up.scales.shape[-1] * up.group_size
    out_dim = up.weight.shape[-2]
    num_experts = up.weight.shape[0]
    dtype = up.scales.dtype

    # Two batch sizes both take the unsorted path (indices.size < 64).
    for B, TOPK in [(1, 8), (4, 8)]:
        x = mx.random.normal(shape=(B, 1, 1, in_dim)).astype(dtype)
        indices = mx.random.randint(0, num_experts, shape=(B, TOPK)).astype(mx.uint32)

        x_up = up(x, indices, sorted_indices=False)
        x_gate = gate(x, indices, sorted_indices=False)
        y_ref = nn.silu(x_gate) * x_up

        y_fused = fused_gate_qmv_silu_mul(
            x, gate["weight"], gate["scales"], gate.get("biases"), indices, x_up
        )
        mx.eval(y_ref, y_fused)

        assert y_ref.shape == y_fused.shape

        max_abs, rel = _max_rel_diff(y_ref, y_fused)
        # 2 % relative covers ~2 bf16 ULPs at typical activation magnitudes;
        # the kernel's fp32 accumulation order matches MLX's qmv_fast_impl so
        # most elements should land within 1 ULP.
        assert rel < 2e-2, f"B={B} TOPK={TOPK}: max_abs={max_abs:.3e} rel={rel:.2%}"


@requires_model
def test_patched_switchglu_matches_unpatched():
    """Full SwitchGLU forward equivalence on both sorted and unsorted paths."""
    from mlx_lm import load

    from sglang.srt.hardware_backend.mlx.moe.fused_swiglu import (
        patch_switch_glu_with_fused_swiglu,
    )

    model, _ = load(os.environ["SGLANG_MLX_TEST_MODEL"])
    sw = model.model.layers[0].mlp.switch_mlp
    in_dim = sw.up_proj.scales.shape[-1] * sw.up_proj.group_size
    num_experts = sw.up_proj.weight.shape[0]
    dtype = sw.up_proj.scales.dtype

    cases = []
    # B=2 TOPK=8 -> indices.size=16 < 64 -> unsorted
    # B=8 TOPK=8 -> indices.size=64 -> sorted
    for B, TOPK, label in [(2, 8, "unsorted"), (8, 8, "sorted")]:
        x = mx.random.normal(shape=(B, in_dim)).astype(dtype)
        indices = mx.random.randint(0, num_experts, shape=(B, TOPK)).astype(mx.uint32)
        out_ref = sw(x, indices)
        mx.eval(out_ref)
        cases.append((label, x, indices, out_ref))

    n_patched = patch_switch_glu_with_fused_swiglu(model)
    assert n_patched > 0, "no SwitchGLU layers were patched"

    for label, x, indices, out_ref in cases:
        out_fused = sw(x, indices)
        mx.eval(out_fused)
        max_abs, rel = _max_rel_diff(out_ref, out_fused)
        # 5 % is generous; in practice we see <0.6 % on 48-layer Qwen3-MoE.
        # The looser bound here absorbs cross-layer ULP propagation through
        # down_proj's quantized matmul.
        assert rel < 5e-2, f"full forward {label}: max_abs={max_abs:.3e} rel={rel:.2%}"


# Learned-bias fallback (synthetic, no model): a gate with a learned bias must
# not fuse, since the kernel has no slot for the bias added after the matmul.
def _quantized_switch_glu(in_dim, hidden, n_experts, gate_bias):
    """Small quantized SwitchGLU; gate carries a learned bias iff gate_bias.

    in_dim=512 keeps K%512==0 and hidden%8==0, inside the Path B v1 regime, so
    the bias-free build is genuinely fusion-eligible (the True control).
    """
    from mlx_lm.models.switch_layers import SwitchGLU

    sw = SwitchGLU(in_dim, hidden, n_experts, bias=False)
    sw.up_proj = sw.up_proj.to_quantized(group_size=64, bits=4, mode="affine")
    sw.down_proj = sw.down_proj.to_quantized(group_size=64, bits=4, mode="affine")
    gate = sw.gate_proj
    if gate_bias:
        # Learned per-expert bias (E, N), nonzero so dropping it would change
        # the result. to_quantized copies it into the QuantizedSwitchLinear.
        gate.bias = mx.random.normal((n_experts, hidden)) * 0.1
    sw.gate_proj = gate.to_quantized(group_size=64, bits=4, mode="affine")
    return sw


def test_can_fuse_excludes_learned_gate_bias():
    """can_fuse: False for a gate with a learned bias, True when bias-free."""
    from sglang.srt.hardware_backend.mlx.moe.fused_swiglu import can_fuse

    sw_free = _quantized_switch_glu(512, 64, 8, gate_bias=False)
    sw_bias = _quantized_switch_glu(512, 64, 8, gate_bias=True)
    assert "bias" not in sw_free.gate_proj
    assert "bias" in sw_bias.gate_proj
    assert can_fuse(sw_free) is True, "bias-free gate in regime should fuse"
    assert can_fuse(sw_bias) is False, "gate with learned bias must fall back"


def test_patch_falls_back_on_gate_bias():
    """Patching a biased-gate SwitchGLU is a no-op; the forward stays bias-correct."""
    import types

    from sglang.srt.hardware_backend.mlx.moe.fused_swiglu import (
        patch_switch_glu_with_fused_swiglu,
    )

    in_dim, hidden, n_experts, top_k, B = 512, 64, 8, 4, 2  # 2*4=8 < 64 -> unsorted
    sw = _quantized_switch_glu(in_dim, hidden, n_experts, gate_bias=True)

    x = mx.random.normal((B, in_dim))
    indices = mx.random.randint(0, n_experts, shape=(B, top_k)).astype(mx.uint32)
    out_before = sw(x, indices)
    mx.eval(out_before)

    # Minimal model stand-in: the patch walks model.model.layers[*].mlp.switch_mlp.
    mlp = types.SimpleNamespace(switch_mlp=sw, top_k=top_k)
    layer = types.SimpleNamespace(mlp=mlp)
    model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[layer]))

    n_patched = patch_switch_glu_with_fused_swiglu(model)
    assert n_patched == 0, "biased gate must not be patched"

    out_after = sw(x, indices)
    mx.eval(out_after)
    d = mx.abs(out_before.astype(mx.float32) - out_after.astype(mx.float32))
    diff = d.max().item()
    assert diff == 0.0, f"forward changed after (no-op) patch: max|delta|={diff:.3e}"


# Model-free numerical equivalence + non-stock-forward guard: the central
# correctness check, runs without a model download (skips where Metal is absent).
def test_fused_matches_unfused_synthetic():
    """Synthetic quantized gate weights: fused kernel vs the unfused
    gather_qmm + silu*x_up path, within the kernel's bf16 bound, plus finiteness."""
    mx.random.seed(0)

    import mlx.nn as nn

    from sglang.srt.hardware_backend.mlx.moe.fused_swiglu import (
        fused_gate_qmv_silu_mul,
    )

    # Gate regime: K%512==0, N%8==0, bits=4, group_size=64, affine.
    E, N, K, TOPK = 4, 16, 512, 2
    dtype = mx.bfloat16
    gate_w = (mx.random.normal((E, N, K)) * 0.02).astype(dtype)
    gwq, gs, gb = mx.quantize(gate_w, group_size=64, bits=4)
    mx.eval(gwq, gs, gb)

    # Two routing patterns: spread (hi=E) and collisions (many tokens, few experts).
    for B, hi in [(2, E), (4, max(1, E // 2))]:
        x = mx.random.normal((B, 1, 1, K)).astype(dtype)
        idx = mx.random.randint(0, hi, shape=(B, TOPK)).astype(mx.uint32)
        x_up = mx.random.normal((B, TOPK, 1, N)).astype(dtype)

        x_gate = mx.gather_qmm(
            x,
            gwq,
            gs,
            gb,
            rhs_indices=idx,
            transpose=True,
            group_size=64,
            bits=4,
            mode="affine",
        )
        y_ref = nn.silu(x_gate) * x_up
        y_fused = fused_gate_qmv_silu_mul(x, gwq, gs, gb, idx, x_up)
        mx.eval(y_ref, y_fused)

        assert y_ref.shape == y_fused.shape
        # A broken kernel must not leak NaN/Inf into the downstream down_proj matmul.
        assert bool(
            mx.all(mx.isfinite(y_fused.astype(mx.float32))).item()
        ), f"B={B} hi={hi}: non-finite fused output"
        # Same bf16 bound as the @requires_model kernel test.
        max_abs, rel = _max_rel_diff(y_ref, y_fused)
        assert rel < 2e-2, f"B={B} hi={hi}: max_abs={max_abs:.3e} rel={rel:.2%}"


def test_can_fuse_declines_nonstock_call():
    """can_fuse: False when SwitchGLU.__call__ is overridden (the fused subclass
    would impose stock semantics and silently bypass the override), True for stock."""
    from mlx_lm.models.switch_layers import SwitchGLU

    from sglang.srt.hardware_backend.mlx.moe.fused_swiglu import can_fuse

    # hidden=64 keeps down_proj's input dim divisible by the quant group size.
    sw_stock = _quantized_switch_glu(512, 64, 4, gate_bias=False)
    assert can_fuse(sw_stock) is True, "stock in-regime SwitchGLU should fuse"

    class _CustomSwitchGLU(SwitchGLU):
        def __call__(self, x, indices):  # overridden forward
            return super().__call__(x, indices)

    sw_custom = _quantized_switch_glu(512, 64, 4, gate_bias=False)
    sw_custom.__class__ = _CustomSwitchGLU  # same swap mechanism the patch uses
    assert can_fuse(sw_custom) is False, "non-stock __call__ must fall back"


def test_can_fuse_declines_non_silu_activation():
    """can_fuse: False for a non SiLU activation (the kernel and the fallback
    both bake in silu, which would silently replace the module's formula),
    True for the stock SwiGLU control."""
    import types

    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchGLU

    from sglang.srt.hardware_backend.mlx.moe.fused_swiglu import (
        can_fuse,
        patch_switch_glu_with_fused_swiglu,
    )

    # Same build as _quantized_switch_glu, but the activation kwarg is the
    # subject under test, so construct directly.
    sw = SwitchGLU(512, 64, 4, activation=nn.gelu, bias=False)
    for name in ("up_proj", "gate_proj", "down_proj"):
        proj = getattr(sw, name)
        setattr(sw, name, proj.to_quantized(group_size=64, bits=4, mode="affine"))
    assert can_fuse(sw) is False, "non SiLU activation must fall back"

    mlp = types.SimpleNamespace(switch_mlp=sw, top_k=4)
    layer = types.SimpleNamespace(mlp=mlp)
    model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[layer]))
    assert patch_switch_glu_with_fused_swiglu(model) == 0, "gelu module must not patch"

    sw_stock = _quantized_switch_glu(512, 64, 4, gate_bias=False)
    assert can_fuse(sw_stock) is True, "stock SwiGLU activation should fuse"


def test_fused_forward_falls_back_on_dtype_mismatch():
    """A runtime activation dtype the fused kernel rejects but gather_qmm
    tolerates (bf16 gate params, fp16 activations) must fall back, not crash,
    and match the unfused forward."""
    import types

    from mlx_lm.models.switch_layers import SwitchGLU

    from sglang.srt.hardware_backend.mlx.moe.fused_swiglu import (
        patch_switch_glu_with_fused_swiglu,
    )

    mx.random.seed(0)
    in_dim, hidden, n_experts, top_k, B = 512, 64, 4, 4, 2  # 2*4=8 < 64 -> unsorted
    sw = SwitchGLU(in_dim, hidden, n_experts, bias=False)
    for name in ("up_proj", "gate_proj", "down_proj"):
        lin = getattr(sw, name)
        lin.weight = lin.weight.astype(mx.bfloat16)  # bf16 weight -> bf16 scales
        setattr(sw, name, lin.to_quantized(group_size=64, bits=4, mode="affine"))
    assert sw.gate_proj.scales.dtype == mx.bfloat16

    # fp16 activations mismatch the bf16 gate params: the fused kernel raises,
    # the unfused gather_qmm tolerates it.
    x = mx.random.normal((B, in_dim)).astype(mx.float16)
    indices = mx.random.randint(0, n_experts, shape=(B, top_k)).astype(mx.uint32)

    out_ref = sw(x, indices)  # stock forward, unpatched
    mx.eval(out_ref)

    mlp = types.SimpleNamespace(switch_mlp=sw, top_k=top_k)
    layer = types.SimpleNamespace(mlp=mlp)
    model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[layer]))
    assert patch_switch_glu_with_fused_swiglu(model) == 1, "layer should patch"

    out_fb = sw(x, indices)  # patched -> kernel raises -> fallback, no crash
    mx.eval(out_fb)
    max_abs, rel = _max_rel_diff(out_ref, out_fb)
    assert rel < 1e-3, f"fallback != unfused: max_abs={max_abs:.3e} rel={rel:.2%}"


# Fallback index contract (synthetic, no model): the fallback must see the
# untouched flat indices. The sorted path's (M_tok, 1) kernel reshape once
# leaked into the fallback and broadcast an M_tok x M_tok cross product
# (PR #26188 review repro).
def _bf16_quantized_switch_glu(in_dim, hidden, n_experts):
    """Quantize from bf16 weights so fp16 activations trip the kernel's runtime
    dtype check while the unfused path tolerates them."""
    from mlx_lm.models.switch_layers import SwitchGLU

    sw = SwitchGLU(in_dim, hidden, n_experts, bias=False)
    for name in ("up_proj", "gate_proj", "down_proj"):
        lin = getattr(sw, name)
        lin.weight = lin.weight.astype(mx.bfloat16)
        setattr(sw, name, lin.to_quantized(group_size=64, bits=4, mode="affine"))
    return sw


def test_sorted_dtype_mismatch_fallback_matches_reference(monkeypatch):
    """Reviewer repro: B*T == 64 takes the sorted path, the kernel rejects fp16
    activations on bf16 params, and the fallback must match the reference in
    shape and value."""
    import types

    import sglang.srt.hardware_backend.mlx.moe.fused_swiglu as fused_swiglu

    monkeypatch.setattr(fused_swiglu, "_fallback_warned", False)
    mx.random.seed(0)
    in_dim, hidden, n_experts, top_k, B = 512, 64, 4, 4, 16  # 16*4 = 64 -> sorted
    sw = _bf16_quantized_switch_glu(in_dim, hidden, n_experts)

    x = mx.random.normal((B, in_dim)).astype(mx.float16)
    indices = mx.random.randint(0, n_experts, shape=(B, top_k)).astype(mx.uint32)
    out_ref = sw(x, indices)
    mx.eval(out_ref)

    mlp = types.SimpleNamespace(switch_mlp=sw, top_k=top_k)
    layer = types.SimpleNamespace(mlp=mlp)
    model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[layer]))
    assert fused_swiglu.patch_switch_glu_with_fused_swiglu(model) == 1

    out_fb = sw(x, indices)
    mx.eval(out_fb)
    assert out_fb.shape == out_ref.shape
    max_abs, rel = _max_rel_diff(out_ref, out_fb)
    # Post fix the fallback runs the same MLX ops as the stock forward, so the
    # bound only absorbs compiled vs eager elementwise ordering (~1 fp16 ULP).
    assert bool(
        mx.allclose(
            out_fb.astype(mx.float32),
            out_ref.astype(mx.float32),
            rtol=2e-3,
            atol=2e-4,
        ).item()
    ), f"sorted fallback != reference: max_abs={max_abs:.3e} rel={rel:.2%}"


def test_unsorted_dtype_mismatch_fallback_matches_reference(monkeypatch):
    """Sibling guard: same dtype mismatch on the unsorted path (B*T < 64)."""
    import types

    import sglang.srt.hardware_backend.mlx.moe.fused_swiglu as fused_swiglu

    monkeypatch.setattr(fused_swiglu, "_fallback_warned", False)
    mx.random.seed(0)
    in_dim, hidden, n_experts, top_k, B = 512, 64, 4, 4, 2  # 2*4 = 8 < 64 -> unsorted
    sw = _bf16_quantized_switch_glu(in_dim, hidden, n_experts)

    x = mx.random.normal((B, in_dim)).astype(mx.float16)
    indices = mx.random.randint(0, n_experts, shape=(B, top_k)).astype(mx.uint32)
    out_ref = sw(x, indices)
    mx.eval(out_ref)

    mlp = types.SimpleNamespace(switch_mlp=sw, top_k=top_k)
    layer = types.SimpleNamespace(mlp=mlp)
    model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[layer]))
    assert fused_swiglu.patch_switch_glu_with_fused_swiglu(model) == 1

    out_fb = sw(x, indices)
    mx.eval(out_fb)
    assert out_fb.shape == out_ref.shape
    max_abs, rel = _max_rel_diff(out_ref, out_fb)
    assert bool(
        mx.allclose(
            out_fb.astype(mx.float32),
            out_ref.astype(mx.float32),
            rtol=2e-3,
            atol=2e-4,
        ).item()
    ), f"unsorted fallback != reference: max_abs={max_abs:.3e} rel={rel:.2%}"


def test_forced_kernel_rejection_falls_back_correctly(monkeypatch):
    """Any NotFusable from the fused kernel, not just a dtype mismatch, must
    take the identical fallback: force one via monkeypatch and check both
    routing paths against the unpatched module."""
    import types

    import sglang.srt.hardware_backend.mlx.moe.fused_swiglu as fused_swiglu

    monkeypatch.setattr(fused_swiglu, "_fallback_warned", False)
    mx.random.seed(0)
    in_dim, hidden, n_experts, top_k = 512, 64, 4, 4
    sw = _quantized_switch_glu(in_dim, hidden, n_experts, gate_bias=False)

    cases = []
    # B=2 -> 8 < 64 -> unsorted; B=16 -> 64 -> sorted.
    for B, label in [(2, "unsorted"), (16, "sorted")]:
        x = mx.random.normal((B, in_dim))
        indices = mx.random.randint(0, n_experts, shape=(B, top_k)).astype(mx.uint32)
        out_ref = sw(x, indices)
        mx.eval(out_ref)
        cases.append((label, x, indices, out_ref))

    # top_k=None skips _aot_warm_kernel: _quantized_switch_glu's fp32 scales
    # are outside _DTYPES, and warm dispatch hits the real kernel with no
    # exception handling, before the raiser below is installed.
    mlp = types.SimpleNamespace(switch_mlp=sw, top_k=None)
    layer = types.SimpleNamespace(mlp=mlp)
    model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[layer]))
    # Patch before installing the raiser: _aot_warm_kernel dispatches the real
    # kernel at patch time and does not catch ValueError.
    assert fused_swiglu.patch_switch_glu_with_fused_swiglu(model) == 1

    def raiser(*args, **kwargs):
        raise fused_swiglu.NotFusable("forced rejection")

    monkeypatch.setattr(fused_swiglu, "fused_gate_qmv_silu_mul", raiser)

    for label, x, indices, out_ref in cases:
        out_fb = sw(x, indices)
        mx.eval(out_fb)
        assert out_fb.shape == out_ref.shape, label
        max_abs, rel = _max_rel_diff(out_ref, out_fb)
        assert bool(
            mx.allclose(
                out_fb.astype(mx.float32),
                out_ref.astype(mx.float32),
                rtol=1e-5,
                atol=1e-6,
            ).item()
        ), f"forced rejection {label}: max_abs={max_abs:.3e} rel={rel:.2%}"


def test_ineligible_input_surfaces_pre_change_message_through_wrapper(
    monkeypatch, caplog
):
    """The K-divisibility diagnostic, the first check in the regime chain,
    must still reach the wrapper's warning log with the same string the
    pre-change ValueError carried, now raised as NotFusable."""
    import logging

    import sglang.srt.hardware_backend.mlx.moe.fused_swiglu as fused_swiglu

    monkeypatch.setattr(fused_swiglu, "_fallback_warned", False)
    mx.random.seed(0)
    # in_dim=64 -> K=64, not divisible by BLOCK_SIZE (512): the K check trips
    # before the N or dtype checks are ever reached.
    in_dim, hidden, n_experts, top_k, B = 64, 64, 4, 2, 2
    sw = _quantized_switch_glu(in_dim, hidden, n_experts, gate_bias=False)
    gate_proj = sw.gate_proj

    x = mx.random.normal((B, 1, 1, in_dim)).astype(mx.float32)
    idx = mx.random.randint(0, n_experts, shape=(B, top_k)).astype(mx.uint32)
    x_up = mx.random.normal((B, top_k, 1, hidden)).astype(mx.float32)

    # Verbatim from the pre-change dispatch_fallback ValueError string.
    expected = (
        "fused_gate_qmv_silu_mul: K=64 not divisible by 512. Use the unfused path."
    )

    with caplog.at_level(logging.WARNING):
        out = fused_swiglu._fused_gate_or_fallback(
            gate_proj, x, idx, x_up, sorted_indices=False
        )
    mx.eval(out)

    assert any(
        expected in record.getMessage() for record in caplog.records
    ), f"expected message not in log: {[r.getMessage() for r in caplog.records]}"


def test_matched_fp32_triple_falls_back_with_membership_message(monkeypatch, caplog):
    """A matched fp32 triple (x, gate_s, gate_b all float32) passes the K, N,
    and mutual-match checks but fails the dtype membership check; the wrapper
    must still fall back cleanly and surface the membership message."""
    import logging

    import sglang.srt.hardware_backend.mlx.moe.fused_swiglu as fused_swiglu

    monkeypatch.setattr(fused_swiglu, "_fallback_warned", False)
    mx.random.seed(0)
    # in_dim=512 -> K=512 (%512==0), hidden=64 -> N=64 (%8==0): in regime.
    # _quantized_switch_glu's default construction quantizes to fp32
    # scales/biases, matching x's fp32 dtype below.
    in_dim, hidden, n_experts, top_k, B = 512, 64, 4, 2, 2
    sw = _quantized_switch_glu(in_dim, hidden, n_experts, gate_bias=False)
    gate_proj = sw.gate_proj
    assert gate_proj.scales.dtype == mx.float32

    x = mx.random.normal((B, 1, 1, in_dim)).astype(mx.float32)
    idx = mx.random.randint(0, n_experts, shape=(B, top_k)).astype(mx.uint32)
    x_up = mx.random.normal((B, top_k, 1, hidden)).astype(mx.float32)

    expected = (
        f"fused_gate_qmv_silu_mul: dtype {mx.float32} not in {fused_swiglu._DTYPES}."
    )

    with caplog.at_level(logging.WARNING):
        out = fused_swiglu._fused_gate_or_fallback(
            gate_proj, x, idx, x_up, sorted_indices=False
        )
    mx.eval(out)

    assert any(
        expected in record.getMessage() for record in caplog.records
    ), f"expected message not in log: {[r.getMessage() for r in caplog.records]}"


def test_patch_skips_warm_and_logs_for_out_of_regime_dtype(monkeypatch, caplog):
    """A matched fp32 SwitchGLU is eligible per can_fuse (which does not check
    dtype membership) and must still patch successfully; _aot_warm_kernel has
    to catch the membership NotFusable at warm time, skip warming, and log
    it, rather than letting the patch crash."""
    import logging
    import types

    import sglang.srt.hardware_backend.mlx.moe.fused_swiglu as fused_swiglu

    mx.random.seed(0)
    # in_dim=512 -> K=512 (%512==0), hidden=64 -> N=64 (%8==0): in regime for
    # can_fuse, which does not check dtype membership against _DTYPES.
    in_dim, hidden, n_experts, top_k = 512, 64, 4, 4
    sw = _quantized_switch_glu(in_dim, hidden, n_experts, gate_bias=False)
    assert sw.gate_proj.scales.dtype == mx.float32

    mlp = types.SimpleNamespace(switch_mlp=sw, top_k=top_k)
    layer = types.SimpleNamespace(mlp=mlp)
    model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[layer]))

    with caplog.at_level(logging.DEBUG, logger=fused_swiglu.logger.name):
        n_patched = fused_swiglu.patch_switch_glu_with_fused_swiglu(model)

    assert n_patched == 1, "matched fp32 model should still patch"
    assert any(
        "skipping warm" in record.getMessage() for record in caplog.records
    ), f"expected skip-warm debug log not found: {[r.getMessage() for r in caplog.records]}"


def test_warmup_specs_enumerates_declared_dtypes():
    from sglang.srt.hardware_backend.mlx.moe.fused_swiglu import (
        _DTYPES,
        FusedGateQmvSiluMulKernel,
    )

    specs = list(FusedGateQmvSiluMulKernel().warmup_specs(model=None))
    assert len(specs) > 0
    assert len(specs) == len(_DTYPES)
    for spec in specs:
        (dtype,) = spec.dtypes
        assert dtype in _DTYPES
        assert spec.shapes == ()
