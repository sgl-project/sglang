"""The DeepSeek-V4.1 gfx950 monokernel spec on CPU (RFC #43949): its spans on the
DeepSeek-V4.1-Flash layer structure, its widths, its registry entry, its
per-output bounds and its refusal reasons."""

from types import SimpleNamespace

from sglang.kernels.ops.mono import spec as registry
from sglang.kernels.ops.mono.deepseek_v41_hip import HF_CONFIG
from sglang.kernels.ops.mono.deepseek_v41_hip import spec as dsv41
from sglang.kernels.ops.mono.spec import NumericsKind, SpanPoint
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

# DeepSeek-V4.1-Flash's layer structure (its config.json)
FLASH = SimpleNamespace(
    num_hidden_layers=40,
    compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
    kv_source_layer_ids=[2, 8, 14, 20],
    index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
    engram_layer_ids=[1, 14],
    **HF_CONFIG,
    model_type="deepseek_v41",
    architectures=["DeepseekV4ForCausalLM"],
)


def test_flash_spans():
    # every backbone layer whole: the first, the Engram layers and the index
    # sources (their compressor and indexer between the launches) too, and
    # DSpark's target layers (37-39), whose captures the spans take
    spans = dsv41.SPEC.spans(FLASH)
    assert spans == {i: (dsv41.LAYER,) for i in range(40)}


def test_other_layers_keep_the_normal_path():
    hashed = SimpleNamespace(**{**vars(FLASH), "num_hash_layers": 4})
    spans = dsv41.SPEC.spans(hashed)
    assert 0 not in spans and 3 not in spans  # hash-routed MoE
    assert spans[4] == (dsv41.LAYER,)  # K1 folds in the normal path's pending post
    ratio4 = SimpleNamespace(**{**vars(FLASH), "compress_ratios": [4] * 43})
    assert dsv41.SPEC.spans(ratio4) == {}  # another compress ratio


def test_the_span_is_the_whole_layer():
    assert (dsv41.LAYER.start, dsv41.LAYER.end) == (
        SpanPoint.LAYER_IN,
        SpanPoint.LAYER_OUT,
    )


def test_widths_are_rows_up_to_48():
    # 48 decode requests, or 8 DSpark verify requests of 6 tokens
    limits = dsv41.SPEC.devices["gfx950"]
    assert limits.widths == tuple(range(1, 49)) and limits.tp_sizes == (2, 4)


def test_registry_entry():
    spec, why = registry.find_spec(["DeepseekV4ForCausalLM"], "gfx950")
    assert spec is dsv41.SPEC, why
    spec, why = registry.find_spec(["DeepseekV4ForCausalLM"], "gfx942")
    assert spec is None and "gfx942" in why


def test_bounds_are_per_output():
    n = dsv41.SPEC.numerics
    assert n.kind is NumericsKind.TOLERANCE
    assert n.bound("k1.kv.max_abs") == 0.0  # the KV records are bit-exact
    for name in ("k1.q.rel", "k2.wo_b.rel", "ffn.moe.cosine"):
        assert n.bound(name) > 0


def _cfg(**kw):
    defaults = {"speculative_algorithm": "DSPARK", "fp8_gemm_runner_backend": "aiter"}
    return SimpleNamespace(**{**defaults, **kw})


def test_config_reasons_name_each_mismatch():
    reasons = dsv41.SPEC.config_disable_reasons(_cfg(), FLASH)
    assert not any("DeepSeek-V4.1-Flash" in r or "speculative" in r for r in reasons), (
        reasons
    )
    other = SimpleNamespace(
        **{**vars(FLASH), "hidden_size": 7168, "model_type": "deepseek_v4"}
    )
    reasons = dsv41.SPEC.config_disable_reasons(
        _cfg(speculative_algorithm="EAGLE"), other
    )
    text = " | ".join(reasons)
    for want in ("hidden_size = 5120", "deepseek_v41", "speculative algorithm EAGLE"):
        assert want in text, text
    reasons = dsv41.SPEC.config_disable_reasons(
        _cfg(enable_encoder_swa_bounded_replay=True), FLASH
    )
    assert any("request window" in r for r in reasons), reasons
    # the default FP8 GEMM backend reorders the attention weights in place
    reasons = dsv41.SPEC.config_disable_reasons(
        _cfg(fp8_gemm_runner_backend="auto"), FLASH
    )
    assert any("--fp8-gemm-backend 'auto'" in r for r in reasons), reasons


def _layer(i, **attn):
    attn = {
        "compressor": None,
        "indexer": None,
        "q_head_norm": False,
        "fuse_wqa_wkv": True,
        "n_local_heads": 16,
        "compress_ratio": 0 if i < 2 else 2 if i < 20 else 1,
        **{lin: SimpleNamespace() for lin in ("wqkv_a", "wq_b", "wo_a", "wo_b")},
        **attn,
    }
    experts = SimpleNamespace(w13_weight=SimpleNamespace(is_shuffled=True))
    return SimpleNamespace(
        hc_boundary_fused=True,
        engram=None,
        self_attn=SimpleNamespace(**attn),
        mlp=SimpleNamespace(is_nextn=False, is_hash=False, experts=experts),
    )


def test_model_reasons(monkeypatch):
    import sglang.srt.runtime_context as rc

    monkeypatch.setattr(rc, "get_parallel", lambda: SimpleNamespace(tp_size=4))
    layers = [_layer(i) for i in range(40)]
    model = SimpleNamespace(layers=layers)
    seen = {
        ("shared expert fp8", i, p, k): None
        for i in range(40)
        for p in dsv41._SHARED
        for k in ("weight", "scale")
    }
    assert dsv41.SPEC.model_disable_reasons(model, FLASH, seen) == []
    # an FP8 linear method shuffled layer 5's wo_b in place; layer 1 (window only)
    # has a compressor; layer 7's shared expert never went through the load hook
    layers[5] = _layer(5, wo_b=SimpleNamespace(aiter_bpreshuffled=True))
    layers[1] = _layer(1, compress_ratio=0, compressor=SimpleNamespace())
    seen = {k: v for k, v in seen.items() if k[1] != 7}
    reasons = dsv41.SPEC.model_disable_reasons(model, FLASH, seen)
    assert len(reasons) == 3, reasons
    assert reasons[0].startswith("layer 1: a compressor or indexer at compress ratio 0")
    assert reasons[1].startswith("layer 5: an FP8 linear method reordered")
    assert "(layers [7])" in reasons[2]
