"""Unit test for the encoder parallel-folding decision.

Verifies `ServerArgs.adjust_pipeline_config` decides *when* to TP-shard an
encoder across the whole DiT replica (`mode=world`) vs leaving it alone, for
every tp/sp/cfg combination — the "encoder uses all idle replica GPUs during
encoding" rule — for *any* text/image encoder, not just T5, without degrading
the pure-TP or dp>1 paths. Also covers the divisibility guard and the config
fields now living on the shared base. Pure logic, no GPU or distributed init.
"""

from types import SimpleNamespace

from sglang.multimodal_gen.configs.models.encoders import (
    EncoderConfig,
    ImageEncoderConfig,
    TextEncoderConfig,
)
from sglang.multimodal_gen.configs.models.encoders.t5 import T5Config
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _run(encoders, tp, sp, cfg, dp=1, disagg=False, num_gpus=None, image=()):
    self = SimpleNamespace(
        tp_size=tp,
        sp_degree=sp,
        cfg_parallel_degree=cfg,
        dp_size=dp,
        disagg_mode=disagg,
        num_gpus=num_gpus if num_gpus is not None else tp * sp * cfg * dp,
        pipeline_config=SimpleNamespace(
            text_encoder_configs=tuple(encoders),
            image_encoder_configs=tuple(image),
        ),
    )
    ServerArgs.adjust_pipeline_config(self)


def _t5(heads=16, d_ff=64):
    enc = T5Config()
    enc.num_heads = heads
    enc.d_ff = d_ff
    enc.parallel_folding = False
    enc.parallel_folding_mode = "sp"
    return enc


def _generic_text(heads=16, inter=64):
    """A non-T5 text encoder that spells its dims num_attention_heads /
    intermediate_size (CLIP/Llama/Qwen/Gemma family)."""
    enc = TextEncoderConfig()
    enc.num_attention_heads = heads
    enc.intermediate_size = inter
    return enc


def _decide(tp, sp, cfg, dp=1, disagg=False, num_gpus=None, heads=16, d_ff=64):
    enc = _t5(heads=heads, d_ff=d_ff)
    _run([enc], tp, sp, cfg, dp=dp, disagg=disagg, num_gpus=num_gpus)
    return (enc.parallel_folding, enc.parallel_folding_mode)


# --- original T5 decision matrix (unchanged behavior) ----------------------


def test_pure_tp_not_folded():
    # replica == tp: encoder already uses every replica GPU; leave untouched.
    assert _decide(tp=2, sp=1, cfg=1) == (False, "sp")


def test_single_gpu_not_folded():
    assert _decide(tp=1, sp=1, cfg=1) == (False, "sp")


def test_cfg_parallel_folds_world():
    # tp=1, cfg=2: was single-GPU encoder; now folds across the 2-GPU replica.
    assert _decide(tp=1, sp=1, cfg=2) == (True, "world")


def test_sp_dp1_folds_world():
    # pure SP on a single replica: replica == sp > tp, uses the world group.
    assert _decide(tp=1, sp=2, cfg=1) == (True, "world")


def test_tp_times_cfg_folds_world():
    # tp=2 x cfg=2 = 4-GPU replica: encoder was on tp_group(2); now spans 4.
    assert _decide(tp=2, sp=1, cfg=2, heads=16, d_ff=64) == (True, "world")


def test_sp_times_cfg_folds_world():
    assert _decide(tp=1, sp=2, cfg=2) == (True, "world")


def test_dp_gt_1_keeps_sp_folding():
    # dp>1: world group spans replicas, so keep the existing per-SP folding.
    assert _decide(tp=1, sp=2, cfg=1, dp=2) == (True, "sp")
    # dp>1 pure cfg has no SP folding to fall back to -> not folded here.
    assert _decide(tp=1, sp=1, cfg=2, dp=2) == (False, "sp")


def test_disagg_not_world_folded():
    assert _decide(tp=1, sp=2, cfg=2, disagg=True) == (True, "sp")


def test_num_gpus_is_authoritative_for_replica():
    # Replica size comes from num_gpus // dp, so any parallelism reflected in
    # the GPU count is covered without enumerating it.
    assert _decide(tp=2, sp=1, cfg=1, num_gpus=8, heads=16, d_ff=64) == (
        True,
        "world",
    )


def test_indivisible_replica_not_folded():
    # replica does not divide the encoder heads/d_ff -> cannot shard -> skip.
    assert _decide(tp=1, sp=1, cfg=3, heads=16, d_ff=64) == (False, "sp")


# --- generalization: any encoder, not just T5 ------------------------------


def test_generic_text_encoder_folds_world():
    # A non-T5 encoder (num_attention_heads / intermediate_size) folds too.
    enc = _generic_text(heads=16, inter=64)
    _run([enc], tp=1, sp=1, cfg=2)
    assert (enc.parallel_folding, enc.parallel_folding_mode) == (True, "world")


def test_generic_text_encoder_indivisible_skipped():
    # replica=3 divides neither 16 heads nor 64 intermediate -> skip.
    enc = _generic_text(heads=16, inter=64)
    _run([enc], tp=1, sp=1, cfg=3)
    assert enc.parallel_folding is False


def test_unknown_dims_encoder_not_folded():
    # A bare encoder whose dims we cannot introspect is left unfolded (safe):
    # TextEncoderArchConfig defaults num_attention_heads=0 and has no
    # intermediate_size, so both dims read as unknown.
    enc = TextEncoderConfig()
    _run([enc], tp=1, sp=1, cfg=2)
    assert enc.parallel_folding is False


def test_mixed_encoders_decided_independently():
    # Two encoders in one pipeline: the divisible one folds, the other is
    # skipped — each decided on its own dims.
    ok = _generic_text(heads=16, inter=64)
    bad = _generic_text(heads=10, inter=30)  # 10 % 4 != 0
    _run([ok, bad], tp=1, sp=1, cfg=4)
    assert ok.parallel_folding is True
    assert bad.parallel_folding is False


def test_image_encoder_configs_also_folded():
    # Image encoders fold when a pipeline declares them (config fields are on
    # the shared base). Here an image encoder is passed via image_encoder_configs.
    img = ImageEncoderConfig()
    img.num_attention_heads = 16
    img.intermediate_size = 64
    _run([_t5()], tp=1, sp=1, cfg=2, image=[img])
    assert (img.parallel_folding, img.parallel_folding_mode) == (True, "world")


# --- config hoist: fields live on the shared base --------------------------


def test_folding_fields_on_base_encoder_config():
    # A bare EncoderConfig / ImageEncoderConfig must carry the folding fields
    # (previously only TextEncoderConfig did -> AttributeError for others).
    assert EncoderConfig().parallel_folding is False
    assert EncoderConfig().parallel_folding_mode == "sp"
    assert ImageEncoderConfig().parallel_folding is False
