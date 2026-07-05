"""Unit test for the encoder parallel-folding decision.

Verifies `ServerArgs.adjust_pipeline_config` decides *when* to TP-shard an
encoder across the whole DiT replica vs leaving it alone, for every tp/sp/cfg
combination. Two config fields, distinct roles:
  - `parallel_folding` (bool, static): per-encoder opt-in. Only encoders measured
    to benefit (T5-XXL family) turn it on; folding a small encoder (CLIP-L, Qwen3
    at bench batch) is a net loss because per-layer all_reduce dominates.
  - `parallel_folding_mode` (str | None, runtime): the group actually folded over,
    set here; stays None when the encoder is not folded this run.
Also covers the divisibility guard and the config defaults. Pure logic, no GPU.
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
    enc = T5Config()  # T5Config opts in (parallel_folding=True) by default
    enc.num_heads = heads
    enc.d_ff = d_ff
    enc.parallel_folding_mode = None  # runtime result, filled by adjust
    return enc


def _generic_text(heads=16, inter=64, opt_in=False):
    """A non-T5 text encoder that spells its dims num_attention_heads /
    intermediate_size (CLIP/Llama/Qwen/Gemma family). Does not opt in by default."""
    enc = TextEncoderConfig()
    enc.num_attention_heads = heads
    enc.intermediate_size = inter
    enc.parallel_folding = opt_in
    return enc


def _decide(tp, sp, cfg, dp=1, disagg=False, num_gpus=None, heads=16, d_ff=64):
    """Return the runtime fold group for an opted-in T5 (None = not folded)."""
    enc = _t5(heads=heads, d_ff=d_ff)
    _run([enc], tp, sp, cfg, dp=dp, disagg=disagg, num_gpus=num_gpus)
    return enc.parallel_folding_mode


# --- opted-in (T5) decision matrix -----------------------------------------


def test_pure_tp_not_folded():
    # replica == tp: encoder already uses every replica GPU; leave untouched.
    assert _decide(tp=2, sp=1, cfg=1) is None


def test_single_gpu_not_folded():
    assert _decide(tp=1, sp=1, cfg=1) is None


def test_cfg_parallel_folds_world():
    # tp=1, cfg=2: was single-GPU encoder; now folds across the 2-GPU replica.
    assert _decide(tp=1, sp=1, cfg=2) == "world"


def test_sp_dp1_folds_world():
    # pure SP on a single replica: replica == sp > tp, uses the world group.
    assert _decide(tp=1, sp=2, cfg=1) == "world"


def test_tp_times_cfg_folds_world():
    # tp=2 x cfg=2 = 4-GPU replica: encoder was on tp_group(2); now spans 4.
    assert _decide(tp=2, sp=1, cfg=2, heads=16, d_ff=64) == "world"


def test_sp_times_cfg_folds_world():
    assert _decide(tp=1, sp=2, cfg=2) == "world"


def test_dp_gt_1_keeps_sp_folding():
    # dp>1: world group spans replicas, so keep the existing per-SP folding.
    assert _decide(tp=1, sp=2, cfg=1, dp=2) == "sp"
    # dp>1 pure cfg has no SP folding to fall back to -> not folded here.
    assert _decide(tp=1, sp=1, cfg=2, dp=2) is None


def test_disagg_not_world_folded():
    assert _decide(tp=1, sp=2, cfg=2, disagg=True) == "sp"


def test_num_gpus_is_authoritative_for_replica():
    # Replica size comes from num_gpus // dp, so any parallelism reflected in
    # the GPU count is covered without enumerating it.
    assert _decide(tp=2, sp=1, cfg=1, num_gpus=8, heads=16, d_ff=64) == "world"


def test_indivisible_replica_not_folded():
    # replica does not divide the encoder heads/d_ff -> cannot shard -> skip.
    assert _decide(tp=1, sp=1, cfg=3, heads=16, d_ff=64) is None


# --- opt-in gate: only encoders that opt in fold ---------------------------


def test_non_opted_in_encoder_skipped_even_if_divisible():
    # A divisible but not-opted-in encoder (CLIP-L / Qwen3: folding measured a
    # net loss) is left replicated even when a fold group exists.
    enc = _generic_text(heads=16, inter=64, opt_in=False)
    _run([enc], tp=1, sp=1, cfg=2)
    assert enc.parallel_folding_mode is None


def test_opted_in_generic_encoder_folds():
    # The mechanism is generic: any encoder that opts in folds, not just T5 —
    # so a future wide encoder can enable it without new plumbing.
    enc = _generic_text(heads=16, inter=64, opt_in=True)
    _run([enc], tp=1, sp=1, cfg=2)
    assert enc.parallel_folding_mode == "world"


def test_opted_in_generic_encoder_indivisible_skipped():
    # opted in but replica=3 divides neither 16 heads nor 64 intermediate -> skip.
    enc = _generic_text(heads=16, inter=64, opt_in=True)
    _run([enc], tp=1, sp=1, cfg=3)
    assert enc.parallel_folding_mode is None


def test_mixed_encoders_only_opted_in_and_divisible_folds():
    # In one pipeline: only the encoder that both opts in AND divides folds.
    ok = _generic_text(heads=16, inter=64, opt_in=True)
    not_opted = _generic_text(heads=16, inter=64, opt_in=False)
    indivisible = _generic_text(heads=10, inter=30, opt_in=True)  # 10 % 4 != 0
    _run([ok, not_opted, indivisible], tp=1, sp=1, cfg=4)
    assert ok.parallel_folding_mode == "world"
    assert not_opted.parallel_folding_mode is None
    assert indivisible.parallel_folding_mode is None


def test_opted_in_image_encoder_folds():
    # Image encoders fold too when opted in (config fields are on the shared base).
    img = ImageEncoderConfig()
    img.num_attention_heads = 16
    img.intermediate_size = 64
    img.parallel_folding = True
    _run([_t5()], tp=1, sp=1, cfg=2, image=[img])
    assert img.parallel_folding_mode == "world"


# --- config defaults -------------------------------------------------------


def test_folding_fields_on_base_encoder_config():
    # A bare EncoderConfig / ImageEncoderConfig must carry the folding fields
    # (previously only TextEncoderConfig did -> AttributeError for others).
    assert EncoderConfig().parallel_folding is False
    assert EncoderConfig().parallel_folding_mode is None
    assert ImageEncoderConfig().parallel_folding_mode is None


def test_opt_in_defaults_off_except_t5():
    # Conservative default: encoders do not fold unless measured beneficial.
    assert EncoderConfig().parallel_folding is False
    assert TextEncoderConfig().parallel_folding is False
    assert ImageEncoderConfig().parallel_folding is False
    # T5 family opts in.
    assert T5Config().parallel_folding is True
