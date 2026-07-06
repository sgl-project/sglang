"""Unit test for the encoder_parallel decision.

adjust_pipeline_config proposes a fold group from the parallelism alone;
finalize_encoder_folding resolves fold-vs-replicate per policy on real dims;
encoder_dp_worthwhile gates the runtime batch data-parallel. Pure logic, no
GPU / distributed init (the fold group is monkeypatched).
"""

from types import SimpleNamespace

from sglang.multimodal_gen.configs.models.encoders import (
    EncoderConfig,
    ImageEncoderConfig,
    TextEncoderConfig,
)
from sglang.multimodal_gen.configs.models.encoders.t5 import T5Config
from sglang.multimodal_gen.runtime.models.encoders import base as _base_mod
from sglang.multimodal_gen.runtime.models.encoders.base import (
    FOLD_MIN_HIDDEN_SIZE,
    _encoder_dims_divide,
    encoder_dp_worthwhile,
    encoder_folding_worthwhile,
    finalize_encoder_folding,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _run(
    encoders,
    tp,
    sp,
    cfg,
    dp=1,
    disagg=False,
    num_gpus=None,
    image=(),
    policy="auto",
    batching_max_size=1,
    explicit=(),
):
    self = SimpleNamespace(
        tp_size=tp,
        sp_degree=sp,
        cfg_parallel_degree=cfg,
        dp_size=dp,
        disagg_mode=disagg,
        encoder_parallel=policy,
        batching_max_size=batching_max_size,
        is_arg_explicitly_set=lambda name: name in explicit,
        num_gpus=num_gpus if num_gpus is not None else tp * sp * cfg * dp,
        pipeline_config=SimpleNamespace(
            text_encoder_configs=tuple(encoders),
            image_encoder_configs=tuple(image),
        ),
    )
    ServerArgs.adjust_pipeline_config(self)
    return self


def _proposed_mode(tp, sp, cfg, dp=1, disagg=False, num_gpus=None, policy="auto"):
    enc = T5Config()
    enc.parallel_folding_mode = None
    _run([enc], tp, sp, cfg, dp=dp, disagg=disagg, num_gpus=num_gpus, policy=policy)
    return enc.parallel_folding_mode


# --- stage 1: adjust proposes a fold group from the parallelism --------------


def test_pure_tp_not_folded():
    # replica == tp: encoder already uses every replica GPU; nothing to fold.
    assert _proposed_mode(tp=2, sp=1, cfg=1) is None


def test_single_gpu_not_folded():
    assert _proposed_mode(tp=1, sp=1, cfg=1) is None


def test_cfg_parallel_proposes_world():
    assert _proposed_mode(tp=1, sp=1, cfg=2) == "world"


def test_sp_dp1_proposes_world():
    assert _proposed_mode(tp=1, sp=2, cfg=1) == "world"


def test_tp_times_cfg_proposes_world():
    assert _proposed_mode(tp=2, sp=1, cfg=2) == "world"


def test_sp_times_cfg_proposes_world():
    assert _proposed_mode(tp=1, sp=2, cfg=2) == "world"


def test_dp_gt_1_keeps_sp():
    # dp>1: world group spans replicas, so fold over the per-replica SP group.
    assert _proposed_mode(tp=1, sp=2, cfg=1, dp=2) == "sp"
    # dp>1 pure cfg has no SP group to fall back to -> nothing proposed.
    assert _proposed_mode(tp=1, sp=1, cfg=2, dp=2) is None


def test_disagg_keeps_sp():
    assert _proposed_mode(tp=1, sp=2, cfg=2, disagg=True) == "sp"


def test_num_gpus_is_authoritative_for_replica():
    assert _proposed_mode(tp=2, sp=1, cfg=1, num_gpus=8) == "world"


def test_all_encoders_get_the_same_proposed_mode():
    # adjust no longer size-gates: every encoder in the pipeline (text + image)
    # gets the proposed group; the loader trims it later by real size.
    t5 = T5Config()
    clip = TextEncoderConfig()
    img = ImageEncoderConfig()
    for e in (t5, clip, img):
        e.parallel_folding_mode = None
    _run([t5, clip], tp=1, sp=2, cfg=1, image=[img])
    assert t5.parallel_folding_mode == "world"
    assert clip.parallel_folding_mode == "world"
    assert img.parallel_folding_mode == "world"


def test_dp_replicate_policy_proposes_nothing():
    # dp/replicate never fold: adjust short-circuits so encoders stay replicated.
    assert _proposed_mode(tp=1, sp=2, cfg=1, policy="dp") is None
    assert _proposed_mode(tp=1, sp=2, cfg=1, policy="replicate") is None


def test_fold_and_auto_policy_still_propose():
    assert _proposed_mode(tp=1, sp=2, cfg=1, policy="fold") == "world"
    assert _proposed_mode(tp=1, sp=2, cfg=1, policy="auto") == "world"


def test_dp_policy_raises_default_batching_max_size():
    # dp needs batch>1 to ever engage; raise the ceiling to the replica size
    # so choosing dp isn't a silent no-op.
    sa = _run([T5Config()], tp=1, sp=2, cfg=1, policy="dp")
    assert sa.batching_max_size == 2


def test_dp_policy_keeps_explicit_batching_max_size():
    sa = _run(
        [T5Config()],
        tp=1,
        sp=2,
        cfg=1,
        policy="dp",
        batching_max_size=1,
        explicit=("batching_max_size",),
    )
    assert sa.batching_max_size == 1


def test_dp_policy_never_lowers_batching_max_size():
    sa = _run([T5Config()], tp=1, sp=2, cfg=1, policy="dp", batching_max_size=8)
    assert sa.batching_max_size == 8


def test_non_dp_policy_does_not_touch_batching_max_size():
    for policy in ("auto", "fold", "replicate"):
        sa = _run([T5Config()], tp=1, sp=2, cfg=1, policy=policy)
        assert sa.batching_max_size == 1


def test_dp_policy_single_replica_does_not_raise_batching_max_size():
    # replica_size == 1: dp has nothing to shard across, nothing to raise.
    sa = _run([T5Config()], tp=1, sp=1, cfg=1, policy="dp")
    assert sa.batching_max_size == 1


# --- stage 2: size + divisibility gate (loader, on real dims) ----------------


def _enc(hidden, heads, inter):
    enc = TextEncoderConfig()
    enc.hidden_size = hidden
    enc.num_attention_heads = heads
    enc.intermediate_size = inter
    return enc


def test_wide_encoder_worth_folding():
    # T5-XXL / Mistral-24B class: hidden >= threshold and dims divide the group.
    assert encoder_folding_worthwhile(_enc(4096, 64, 10240), group_size=2) is True
    assert encoder_folding_worthwhile(_enc(5120, 32, 32768), group_size=2) is True


def test_narrow_encoder_not_worth_folding():
    # Qwen3 (2560) measured a net loss -> below the bar.
    assert encoder_folding_worthwhile(_enc(2560, 32, 9728), group_size=2) is False


def test_tiny_encoder_not_worth_folding():
    # CLIP-L (512): far too small.
    assert encoder_folding_worthwhile(_enc(512, 8, 2048), group_size=2) is False


def test_indivisible_dims_not_folded():
    # wide enough but heads/intermediate do not divide the group -> cannot shard.
    assert encoder_folding_worthwhile(_enc(4096, 6, 10240), group_size=4) is False
    assert encoder_folding_worthwhile(_enc(4096, 64, 10250), group_size=4) is False


def test_group_size_one_not_folded():
    assert encoder_folding_worthwhile(_enc(4096, 64, 10240), group_size=1) is False


def test_unknown_dims_not_folded():
    # a bare encoder whose dims we cannot introspect is left replicated (safe).
    assert encoder_folding_worthwhile(TextEncoderConfig(), group_size=2) is False


def test_threshold_is_the_boundary():
    assert encoder_folding_worthwhile(_enc(FOLD_MIN_HIDDEN_SIZE, 8, 8192), 2) is True
    assert (
        encoder_folding_worthwhile(_enc(FOLD_MIN_HIDDEN_SIZE - 128, 8, 8192), 2)
        is False
    )


def test_dims_divide():
    # divisibility only (size-agnostic): the hard constraint to shard at all.
    assert _encoder_dims_divide(_enc(2560, 32, 9728), 2) is True
    assert _encoder_dims_divide(_enc(4096, 6, 10240), 4) is False  # heads
    assert _encoder_dims_divide(_enc(4096, 64, 10250), 4) is False  # intermediate
    assert _encoder_dims_divide(_enc(4096, 64, 10240), 1) is False  # group of 1


def test_dp_worthwhile():
    # batch data-parallel pays only above the latency-bound width and batch>1.
    assert encoder_dp_worthwhile(_enc(4096, 64, 10240), batch_size=2) is True
    assert encoder_dp_worthwhile(_enc(2560, 32, 9728), batch_size=4) is True
    assert encoder_dp_worthwhile(_enc(768, 12, 3072), batch_size=8) is False  # CLIP-L
    assert encoder_dp_worthwhile(_enc(4096, 64, 10240), batch_size=1) is False
    assert encoder_dp_worthwhile(TextEncoderConfig(), batch_size=4) is False


# --- stage 3: finalize dispatches on the encoder_parallel policy --------------


def _finalize(monkeypatch, hidden, heads, inter, policy, mode="world", group_size=2):
    monkeypatch.setattr(
        _base_mod,
        "get_folding_tp_group",
        lambda config: SimpleNamespace(world_size=group_size),
    )
    enc = _enc(hidden, heads, inter)
    enc.parallel_folding_mode = mode
    finalize_encoder_folding(enc, policy)
    return enc.parallel_folding_mode


def test_finalize_dp_replicate_never_fold(monkeypatch):
    # policy alone clears the proposed fold, even for a huge encoder.
    assert _finalize(monkeypatch, 5120, 32, 32768, "dp") is None
    assert _finalize(monkeypatch, 5120, 32, 32768, "replicate") is None


def test_finalize_auto_keeps_wide_clears_narrow(monkeypatch):
    assert _finalize(monkeypatch, 4096, 64, 10240, "auto") == "world"
    assert _finalize(monkeypatch, 2560, 32, 9728, "auto") is None  # below threshold


def test_finalize_fold_ignores_size_but_needs_divisible(monkeypatch):
    # "fold" folds a narrow encoder that "auto" would reject...
    assert _finalize(monkeypatch, 2560, 32, 9728, "fold") == "world"
    # ...but it still must divide the group.
    assert _finalize(monkeypatch, 2560, 6, 9728, "fold", group_size=4) is None


def test_finalize_mode_none_is_noop(monkeypatch):
    # nothing proposed -> stays replicated regardless of policy.
    assert _finalize(monkeypatch, 5120, 32, 32768, "auto", mode=None) is None


# --- config defaults ---------------------------------------------------------


def test_parallel_folding_mode_defaults_none():
    assert EncoderConfig().parallel_folding_mode is None
    assert TextEncoderConfig().parallel_folding_mode is None
    assert ImageEncoderConfig().parallel_folding_mode is None
    assert T5Config().parallel_folding_mode is None
