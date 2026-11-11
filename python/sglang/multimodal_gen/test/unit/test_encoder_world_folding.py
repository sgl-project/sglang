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
from sglang.multimodal_gen.configs.pipeline_configs import PipelineConfig
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
    image=None,
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
        pipeline_config=PipelineConfig(
            text_encoder_configs=tuple(encoders),
            image_encoder_config=(image if image is not None else ImageEncoderConfig()),
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
    # replica == tp under auto: each rank keeps a full encoder replica; auto
    # never proposes folding here (the trade-off stays opt-in, see below).
    assert _proposed_mode(tp=2, sp=1, cfg=1) is None


def test_explicit_fold_proposes_replica_for_any_shape():
    # Encoder layout is independent of the DiT's parallelism: an operator's
    # explicit fold is honored on every multi-rank replica instead of being
    # silently ignored (pure TP) or narrowed to the SP group (dp > 1).
    assert _proposed_mode(tp=2, sp=1, cfg=1, policy="fold") == "replica"
    assert _proposed_mode(tp=1, sp=2, cfg=1, policy="fold") == "replica"
    assert (
        _proposed_mode(tp=2, sp=2, cfg=1, dp=2, num_gpus=8, policy="fold") == "replica"
    )
    # single-GPU replica: nothing to shard over
    assert _proposed_mode(tp=1, sp=1, cfg=1, policy="fold") is None
    assert _proposed_mode(tp=1, sp=1, cfg=1, dp=2, num_gpus=2, policy="fold") is None


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
    _run([t5, clip], tp=1, sp=2, cfg=1, image=img)
    assert t5.parallel_folding_mode == "world"
    assert clip.parallel_folding_mode == "world"
    assert img.parallel_folding_mode == "world"


def test_image_encoder_gets_each_policy_proposal():
    expected_modes = {"auto": "world", "fold": "replica", "replicate": "world"}
    for policy, expected_mode in expected_modes.items():
        image = ImageEncoderConfig()
        _run([], tp=1, sp=2, cfg=1, image=image, policy=policy)
        assert image.parallel_folding_mode == expected_mode


def test_adjust_proposal_policy_dependence():
    # adjust reads the parallelism only for auto/dp/replicate; finalize owns
    # those policy decisions. An explicit fold is the one exception: it widens
    # the proposal to the whole replica, because finalize cannot conjure a
    # group that was never proposed (pure-TP replicas had none at all).
    for policy in ("auto", "dp", "replicate"):
        assert _proposed_mode(tp=1, sp=2, cfg=1, policy=policy) == "world", policy
    assert _proposed_mode(tp=1, sp=2, cfg=1, policy="fold") == "replica"


def test_no_policy_touches_batching_max_size():
    # An encoder flag must not switch DiT batching on: that changes the denoise
    # batch shape, and with it the output, for every serve deployment. dp simply
    # stays inactive until the operator raises the ceiling themselves.
    for policy in ("auto", "fold", "dp", "replicate"):
        sa = _run([T5Config()], tp=1, sp=2, cfg=1, policy=policy)
        assert sa.batching_max_size == 1, policy


def test_explicit_batching_max_size_is_preserved():
    sa = _run([T5Config()], tp=1, sp=2, cfg=1, policy="dp", batching_max_size=8)
    assert sa.batching_max_size == 8


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


def test_image_encoder_fold_requires_divisible_dims(monkeypatch):
    monkeypatch.setattr(
        _base_mod,
        "get_folding_tp_group",
        lambda config: SimpleNamespace(world_size=4),
    )
    for heads, expected_mode in ((64, "world"), (6, None)):
        image = ImageEncoderConfig()
        image.hidden_size = 4096
        image.num_attention_heads = heads
        image.intermediate_size = 10240
        image.parallel_folding_mode = "world"
        finalize_encoder_folding(image, "fold")
        assert image.parallel_folding_mode == expected_mode


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
    # dp pays above the latency-bound width, with a batch, on a measured topology
    wide = _enc(4096, 64, 10240)
    assert encoder_dp_worthwhile(wide, 2, True) is True
    assert encoder_dp_worthwhile(_enc(2560, 32, 9728), 4, True) is True
    assert encoder_dp_worthwhile(_enc(768, 12, 3072), 8, True) is False  # CLIP-L
    assert encoder_dp_worthwhile(wide, 1, True) is False  # unbatched
    assert encoder_dp_worthwhile(wide, 4, False) is False  # no peer-to-peer
    assert encoder_dp_worthwhile(TextEncoderConfig(), 4, True) is False


# --- stage 3: finalize dispatches on the encoder_parallel policy --------------


def _finalize(
    monkeypatch,
    hidden,
    heads,
    inter,
    policy,
    mode="world",
    group_size=2,
    prefer_dp=False,
    measured=True,
):
    monkeypatch.setattr(
        _base_mod,
        "get_folding_tp_group",
        lambda config: SimpleNamespace(world_size=group_size),
    )
    monkeypatch.setattr(
        _base_mod, "group_has_measured_topology", lambda group: measured
    )
    enc = _enc(hidden, heads, inter)
    enc.parallel_folding_mode = mode
    finalize_encoder_folding(enc, policy, prefer_dp=prefer_dp)
    return enc.parallel_folding_mode


def test_finalize_dp_replicate_never_fold(monkeypatch):
    # policy alone clears the proposed fold, even for a huge encoder.
    assert _finalize(monkeypatch, 5120, 32, 32768, "dp") is None
    assert _finalize(monkeypatch, 5120, 32, 32768, "replicate") is None


def test_finalize_auto_keeps_wide_clears_narrow(monkeypatch):
    assert _finalize(monkeypatch, 4096, 64, 10240, "auto") == "world"
    assert _finalize(monkeypatch, 2560, 32, 9728, "auto") is None  # below threshold


def test_finalize_auto_leaves_dp_capable_unsharded_when_dp_preferred(monkeypatch):
    # with a batch, dp (one all_gather) beats folding (an all_reduce per layer)
    assert _finalize(monkeypatch, 4096, 64, 10240, "auto", prefer_dp=True) is None
    # TP or DiT-DP makes encoder DP ineligible, so keep the useful fold
    assert _finalize(monkeypatch, 4096, 64, 10240, "auto") == "world"
    # CLIP-L cannot dp either, so folding remains the only question
    assert _finalize(monkeypatch, 768, 12, 3072, "auto", prefer_dp=True) is None


def test_finalize_auto_needs_a_measured_topology(monkeypatch):
    assert _finalize(monkeypatch, 4096, 64, 10240, "auto", measured=False) is None
    # explicit fold is the operator's call, topology included
    assert _finalize(monkeypatch, 4096, 64, 10240, "fold", measured=False) == "world"


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
