"""Unit test for the encoder parallel-folding decision (two stages).

Stage 1 - ServerArgs.adjust_pipeline_config proposes a fold group from the
parallelism alone (mode = "world"/"sp"/None), the same for every encoder.

Stage 2 - encoder_folding_worthwhile (applied by the loader once real dims are
known) keeps the fold only for encoders wide enough to benefit and whose heads
and MLP divide the group. Being size-based (not per-architecture) it handles the
same encoder family at different parameter counts.

Pure logic, no GPU / distributed init.
"""

from types import SimpleNamespace

from sglang.multimodal_gen.configs.models.encoders import (
    EncoderConfig,
    ImageEncoderConfig,
    TextEncoderConfig,
)
from sglang.multimodal_gen.configs.models.encoders.t5 import T5Config
from sglang.multimodal_gen.runtime.models.encoders.base import (
    FOLD_MIN_HIDDEN_SIZE,
    encoder_folding_worthwhile,
)
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


def _proposed_mode(tp, sp, cfg, dp=1, disagg=False, num_gpus=None):
    enc = T5Config()
    enc.parallel_folding_mode = None
    _run([enc], tp, sp, cfg, dp=dp, disagg=disagg, num_gpus=num_gpus)
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


# --- config defaults ---------------------------------------------------------


def test_parallel_folding_mode_defaults_none():
    assert EncoderConfig().parallel_folding_mode is None
    assert TextEncoderConfig().parallel_folding_mode is None
    assert ImageEncoderConfig().parallel_folding_mode is None
    assert T5Config().parallel_folding_mode is None
