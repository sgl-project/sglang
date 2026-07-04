"""Unit test for the T5 text-encoder parallel-folding decision.

Verifies `ServerArgs.adjust_pipeline_config` decides *when* to TP-shard the T5
encoder across the whole DiT replica (`mode=world`) vs leaving it alone, for
every tp/sp/cfg combination — the "encoder uses all idle replica GPUs during
encoding" rule — without degrading the pure-TP or dp>1 paths. Pure logic, no
GPU or distributed init.
"""

from types import SimpleNamespace

from sglang.multimodal_gen.configs.models.encoders.t5 import T5Config
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _decide(tp, sp, cfg, dp=1, disagg=False, num_gpus=None, heads=16, d_ff=64):
    enc = T5Config()
    enc.num_heads = heads
    enc.d_ff = d_ff
    enc.parallel_folding = False
    enc.parallel_folding_mode = "sp"
    self = SimpleNamespace(
        tp_size=tp,
        sp_degree=sp,
        cfg_parallel_degree=cfg,
        dp_size=dp,
        disagg_mode=disagg,
        num_gpus=num_gpus if num_gpus is not None else tp * sp * cfg * dp,
        pipeline_config=SimpleNamespace(text_encoder_configs=(enc,)),
    )
    ServerArgs.adjust_pipeline_config(self)
    return (enc.parallel_folding, enc.parallel_folding_mode)


def test_pure_tp_not_folded():
    # replica == tp: encoder already uses every replica GPU; leave untouched.
    assert _decide(tp=2, sp=1, cfg=1) == (False, "sp")


def test_single_gpu_not_folded():
    assert _decide(tp=1, sp=1, cfg=1) == (False, "sp")


def test_cfg_parallel_folds_world():
    # tp=1, cfg=2: was single-GPU encoder; now folds across the 2-GPU replica.
    assert _decide(tp=1, sp=1, cfg=2) == (True, "world")


def test_sp_dp1_folds_world():
    # pure SP on a single replica: replica == sp > tp, uses the world group
    # (same GPUs as the old sp group).
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
    # the GPU count (beyond the tp/sp/cfg named here — e.g. a future axis) is
    # covered without enumerating it: num_gpus=8, tp=2 -> replica 8 > tp -> fold.
    assert _decide(tp=2, sp=1, cfg=1, num_gpus=8, heads=16, d_ff=64) == (
        True,
        "world",
    )


def test_indivisible_replica_not_folded():
    # replica does not divide the encoder heads/d_ff -> cannot shard -> skip.
    assert _decide(tp=1, sp=1, cfg=3, heads=16, d_ff=64) == (False, "sp")
