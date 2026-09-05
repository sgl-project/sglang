"""Unit test for the batched text-encode data-parallel gate.

_text_encode_dp_group decides whether a batched encode may split across
encoder copies. It composes with DiT TP through the orthogonal encoder-DP
group and stays off for folded encoders, single-copy replicas, and non-DP
policies. Pure logic, no GPU / distributed init.
"""

from types import SimpleNamespace
from unittest.mock import Mock

from sglang.multimodal_gen.configs.models.encoders import TextEncoderConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    _get_encoder_data_parallel_group_ranks,
)
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder
from sglang.multimodal_gen.runtime.pipelines_core.stages import text_encoding as _te_mod
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.distributed import RankGenerator


def _enc(hidden=4096, heads=64, inter=10240, folding_mode=None):
    enc = TextEncoderConfig()
    enc.hidden_size = hidden
    enc.num_attention_heads = heads
    enc.intermediate_size = inter
    enc.parallel_folding_mode = folding_mode
    return enc


def _gate(
    monkeypatch,
    policy="dp",
    tp=1,
    dp=1,
    folding_mode=None,
    batch_size=4,
    world_size=4,
    group_available=True,
    measured=True,
    supports_dp=True,
):
    group = SimpleNamespace(world_size=world_size) if group_available else None
    monkeypatch.setattr(_te_mod, "get_encoder_data_parallel_group", lambda: group)
    monkeypatch.setattr(_te_mod, "group_has_measured_topology", lambda group: measured)
    encoder = Mock(spec=TextEncoder)
    encoder.supports_dp_encode = supports_dp
    server_args = SimpleNamespace(encoder_parallel=policy, tp_size=tp, dp_size=dp)
    self = SimpleNamespace(_log_dp_choice=lambda batch, world: None)
    return _te_mod.TextEncodingStage._text_encode_dp_group(
        self, server_args, _enc(folding_mode=folding_mode), batch_size, encoder
    )


def test_dp_engages_for_replicated_encoder(monkeypatch):
    assert _gate(monkeypatch) is not None


def test_dit_tp_composes_with_encoder_dp_group(monkeypatch):
    # TP ranks jointly run one encoder copy; the orthogonal group distributes
    # batch slices across those TP-sharded copies.
    assert _gate(monkeypatch, tp=4) is not None
    assert _gate(monkeypatch, policy="auto", tp=4) is not None


def test_pure_tp_has_no_independent_encoder_copy(monkeypatch):
    # A pure-TP replica already uses every GPU for one encoder copy.
    assert _gate(monkeypatch, tp=4, group_available=False) is None


def test_folded_encoder_blocks_dp(monkeypatch):
    # folding shards the weights over the folding group; a rank cannot
    # encode alone, whatever the policy says
    assert _gate(monkeypatch, folding_mode="world") is None
    assert _gate(monkeypatch, tp=4, folding_mode="world") is None


def test_multi_replica_uses_per_replica_encoder_group(monkeypatch):
    # dp>1 no longer blocks: the encoder-DP group stays within one pipeline
    # replica. A replica with one encoder copy uses its normal TP forward.
    assert _gate(monkeypatch, dp=2) is not None
    assert _gate(monkeypatch, dp=2, world_size=1) is None


def test_policy_and_encoder_gates(monkeypatch):
    assert _gate(monkeypatch, policy="replicate") is None
    assert _gate(monkeypatch, policy="fold") is None
    assert _gate(monkeypatch, supports_dp=False) is None
    assert _gate(monkeypatch, world_size=1) is None
    assert _gate(monkeypatch, batch_size=1) is None  # unbatched: not worthwhile
    # explicit dp trusts the operator on an unmeasured topology; auto does not
    assert _gate(monkeypatch, policy="dp", measured=False) is not None
    assert _gate(monkeypatch, policy="auto", measured=False) is None


def _validate_batching(encoder_parallel, tp, dp):
    args = SimpleNamespace(
        batching_mode="dynamic",
        batching_max_size=1,
        batching_delay_ms=0,
        encoder_parallel=encoder_parallel,
        tp_size=tp,
        dp_size=dp,
    )
    ServerArgs._validate_batching(args)


def test_server_args_accepts_dp_for_any_parallel_shape():
    # Encoder DP composes with TP and stays within each pipeline replica.
    _validate_batching("dp", tp=4, dp=1)
    _validate_batching("dp", tp=1, dp=2)
    _validate_batching("dp", tp=4, dp=2)


def test_encoder_dp_groups_are_tp_orthogonal_and_replica_local():
    ranks = _get_encoder_data_parallel_group_ranks(
        RankGenerator(tp=2, sp=2, pp=1, cfg=1, dp=2, order="tp-sp-pp-cfg-dp")
    )
    assert ranks == [[0, 2], [1, 3], [4, 6], [5, 7]]
