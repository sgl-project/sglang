"""Unit test for the batched text-encode data-parallel gate.

_text_encode_dp_group decides whether a batched encode may split across
ranks. It must engage for any replicated encoder regardless of the DiT's
TP layout (encoders are only sharded through folding), and stay off for
folded encoders, multi-replica servers, and non-dp policies. Pure logic,
no GPU / distributed init (the world group is monkeypatched).
"""

from types import SimpleNamespace
from unittest.mock import Mock

from sglang.multimodal_gen.configs.models.encoders import TextEncoderConfig
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder
from sglang.multimodal_gen.runtime.pipelines_core.stages import text_encoding as _te_mod
from sglang.multimodal_gen.runtime.server_args import ServerArgs


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
    measured=True,
    supports_dp=True,
):
    monkeypatch.setattr(
        _te_mod, "get_replica_group", lambda: SimpleNamespace(world_size=world_size)
    )
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


def test_dit_tp_does_not_block_dp(monkeypatch):
    # pure-TP DiT deployments never fold the encoder (replica == tp), so each
    # rank holds a full replica and can encode a batch slice alone
    assert _gate(monkeypatch, tp=4) is not None
    assert _gate(monkeypatch, policy="auto", tp=4) is not None


def test_folded_encoder_blocks_dp(monkeypatch):
    # folding shards the weights over the folding group; a rank cannot
    # encode alone, whatever the policy says
    assert _gate(monkeypatch, folding_mode="world") is None
    assert _gate(monkeypatch, tp=4, folding_mode="world") is None


def test_multi_replica_uses_replica_group(monkeypatch):
    # dp>1 no longer blocks: the gate operates on the replica group, which
    # only spans the ranks sharing this batch. A single-GPU replica still
    # falls back to the replicated forward via the world_size gate.
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
    # encoder parallelism is independent of the DiT layout: no combination
    # of tp/dp is rejected at validation time
    _validate_batching("dp", tp=4, dp=1)
    _validate_batching("dp", tp=1, dp=2)
    _validate_batching("dp", tp=4, dp=2)
