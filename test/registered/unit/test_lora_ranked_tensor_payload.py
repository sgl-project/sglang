import pytest
import torch

from sglang.srt.managers.tp_worker import (
    _filter_lora_experts_for_ep_rank,
    _select_ranked_lora_payload,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_ranked_lora_payload_selects_the_local_tp_rank():
    """The trainer now gathers one CUDA IPC payload per engine rank and sends
    the whole list, so each TP rank must pick its own entry -- taking another
    rank's handle imports the wrong shard. A plain string is still accepted for
    single-payload senders.
    """
    payloads = ["rank0", "rank1", "rank2", "rank3"]
    assert _select_ranked_lora_payload(payloads, tp_rank=2, tp_size=4) == "rank2"
    assert _select_ranked_lora_payload("payload", tp_rank=2, tp_size=4) == "payload"

    with pytest.raises(ValueError, match="2 payloads for TP size 4"):
        _select_ranked_lora_payload(["rank0", "rank1"], tp_rank=0, tp_size=4)


def test_ranked_lora_payload_keeps_only_local_ep_experts():
    """Under EP the adapter stream carries every expert, but a rank may only
    install its own slice: 896 experts over 32 ranks puts experts 0..27 on rank
    0 and 28.. elsewhere. Non-expert tensors and shared-outer expert tensors
    (no expert index in the name) must pass through untouched.
    """
    tensors = {
        "model.layers.1.mlp.experts.gate_proj.lora_A.weight": torch.ones(1),
        "model.layers.1.mlp.experts.0.gate_proj.lora_B.weight": torch.ones(1),
        "model.layers.1.mlp.experts.27.gate_proj.lora_B.weight": torch.ones(1),
        "model.layers.1.mlp.experts.28.gate_proj.lora_B.weight": torch.ones(1),
        "model.layers.1.self_attn.q_proj.lora_A.weight": torch.ones(1),
    }

    filtered = _filter_lora_experts_for_ep_rank(
        tensors,
        num_experts=896,
        ep_rank=0,
        ep_size=32,
    )

    assert set(filtered) == {
        "model.layers.1.mlp.experts.gate_proj.lora_A.weight",
        "model.layers.1.mlp.experts.0.gate_proj.lora_B.weight",
        "model.layers.1.mlp.experts.27.gate_proj.lora_B.weight",
        "model.layers.1.self_attn.q_proj.lora_A.weight",
    }

    # An uneven split would silently drop the tail experts.
    with pytest.raises(AssertionError, match="even EP split"):
        _filter_lora_experts_for_ep_rank({}, num_experts=7, ep_rank=0, ep_size=2)
