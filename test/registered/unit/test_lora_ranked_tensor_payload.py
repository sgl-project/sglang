import hashlib

import pytest
import torch

from sglang.srt.managers.tp_worker import (
    _filter_lora_experts_for_ep_rank,
    _validate_lora_tensor_checksums,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_ranked_lora_payload_keeps_only_local_ep_experts():
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


def test_ranked_lora_payload_rejects_uneven_ep_split():
    with pytest.raises(AssertionError, match="even EP split"):
        _filter_lora_experts_for_ep_rank(
            {},
            num_experts=7,
            ep_rank=0,
            ep_size=2,
        )


def _checksum(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def test_lora_tensor_checksum_validation():
    tensors = {"a.lora_A.weight": torch.arange(8, dtype=torch.bfloat16)}
    _validate_lora_tensor_checksums(
        tensors,
        {"a.lora_A.weight": _checksum(tensors["a.lora_A.weight"])},
        tp_rank=3,
    )


@pytest.mark.parametrize(
    ("expected", "match"),
    [
        ({"a.lora_A.weight": "bad"}, "value_diff"),
        ({"missing.lora_A.weight": "bad"}, "missing"),
        ({}, "extra"),
    ],
)
def test_lora_tensor_checksum_validation_rejects_mismatch(expected, match):
    tensors = {"a.lora_A.weight": torch.arange(8, dtype=torch.bfloat16)}
    with pytest.raises(RuntimeError, match=match):
        _validate_lora_tensor_checksums(tensors, expected, tp_rank=1)
