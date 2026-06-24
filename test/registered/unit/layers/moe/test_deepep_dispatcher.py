import pytest
import torch

import sglang.srt.layers.moe.token_dispatcher.deepep as deepep
from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_dispatcher(monkeypatch, configured_capacity, attn_tp_size):
    monkeypatch.setattr(deepep, "use_deepep", True)
    monkeypatch.setattr(deepep, "get_attention_tp_size", lambda: attn_tp_size)
    monkeypatch.setattr(
        deepep._DeepEPDispatcherImplBase,
        "set_deepep_dispatcher_dtype",
        lambda self: None,
    )

    with envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(
        configured_capacity
    ):
        return deepep._DeepEPDispatcherImplBase(
            group=None,
            router_topk=8,
            permute_fusion=False,
            num_experts=16,
            num_local_experts=2,
            hidden_size=1024,
            params_dtype=torch.bfloat16,
            deepep_mode=DeepEPMode.AUTO,
        )


@pytest.mark.parametrize(
    ("configured_capacity", "attn_tp_size", "expected_capacity"),
    [
        (128, 1, 128),
        (128, 4, 32),
        (129, 4, 33),
    ],
)
def test_ll_capacity_uses_attention_tp_shard(
    monkeypatch,
    configured_capacity,
    attn_tp_size,
    expected_capacity,
):
    dispatcher = _make_dispatcher(
        monkeypatch,
        configured_capacity=configured_capacity,
        attn_tp_size=attn_tp_size,
    )

    assert dispatcher.num_max_dispatch_tokens_per_rank == expected_capacity


def test_ll_capacity_limit_applies_after_scatter(monkeypatch):
    dispatcher = _make_dispatcher(
        monkeypatch,
        configured_capacity=4096,
        attn_tp_size=4,
    )
    assert dispatcher.num_max_dispatch_tokens_per_rank == 1024

    with pytest.raises(AssertionError):
        _make_dispatcher(
            monkeypatch,
            configured_capacity=4097,
            attn_tp_size=4,
        )
