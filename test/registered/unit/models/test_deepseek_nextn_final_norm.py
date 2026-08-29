from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.models import deepseek_nextn
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class _Decoder:
    def __init__(self, *, publish_marker, return_residual=True):
        self.publish_marker = publish_marker
        self.return_residual = return_residual

    def __call__(self, *args, **kwargs):
        output = torch.tensor([[1.0, 2.0]])
        if self.publish_marker:
            output._sglang_needs_allreduce_fusion = True
        residual = torch.tensor([[3.0, 4.0]]) if self.return_residual else None
        return output, residual, None


class _FinalNorm:
    def __init__(self):
        self.plain_calls = 0
        self.fused_groups = []

    def __call__(self, hidden_states, residual=None):
        self.plain_calls += 1
        if residual is None:
            return hidden_states
        return hidden_states + residual, residual

    def forward_with_allreduce_fusion(
        self, hidden_states, residual, *, use_attn_tp_group
    ):
        self.fused_groups.append(use_attn_tp_group)
        return hidden_states + residual, residual


class _TopKState:
    topk_indices = None
    should_publish = False

    def update(self, topk_indices):
        pass

    def publish(self):
        pass


def _run(*, publish_marker, return_residual=True):
    final_norm = _FinalNorm()
    model = SimpleNamespace(
        quant_config=None,
        rot_weight=None,
        enorm=lambda value: value,
        hnorm=lambda value: value,
        eh_proj=lambda value: value,
        decoder=_Decoder(
            publish_marker=publish_marker,
            return_residual=return_residual,
        ),
        shared_head=SimpleNamespace(norm=final_norm),
        dsa_enable_prefill_cp=False,
        mla_enable_prefill_cp=False,
    )
    forward_batch = SimpleNamespace(
        spec_info=SimpleNamespace(hidden_states=torch.tensor([[5.0, 6.0]])),
        forward_mode=SimpleNamespace(is_idle=lambda: False),
    )

    with (
        patch.object(deepseek_nextn, "_is_cuda", False),
        patch.object(deepseek_nextn, "_is_npu", False),
        patch.object(deepseek_nextn, "is_cp_v2_active", return_value=False),
        patch.object(deepseek_nextn, "dsa_use_prefill_cp", return_value=False),
        patch.object(deepseek_nextn, "mla_use_prefill_cp", return_value=False),
        patch.object(
            deepseek_nextn.IndexTopKShareState,
            "from_mtp_carry",
            return_value=_TopKState(),
        ),
        patch.object(
            deepseek_nextn,
            "get_global_expert_distribution_recorder",
            return_value=SimpleNamespace(disable_this_region=nullcontext),
        ),
    ):
        output = deepseek_nextn.DeepseekModelNextN.forward(
            model,
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0]),
            forward_batch=forward_batch,
            input_embeds=torch.tensor([[7.0, 8.0]]),
        )

    return output, final_norm


def test_owned_reduction_uses_moe_group_fused_final_norm():
    output, norm = _run(publish_marker=True)

    assert norm.plain_calls == 0
    assert norm.fused_groups == [False]
    torch.testing.assert_close(output, torch.tensor([[4.0, 6.0]]))


def test_unmarked_output_keeps_plain_final_norm():
    output, norm = _run(publish_marker=False)

    assert norm.plain_calls == 1
    assert norm.fused_groups == []
    torch.testing.assert_close(output, torch.tensor([[4.0, 6.0]]))


def test_owned_reduction_without_residual_fails_loudly():
    with pytest.raises(AssertionError):
        _run(publish_marker=True, return_residual=False)
