"""Incremental EP and PD coverage on top of upstream GLM interleave BCG."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.arg_groups import cp_tp_group_sharing_hook as sharing
from sglang.srt.layers.cp.bcg import interleave_bcg as interleave
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GLM = "Glm5NextForConditionalGeneration"
QWEN = "Qwen4ExpForConditionalGeneration"


def config(**updates):
    cfg = dict(
        enable_prefill_cp=True,
        tp_size=4,
        attn_cp_size=4,
        dp_size=1,
        pp_size=1,
        cp_strategy="interleave",
        speculative_algorithm=None,
        speculative_draft_model_path=None,
        speculative_eagle_topk=1,
        model_path="model",
        enable_dsa_cache_layer_split=False,
        dcp_size=1,
        enable_mixed_chunk=False,
        moe_dp_size=1,
        disaggregation_mode="null",
        disaggregation_transfer_backend="nixl",
        ep_size=4,
        moe_a2a_backend="none",
        enable_waterfill=False,
        cuda_graph_config=object(),
    )
    cfg.update(updates)
    return SimpleNamespace(**cfg)


@contextmanager
def policy(cfg, architecture=GLM):
    model = SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture]))
    with (
        patch.object(sharing, "resolving_view", return_value=cfg),
        patch.object(sharing, "resolved_view", return_value=cfg),
        patch.object(interleave, "resolving_view", return_value=cfg),
        patch.object(interleave, "resolved_view", return_value=cfg),
        patch.object(interleave, "model_config_of", return_value=model),
        patch.object(
            sharing,
            "mambaish_config",
            return_value=SimpleNamespace(
                linear_num_key_heads=16, linear_num_value_heads=64
            ),
        ),
        patch.object(
            sharing.envs.SGLANG_DISAGG_STAGING_BUFFER, "get", return_value=False
        ),
    ):
        yield model


@pytest.mark.parametrize("ep_size", [1, 2, 4])
@pytest.mark.parametrize("mode", ["null", "prefill"])
def test_ep_bcg_and_nixl_prefill_preserve_graph_config(ep_size, mode):
    cfg = config(ep_size=ep_size, disaggregation_mode=mode)
    with policy(cfg) as model, patch.object(sharing, "declare_resolution") as declare:
        assert interleave.InterleaveCPBCGInput.supports(cfg)
        sharing.resolve_cp_tp_group_sharing(cfg, model)
    assert declare.call_args.kwargs == {"enable_cp_tp_group_sharing": True}


@pytest.mark.parametrize("spec", ["EAGLE", "NEXTN", "DFLASH"])
def test_colocated_spec_policy_is_preserved(spec):
    cfg = config(speculative_algorithm=spec)
    with policy(cfg) as model, patch.object(sharing, "declare_resolution"):
        sharing.resolve_cp_tp_group_sharing(cfg, model)


@pytest.mark.parametrize(
    "updates",
    [
        {"disaggregation_mode": "decode"},
        {
            "disaggregation_mode": "prefill",
            "disaggregation_transfer_backend": "mooncake",
        },
        {"disaggregation_mode": "prefill", "speculative_algorithm": "EAGLE"},
        {"dp_size": 2},
        {"pp_size": 2},
        {"attn_cp_size": 2},
        {"enable_dsa_cache_layer_split": True},
    ],
)
def test_unqualified_glm_pd_topologies_are_rejected(updates):
    cfg = config(**updates)
    with policy(cfg) as model, pytest.raises(ValueError):
        sharing.resolve_cp_tp_group_sharing(cfg, model)


def test_staging_is_still_rejected():
    cfg = config(disaggregation_mode="prefill")
    with (
        policy(cfg) as model,
        patch.object(
            sharing.envs.SGLANG_DISAGG_STAGING_BUFFER, "get", return_value=True
        ),
        pytest.raises(ValueError, match="STAGING_BUFFER"),
    ):
        sharing.resolve_cp_tp_group_sharing(cfg, model)


def test_ep_all_to_all_is_not_silently_enabled_for_bcg():
    cfg = config(moe_a2a_backend="deepep")
    with policy(cfg):
        assert not interleave.InterleaveCPBCGInput.supports(cfg)


def test_qwen_pd_policy_remains_unchanged():
    cfg = config(
        cp_strategy="zigzag",
        ep_size=1,
        dp_size=2,
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="mooncake",
    )
    with (
        policy(cfg, QWEN) as model,
        patch.object(sharing, "declare_resolution") as declare,
        patch.object(sharing, "with_phase", return_value="qwen-prefill-disabled"),
    ):
        sharing.resolve_cp_tp_group_sharing(cfg, model)
    assert declare.call_args.kwargs["enable_cp_tp_group_sharing"]
    assert declare.call_args.kwargs["cuda_graph_config"] == "qwen-prefill-disabled"


def test_replay_keeps_upstream_exact_prefix_free_policy():
    inputs = interleave.InterleaveCPBCGInput(torch.empty(0), torch.empty(0), {16: 4})
    assert inputs.allows_replay(2, 16, [0, 0], False)
    assert not inputs.allows_replay(2, 15, [0, 0], False)
    assert not inputs.allows_replay(2, 16, [8, 0], False)
    assert not inputs.allows_replay(2, 16, [0, 0], True)


def test_ep_count_uses_persistent_storage_across_prepare_calls():
    parallel = SimpleNamespace(attn_cp_size=4, attn_cp_rank=0)
    runner = SimpleNamespace(
        max_num_tokens=32,
        device="cpu",
        model_runner=SimpleNamespace(
            dtype=torch.float32, model_config=SimpleNamespace(hidden_size=2)
        ),
    )
    with patch.object(interleave, "get_parallel", return_value=parallel):
        inputs = interleave.InterleaveCPBCGInput.create(runner)
    address = inputs.moe_local_token_count.data_ptr()
    for tokens in (16, 32, 16):
        batch = SimpleNamespace(
            positions=torch.arange(tokens),
            out_cache_loc=torch.arange(tokens),
            input_ids=torch.arange(tokens),
            attn_cp_metadata=SimpleNamespace(per_rank_actual_token=[tokens // 4] * 4),
        )
        with (
            patch.object(interleave, "get_parallel", return_value=parallel),
            patch.object(interleave.PrefillCPBCGInput, "prepare"),
            patch.object(
                interleave, "cp_interleave_input_ids", return_value=batch.input_ids
            ),
        ):
            inputs.prepare(runner, batch, static_num_tokens=tokens, capture=False)
        assert batch.attn_cp_metadata.moe_local_token_count.data_ptr() == address
        assert batch.attn_cp_metadata.moe_local_token_count.item() == tokens // 4
        assert torch.equal(batch.positions, torch.arange(tokens))
