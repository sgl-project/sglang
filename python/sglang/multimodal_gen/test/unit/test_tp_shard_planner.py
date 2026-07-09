import json

import pytest

from sglang.multimodal_gen.runtime.layers.tp_shard_planner import (
    PLAN_AGGRESSIVE,
    PLAN_AUTO,
    PLAN_FULL,
    ShardScheme,
    TPShardPlan,
    TPShardPlanner,
    parse_workload,
)


def _planner(tp_size, mode=PLAN_AUTO, rules=None, workload_area=None):
    plan = TPShardPlan(
        mode=mode, rules=rules or {}, workload_area=workload_area, source="test"
    )
    return TPShardPlanner(plan, tp_size)


class TestMeasuredRules:
    def test_qwen_text_ffn_replicates_at_tp2(self):
        planner = _planner(tp_size=2)
        assert (
            planner.decide_ffn(model_family="qwen_image", branch="text")
            is ShardScheme.REPLICATE
        )

    def test_qwen_text_ffn_shards_at_tp4(self):
        # tp=4 measured a consistent regression for replication.
        planner = _planner(tp_size=4)
        assert (
            planner.decide_ffn(model_family="qwen_image", branch="text")
            is ShardScheme.SHARD
        )

    def test_qwen_image_ffn_shards_by_default(self):
        planner = _planner(tp_size=2)
        assert (
            planner.decide_ffn(model_family="qwen_image", branch="image")
            is ShardScheme.SHARD
        )

    def test_qwen_image_ffn_replicates_only_aggressive_small_workload(self):
        small = parse_workload("512x512")
        large = parse_workload("1024x1024")
        aggressive_small = _planner(
            tp_size=2, mode=PLAN_AGGRESSIVE, workload_area=small
        )
        assert (
            aggressive_small.decide_ffn(model_family="qwen_image", branch="image")
            is ShardScheme.REPLICATE
        )
        # Same workload without aggressive mode: stay sharded.
        auto_small = _planner(tp_size=2, mode=PLAN_AUTO, workload_area=small)
        assert (
            auto_small.decide_ffn(model_family="qwen_image", branch="image")
            is ShardScheme.SHARD
        )
        # Aggressive but standard resolution: stay sharded (regresses >=1024^2).
        aggressive_large = _planner(
            tp_size=2, mode=PLAN_AGGRESSIVE, workload_area=large
        )
        assert (
            aggressive_large.decide_ffn(model_family="qwen_image", branch="image")
            is ShardScheme.SHARD
        )
        # Aggressive with no workload hint: workload-gated rules must not fire.
        aggressive_unknown = _planner(tp_size=2, mode=PLAN_AGGRESSIVE)
        assert (
            aggressive_unknown.decide_ffn(model_family="qwen_image", branch="image")
            is ShardScheme.SHARD
        )

    def test_unknown_model_defaults_to_shard(self):
        # The qwen rule measurably does not transfer to FLUX; un-measured
        # (model, branch) pairs must never get a speculative layout.
        planner = _planner(tp_size=2)
        assert (
            planner.decide_ffn(model_family="flux", branch="text")
            is ShardScheme.SHARD
        )


class TestModesAndOverrides:
    def test_full_mode_shards_everything(self):
        planner = _planner(tp_size=2, mode=PLAN_FULL)
        assert (
            planner.decide_ffn(model_family="qwen_image", branch="text")
            is ShardScheme.SHARD
        )

    def test_tp1_is_noop(self):
        planner = _planner(tp_size=1)
        assert (
            planner.decide_ffn(model_family="qwen_image", branch="text")
            is ShardScheme.SHARD
        )

    def test_plan_rule_overrides_registry(self):
        planner = _planner(
            tp_size=2, rules={"*.txt_mlp": ShardScheme.SHARD}
        )
        assert (
            planner.decide_ffn(
                model_family="qwen_image",
                branch="text",
                prefix="transformer_blocks.3.txt_mlp",
            )
            is ShardScheme.SHARD
        )

    def test_plan_rule_can_replicate_unmeasured_model(self):
        planner = _planner(
            tp_size=2, rules={"*.ff_context": ShardScheme.REPLICATE}
        )
        assert (
            planner.decide_ffn(
                model_family="flux",
                branch="text",
                prefix="transformer_blocks.0.ff_context",
            )
            is ShardScheme.REPLICATE
        )

    def test_attn_out_is_structurally_sharded(self):
        planner = _planner(tp_size=2)
        assert (
            planner.decide(
                role="attn_out", model_family="qwen_image", branch="image"
            )
            is ShardScheme.SHARD
        )

    def test_plan_rule_rejects_unsupported_scheme(self):
        planner = _planner(
            tp_size=2, rules={"*.to_out*": ShardScheme.REPLICATE}
        )
        with pytest.raises(ValueError, match="only supports"):
            planner.decide(
                role="attn_out",
                model_family="qwen_image",
                branch="image",
                prefix="transformer_blocks.0.to_out.0",
            )


class TestPlanLoading:
    def test_load_mode(self):
        plan = TPShardPlan.load("full", None)
        assert plan.mode == PLAN_FULL and not plan.rules

    def test_load_file(self, tmp_path):
        path = tmp_path / "plan.json"
        path.write_text(
            json.dumps(
                {
                    "mode": "auto",
                    "workload": "512x512",
                    "rules": {"*.txt_mlp": "replicate"},
                }
            )
        )
        plan = TPShardPlan.load(str(path), None)
        assert plan.rules == {"*.txt_mlp": ShardScheme.REPLICATE}
        assert plan.workload_area == 512 * 512
        # CLI workload takes precedence over the plan file's.
        plan = TPShardPlan.load(str(path), "1024x1024")
        assert plan.workload_area == 1024 * 1024

    def test_parse_workload(self):
        assert parse_workload(None) is None
        assert parse_workload("1024x1024") == 1024 * 1024
        assert parse_workload("1280x720x121") == 1280 * 720
        with pytest.raises(ValueError):
            parse_workload("huge")
