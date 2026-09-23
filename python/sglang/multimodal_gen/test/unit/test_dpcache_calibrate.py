# SPDX-License-Identifier: Apache-2.0
"""DPCache calibration: online PACT scoring, request lifecycle, planner, CLI; CPU only."""

import json
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.cache.dpcache import (
    DPCacheCalibrationState,
    load_calibration_capture,
    pact_costs,
    pact_step_errors,
    plan_schedule,
    plan_schedules,
    save_calibration_capture,
    validate_calibration_request,
    validate_schedule,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.test.unit.test_dpcache import (  # noqa: F401
    SIGNATURE,
    _TinyDiT,
    denoise_must_not_run,
    forward_batch,
    make_batch,
    make_stage,
    run_dit,
    scheduler_request,
    single_gpu,
)
from sglang.multimodal_gen.tools import dpcache_calibrate as tool

NUM_STEPS = len(SIGNATURE.timesteps)


def random_features(num_steps, seed, shape=(3, 4)):
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(shape, generator=generator).bfloat16() for _ in range(num_steps)
    ]


def calibrated_state(features, max_gap):
    state = DPCacheCalibrationState(len(features), max_gap)
    for step, feature in enumerate(features):
        state.record(step, feature)
    return state


@pytest.mark.parametrize("max_gap", [None, 1, 3])
def test_online_scoring_matches_offline_scoring(max_gap):
    features = random_features(9, seed=0)
    state = calibrated_state(features, max_gap)
    torch.testing.assert_close(
        state.errors,
        pact_step_errors(features, max_gap=max_gap),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert state.complete


def test_online_scoring_keeps_only_usable_features():
    state = DPCacheCalibrationState(12, max_gap=2)
    for step, feature in enumerate(random_features(12, seed=1)):
        state.record(step, feature)
        assert len(state._features) <= 2 * 2 + 1


def test_calibration_state_rejects_skipped_steps_and_non_bf16():
    state = DPCacheCalibrationState(4)
    assert state.is_full_step(3)
    with pytest.raises(RuntimeError, match="every step in order"):
        state.record(1, torch.zeros(2, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="BF16"):
        state.record(0, torch.zeros(2))
    with pytest.raises(RuntimeError, match="runs every step"):
        state.predict(0)


def test_hook_runs_every_block_while_calibrating():
    dit = _TinyDiT()
    batch = forward_batch({False: DPCacheCalibrationState(3)})
    for step in range(3):
        x = torch.full((1,), float(step), dtype=torch.bfloat16)
        assert run_dit(dit, step, x, batch).tolist() == [2.0 * step]
    assert dit.block_calls == 3
    assert batch.dpcache_states[False].complete


@pytest.mark.parametrize(
    "calibration",
    [
        [],
        {"max_gap": 8},
        {"output": ""},
        {"output": "a.pt", "max_gap": 0},
        {"output": "a.pt", "max_gap": True},
        {"output": "a.pt", "extra": 1},
    ],
)
def test_malformed_calibration_requests_are_rejected(calibration):
    with pytest.raises((TypeError, ValueError)):
        validate_calibration_request(calibration)
    with pytest.raises((TypeError, ValueError)):
        SamplingParams(dpcache_calibration=calibration)


def test_calibration_and_budget_are_exclusive():
    validate_calibration_request({"output": "a.pt", "max_gap": None})
    with pytest.raises(ValueError, match="exclusive"):
        SamplingParams(dpcache_budget=4, dpcache_calibration={"output": "a.pt"})
    SamplingParams(dpcache_budget=0, dpcache_calibration={"output": "a.pt"})


def fill_every_step(self, batch, server_args):
    """Stand-in denoise that records every step, like the DiT hook does."""
    for step in range(len(batch.timesteps)):
        batch.dpcache_states[False].record(
            step, torch.full((2,), float(step * step), dtype=torch.bfloat16)
        )
    return batch


def test_stage_writes_a_capture_after_a_calibration_request(monkeypatch, tmp_path):
    stage, server_args = make_stage()
    output = tmp_path / "captures" / "00000.pt"
    batch = make_batch(None, dpcache_calibration={"output": str(output), "max_gap": 2})
    monkeypatch.setattr(DenoisingStage, "_denoise", fill_every_step)
    assert stage.forward(batch, server_args) is batch
    assert batch.dpcache_states is None
    capture = load_calibration_capture(str(output))
    assert capture["signature"]["checkpoint"] == SIGNATURE.checkpoint
    assert capture["signature"]["timesteps"] == list(SIGNATURE.timesteps)
    assert (capture["prompt"], capture["seed"], capture["max_gap"]) == ("a cat", 42, 2)
    assert capture["errors"].shape == (NUM_STEPS,) * 3


def test_failed_or_incomplete_calibration_writes_nothing(monkeypatch, tmp_path):
    stage, server_args = make_stage()
    output = tmp_path / "00000.pt"
    batch = make_batch(None, dpcache_calibration={"output": str(output)})

    def failing_denoise(self, batch, server_args):
        raise RuntimeError("boom")

    monkeypatch.setattr(DenoisingStage, "_denoise", failing_denoise)
    with pytest.raises(RuntimeError, match="boom"):
        stage.forward(batch, server_args)
    monkeypatch.setattr(DenoisingStage, "_denoise", lambda self, b, s: b)
    with pytest.raises(RuntimeError, match="scored 0 of"):
        stage.forward(batch, server_args)
    assert not output.exists()
    assert batch.dpcache_states is None


def test_warmup_calibration_request_runs_natively(monkeypatch, tmp_path):
    stage, server_args = make_stage()
    output = tmp_path / "00000.pt"
    batch = make_batch(None, dpcache_calibration={"output": str(output)})
    batch.is_warmup = True
    monkeypatch.setattr(DenoisingStage, "_denoise", lambda self, b, s: b)
    stage.forward(batch, server_args)
    assert not output.exists()


def test_stage_applies_the_scope_checks_to_calibration(monkeypatch, tmp_path):
    calibration = {"output": str(tmp_path / "00000.pt")}
    denoise_must_not_run(monkeypatch)
    stage, server_args = make_stage(supported=False)
    with pytest.raises(ValueError, match="not supported"):
        stage.forward(make_batch(None, dpcache_calibration=calibration), server_args)
    stage, server_args = make_stage()
    batch = make_batch(
        None, dpcache_calibration=calibration, do_classifier_free_guidance=True
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        stage.forward(batch, server_args)
    batch = make_batch(4, dpcache_calibration=calibration)
    with pytest.raises(ValueError, match="exclusive"):
        stage.forward(batch, server_args)


def test_calibration_request_ignores_the_server_default_budget(monkeypatch, tmp_path):
    """A calibration run must see every step, whatever the server serves."""
    stage, server_args = make_stage(dpcache_default_budget=4)
    output = tmp_path / "00000.pt"
    batch = make_batch(None, dpcache_calibration={"output": str(output)})
    monkeypatch.setattr(DenoisingStage, "_denoise", fill_every_step)
    stage.forward(batch, server_args)
    assert load_calibration_capture(str(output))["errors"].shape == (NUM_STEPS,) * 3


def test_calibration_requests_are_not_dynamically_batched():
    scheduler = object.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            supports_sequential_multi_output_inference=lambda: False,
            supports_sequential_dit_inference=lambda: False,
        )
    )
    request = scheduler_request(None, calibration={"output": "a.pt"})
    assert scheduler._get_dynamic_batch_reject_reason(request, request) == "dpcache"


def write_captures(directory, seeds, max_gap=3):
    """Captures of one request configuration, as the stage writes them."""
    for index, seed in enumerate(seeds):
        save_calibration_capture(
            str(directory / f"{index:05d}.pt"),
            state=calibrated_state(random_features(NUM_STEPS, seed), max_gap),
            signature=SIGNATURE,
            prompt=f"prompt {seed}",
            seed=seed,
        )


def test_planner_averages_captures_like_the_single_sample_planner(tmp_path):
    write_captures(tmp_path, seeds=(0, 1, 2))
    captures = [load_calibration_capture(str(p)) for p in sorted(tmp_path.iterdir())]
    schedules = plan_schedules(captures, [4, 5], source_commit="abc", max_gap=3)
    mean = (captures[0]["errors"] + captures[1]["errors"] + captures[2]["errors"]) / 3
    costs = pact_costs(mean, max_gap=3)
    for budget, schedule in schedules.items():
        full_steps, cost = plan_schedule(costs, budget, max_gap=3)
        assert schedule["full_steps"] == full_steps
        assert schedule["calibrated_cost"] == cost
        assert schedule["calibration"]["num_samples"] == 3
        assert validate_schedule(json.loads(json.dumps(schedule)))


def test_planner_manifest_ignores_capture_order(tmp_path):
    write_captures(tmp_path, seeds=(0, 1))
    captures = [load_calibration_capture(str(p)) for p in sorted(tmp_path.iterdir())]
    forward = plan_schedules(captures, [4], source_commit="abc", max_gap=3)
    backward = plan_schedules(captures[::-1], [4], source_commit="abc", max_gap=3)
    assert (
        forward[4]["calibration"]["manifest_sha256"]
        == backward[4]["calibration"]["manifest_sha256"]
    )


def test_planner_rejects_mixed_or_under_scored_captures(tmp_path):
    write_captures(tmp_path, seeds=(0, 1))
    captures = [load_calibration_capture(str(p)) for p in sorted(tmp_path.iterdir())]
    with pytest.raises(ValueError, match="looser"):
        plan_schedules(captures, [4], source_commit="abc", max_gap=4)
    with pytest.raises(ValueError, match="looser"):
        plan_schedules(captures, [4], source_commit="abc", max_gap=None)
    captures[1]["signature"] = {**captures[1]["signature"], "height": 128}
    with pytest.raises(ValueError, match="different requests"):
        plan_schedules(captures, [4], source_commit="abc", max_gap=3)


def test_prompts_file_accepts_text_and_jsonl(tmp_path):
    text = tmp_path / "prompts.txt"
    text.write_text("a cat\n\na dog\n")
    assert tool.load_prompts(str(text), default_seed=7) == [
        {"prompt": "a cat", "seed": 7},
        {"prompt": "a dog", "seed": 7},
    ]
    jsonl = tmp_path / "prompts.jsonl"
    jsonl.write_text('{"prompt": "a cat", "seed": 1}\n{"prompt": "a dog"}\n')
    assert tool.load_prompts(str(jsonl), default_seed=7) == [
        {"prompt": "a cat", "seed": 1},
        {"prompt": "a dog", "seed": 7},
    ]


def test_shards_partition_the_prompts():
    shards = [tool.shard_indices(7, f"{i}/3") for i in range(3)]
    assert sorted(sum(shards, [])) == list(range(7))
    assert tool.shard_indices(3, None) == [0, 1, 2]
    with pytest.raises(ValueError):
        tool.shard_indices(3, "3/3")


def test_record_request_asks_for_a_native_calibration_run(tmp_path):
    args = tool.build_parser().parse_args(
        ["record", "--prompts-file", "p.txt", "--capture-dir", str(tmp_path)]
        + ["--height", "64", "--max-gap", "0"]
    )
    kwargs = tool.sampling_kwargs(
        args, {"prompt": "a cat", "seed": 3}, tmp_path / "00000.pt"
    )
    assert kwargs["dpcache_calibration"] == {
        "output": str((tmp_path / "00000.pt").resolve()),
        "max_gap": None,
    }
    assert (kwargs["prompt"], kwargs["seed"], kwargs["height"]) == ("a cat", 3, 64)
    assert "width" not in kwargs and kwargs["save_output"] is False
    SamplingParams(**kwargs)


def test_plan_command_writes_valid_schedules(tmp_path, capsys):
    write_captures(tmp_path / "captures", seeds=(0, 1))
    tool.main(
        ["plan", "--capture-dir", str(tmp_path / "captures")]
        + ["--out-dir", str(tmp_path / "schedules"), "--budgets", "4", "5"]
        + ["--source-commit", "abc"]
    )
    for budget in (4, 5):
        schedule = json.loads((tmp_path / "schedules" / f"K{budget}.json").read_text())
        assert validate_schedule(schedule)
        assert schedule["max_gap"] == 3 and schedule["source_commit"] == "abc"
    assert "planned from 2 captures" in capsys.readouterr().out
