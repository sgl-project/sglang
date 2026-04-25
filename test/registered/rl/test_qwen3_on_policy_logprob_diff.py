from __future__ import annotations

"""Regression test for Qwen3-0.6B true on-policy logprob alignment."""

import multiprocessing as mp
import os
import unittest
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=120,
    suite="stage-b-test-1-gpu-large",
)

REFERENCE_DATASET = "zyzshishui0627/qwen3-0.6b-fsdp2"
REFERENCE_FILE = "compare_sample_train_data.pt"

FALLBACK_MODEL = "Qwen/Qwen3-0.6B"
FALLBACK_ATTENTION_BACKEND = "fa3"
FALLBACK_RL_TARGET = "fsdp"
FALLBACK_KL_THRESHOLD = 1e-6

REFERENCE_PATH_ENV = "SGLANG_ON_POLICY_LOGPROB_REFERENCE_PATH"


@dataclass
class ReferenceSpec:
    source_name: str
    data_path: Path
    input_ids: list[int] | torch.Tensor
    trainer_logprobs: list[float] | torch.Tensor
    model_path: str
    tp_size: int
    attention_backend: str
    rl_on_policy_target: str
    kl_threshold: float
    expect_on_policy_better_than_baseline: bool
    disable_cuda_graph: bool
    disable_piecewise_cuda_graph: bool


def kl_v2(a, b):
    a = torch.tensor(a) if not torch.is_tensor(a) else a
    b = torch.tensor(b) if not torch.is_tensor(b) else b
    return (((a - b) ** 2) * 0.5).mean().item()


def _payload_get(payload, *keys):
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _load_reference_payload():
    reference_path = os.getenv(REFERENCE_PATH_ENV)
    if reference_path:
        data_path = Path(reference_path).expanduser().resolve()
        if data_path.is_dir():
            data_path = data_path / REFERENCE_FILE
        source_name = str(data_path)
    else:
        data_path = Path(
            hf_hub_download(
                repo_id=REFERENCE_DATASET,
                repo_type="dataset",
                filename=REFERENCE_FILE,
            )
        ).resolve()
        source_name = REFERENCE_DATASET

    payload = torch.load(data_path, weights_only=False)
    return source_name, data_path, payload


def load_reference_spec():
    source_name, data_path, payload = _load_reference_payload()

    return ReferenceSpec(
        source_name=source_name,
        data_path=data_path,
        input_ids=_payload_get(payload, "tokens", "input_ids"),
        trainer_logprobs=payload["training_logprobs"],
        model_path=payload.get("model_path", FALLBACK_MODEL),
        tp_size=int(payload.get("tp_size", 1)),
        attention_backend=payload.get("attention_backend", FALLBACK_ATTENTION_BACKEND),
        rl_on_policy_target=payload.get("rl_on_policy_target", FALLBACK_RL_TARGET),
        kl_threshold=float(payload.get("kl_threshold", FALLBACK_KL_THRESHOLD)),
        expect_on_policy_better_than_baseline=bool(
            payload.get("expect_on_policy_better_than_baseline", False)
        ),
        disable_cuda_graph=bool(payload.get("disable_cuda_graph", False)),
        disable_piecewise_cuda_graph=bool(
            payload.get("disable_piecewise_cuda_graph", False)
        ),
    )


def build_engine(spec, *, on_policy):
    kwargs = dict(
        model_path=spec.model_path,
        tp_size=spec.tp_size,
        attention_backend=spec.attention_backend,
        disable_cuda_graph=spec.disable_cuda_graph,
        disable_piecewise_cuda_graph=spec.disable_piecewise_cuda_graph,
    )
    if on_policy:
        kwargs["rl_on_policy_target"] = spec.rl_on_policy_target
    return sgl.Engine(**kwargs)


def get_prompt_logprobs(engine, input_ids):
    out = engine.generate(
        input_ids=input_ids,
        sampling_params={"max_new_tokens": 0, "temperature": 0.0},
        return_logprob=True,
        logprob_start_len=0,
    )
    return [logprob for logprob, _, _ in out["meta_info"]["input_token_logprobs"]][1:]


def _shutdown_engine(engine):
    engine.shutdown()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class TestQwen3OnPolicyLogprobDiff(CustomTestCase):
    def test_qwen3_on_policy_logprob_alignment(self):
        spec = load_reference_spec()

        print(f"[REFERENCE] source={spec.source_name}")
        print(f"[REFERENCE] file={spec.data_path}")
        print(
            f"[REFERENCE] model={spec.model_path}, tp={spec.tp_size}, "
            f"attention_backend={spec.attention_backend}, "
            f"rl_target={spec.rl_on_policy_target}"
        )

        baseline_engine = build_engine(spec, on_policy=False)
        try:
            baseline_logprobs = get_prompt_logprobs(baseline_engine, spec.input_ids)
        finally:
            _shutdown_engine(baseline_engine)

        on_policy_engine = build_engine(spec, on_policy=True)
        try:
            on_policy_logprobs = get_prompt_logprobs(on_policy_engine, spec.input_ids)
        finally:
            _shutdown_engine(on_policy_engine)

        kl_baseline_trainer = kl_v2(spec.trainer_logprobs, baseline_logprobs)
        kl_on_policy_trainer = kl_v2(spec.trainer_logprobs, on_policy_logprobs)
        kl_on_policy_baseline = kl_v2(on_policy_logprobs, baseline_logprobs)

        print(f"KL(baseline, trainer)      = {kl_baseline_trainer:.6e}")
        print(f"KL(on_policy, trainer)     = {kl_on_policy_trainer:.6e}")
        print(f"KL(on_policy, baseline)    = {kl_on_policy_baseline:.6e}")

        self.assertLessEqual(
            kl_on_policy_trainer,
            spec.kl_threshold,
            f"KL(on_policy, trainer) = {kl_on_policy_trainer:.6e} exceeds "
            f"threshold {spec.kl_threshold}",
        )

        if spec.expect_on_policy_better_than_baseline:
            self.assertLess(
                kl_on_policy_trainer,
                kl_baseline_trainer,
                "Expected on-policy inference to be closer to trainer than "
                "baseline inference",
            )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    try:
        unittest.main(warnings="ignore", verbosity=2)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
