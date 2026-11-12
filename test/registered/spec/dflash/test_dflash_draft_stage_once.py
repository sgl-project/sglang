# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""SGLANG_OPT_DFLASH_DRAFT_STAGE_ONCE: the selector draft sampler stages its temperatures / greedy-mask rows only when
the batch's sampling tensors, their contents or bs change. After every call in a scripted sequence the knob-on
sampler's buffers must equal a knob-off sampler's bit for bit, and it must skip exactly where the batch is unchanged."""

import dataclasses
import types

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

DEV = "cuda"
MAX_BS = 4


@dataclasses.dataclass
class _Info:
    """The two SamplingBatchInfo fields the staging reads (copy_for_forward is a dataclasses.replace)."""

    temperatures: torch.Tensor
    top_ks: torch.Tensor


def _info(temps, top_ks):
    return _Info(
        temperatures=torch.tensor(temps, dtype=torch.float32, device=DEV).view(-1, 1),
        top_ks=torch.tensor(top_ks, dtype=torch.int32, device=DEV),
    )


def _sampler(stage_once: bool):
    from sglang.srt.speculative.dflash_worker_v2 import _SelectorDraftSampler

    model = types.SimpleNamespace(candidate_selector=types.SimpleNamespace(top_k=8))
    with envs.SGLANG_OPT_DFLASH_FUSED_SELECTOR.override(False), envs.SGLANG_OPT_DFLASH_DRAFT_STAGE_ONCE.override(
        stage_once
    ):
        return _SelectorDraftSampler(draft_model=model, block_size=8, max_bs=MAX_BS, device=DEV, sampling_enabled=True)


def _bufs(s):
    return s.temperatures.clone(), s.greedy_mask.clone()


def _kernels(fn) -> int:
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)


def test_stage_once_matches_served_staging():
    """Catches a no-sampling staging that leaves the old key (a return to the earlier batch tensors skips over rows
    the no-sampling branch overwrote), a skip that ignores bs (on the no-sampling branch, where bs is the whole key),
    an in-place write to the batch tensors (version), a new batch tensor with equal versions (identity), or a skip
    on inference tensors (no version counter). Each of these makes a buffer differ from the served one at the step
    marked for it. As in the served call, a sampling batch's tensors hold exactly bs rows."""
    served, once = _sampler(False), _sampler(True)
    u = _info([0.5, 0.6, 0.7], [1, 5, 1])
    t = _info([1.3, 1.1, 0.9], [4, 1, 8])
    t2 = _info([0.2, 0.3, 0.4], [2, 2, 1])
    t3 = _info([1.7, 0.45, 0.66], [1, 9, 2])  # new tensors with the same bs and versions as t2's

    def bump():
        t.temperatures.mul_(2.0)  # in place: same tensor, new version

    with torch.inference_mode():
        inf = _info([0.8, 0.85, 0.95], [1, 1, 3])

    def bump_inf():
        with torch.inference_mode():
            inf.temperatures.mul_(0.5)  # in place on an inference tensor: no version counter sees it

    # (bs, info or (callable run first, info), expected skip count after the call); expected counts are by hand.
    steps = [
        (3, u, 0),
        (3, u, 1),
        (3, dataclasses.replace(u), 2),  # copy_for_forward: new object, same tensors
        (3, None, 2),
        (3, u, 2),  # back from the no-sampling branch to the same tensors and versions: must restage (stale key)
        (1, None, 2),  # rows 1-2 keep u's values
        (3, None, 2),  # bs grows on the no-sampling branch: rows 1-2 must be refilled (bs)
        (3, None, 3),
        (3, t, 3),
        (3, (bump, t), 3),  # same tensor, new version (version)
        (3, t, 4),
        (3, t2, 4),
        (3, t3, 4),  # new tensors with t2's bs and versions (identity)
        (3, t3, 5),
        (3, inf, 5),
        (3, (bump_inf, inf), 5),  # inference tensors carry no version: never skipped (inference)
        (3, inf, 5),
    ]
    for i, (bs, info, want_skips) in enumerate(steps):
        if isinstance(info, tuple):
            info[0]()
            info = info[1]
        with torch.inference_mode(info is inf):
            served.stage_sampling_params(bs=bs, sampling_info=info)
            once.stage_sampling_params(bs=bs, sampling_info=info)
        torch.cuda.synchronize()
        for name, a, b in zip(("temperatures", "greedy_mask"), _bufs(served), _bufs(once)):
            assert torch.equal(a, b), f"step {i}: {name} {b.tolist()} != served {a.tolist()}"
        assert once.stage_skip_ct == want_skips, f"step {i}: skips {once.stage_skip_ct}, want {want_skips}"
    assert served.stage_skip_ct == 0


def test_skip_launches_nothing():
    """The skipped call launches no kernel; the served call launches the clamp, the mask and its copy."""
    served, once = _sampler(False), _sampler(True)
    info = _info([0.7], [1])
    once.stage_sampling_params(bs=1, sampling_info=info)
    n_skip = _kernels(lambda: once.stage_sampling_params(bs=1, sampling_info=info))
    n_served = _kernels(lambda: served.stage_sampling_params(bs=1, sampling_info=info))
    assert once.stage_skip_ct == 1
    assert n_served >= 3, n_served
    assert n_skip == 0, n_skip


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
