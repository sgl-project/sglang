# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.vla.pi05_segment import (
    Pi05FlowState,
    advance_pi05_segment,
    collate_prefixes,
)
from sglang.multimodal_gen.runtime.vla.prefix_cache import (
    PrefixContext,
    VLADensePrefixCache,
)


def prefix(length=3, value=0.2):
    kv = torch.full((1, 2, length, 4), value)
    return PrefixContext(
        VLADensePrefixCache([(kv, kv.clone(), None)], read_only=True),
        torch.ones(1, length, dtype=torch.bool),
        length,
        {"full_attention": True},
    )


class Denoiser:
    def __init__(self):
        self.core_model = SimpleNamespace(prepare_denoise_layout=lambda *a, **kw: None)
        self.times = []

    def denoise_step(self, pc, x, t, **kwargs):
        assert kwargs["use_cuda_graph"] is False
        assert kwargs["action_sp_enabled"] is False
        self.times.append(t.clone())
        signal = (pc.past_key_values[0][0][:, 0, :, 0] * pc.prefix_pad_masks).sum(-1)
        return 0.2 * x.sin() + 0.1 * t[:, None, None] + 0.03 * signal[:, None, None]


def reference(noise, pc, steps):
    # Independent whole-trajectory loop, matching native sample_actions.
    model = Denoiser()
    x = noise.clone()
    for t in torch.linspace(1.0, 1.0 / steps, steps):
        x.add_(
            model.denoise_step(
                pc, x, t.expand(1), use_cuda_graph=False, action_sp_enabled=False
            ),
            alpha=-1.0 / steps,
        )
    return x


@pytest.mark.parametrize("capacity", [1, 2, 4])
@pytest.mark.parametrize("per_step", [False, True])
def test_pause_resume_refill_matches_independent_trajectories(capacity, per_step):
    specs = [(3, 2), (5, 7), (4, 4), (1, 2), (8, 5), (8, 3)]
    pending, expected, noises = [], {}, []
    for i, (steps, plen) in enumerate(specs):
        noise = torch.randn((1, 3, 4), generator=torch.Generator().manual_seed(i))
        pc = prefix(plen, value=(i + 1) / 10)
        flow = Pi05FlowState.create(noise, pc, steps)
        pending.append(flow)
        expected[id(flow)] = reference(noise, pc, steps)
        noises.append((noise, noise.clone()))
    active, done = [], []
    model = Denoiser()
    while active or pending:
        while pending and len(active) < capacity:
            active.append(pending.pop(0))
        length = 1 if per_step else min(s.remaining for s in active)
        advance_pi05_segment(model, active, length)
        done.extend(s for s in active if s.remaining == 0)
        active = [s for s in active if s.remaining]
    for flow in done:
        torch.testing.assert_close(flow.x_t, expected[id(flow)], rtol=1e-6, atol=1e-6)
        assert flow.steps_done == flow.num_steps
        assert flow.dt == -1.0 / flow.num_steps
    for noise, original in noises:
        torch.testing.assert_close(noise, original, rtol=0, atol=0)


def test_resume_uses_original_grid_and_dt():
    a = Pi05FlowState.create(torch.zeros(1, 3, 4), prefix(), 3)
    b = Pi05FlowState.create(torch.zeros(1, 3, 4), prefix(), 5)
    model = Denoiser()
    advance_pi05_segment(model, [a, b], 3)
    c = Pi05FlowState.create(torch.zeros(1, 3, 4), prefix(), 4)
    advance_pi05_segment(model, [b, c], 2)
    torch.testing.assert_close(
        torch.stack([t[0] for t in model.times[3:]]), torch.linspace(1.0, 0.2, 5)[3:]
    )
    assert b.num_steps == 5 and b.dt == -0.2 and b.remaining == 0
    assert c.remaining == 2


def test_masked_padding_preserves_original_prefixes():
    a, b = prefix(2), prefix(5)
    result = collate_prefixes([a, b])
    assert result.prefix_len == 5
    assert result.layout["cuda_graph_eligible"] is False
    assert result.layout["full_attention"] is False
    assert result.prefix_pad_masks.tolist() == [
        [True, True, False, False, False],
        [True] * 5,
    ]
    assert torch.count_nonzero(result.past_key_values[0][0][0, :, 2:]) == 0
    assert a.prefix_len == 2 and a.past_key_values[0][0].shape[-2] == 2


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_invalid_budget(steps):
    with pytest.raises(ValueError):
        Pi05FlowState.create(torch.zeros(1, 3, 4), prefix(), steps)


@pytest.mark.parametrize("length", [0, -1, 4, True])
def test_invalid_segment_does_not_advance(length):
    flow = Pi05FlowState.create(torch.zeros(1, 3, 4), prefix(), 3)
    with pytest.raises(ValueError):
        advance_pi05_segment(Denoiser(), [flow], length)
    assert flow.steps_done == 0


def test_failed_segment_does_not_commit_partial_state():
    flow = Pi05FlowState.create(torch.ones(1, 3, 4), prefix(), 3)
    original = flow.x_t.clone()
    model = Denoiser()
    step = model.denoise_step

    def fail(pc, x, t, **kwargs):
        if model.times:
            raise RuntimeError("simulated device failure")
        return step(pc, x, t, **kwargs)

    model.denoise_step = fail
    with pytest.raises(RuntimeError):
        advance_pi05_segment(model, [flow], 2)
    assert flow.steps_done == 0
    torch.testing.assert_close(flow.x_t, original, rtol=0, atol=0)


def test_rejects_incompatible_prefixes_and_duplicate_state():
    a, b = prefix(), prefix()
    b.past_key_values = VLADensePrefixCache(
        [(torch.zeros(1, 3, 3, 4), torch.zeros(1, 3, 3, 4), None)]
    )
    with pytest.raises(ValueError, match="Incompatible"):
        collate_prefixes([a, b])
    flow = Pi05FlowState.create(torch.zeros(1, 3, 4), a, 3)
    with pytest.raises(ValueError, match="distinct"):
        advance_pi05_segment(Denoiser(), [flow, flow], 1)
