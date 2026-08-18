# SPDX-License-Identifier: Apache-2.0
"""Integration test for StepReuseMixin, wired the same way
``WanTransformer3DModel`` (runtime/models/dits/wanvideo.py) actually uses it:
``should_skip_forward_for_step_reuse`` -> ``retrieve_step_reuse_prediction``,
else run the transformer blocks then ``maybe_record_step_reuse``.

Uses real torch tensors and the real ``set_forward_context`` context manager
-- not mocks -- to validate the actual model-integration surface (state
reset, CFG-branch isolation, forced boundary steps) rather than only the
pure StepReuseController contract (covered separately in
test_step_reuse.py).
"""

from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.sample.step_reuse import StepReuseParams
from sglang.multimodal_gen.runtime.cache.step_reuse import StepReuseMixin
from sglang.multimodal_gen.runtime.managers.forward_context import (
    set_forward_context,
)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _device()


class _FakeDiT(StepReuseMixin):
    """Minimal stand-in for a CachableDiT subclass, mirroring
    WanTransformer3DModel's exact step-reuse call pattern."""

    def __init__(self, dim: int = 8):
        self._init_step_reuse_state()
        self.enable_step_reuse = True
        self.linear = torch.nn.Linear(dim, dim).to(DEVICE)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.should_skip_forward_for_step_reuse():
            return self.retrieve_step_reuse_prediction(hidden_states)

        modulated_inp = hidden_states.detach()
        original_hidden_states = hidden_states.clone()
        hidden_states = torch.tanh(self.linear(hidden_states))
        self.maybe_record_step_reuse(
            hidden_states, original_hidden_states, modulated_inp=modulated_inp
        )
        return hidden_states


def _make_forward_batch(
    num_inference_steps=6, do_cfg=False, is_cfg_negative=False, threshold=0.5
):
    return SimpleNamespace(
        enable_step_reuse=True,
        step_reuse_params=StepReuseParams(
            threshold=threshold, max_skip_steps=3, history_size=1
        ),
        num_inference_steps=num_inference_steps,
        do_classifier_free_guidance=do_cfg,
        is_cfg_negative=is_cfg_negative,
    )


class TestStepReuseMixinRealRollout:
    def test_first_step_always_real_then_skips_open(self):
        model = _FakeDiT()
        batch = _make_forward_batch(num_inference_steps=6, threshold=0.9)
        # Near-constant input across steps keeps relative-L1 low so a skip
        # window opens once history exists.
        x = torch.ones(4, 8, device=DEVICE) * 0.01

        outputs = []
        for step in range(6):
            with set_forward_context(
                current_timestep=step, attn_metadata=None, forward_batch=batch
            ):
                outputs.append(model.forward(x))

        assert not torch.equal(outputs[0], x)

        metrics = model._step_reuse_controller.metrics(("cfg_positive",))
        assert metrics["real_forwards"] >= 1
        assert metrics["reused_steps"] >= 1
        assert metrics["total_steps_seen"] == 6

    def test_terminal_step_is_forced_real_even_mid_skip_window(self):
        model = _FakeDiT()
        batch = _make_forward_batch(num_inference_steps=4, threshold=0.9)
        x = torch.ones(4, 8, device=DEVICE) * 0.01

        for step in range(4):
            with set_forward_context(
                current_timestep=step, attn_metadata=None, forward_batch=batch
            ):
                model.forward(x)

        # Real forwards must include step 0 (first_one) and step 3
        # (terminal) at minimum -- not all 4 steps were reused, even though
        # max_skip_steps=3 would otherwise cover the whole rollout after
        # step 0 opens a window.
        state = model._step_reuse_controller._peek_state(("cfg_positive",))
        assert state.real_forwards >= 2

    def test_cfg_positive_and_negative_branches_independent(self):
        # Deliberately calls positive before negative at each timestep, to
        # verify the reset guard is not order-dependent on which CFG branch
        # is dispatched first (see _maybe_reset_step_reuse_for_new_task).
        model = _FakeDiT()
        batch_pos = _make_forward_batch(
            num_inference_steps=6, do_cfg=True, is_cfg_negative=False, threshold=0.9
        )
        batch_neg = _make_forward_batch(
            num_inference_steps=6, do_cfg=True, is_cfg_negative=True, threshold=0.9
        )
        x_pos = torch.ones(4, 8, device=DEVICE) * 0.01
        x_neg = torch.ones(4, 8, device=DEVICE) * 5.0

        for step in range(3):
            with set_forward_context(
                current_timestep=step, attn_metadata=None, forward_batch=batch_pos
            ):
                model.forward(x_pos)
            with set_forward_context(
                current_timestep=step, attn_metadata=None, forward_batch=batch_neg
            ):
                model.forward(x_neg)

        pos_metrics = model._step_reuse_controller.metrics(("cfg_positive",))
        neg_metrics = model._step_reuse_controller.metrics(("cfg_negative",))
        assert pos_metrics["total_steps_seen"] == 3
        assert neg_metrics["total_steps_seen"] == 3

    def test_new_request_resets_state_at_timestep_zero(self):
        model = _FakeDiT()
        batch = _make_forward_batch(num_inference_steps=3, threshold=0.9)
        x = torch.ones(4, 8, device=DEVICE) * 0.01

        for step in range(3):
            with set_forward_context(
                current_timestep=step, attn_metadata=None, forward_batch=batch
            ):
                model.forward(x)

        metrics_before = model._step_reuse_controller.metrics(("cfg_positive",))
        assert metrics_before["total_steps_seen"] == 3

        with set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=batch
        ):
            model.forward(x)

        metrics_after = model._step_reuse_controller.metrics(("cfg_positive",))
        # A new request starting at timestep 0 resets state, so only its own
        # single step is counted -- not 3 (old request) + 1.
        assert metrics_after["total_steps_seen"] == 1

    def test_disabled_flag_never_skips(self):
        model = _FakeDiT()
        batch = _make_forward_batch(num_inference_steps=4, threshold=0.9)
        batch.enable_step_reuse = False

        with set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=batch
        ):
            skip = model.should_skip_forward_for_step_reuse()

        assert skip is False
        assert model._step_reuse_controller is None
