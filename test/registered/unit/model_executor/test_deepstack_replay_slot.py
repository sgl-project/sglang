"""Unit tests for the optional DeepStack BCG replay slot.

Covers the allocation contract (registry gating, buffer allocation,
buffer-to-registry adoption), the model capability opt-in, and the
per-replay refresh of the slot (``_refresh_deepstack_replay_slot``).

CPU-only; the logic under test is GPU-agnostic.
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    build_prefill_registry,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
    _refresh_deepstack_replay_slot,
)
from sglang.srt.model_executor.runner_utils.buffers import PrefillInputBuffers
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_DEVICE = torch.device("cpu")


def _reg(**overrides):
    base = dict(
        device=_DEVICE,
        max_bs=1,
        max_num_token=128,
        cache_loc_dtype=torch.int32,
        is_multimodal=True,
        hidden_size=64,
        embed_dtype=torch.bfloat16,
        deepstack_replay_width=0,
    )
    base.update(overrides)
    return build_prefill_registry(**base)


def _buffers(**overrides):
    base = dict(
        device=_DEVICE,
        max_bs=1,
        max_num_tokens=128,
        cache_loc_dtype=torch.int32,
        is_multimodal=True,
        hidden_size=64,
        dtype=torch.bfloat16,
        enable_mamba_track=False,
        deepstack_replay_width=0,
    )
    base.update(overrides)
    return PrefillInputBuffers.create(**base)


class TestDeepStackReplaySlotRegistration(CustomTestCase):
    def test_slot_registration_follows_the_gates(self):
        cases = [
            (dict(), False),
            (dict(deepstack_replay_width=192), True),
            (dict(deepstack_replay_width=192, is_multimodal=False), False),
            (dict(deepstack_replay_width=192, register_input_embeds=False), False),
        ]
        for overrides, expected in cases:
            with self.subTest(**overrides):
                reg = _reg(**overrides)
                self.assertEqual(reg.has_slot("input_deepstack_embeds"), expected)

    def test_slot_shape_and_dtype_match_contract(self):
        buf = _reg(deepstack_replay_width=192).get_slot("input_deepstack_embeds").buffer
        self.assertEqual(buf.shape[-1], 192)
        self.assertEqual(buf.dtype, torch.bfloat16)


class TestPrefillInputBuffersDeepStackField(CustomTestCase):
    def test_buffer_allocation_follows_the_gates(self):
        cases = [
            (dict(), False),
            (dict(deepstack_replay_width=192), True),
            (dict(deepstack_replay_width=192, is_multimodal=False), False),
        ]
        for overrides, expected in cases:
            with self.subTest(**overrides):
                buf = _buffers(**overrides).input_deepstack_embeds
                if expected:
                    self.assertEqual(buf.shape, (128, 192))
                    self.assertEqual(buf.dtype, torch.bfloat16)
                else:
                    self.assertIsNone(buf)

    def test_registry_adopts_the_buffer_tensor(self):
        # Adoption looks the slot up by name on the source; a field rename
        # silently breaks the wiring without this pin.
        buf = _buffers(deepstack_replay_width=192)
        reg = build_prefill_registry(
            device=_DEVICE,
            max_bs=1,
            max_num_token=128,
            cache_loc_dtype=torch.int32,
            is_multimodal=True,
            hidden_size=64,
            embed_dtype=torch.bfloat16,
            deepstack_replay_width=192,
            source=buf,
        )
        self.assertIs(
            reg.get_slot("input_deepstack_embeds").buffer,
            buf.input_deepstack_embeds,
        )


class TestQwen3VLCapabilityOptIn(CustomTestCase):
    def test_only_deepstack_capable_models_opt_in(self):
        from sglang.srt.models.qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration,
        )
        from sglang.srt.models.qwen3 import Qwen3ForCausalLM
        from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
        from sglang.srt.models.qwen3_vl_moe import (
            Qwen3VLMoeForConditionalGeneration,
        )

        for cls, expected in [
            (Qwen3VLForConditionalGeneration, True),
            (Qwen3VLMoeForConditionalGeneration, True),
            (Qwen2_5_VLForConditionalGeneration, False),
            (Qwen3ForCausalLM, False),
        ]:
            with self.subTest(cls=cls.__name__):
                self.assertEqual(
                    getattr(cls, "supports_bcg_deepstack_replay", False), expected
                )


class TestDeepStackReplaySlotRefresh(CustomTestCase):
    """The slot persists across requests sharing a token bucket, so each
    replay must fully define its contents."""

    NUM_TOKENS = 8
    WIDTH = 192
    DTYPE = torch.bfloat16

    def _slot(self) -> torch.Tensor:
        return torch.zeros(
            (self.NUM_TOKENS, self.WIDTH), dtype=self.DTYPE, device=_DEVICE
        )

    def _embeds(self, num_rows, value, dtype=None) -> torch.Tensor:
        return torch.full(
            (num_rows, self.WIDTH), value, dtype=dtype or self.DTYPE, device=_DEVICE
        )

    def test_no_stale_rows_survive_into_the_next_request(self):
        # The LM applies the slot with ``add_``, not through attention, so
        # uncleared rows corrupt real tokens instead of being masked out.
        slot = self._slot()
        _refresh_deepstack_replay_slot(
            slot=slot, deepstack_embeds=self._embeds(self.NUM_TOKENS, 3.0)
        )
        _refresh_deepstack_replay_slot(slot=slot, deepstack_embeds=self._embeds(3, 5.0))
        self.assertTrue(torch.all(slot[:3] == 5.0))
        self.assertTrue(torch.all(slot[3:] == 0.0))

        _refresh_deepstack_replay_slot(slot=slot, deepstack_embeds=None)
        self.assertTrue(torch.all(slot == 0.0))

    def _malformed_inputs(self):
        return [
            self._embeds(4, 1.0, dtype=torch.float32),
            torch.full((4, 1), 1.0, dtype=self.DTYPE, device=_DEVICE),
            self._embeds(self.NUM_TOKENS + 1, 1.0),
            torch.tensor(1.0, dtype=self.DTYPE, device=_DEVICE),
        ]

    def test_malformed_deepstack_fails_closed(self):
        # copy_ silently casts dtype drift and broadcasts an (n, 1) source
        # across the full width, so the explicit guard is the only check.
        for bad in self._malformed_inputs():
            with self.subTest(shape=tuple(bad.shape), dtype=bad.dtype):
                with self.assertRaises(RuntimeError):
                    _refresh_deepstack_replay_slot(
                        slot=self._slot(), deepstack_embeds=bad
                    )

    def test_fail_closed_leaves_slot_unmodified(self):
        # copy_ would have written every rejected input above (cast or
        # broadcast), so they are what prove validate-before-write.
        for bad in self._malformed_inputs():
            with self.subTest(shape=tuple(bad.shape), dtype=bad.dtype):
                slot = self._slot()
                _refresh_deepstack_replay_slot(
                    slot=slot, deepstack_embeds=self._embeds(self.NUM_TOKENS, 3.0)
                )
                with self.assertRaises(RuntimeError):
                    _refresh_deepstack_replay_slot(slot=slot, deepstack_embeds=bad)
                self.assertTrue(torch.all(slot == 3.0))

    def test_empty_tensor_clears_like_none(self):
        slot = self._slot()
        _refresh_deepstack_replay_slot(
            slot=slot, deepstack_embeds=self._embeds(self.NUM_TOKENS, 3.0)
        )
        _refresh_deepstack_replay_slot(slot=slot, deepstack_embeds=self._embeds(0, 1.0))
        self.assertTrue(torch.all(slot == 0.0))


class TestDeepStackReplaySlotWiring(CustomTestCase):
    """Drives ``_execute_body_capture`` so the slot is exercised through the
    replay closure as wired; the helper tests above cannot catch the refresh
    call or the capture binding being dropped."""

    NUM_TOKENS = 8
    WIDTH = 6
    DTYPE = torch.bfloat16

    def _runner(self, replay):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner._qwen_bcg_hc_sidechannel = False
        runner._use_draft_input_embeds = False
        runner._input_embeds_arg_idx = None
        runner.buffer_registry = build_prefill_registry(
            device=_DEVICE,
            max_bs=1,
            max_num_token=self.NUM_TOKENS,
            cache_loc_dtype=torch.int32,
            is_multimodal=True,
            hidden_size=4,
            embed_dtype=self.DTYPE,
            deepstack_replay_width=self.WIDTH,
        )
        runner._fill_input_embeds_slot = Mock()
        runner.backend = SimpleNamespace(replay=replay)
        runner.layer_model = SimpleNamespace(forward=None)
        runner._prefill_forward_context = lambda *_args, **_kwargs: nullcontext()
        return runner

    def _drive(self, runner, deepstack_embeds):
        def model_forward(ids, positions, batch, **_kwargs):
            return runner.layer_model.forward(
                ids, positions, batch, input_deepstack_embeds=deepstack_embeds
            )

        runner.model_runner = SimpleNamespace(
            pp_group=SimpleNamespace(is_first_rank=True),
            model=SimpleNamespace(forward=model_forward),
        )
        batch = SimpleNamespace(input_ids=None, positions=None, mm_input_embeds=None)
        return runner._execute_body_capture(
            batch,
            batch,
            static_num_tokens=self.NUM_TOKENS,
            raw_num_tokens=3,
            shape_key=object(),
        )

    def _slot_contents(self, runner):
        return runner.buffer_registry.get_slot("input_deepstack_embeds").buffer.clone()

    def test_replay_refreshes_the_captured_slot(self):
        seen = {}

        def replay(*_args, **_kwargs):
            seen["slot"] = self._slot_contents(runner)
            return "replayed"

        runner = self._runner(replay)
        full = torch.full((3, self.WIDTH), 3.0, dtype=self.DTYPE, device=_DEVICE)
        self.assertEqual(self._drive(runner, full), "replayed")
        self.assertTrue(torch.all(seen["slot"][:3] == 3.0))
        self.assertTrue(torch.all(seen["slot"][3:] == 0.0))

        short = torch.full((2, self.WIDTH), 5.0, dtype=self.DTYPE, device=_DEVICE)
        self._drive(runner, short)
        self.assertTrue(torch.all(seen["slot"][:2] == 5.0))
        self.assertTrue(torch.all(seen["slot"][2:] == 0.0))

        self._drive(runner, None)
        self.assertTrue(torch.all(seen["slot"] == 0.0))

    def test_malformed_input_prevents_replay(self):
        replayed = []

        def replay(*_args, **_kwargs):
            replayed.append(True)
            return "replayed"

        runner = self._runner(replay)
        bad = torch.full((3, self.WIDTH), 1.0, dtype=torch.float32, device=_DEVICE)
        with self.assertRaises(RuntimeError):
            self._drive(runner, bad)
        self.assertFalse(replayed)
        self.assertIsNone(runner.layer_model.forward)


if __name__ == "__main__":
    unittest.main()
