"""Unit tests for the optional DeepStack BCG replay slot.

Covers the allocation contract (registry gating, buffer allocation,
buffer-to-registry adoption), the model capability opt-in, and the
per-replay refresh of the slot (``_refresh_deepstack_replay_slot``).

CPU-only; the logic under test is GPU-agnostic.
"""

import unittest

import torch

from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    build_prefill_registry,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
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

    def test_malformed_deepstack_fails_closed(self):
        # Row/width mismatches are rejected by copy_ itself; dtype drift is
        # silently cast, so the guard is the only failing-closed check.
        with self.assertRaises(RuntimeError):
            _refresh_deepstack_replay_slot(
                slot=self._slot(),
                deepstack_embeds=self._embeds(4, 1.0, dtype=torch.float32),
            )

    def test_fail_closed_leaves_slot_unmodified(self):
        # A dtype-mismatched tensor is the only rejected input copy_ would
        # have modified the slot with, so it is what proves validate-before-write.
        slot = self._slot()
        _refresh_deepstack_replay_slot(
            slot=slot, deepstack_embeds=self._embeds(self.NUM_TOKENS, 3.0)
        )
        with self.assertRaises(RuntimeError):
            _refresh_deepstack_replay_slot(
                slot=slot,
                deepstack_embeds=self._embeds(4, 9.0, dtype=torch.float32),
            )
        self.assertTrue(torch.all(slot == 3.0))


if __name__ == "__main__":
    unittest.main()
