"""Unit tests for the optional DeepStack BCG replay slot.

Covers the three-site allocation contract:

  * ``build_prefill_registry`` registers an ``input_deepstack_embeds``
    slot iff ``is_multimodal AND register_input_embeds AND
    deepstack_replay_width > 0``.
  * ``PrefillInputBuffers.create`` allocates the matching
    ``input_deepstack_embeds`` field iff both ``is_multimodal`` and
    ``deepstack_replay_width > 0`` hold.
  * ``Qwen3VLForConditionalGeneration`` declares the explicit opt-in
    class attribute ``supports_bcg_deepstack_replay = True`` and its
    MoE subclass inherits it; ``Qwen2_5_VLForConditionalGeneration``
    and text-only ``Qwen3ForCausalLM`` do not.

and the per-replay refresh of the allocated slot
(``_refresh_deepstack_replay_slot``), which is where a stale or
malformed contribution would reach the captured graph.

All tests are CPU-only — the logic under test is GPU-agnostic;
``torch.zeros`` on the ``cpu`` device is sufficient.
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

    def test_slot_absent_when_any_gate_is_off(self):
        """The DeepStack slot is registered only when all three gates
        pass: is_multimodal, register_input_embeds, and
        deepstack_replay_width > 0. Any missing gate → no slot."""
        cases = [
            dict(is_multimodal=False, deepstack_replay_width=192),
            dict(is_multimodal=True, deepstack_replay_width=0),
            dict(
                is_multimodal=True,
                deepstack_replay_width=192,
                register_input_embeds=False,
            ),
        ]
        for kwargs in cases:
            with self.subTest(**kwargs):
                reg = _reg(**kwargs)
                self.assertFalse(reg.has_slot("input_deepstack_embeds"))

    def test_slot_present_when_all_gates_pass(self):
        reg = _reg(is_multimodal=True, deepstack_replay_width=192)
        self.assertTrue(reg.has_slot("input_deepstack_embeds"))

    def test_slot_shape_and_dtype_match_contract(self):
        reg = _reg(
            is_multimodal=True,
            hidden_size=64,
            embed_dtype=torch.bfloat16,
            deepstack_replay_width=192,
        )
        buf = reg.get_slot("input_deepstack_embeds").buffer
        self.assertEqual(buf.shape[-1], 192)
        self.assertEqual(buf.dtype, torch.bfloat16)


class TestPrefillInputBuffersDeepStackField(CustomTestCase):

    def test_buffer_none_when_width_zero(self):
        buf = _buffers(is_multimodal=True, deepstack_replay_width=0)
        self.assertIsNone(buf.input_deepstack_embeds)

    def test_buffer_none_when_not_multimodal(self):
        buf = _buffers(is_multimodal=False, deepstack_replay_width=192)
        self.assertIsNone(buf.input_deepstack_embeds)

    def test_buffer_allocated_when_multimodal_and_width_positive(self):
        buf = _buffers(
            is_multimodal=True,
            hidden_size=64,
            dtype=torch.bfloat16,
            deepstack_replay_width=192,
            max_num_tokens=128,
        )
        self.assertIsNotNone(buf.input_deepstack_embeds)
        self.assertEqual(buf.input_deepstack_embeds.shape, (128, 192))
        self.assertEqual(buf.input_deepstack_embeds.dtype, torch.bfloat16)

    def test_buffer_field_backs_registry_adoption(self):
        """The registry uses ``getattr(source, slot.name, None)`` when
        adopting; the buffer field must exist and be a tensor when the
        slot is registered so adoption succeeds."""
        buf = _buffers(is_multimodal=True, deepstack_replay_width=192)
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
        adopted = reg.get_slot("input_deepstack_embeds").buffer
        self.assertIs(adopted, buf.input_deepstack_embeds)


class TestQwen3VLCapabilityOptIn(CustomTestCase):

    def test_qwen3vl_declares_capability(self):
        from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration

        self.assertTrue(
            getattr(
                Qwen3VLForConditionalGeneration,
                "supports_bcg_deepstack_replay",
                False,
            )
        )

    def test_qwen3vl_moe_inherits_capability(self):
        from sglang.srt.models.qwen3_vl_moe import (
            Qwen3VLMoeForConditionalGeneration,
        )

        self.assertTrue(
            getattr(
                Qwen3VLMoeForConditionalGeneration,
                "supports_bcg_deepstack_replay",
                False,
            )
        )

    def test_qwen2_5_vl_does_not_declare_capability(self):
        from sglang.srt.models.qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration,
        )

        self.assertFalse(
            getattr(
                Qwen2_5_VLForConditionalGeneration,
                "supports_bcg_deepstack_replay",
                False,
            )
        )

    def test_text_only_qwen3_does_not_declare_capability(self):
        from sglang.srt.models.qwen3 import Qwen3ForCausalLM

        self.assertFalse(
            getattr(Qwen3ForCausalLM, "supports_bcg_deepstack_replay", False)
        )


class TestDeepStackReplaySlotRefresh(CustomTestCase):
    """Per-replay refresh of the stable slot.

    The slot is a persistent buffer shared by every request that lands in
    the same token bucket, so each replay must fully define its contents.
    These cases guard the two ways that breaks: rows surviving into a
    later request, and a malformed contribution being swapped for zeros
    and served as if the request had carried no DeepStack at all.
    """

    NUM_TOKENS = 8
    WIDTH = 192
    DTYPE = torch.bfloat16

    def _slot(self) -> torch.Tensor:
        return torch.zeros(
            (self.NUM_TOKENS, self.WIDTH), dtype=self.DTYPE, device=_DEVICE
        )

    def _embeds(self, num_rows, value, width=None, dtype=None) -> torch.Tensor:
        return torch.full(
            (num_rows, width or self.WIDTH),
            value,
            dtype=dtype or self.DTYPE,
            device=_DEVICE,
        )

    def test_shorter_request_does_not_inherit_stale_rows(self):
        """A shorter request reusing the bucket sees its own rows followed
        by zeros. The LM applies the slot with ``add_`` rather than through
        attention, so an uncleared tail would corrupt its real tokens."""
        slot = self._slot()
        _refresh_deepstack_replay_slot(
            slot=slot, deepstack_embeds=self._embeds(self.NUM_TOKENS, 3.0)
        )
        _refresh_deepstack_replay_slot(slot=slot, deepstack_embeds=self._embeds(3, 5.0))
        self.assertTrue(torch.all(slot[:3] == 5.0))
        self.assertTrue(torch.all(slot[3:] == 0.0))

    def test_absent_deepstack_clears_stale_rows(self):
        """A text-only request carries no DeepStack, so the slot is cleared
        instead of left holding the previous image request's rows."""
        slot = self._slot()
        _refresh_deepstack_replay_slot(
            slot=slot, deepstack_embeds=self._embeds(self.NUM_TOKENS, 3.0)
        )
        _refresh_deepstack_replay_slot(slot=slot, deepstack_embeds=None)
        self.assertTrue(torch.all(slot == 0.0))

    def test_empty_deepstack_clears_stale_rows(self):
        """An empty tensor states absence rather than a malformed
        contribution, so it clears the slot instead of failing closed."""
        slot = self._slot()
        _refresh_deepstack_replay_slot(
            slot=slot, deepstack_embeds=self._embeds(self.NUM_TOKENS, 3.0)
        )
        _refresh_deepstack_replay_slot(slot=slot, deepstack_embeds=self._embeds(0, 0.0))
        self.assertTrue(torch.all(slot == 0.0))

    def test_malformed_deepstack_fails_closed(self):
        """A non-empty contribution that does not fit the captured slot
        raises. Zeroing it and replaying anyway would answer the request
        with zero DeepStack while looking successful — the silent
        corruption this slot exists to prevent."""
        cases = {
            "narrower than the slot": self._embeds(4, 1.0, width=self.WIDTH // 2),
            "dtype the graph cannot read": self._embeds(4, 1.0, dtype=torch.float32),
            "more rows than the bucket": self._embeds(self.NUM_TOKENS + 1, 1.0),
        }
        for name, embeds in cases.items():
            with self.subTest(name):
                with self.assertRaises(RuntimeError):
                    _refresh_deepstack_replay_slot(
                        slot=self._slot(), deepstack_embeds=embeds
                    )

    def test_fail_closed_leaves_slot_unmodified(self):
        """Validation precedes every write, so a rejected contribution
        cannot leave the slot half-updated for a caller that catches the
        error and replays."""
        slot = self._slot()
        _refresh_deepstack_replay_slot(
            slot=slot, deepstack_embeds=self._embeds(self.NUM_TOKENS, 3.0)
        )
        with self.assertRaises(RuntimeError):
            _refresh_deepstack_replay_slot(
                slot=slot,
                deepstack_embeds=self._embeds(4, 9.0, width=self.WIDTH // 2),
            )
        self.assertTrue(torch.all(slot == 3.0))


if __name__ == "__main__":
    unittest.main()
