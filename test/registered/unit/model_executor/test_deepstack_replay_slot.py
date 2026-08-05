"""Unit tests for the optional DeepStack BCG replay slot.

Covers the three-site contract:

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

All tests are CPU-only — the registry / dataclass logic under test is
GPU-agnostic; ``torch.zeros`` on the ``cpu`` device is sufficient.
"""

import unittest

import torch

from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    build_prefill_registry,
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


if __name__ == "__main__":
    unittest.main()
