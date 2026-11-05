"""State consumers keep aligned temporal rows as entry payloads compose."""

import unittest

import torch

from sglang.srt.mem_cache.kv.decl import (
    MHAGeometry,
    StateGeometry,
    StateReplay,
    TensorPartDecl,
    kv_layer_decls,
    with_state_replay,
)
from sglang.srt.mem_cache.kv.layout.parts import TensorEntryLayout
from sglang.srt.mem_cache.kv.layout.state import MambaSubPoolSpec, StateView
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.srt.mem_cache.layout.fused_draft import (
    DraftStateGeometry,
    DraftStateRegion,
    FusedDraftPlacement,
)
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import arena_of, setup_allocator_context

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def replay_plan(dtype=torch.float32):
    geometry = with_state_replay(
        StateGeometry(
            conv_state_shapes=((3, 3072),),
            conv_dtype=torch.bfloat16,
            temporal_state_shape=(8, 128, 128),
            temporal_dtype=dtype,
        ),
        StateReplay(
            decode=True, speculative=False, is_kda=True, cache_len=16, key_heads=8
        ),
        layer_ids=(1, 2),
        speculative_num_draft_tokens=None,
    )
    return plan_kv_storage(
        kv_layer_decls(
            full_layer_ids=(0,),
            full_geometry=MHAGeometry(
                head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.bfloat16
            ),
            state_layer_ids=(1, 2),
            state_geometry=geometry,
            checkpoint_interval=64,
        ),
        page_size=1,
    )


class TestStateEntryAlignment(CustomTestCase):
    def setUp(self):
        setup_allocator_context()
        self.addCleanup(reset_context)

    def assert_temporal_alignment(self, temporal):
        alignment = max(32, 16 * temporal.element_size())
        self.assertEqual(temporal.data_ptr() % alignment, 0)
        self.assertEqual(temporal.stride(0) * temporal.element_size() % alignment, 0)
        self.assertEqual(temporal.stride(1) * temporal.element_size() % alignment, 0)

    def test_replay_pitch_is_the_sized_state_pitch(self):
        setup_allocator_context()
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                plan = replay_plan(dtype)
                spec = plan.sub_pool("mamba").spec
                sized = size_plan(
                    plan,
                    KVSizingInputs(
                        full_tokens=32,
                        state_slots=8,
                        max_running_requests=1,
                        context_len=16,
                        alignment=1,
                    ),
                )
                expected = (
                    32 * plan.owner().spec.entry_bytes()
                    + 8 * spec.entry_bytes()
                    + plan.sink_bytes()
                )
                self.assertEqual(sized.total_bytes, expected)
                arena = arena_of(
                    total_bytes=sized.total_bytes,
                    sub_pool_specs=plan.specs(),
                    device="cpu",
                    enable_memory_saver=False,
                )
                parts = spec.part_views(arena)
                temporal = parts["temporal"]
                self.assert_temporal_alignment(temporal)
                self.assertEqual(
                    temporal.stride(1) * temporal.element_size(), spec.entry_bytes()
                )
                self.assertTrue(
                    all(stride % 16 == 0 for stride in temporal[1].stride()[:-1])
                )
                if dtype == torch.float32:
                    self.assertEqual(spec.entry_bytes(), 1_478_720)
                for slot in (1, 2):
                    temporal[:, slot].fill_(slot)
                    parts["replayssm_write_pos"][slot] = slot + 10
                self.assertTrue(torch.all(temporal[:, 1] == 1))
                self.assertTrue(torch.all(temporal[:, 2] == 2))
                self.assertEqual(parts["replayssm_write_pos"][1].item(), 11)

    def test_odd_layer_rows_and_fused_draft_keep_independent_alignment(self):
        for target_dtype, draft_dtype in (
            (torch.bfloat16, torch.float32),
            (torch.float32, torch.bfloat16),
        ):
            with self.subTest(target=target_dtype, draft=draft_dtype):
                draft = DraftStateRegion(
                    lane_num=3,
                    state=DraftStateGeometry(
                        conv_state_shapes=((1, 3),),
                        conv_dtype=torch.bfloat16,
                        temporal_state_shape=(3,),
                        temporal_dtype=draft_dtype,
                    ),
                )
                spec = MambaSubPoolSpec(
                    name="mamba",
                    layer_num=3,
                    grow_direction="up",
                    conv_state_shapes=((1, 1),),
                    conv_dtype=torch.bfloat16,
                    temporal_state_shape=(3,),
                    temporal_dtype=target_dtype,
                    extra_parts=(
                        TensorPartDecl(name="cursor", shape=(), dtype=torch.int32),
                    ),
                    draft_region=draft,
                )
                full = replay_plan().owner().spec
                arena = arena_of(
                    total_bytes=8 * spec.entry_bytes() + 8 * full.entry_bytes(),
                    sub_pool_specs=[full, spec],
                    device="cpu",
                    enable_memory_saver=False,
                    fused_draft=FusedDraftPlacement.from_counts(
                        counts={"mamba": [3]}, regions={"mamba": draft}
                    ),
                )
                target_conv, target = spec.views(arena)
                draft_conv, drafted = spec.draft_views(arena)
                for temporal in (target, drafted):
                    self.assert_temporal_alignment(temporal)
                parts = spec.part_views(arena)
                target[:, 2].fill_(3)
                drafted[:, 2].fill_(7)
                parts["cursor"][2] = 11
                target_conv[0][:, 2].fill_(13)
                draft_conv[0][:, 2].fill_(17)
                state = StateView(arena=arena, spec=spec, mamba_layer_ids=[1, 2, 3])
                state.copy_from(torch.tensor([2]), torch.tensor([3]))
                for tensor, value in (
                    (target, 3),
                    (drafted, 7),
                    (target_conv[0], 13),
                    (draft_conv[0], 17),
                ):
                    self.assertTrue(torch.all(tensor[:, 3] == value))
                self.assertEqual(parts["cursor"][3].item(), 11)

    def test_zero_temporal_and_generic_entries_keep_dtype_alignment(self):
        generic = TensorEntryLayout(
            (
                TensorPartDecl(name="value", shape=(1,), dtype=torch.bfloat16),
                TensorPartDecl(name="cursor", shape=(), dtype=torch.int32),
            )
        )
        self.assertEqual(generic.entry_bytes, 8)
        spec = MambaSubPoolSpec(
            name="mamba",
            layer_num=2,
            grow_direction="up",
            conv_state_shapes=((3, 3),),
            conv_dtype=torch.bfloat16,
            temporal_state_shape=(0, 0, 0),
            temporal_dtype=torch.float32,
        )
        self.assertEqual(spec.entry_bytes(), 36)

    def test_embedding_cannot_weaken_declared_alignment(self):
        layout = TensorEntryLayout(
            (
                TensorPartDecl(
                    name="value", shape=(3,), dtype=torch.float32, layer_ids=(0, 1)
                ),
            ),
            row_alignments={"value": 64},
        )
        raw = torch.empty(1024, dtype=torch.uint8)
        for kwargs in ({"entry_bytes": 132}, {"offset_bytes": 4}):
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(ValueError, "row alignment"),
            ):
                layout.views(raw, 2, **kwargs)
        with self.assertRaisesRegex(ValueError, "row alignment"):
            layout.views(raw[4:], 2)
        raw.zero_()
        aligned = layout.views(raw[64:], 2)["value"]
        aligned.fill_(5)
        self.assertEqual(aligned.data_ptr(), raw.data_ptr() + 64)
        self.assertTrue(torch.all(raw[:64] == 0))

    def test_cutedsl_prefill_wrapper_receives_aligned_replay_state(self):
        try:
            from sglang.srt.layers.attention.linear.kernels.kda_cutedsl import (
                CuteDSLKDAKernel,
            )
        except ImportError as error:
            self.skipTest(str(error))
        spec = replay_plan().sub_pool("mamba").spec
        raw = torch.empty(3 * spec.entry_bytes(), dtype=torch.uint8)
        state = spec.entry_layout().views(raw, 3)["temporal"][1]
        kernel = CuteDSLKDAKernel()
        calls = []

        def compute(q, k, v, g, beta, initial, cu, **kwargs):
            self.assertIs(initial, state)
            self.assertTrue(all(s % 16 == 0 for s in initial.stride()[:-1]))
            self.assertEqual(initial.data_ptr() % 32, 0)
            calls.append(initial)
            return torch.zeros_like(v), initial

        kernel._extend_fn = compute
        kernel._l2norm_fn = lambda tensor: tensor
        qkv = torch.ones((1, 2, 8, 128), dtype=torch.bfloat16)
        output, intermediate = kernel.extend(
            qkv,
            qkv,
            qkv,
            qkv.float(),
            torch.ones((1, 2, 8)),
            ssm_states=state,
            cache_indices=torch.tensor([1]),
            query_start_loc=torch.tensor([0, 2]),
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(output.shape, qkv.shape)
        self.assertIsNone(intermediate)


if __name__ == "__main__":
    unittest.main()
