# SPDX-License-Identifier: Apache-2.0
"""Unit tests for continuous batching varlen packing and the batched solver."""

import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.managers.continuous_batching import (
    build_denoising_batch_key,
)
from sglang.multimodal_gen.runtime.pipelines_core.batched_solver import (
    SolverRejection,
    build_batched_solver,
    scheduler_is_batchable,
    sigma_table_cache,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)


class _VarlenModel:
    supports_varlen_step_packing = True
    zero_cond_t = False


def _make_state(latents, step_index=0, scheduler=None):
    return SimpleNamespace(
        denoising_context=SimpleNamespace(latents=latents, scheduler=scheduler),
        step_index=step_index,
    )


def _make_branch_kwargs(txt_len, img_cache_len, txt_cache_len, rows=1, mask=True):
    encoder = torch.randn(rows, txt_len, 16)
    kwargs = {
        "encoder_hidden_states": encoder,
        "encoder_hidden_states_mask": (
            torch.ones(rows, txt_len, dtype=torch.bool) if mask else None
        ),
        "txt_seq_lens": [txt_len] * rows,
        "img_shapes": [[(1, 2, 2)]] * rows,
        "freqs_cis": (
            torch.randn(img_cache_len, 8),
            torch.randn(txt_cache_len, 8),
        ),
    }
    return kwargs


class TestVarlenKwargs(unittest.TestCase):
    def setUp(self):
        self.stage = DenoisingStage.__new__(DenoisingStage)

    def test_pad_seq_dim(self):
        tensor = torch.ones(2, 3, 4)
        padded = DenoisingStage._pad_seq_dim(tensor, 5)
        self.assertEqual(tuple(padded.shape), (2, 5, 4))
        self.assertTrue(torch.all(padded[:, 3:] == 0))
        self.assertIs(DenoisingStage._pad_seq_dim(tensor, 3), tensor)

    def test_build_varlen_positions(self):
        positions = DenoisingStage._build_varlen_positions(
            offsets=[0, 10],
            row_lens=[3, 5],
            row_state_index=[0, 1],
            seq_len=5,
            device=torch.device("cpu"),
        )
        self.assertEqual(positions.tolist(), [0, 1, 2, 2, 2, 10, 11, 12, 13, 14])

    def test_prepare_and_inject_varlen_kwargs(self):
        states = [
            _make_state(torch.randn(1, 4, 8)),
            _make_state(torch.randn(1, 9, 8)),
        ]
        branch_kwargs_by_index = [
            [
                _make_branch_kwargs(5, img_cache_len=4, txt_cache_len=5),
                _make_branch_kwargs(7, img_cache_len=9, txt_cache_len=7, mask=False),
            ],
            [
                _make_branch_kwargs(3, img_cache_len=4, txt_cache_len=5),
                _make_branch_kwargs(6, img_cache_len=9, txt_cache_len=7),
            ],
        ]
        normalized, varlen_ctx = self.stage._prepare_varlen_branch_kwargs(
            states, branch_kwargs_by_index
        )

        self.assertEqual(varlen_ctx["txt_pad"], 7)
        self.assertEqual(varlen_ctx["img_offsets"], [0, 4])
        self.assertEqual(varlen_ctx["txt_offsets"], [0, 5])
        self.assertEqual(tuple(varlen_ctx["img_table"].shape), (13, 8))
        self.assertEqual(tuple(varlen_ctx["txt_table"].shape), (12, 8))
        self.assertEqual(varlen_ctx["txt_lens_by_branch"], [[5, 7], [3, 6]])
        for branch in normalized:
            for kwargs in branch:
                self.assertNotIn("freqs_cis", kwargs)
                self.assertEqual(kwargs["encoder_hidden_states"].shape[1], 7)
                self.assertEqual(kwargs["encoder_hidden_states_mask"].shape[1], 7)
        # Original kwargs are untouched.
        self.assertIn("freqs_cis", branch_kwargs_by_index[0][0])

        merged = tuple(
            self.stage._merge_step_input_kwargs(branch) for branch in normalized
        )
        folded = self.stage._fold_branch_kwargs_along_batch(merged)
        self.stage._inject_varlen_kwargs(
            merged,
            folded,
            varlen_ctx,
            row_seq_lens=[4, 9],
            row_state_index=[0, 1],
            padded_rows=2,
            device=torch.device("cpu"),
        )

        for kwargs in merged:
            self.assertEqual(kwargs["image_seq_lens"], [4, 9])
            self.assertEqual(
                kwargs["attention_kwargs"]["img_rope_positions"].numel(), 2 * 9
            )
            self.assertEqual(
                kwargs["attention_kwargs"]["txt_rope_positions"].numel(), 2 * 7
            )
        # Row 0 image positions clamp at its own cache end (offset 0, len 4).
        img_positions = merged[0]["attention_kwargs"]["img_rope_positions"]
        self.assertEqual(img_positions[:9].max().item(), 3)
        self.assertEqual(img_positions[9:].tolist(), [4 + i for i in range(9)])
        # Branch txt positions clamp by that branch's per-row lengths.
        txt_positions = merged[1]["attention_kwargs"]["txt_rope_positions"]
        self.assertEqual(txt_positions[:7].max().item(), 2)

        self.assertEqual(folded["image_seq_lens"], [4, 9, 4, 9])
        self.assertEqual(
            folded["attention_kwargs"]["img_rope_positions"].numel(), 4 * 9
        )
        self.assertEqual(
            folded["attention_kwargs"]["txt_rope_positions"].numel(), 4 * 7
        )
        self.assertEqual(folded["encoder_hidden_states"].shape, (4, 7, 16))


class TestBatchedSolver(unittest.TestCase):
    def _make_scheduler(self, num_inference_steps):
        try:
            from diffusers import FlowMatchEulerDiscreteScheduler
        except ImportError:
            self.skipTest("diffusers is not installed")

        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        return scheduler

    def test_step_rows_matches_scheduler_step(self):
        scheduler_a = self._make_scheduler(6)
        scheduler_b = self._make_scheduler(4)
        latents_a = torch.randn(1, 4, 8)
        latents_b = torch.randn(1, 9, 8)
        state_a = _make_state(latents_a, step_index=2, scheduler=scheduler_a)
        state_b = _make_state(latents_b, step_index=0, scheduler=scheduler_b)

        solver = build_batched_solver([state_a, state_b], torch.device("cpu"))
        self.assertIsNotNone(solver)

        packed = torch.cat(
            [
                DenoisingStage._pad_seq_dim(latents_a, 9),
                DenoisingStage._pad_seq_dim(latents_b, 9),
            ],
            dim=0,
        )
        noise = torch.randn_like(packed)
        new_packed = solver.step_rows(noise, packed, [state_a, state_b])

        ref_scheduler_a = self._make_scheduler(6)
        ref_scheduler_a._step_index = 2
        expected_a = ref_scheduler_a.step(
            model_output=noise[0:1, :4],
            timestep=ref_scheduler_a.timesteps[2],
            sample=latents_a,
            return_dict=False,
        )[0]
        ref_scheduler_b = self._make_scheduler(4)
        ref_scheduler_b._step_index = 0
        expected_b = ref_scheduler_b.step(
            model_output=noise[1:2, :9],
            timestep=ref_scheduler_b.timesteps[0],
            sample=latents_b,
            return_dict=False,
        )[0]

        torch.testing.assert_close(new_packed[0:1, :4], expected_a)
        torch.testing.assert_close(new_packed[1:2, :9], expected_b)
        # Host-side counters mirror the per-request scheduler.step behavior.
        self.assertEqual(scheduler_a._step_index, 3)
        self.assertEqual(scheduler_b._step_index, 1)

    def test_build_rejects_unknown_scheduler(self):
        scheduler = SimpleNamespace(sigmas=torch.linspace(1, 0, 5), order=1)
        state = _make_state(torch.randn(1, 4, 8), scheduler=scheduler)
        self.assertFalse(scheduler_is_batchable(scheduler))
        with self.assertRaises(SolverRejection):
            build_batched_solver([state], torch.device("cpu"))

    def test_scheduler_batchable_accepts_flow_match_euler(self):
        scheduler = self._make_scheduler(4)
        self.assertTrue(scheduler_is_batchable(scheduler))

    def test_sigma_table_cache_reuses_rows(self):
        scheduler = self._make_scheduler(6)
        latents = torch.randn(1, 4, 8)
        sigma_table_cache.clear()
        state = _make_state(latents, step_index=0, scheduler=scheduler)
        build_batched_solver([state], torch.device("cpu"))
        self.assertEqual(sigma_table_cache.misses, 1)
        build_batched_solver([state], torch.device("cpu"))
        self.assertEqual(sigma_table_cache.hits, 1)
        self.assertEqual(sigma_table_cache.misses, 1)

    def test_repeated_steps_to_terminal_match_scheduler(self):
        """Drive both members to their terminal step and compare each update."""
        num_steps_a, num_steps_b = 4, 6
        scheduler_a = self._make_scheduler(num_steps_a)
        scheduler_b = self._make_scheduler(num_steps_b)
        ref_a = self._make_scheduler(num_steps_a)
        ref_b = self._make_scheduler(num_steps_b)

        latents_a = torch.randn(1, 4, 8)
        latents_b = torch.randn(1, 4, 8)
        ref_latents_a = latents_a.clone()
        ref_latents_b = latents_b.clone()
        state_a = _make_state(latents_a, step_index=2, scheduler=scheduler_a)
        state_b = _make_state(latents_b, step_index=0, scheduler=scheduler_b)
        ref_a._step_index = 2
        ref_b._step_index = 0

        solver = build_batched_solver([state_a, state_b], torch.device("cpu"))
        packed = torch.cat([latents_a, latents_b], dim=0)

        # Step until member A hits its terminal step (indices 2..3).
        for offset in range(num_steps_a - 2):
            noise = torch.randn_like(packed)
            packed = solver.step_rows(noise, packed, [state_a, state_b])

            ref_latents_a = ref_a.step(
                model_output=noise[0:1],
                timestep=ref_a.timesteps[2 + offset],
                sample=ref_latents_a,
                return_dict=False,
            )[0]
            ref_latents_b = ref_b.step(
                model_output=noise[1:2],
                timestep=ref_b.timesteps[offset],
                sample=ref_latents_b,
                return_dict=False,
            )[0]
            state_a.step_index += 1
            state_b.step_index += 1

            torch.testing.assert_close(packed[0:1], ref_latents_a)
            torch.testing.assert_close(packed[1:2], ref_latents_b)

        # Member A is finished; scheduler counters stayed mirrored throughout.
        self.assertEqual(scheduler_a._step_index, num_steps_a)
        self.assertEqual(scheduler_b._step_index, 2)


class TestVarlenBatchKey(unittest.TestCase):
    def _make_key_state(self, seq_len):
        req = SimpleNamespace(
            raw_latent_shape=(1, 1, 16, seq_len, 8),
            image_latent=None,
            do_classifier_free_guidance=True,
            enable_sequence_shard=None,
            did_sp_shard_latents=False,
            sp_video_start_frame=0,
        )
        ctx = SimpleNamespace(
            latents=torch.randn(1, seq_len, 8),
            scheduler=SimpleNamespace(order=1),
            cfg_policy=None,
            target_dtype=torch.bfloat16,
        )
        step = SimpleNamespace(
            t_device=torch.tensor(1.0),
            current_model=_VarlenModel(),
            attn_metadata=None,
        )
        return SimpleNamespace(req=req, denoising_context=ctx, current_step=step)

    def _make_server_args(self, varlen):
        pipeline_config = SimpleNamespace(supports_varlen_step_packing=True)
        return SimpleNamespace(
            pipeline_config=pipeline_config,
            cb_varlen_packing=varlen,
            sp_degree=1,
            ulysses_degree=1,
            ring_degree=1,
            tp_size=1,
            cfg_parallel_degree=1,
            enable_cfg_parallel=False,
        )

    def test_varlen_key_groups_resolutions(self):
        stage = SimpleNamespace(attn_backend=None)
        server_args = self._make_server_args(varlen=True)
        key_small = build_denoising_batch_key(
            self._make_key_state(16), stage, server_args
        )
        key_large = build_denoising_batch_key(
            self._make_key_state(64), stage, server_args
        )
        self.assertTrue(key_small.varlen_packed)
        self.assertEqual(key_small, key_large)

    def test_key_separates_resolutions_without_varlen(self):
        stage = SimpleNamespace(attn_backend=None)
        server_args = self._make_server_args(varlen=False)
        key_small = build_denoising_batch_key(
            self._make_key_state(16), stage, server_args
        )
        key_large = build_denoising_batch_key(
            self._make_key_state(64), stage, server_args
        )
        self.assertFalse(key_small.varlen_packed)
        self.assertNotEqual(key_small, key_large)


if __name__ == "__main__":
    unittest.main()
