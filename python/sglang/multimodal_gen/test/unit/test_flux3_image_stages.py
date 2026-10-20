# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context,
)
from sglang.multimodal_gen.runtime.models.dits.flux3_image import TEXT, pack
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.flux3_image import (
    Flux3ImageDenoiseStage,
    build_timesteps,
    crop_output,
    denoise_flow_map,
    denoise_packed,
    latent_ids,
    load_references,
    noise_tokens,
    output_size,
    scaled_ids,
    validate_generation,
)


def _pack(image, texts):
    return pack(
        [{"image": image, "image_cond": None, TEXT: text} for text in texts],
        {"image": 2, "image_cond": 2, TEXT: 3},
    )


def _text_mean_velocity(packed):
    """Each sample's image velocity is the mean of its own text tokens."""
    return {
        "image": torch.cat(
            [
                torch.full((end - start, 2), packed.data[TEXT][lo:hi].mean())
                for (start, end), (lo, hi) in zip(
                    packed.mode_indices["image"], packed.mode_indices[TEXT]
                )
            ]
        )
    }


class TestFlux3PackedConditioning(unittest.TestCase):
    def test_image_ids_use_zero_time_and_sequence_axes(self):
        ids = latent_ids(2, 3)
        self.assertTrue(torch.equal(ids[:, 0], torch.zeros(6)))
        self.assertTrue(torch.equal(ids[:, 3], torch.zeros(6)))
        scaled = scaled_ids(ids, 0.5)
        torch.testing.assert_close(scaled[:, 1:3], ids[:, 1:3].float() * 0.5)

    def test_i2i_without_an_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires image_path"):
            validate_generation("i2i", prompt="prompt", image_path=None)

    def test_references_follow_the_reference_preprocessing(self):
        """BFL's rules: one reference is fitted under 4 MP and reflect-padded to
        the 16-px grid, sets the output size, and is cropped back out; several
        are each fitted under 1 MP and centre-cropped; at most 10."""
        rng = np.random.default_rng(0)
        odd = Image.fromarray(rng.integers(0, 255, (1001, 1503, 3), dtype=np.uint8))
        (tensor,), size, crop = load_references([odd])
        self.assertEqual(size, (1504, 1008))
        self.assertEqual(tuple(tensor.shape), (3, 1008, 1504))
        self.assertEqual(crop, (0, 3, 1503, 1004))
        torch.testing.assert_close(
            crop_output(tensor, crop),
            torch.from_numpy(np.asarray(odd)).permute(2, 0, 1).float() / 127.5 - 1,
        )
        # Above 4 MP the canvas shrinks under the limit, padding included.
        _, size, _ = load_references([Image.new("RGB", (4000, 3000))])
        self.assertLessEqual(size[0] * size[1], 4 * 1024 * 1024)
        self.assertEqual((size[0] % 16, size[1] % 16), (0, 0))

        tensors, size, crop = load_references(
            [Image.new("RGB", (2000, 1000)), Image.new("RGB", (500, 333))]
        )
        self.assertEqual(
            [tuple(t.shape) for t in tensors], [(3, 720, 1440), (3, 320, 496)]
        )
        self.assertEqual((size, crop), ((1440, 720), None))
        with self.assertRaisesRegex(ValueError, "1 to 10"):
            load_references([Image.new("RGB", (64, 64))] * 11)

        # Unset sides take the first reference's size; the crop needs both unset.
        self.assertEqual(
            output_size(
                None, None, (1504, 1008), crop=(0, 3, 1503, 1004), default=(1024, 1024)
            ),
            (1504, 1008, (0, 3, 1503, 1004)),
        )
        self.assertEqual(
            output_size(
                1024, None, (1504, 1008), crop=(0, 3, 1503, 1004), default=(1024, 1024)
            ),
            (1024, 1008, None),
        )
        self.assertEqual(
            output_size(None, None, None, crop=None, default=(1024, 1024)),
            (1024, 1024, None),
        )


class TestFlux3PackedSampler(unittest.TestCase):
    def test_multiple_requests_keep_cfg_pairs_and_flow_map_targets_isolated(self):
        samples = [
            {
                "image": (torch.full((2, 2), float(index)), torch.zeros(2, 4)),
                "image_cond": None,
                TEXT: (torch.full((length, 3), float(value)), torch.zeros(length, 4)),
            }
            for index, length, value in ((0, 1, 0), (1, 2, 2), (0, 3, 1), (1, 4, 4))
        ]
        channels = {"image": 2, "image_cond": 2, TEXT: 3}
        schedule = torch.tensor([1.0, 0.0])
        expected = []
        for index in range(2):
            expected.append(
                denoise_packed(
                    _text_mean_velocity,
                    pack([samples[index], samples[index + 2]], channels),
                    schedule,
                    3.5,
                )
            )
        for sequential in (False, True):
            actual = denoise_packed(
                _text_mean_velocity,
                pack(samples, channels),
                schedule,
                3.5,
                sequential_cfg=sequential,
            )
            torch.testing.assert_close(actual, torch.cat(expected), rtol=0, atol=0)
        flow = pack(samples[2:], channels)
        sliced = flow.slice_samples(1, 2)
        self.assertEqual(sliced.mode_indices[TEXT], [(0, 4)])
        out = denoise_flow_map(_text_mean_velocity, flow, schedule)
        torch.testing.assert_close(
            out, torch.cat([torch.full((2, 2), -1.0), torch.full((2, 2), -3.0)])
        )
        self.assertEqual(flow.timesteps_ref.shape, (2,))

    def test_flow_map_jumps_to_the_next_time_without_cfg(self):
        """The reference flow map embeds the next schedule time as the jump
        target and steps ``x += (r - t) * u`` on the conditional sample alone;
        each step's forward context carries its step index."""
        schedule = build_timesteps(3, alpha=6.93)
        calls = []

        def flow_map(packed):
            calls.append(
                (
                    get_forward_context().current_timestep,
                    packed.timesteps["image"][0].item(),
                    packed.timesteps_ref.tolist(),
                )
            )
            return {"image": torch.ones_like(packed.data["image"])}

        packed = _pack(
            (torch.zeros(4, 2), torch.zeros(4, 4)),
            [(torch.zeros(2, 3), torch.zeros(2, 4))],
        )
        out = denoise_flow_map(flow_map, packed, schedule)
        self.assertEqual([step for step, _, _ in calls], [0, 1, 2])
        self.assertEqual(
            [ref for _, _, ref in calls], [[r] for r in schedule[1:].tolist()]
        )
        for (_, timestep, _), expected in zip(calls, schedule[:-1].tolist()):
            self.assertAlmostEqual(timestep, expected, places=2)
        # Constant velocity 1 from t=1 to 0 lands at -1.
        torch.testing.assert_close(out, torch.full((4, 2), -1.0), atol=1e-2, rtol=0)

    def test_noise_keeps_the_sampling_state_in_fp32(self):
        """The reference rounds the noise to bf16 but steps in fp32; a bf16
        state drops the small early updates (dt ~ 1e-2 near t = 1)."""
        config = SimpleNamespace(
            dit=SimpleNamespace(in_channels={"image": 128}, image_rope_scale=0.5)
        )
        (noise, _), _ = noise_tokens(64, 64, 7, config, torch.device("cpu"))
        self.assertEqual(noise.dtype, torch.float32)
        torch.testing.assert_close(noise, noise.bfloat16().float(), atol=0, rtol=0)

    def test_batched_and_sequential_denoise_match(self):
        image = (torch.full((2, 2), 2.0), torch.zeros(2, 4))
        texts = [
            (torch.zeros(1, 3), torch.zeros(1, 4)),
            (torch.ones(1, 3), torch.zeros(1, 4)),
        ]
        for sequential in (False, True):
            with self.subTest(sequential=sequential):
                packed = _pack(image, texts)
                original_text = packed.data[TEXT].clone()
                model = mock.Mock(side_effect=_text_mean_velocity)
                result = denoise_packed(
                    model,
                    packed,
                    torch.tensor([1.0, 0.0]),
                    guidance=3.5,
                    sequential_cfg=sequential,
                )
                torch.testing.assert_close(result, torch.full((2, 2), -1.5))
                self.assertTrue(torch.equal(packed.data[TEXT], original_text))
                self.assertTrue(torch.equal(packed.timesteps[TEXT], torch.zeros(2)))
                self.assertEqual(model.call_count, 2 if sequential else 1)

    def test_cache_dit_runs_only_where_one_forward_covers_a_step(self):
        """Packed CFG and the flow map are one forward per step, so cache-dit
        must use a single context (two would alternate on every step and mix
        samples). Ulysses SP and the memory-bound sequential CFG pair are not
        cacheable: both hooks decline there, so torch.compile is not deferred."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
            flux3_image as stages,
        )

        stage = Flux3ImageDenoiseStage.__new__(Flux3ImageDenoiseStage)
        batch = SimpleNamespace(
            sampling_params=SimpleNamespace(enable_cache_dit=True),
            do_classifier_free_guidance=True,
        )
        self.assertFalse(stage._cache_dit_has_separate_cfg(batch))
        with mock.patch.dict(os.environ, {"SGLANG_CACHE_DIT_ENABLED": "true"}):
            for sp_size, sequential_cfg, expected in (
                (1, False, True),
                (1, True, False),
                (2, False, False),
            ):
                with (
                    self.subTest(sp_size=sp_size, sequential_cfg=sequential_cfg),
                    mock.patch.object(
                        stages, "get_sp_world_size", return_value=sp_size
                    ),
                ):
                    stage._sequential_cfg_active = sequential_cfg
                    self.assertEqual(
                        stage._cache_dit_requested_for_batch(batch), expected
                    )
                    self.assertEqual(stage._cache_dit_requested(), sp_size == 1)


if __name__ == "__main__":
    unittest.main()
