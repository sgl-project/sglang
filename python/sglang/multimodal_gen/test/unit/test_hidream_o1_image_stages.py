"""Derived properties of the HiDream-O1-Image stages.

The mrope grid, the hybrid attention mask and the clean-image -> velocity
reordering are closed forms derived from the reference implementation rather
than direct ports, so each one is pinned here against the property it was
derived from.
"""

import unittest
from types import SimpleNamespace

import torch
from einops import rearrange

from sglang.multimodal_gen.configs.models.dits.hidream_o1_image import (
    HiDreamO1ImageDitConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.hidream_o1_image import (
    HiDreamO1ImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.hidream_o1_image import (
    HiDreamO1ImageSamplingParams,
)
from sglang.multimodal_gen.registry import _get_config_info, get_model_info
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGBranch, CFGPolicy
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines.hidream_o1_image import (
    HIDREAM_O1_FLOW_SHIFT,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hidream_o1_image import (
    HIDREAM_O1_MROPE_FIX_POINT,
    HIDREAM_O1_NOISE_SCALE,
    HIDREAM_O1_T_EPS,
    _resolve_single_prompt,
    build_hidream_o1_attention_mask,
    build_hidream_o1_position_ids,
    patchify_pixels,
    unpatchify_pixels,
)

PATCH_SIZE = 32
CPU = torch.device("cpu")


class TestHiDreamO1PositionIds(unittest.TestCase):
    def test_text_axes_count_up_from_zero(self) -> None:
        position_ids = build_hidream_o1_position_ids(
            text_len=7, h_patches=2, w_patches=3, device=CPU
        )
        expected = torch.arange(7).view(1, 1, -1).expand(3, 1, -1)
        self.assertTrue(torch.equal(position_ids[:, :, :7], expected))

    def test_image_grid_is_fix_point_plus_row_col(self) -> None:
        """Image tokens restart at the fix point as (t, row, col), row-major."""
        h_patches, w_patches = 2, 3
        position_ids = build_hidream_o1_position_ids(
            text_len=7, h_patches=h_patches, w_patches=w_patches, device=CPU
        )
        image = position_ids[:, 0, 7:] - HIDREAM_O1_MROPE_FIX_POINT
        rows = torch.arange(h_patches).repeat_interleave(w_patches)
        cols = torch.arange(w_patches).repeat(h_patches)

        self.assertTrue(torch.equal(image[0], torch.zeros(h_patches * w_patches)))
        self.assertTrue(torch.equal(image[1], rows))
        self.assertTrue(torch.equal(image[2], cols))

    def test_image_grid_is_prompt_length_invariant(self) -> None:
        """The "fix point" is what keeps the grid from shifting with prompt length.

        A regression that continued the text positions instead would still have
        the right shape and monotonic order, so only this comparison catches it.
        """
        short = build_hidream_o1_position_ids(
            text_len=5, h_patches=4, w_patches=4, device=CPU
        )
        long = build_hidream_o1_position_ids(
            text_len=61, h_patches=4, w_patches=4, device=CPU
        )
        self.assertTrue(torch.equal(short[:, 0, 5:], long[:, 0, 61:]))


class TestHiDreamO1AttentionMask(unittest.TestCase):
    def _visible(self, text_len: int, image_len: int) -> torch.Tensor:
        mask = build_hidream_o1_attention_mask(
            text_len=text_len, image_len=image_len, dtype=torch.float32, device=CPU
        )
        self.assertEqual(mask.shape, (1, 1, text_len + image_len, text_len + image_len))
        return mask[0, 0] == 0

    def test_prompt_rows_stay_causal(self) -> None:
        visible = self._visible(text_len=6, image_len=4)
        for row in range(5):
            self.assertTrue(bool(visible[row, : row + 1].all()), row)
            self.assertFalse(bool(visible[row, row + 1 :].any()), row)

    def test_last_prompt_row_and_image_rows_see_everything(self) -> None:
        """The timestep token is the last prompt token and joins the image span."""
        text_len, image_len = 6, 4
        visible = self._visible(text_len=text_len, image_len=image_len)
        self.assertTrue(bool(visible[text_len - 1 :].all()))

    def test_row_before_the_timestep_token_is_still_causal(self) -> None:
        """Guards the off-by-one in the full-attention span start."""
        text_len = 6
        visible = self._visible(text_len=text_len, image_len=4)
        self.assertFalse(bool(visible[text_len - 2, text_len - 1 :].any()))


class TestHiDreamO1Patchify(unittest.TestCase):
    def test_patchify_matches_the_reference_rearrangement(self) -> None:
        """Patch order is row-major over (H W) with (C p1 p2) inside."""
        pixels = torch.arange(2 * 3 * 64 * 96, dtype=torch.float32).reshape(
            2, 3, 64, 96
        )
        expected = rearrange(
            pixels,
            "B C (H p1) (W p2) -> B (H W) (C p1 p2)",
            p1=PATCH_SIZE,
            p2=PATCH_SIZE,
        )
        self.assertTrue(torch.equal(patchify_pixels(pixels, PATCH_SIZE), expected))

    def test_unpatchify_inverts_patchify(self) -> None:
        pixels = torch.randn(2, 3, 64, 96)
        patches = patchify_pixels(pixels, PATCH_SIZE)
        restored = unpatchify_pixels(
            patches, channels=3, patch_size=PATCH_SIZE, h_patches=2, w_patches=3
        )
        self.assertTrue(torch.equal(restored, pixels))


class TestHiDreamO1Noise(unittest.TestCase):
    def test_seed_offset_is_one(self) -> None:
        """The reference draws noise from ``seed + 1`` on a CPU generator.

        Dropping the offset would still produce a valid image, just a different
        one than every published sample, so nothing else would flag it.
        """
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hidream_o1_image import (
            HiDreamO1ImageBeforeDenoisingStage,
        )

        stage = HiDreamO1ImageBeforeDenoisingStage.__new__(
            HiDreamO1ImageBeforeDenoisingStage
        )
        latents = stage._prepare_latents(
            seeds=[41],
            height=64,
            width=64,
            patch_size=PATCH_SIZE,
            channels=3,
            dtype=torch.float32,
            device=CPU,
        )

        def reference(seed: int) -> torch.Tensor:
            generator = torch.Generator("cpu").manual_seed(seed)
            noise = HIDREAM_O1_NOISE_SCALE * torch.randn(
                (1, 3, 64, 64), generator=generator
            )
            return patchify_pixels(noise, PATCH_SIZE)

        self.assertTrue(torch.equal(latents, reference(42)))
        self.assertFalse(torch.equal(latents, reference(41)))


class TestHiDreamO1VelocityReorder(unittest.TestCase):
    """The denoising override guides ``x_pred`` and converts to velocity after.

    The reference guides the velocity of each branch instead. The two agree only
    because ``v = (x_pred - sample) / sigma`` is affine and both branches share
    ``sample`` and ``sigma``; if a future CFG formula stops being affine in the
    predictions, these cases go red.
    """

    def _combine(self, pos, neg, *, scale: float, cfg_parallel: bool):
        policy = CFGPolicy(
            branches=[
                CFGBranch("conditional", True, {}),
                CFGBranch("unconditional", False, {}),
            ]
        )
        batch = SimpleNamespace(cfg_normalization=0, guidance_rescale=0.0)
        return policy.combine(
            [pos, neg],
            batch,
            scale,
            HiDreamO1ImagePipelineConfig(),
            cfg_parallel=cfg_parallel,
        )

    def _assert_equivalent(self, *, cfg_parallel: bool) -> None:
        torch.manual_seed(0)
        sample = torch.randn(2, 8, 12, dtype=torch.float64)
        x_pos = torch.randn(2, 8, 12, dtype=torch.float64)
        x_neg = torch.randn(2, 8, 12, dtype=torch.float64)
        sigma = torch.tensor(0.37, dtype=torch.float64)
        scale = 5.0

        v_pos = (x_pos - sample) / sigma
        v_neg = (x_neg - sample) / sigma
        reference = -self._combine(v_pos, v_neg, scale=scale, cfg_parallel=cfg_parallel)

        x_guided = self._combine(x_pos, x_neg, scale=scale, cfg_parallel=cfg_parallel)
        ours = (sample - x_guided) / sigma

        torch.testing.assert_close(ours, reference)

    def test_serial_cfg_is_equivalent(self) -> None:
        self._assert_equivalent(cfg_parallel=False)

    def test_cfg_parallel_is_equivalent(self) -> None:
        self._assert_equivalent(cfg_parallel=True)


class TestHiDreamO1SchedulerContract(unittest.TestCase):
    """Pins the scheduler-side half of the clean-image -> velocity conversion.

    ``_run_denoising_step`` hands the scheduler ``(sample - x_pred) / sigma`` on
    the premise that ``convert_model_output`` is its exact inverse. That premise
    lives in FlowUniPCMultistepScheduler, so a diffusers-side change to the
    flow_prediction branch is what this guards -- not the stage itself.
    """

    def test_x_pred_survives_the_velocity_round_trip(self) -> None:
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=HIDREAM_O1_FLOW_SHIFT,
            prediction_type="flow_prediction",
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(50, device=CPU)

        torch.manual_seed(0)
        sample = torch.randn(1, 4, 12)
        x_pred = torch.randn(1, 4, 12)

        for step_index in range(len(scheduler.timesteps)):
            with self.subTest(step_index=step_index):
                scheduler._step_index = step_index
                sigma = scheduler.sigmas[step_index].float()
                model_output = (sample - x_pred) / sigma.clamp_min(HIDREAM_O1_T_EPS)
                recovered = scheduler.convert_model_output(model_output, sample=sample)
                torch.testing.assert_close(recovered, x_pred)


class TestHiDreamO1Config(unittest.TestCase):
    def test_checkpoint_literals(self) -> None:
        """Values dictated by the released checkpoint, not by this repo."""
        arch_config = HiDreamO1ImageDitConfig().arch_config
        self.assertEqual(arch_config.patch_size, 32)
        self.assertEqual(arch_config.in_channels, 3)
        self.assertEqual(arch_config.tms_token_id, 151673)

    def test_mrope_section_must_fit_the_interleaved_layout(self) -> None:
        """An h/w section too wide for its stride would be silently truncated."""
        arch_config = HiDreamO1ImageDitConfig().arch_config
        arch_config.__post_init__()  # the shipped (24, 20, 20) is accepted

        arch_config.mrope_section = (20, 24, 20)  # same sum, h runs off the end
        with self.assertRaises(ValueError):
            arch_config.__post_init__()

    def test_generator_device_is_the_host(self) -> None:
        """Seeded noise must reproduce across accelerators, so seeds stay on CPU."""
        self.assertEqual(HiDreamO1ImagePipelineConfig().generator_device, "cpu")

    def test_sequence_parallelism_is_refused(self) -> None:
        config = HiDreamO1ImagePipelineConfig()
        with self.assertRaisesRegex(ValueError, "cannot be sharded"):
            config.validate_server_args(SimpleNamespace(sp_degree=2))

    def test_multi_prompt_is_refused(self) -> None:
        self.assertEqual(_resolve_single_prompt("a cat", field_name="prompt"), "a cat")
        self.assertEqual(
            _resolve_single_prompt(["a cat"], field_name="prompt"), "a cat"
        )
        with self.assertRaises(ValueError):
            _resolve_single_prompt(["a cat", "a dog"], field_name="prompt")


class TestHiDreamO1Registry(unittest.TestCase):
    """The checkpoint has no model_index.json, so resolution is registry-only.

    A typo in either registry entry falls through to the diffusers loader
    instead of raising, and the pattern key must be lowercase because the
    lookup lowercases the path before comparing.
    """

    def _assert_resolves(self, model_path: str) -> None:
        get_model_info.cache_clear()
        _get_config_info.cache_clear()

        info = get_model_info(model_path, backend="sglang")

        self.assertEqual(info.pipeline_cls.__name__, "HiDreamO1ImagePipeline")
        self.assertIs(info.pipeline_config_cls, HiDreamO1ImagePipelineConfig)
        self.assertIs(info.sampling_param_cls, HiDreamO1ImageSamplingParams)

    def test_registry_resolves_the_hf_model_id(self) -> None:
        self._assert_resolves("HiDream-ai/HiDream-O1-Image")

    def test_registry_resolves_a_local_directory(self) -> None:
        self._assert_resolves("/data/models/HiDream-O1-Image")

    def test_registry_resolves_an_hf_cache_snapshot(self) -> None:
        self._assert_resolves(
            "/root/.cache/huggingface/hub/"
            "models--HiDream-ai--HiDream-O1-Image/snapshots/abc123"
        )


if __name__ == "__main__":
    unittest.main()
