import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.multimodal_gen.configs.pipeline_configs import qwen_image as qwen_config
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageEditPlusPipelineConfig,
    QwenImageLayeredPipelineConfig,
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.qwenimage import (
    QwenImageLayeredSamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.entrypoints import diffusion_generator as dg
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.entrypoints.utils import map_request_outputs
from sglang.multimodal_gen.runtime.managers import gpu_worker
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    _qwen_modulation_cache_key,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    qwen_image_layered,
)
from sglang.test.test_utils import CustomTestCase


class TestQwenImageLayeredPipelineConfig(unittest.TestCase):
    def test_unpack_uses_layered_img_shapes_not_stale_request_size(self):
        config = QwenImageLayeredPipelineConfig()
        channels = config.dit_config.arch_config.in_channels
        generated_layers = 2
        latent_height = 40
        latent_width = 40
        latents = torch.empty(
            1,
            generated_layers * latent_height * latent_width,
            channels,
        )
        batch = SimpleNamespace(
            height=512,
            width=512,
            raw_latent_shape=latents.shape,
            img_shapes=[
                [
                    (1, latent_height, latent_width),
                    (1, latent_height, latent_width),
                    (1, latent_height, latent_width),
                ]
            ],
        )

        unpacked, batch_size, unpacked_channels, height, width = (
            config._unpad_and_unpack_latents(latents, batch)
        )

        self.assertEqual(batch_size, 1)
        self.assertEqual(unpacked_channels, channels)
        self.assertEqual((height, width), (80, 80))
        self.assertEqual(unpacked.shape, (1, channels // 4, generated_layers, 80, 80))


class TestLayeredOutputMapping(CustomTestCase):
    def _params(self, **kwargs):
        return QwenImageLayeredSamplingParams(
            prompt="layers",
            num_frames=4,
            output_path="/tmp",
            output_file_name="layers.png",
            **kwargs,
        )

    def _generate(self, params, response, save_mock=None):
        generator = object.__new__(DiffGenerator)
        generator.local_scheduler_process = []
        generator.owns_scheduler_client = False
        generator.server_args = SimpleNamespace(
            model_path="Qwen/Qwen-Image-Layered",
            prompt_file_path=None,
            warmup_mode="off",
            batching_max_size=1,
        )
        with (
            patch.object(
                SamplingParams, "from_user_sampling_params_args", return_value=params
            ),
            patch.object(
                dg,
                "prepare_request",
                side_effect=lambda **kw: Req(sampling_params=kw["sampling_params"]),
            ),
            patch.object(
                generator,
                "_send_to_scheduler_and_wait_for_response",
                return_value=response,
            ),
            patch.object(dg, "save_outputs", side_effect=save_mock),
        ):
            return generator.generate(
                {"prompt": "layers", "output_file_name": "layers.png"}
            )

    def test_all_four_saved_layers_are_returned(self):
        paths = [f"/tmp/layers_{i}.png" for i in range(4)]
        result = self._generate(
            self._params(save_output=True, return_file_paths_only=True),
            OutputBatch(output_file_paths=paths),
        )
        self.assertEqual([r.output_file_path for r in result], paths)
        self.assertEqual([r.prompt_index for r in result], [0, 1, 2, 3])

    def test_grouped_layers_keep_their_request_metrics(self):
        paths = [
            f"/tmp/layers_{draw}_{layer}.png" for draw in range(2) for layer in range(4)
        ]
        metrics = [
            SimpleNamespace(to_dict=lambda index=index: {"request": index})
            for index in range(2)
        ]
        result = self._generate(
            self._params(
                save_output=True, return_file_paths_only=True, num_outputs_per_prompt=2
            ),
            OutputBatch(output_file_paths=paths, metrics_list=metrics),
        )
        self.assertEqual([r.output_file_path for r in result], paths)
        self.assertEqual([r.metrics["request"] for r in result], [0] * 4 + [1] * 4)
        self.assertEqual([r.prompt_index for r in result], list(range(8)))

    def test_returned_samples_use_unique_layer_filenames(self):
        samples = [np.full((2, 2, 4), index, dtype=np.uint8) for index in range(4)]
        paths = []

        def save(outputs, data_type, fps, should_save, build_path, **kwargs):
            paths.extend(build_path(index) for index in range(len(outputs)))
            kwargs["samples_out"].extend(samples)
            kwargs["frames_out"].extend([None] * 4)
            kwargs["audios_out"].extend([None] * 4)

        result = self._generate(
            self._params(save_output=False, return_file_paths_only=False),
            OutputBatch(output=samples),
            save_mock=save,
        )
        self.assertEqual(paths, [f"/tmp/layers_{i}.png" for i in range(4)])
        self.assertEqual([r.output_file_path for r in result], paths)
        for actual, expected in zip(result, samples):
            self.assertIs(actual.samples, expected)

    def test_partial_layer_response_is_rejected(self):
        with self.assertLogs(dg.logger, level="ERROR") as logs:
            result = self._generate(
                self._params(save_output=True, return_file_paths_only=True),
                OutputBatch(output_file_paths=["a.png", "b.png", "c.png"]),
            )
        self.assertIsNone(result)
        self.assertTrue(
            any("Expected 4 outputs, got 3" in line for line in logs.output)
        )

    def test_video_frames_do_not_expand_into_separate_results(self):
        req = Req(
            sampling_params=SamplingParams(data_type=DataType.VIDEO, num_frames=121)
        )
        outputs = map_request_outputs([req])
        self.assertEqual(len(outputs), 1)
        self.assertIs(outputs[0].request, req)

    def test_worker_group_saves_each_layer_under_its_parent(self):
        requests = [
            Req(sampling_params=self._params(request_id=f"draw{i}")) for i in range(2)
        ]
        for index, req in enumerate(requests):
            req.output_file_name = f"draw{index}.png"
        paths = []

        def save(outputs, data_type, fps, should_save, build_path, **kwargs):
            paths.extend(build_path(index) for index in range(len(outputs)))
            return paths

        batch = OutputBatch(output=[None] * 8)
        with patch.object(gpu_worker, "save_outputs", side_effect=save):
            GPUWorker._save_group_output_paths(
                SimpleNamespace(is_output_rank=True), requests, batch
            )
        expected = [
            f"/tmp/draw{draw}_{layer}.png" for draw in range(2) for layer in range(4)
        ]
        self.assertEqual(paths, expected)
        self.assertEqual(batch.output_file_paths, expected)


class TestLayeredCFGOrder(CustomTestCase):
    def test_layered_parallel_uses_serial_formula_and_postprocess(self):
        config = QwenImageLayeredPipelineConfig()
        req = SimpleNamespace(
            do_classifier_free_guidance=True,
            true_cfg_scale=7.0,
            cfg_normalization=0,
            guidance_rescale=0,
            cfg_normalize=True,
        )
        policy = config.cfg_policy.build(req, {}, {}, {})
        self.assertTrue(policy.parallel_uses_serial_arithmetic)
        self.assertFalse(
            QwenImagePipelineConfig().cfg_policy.parallel_uses_serial_arithmetic
        )
        pos = torch.tensor([[1.0, 0.3]], dtype=torch.bfloat16)
        neg = torch.tensor([[0.1, -0.2]], dtype=torch.bfloat16)
        serial = policy.combine([pos, neg], req, 7.0, config)
        parallel = policy.combine([pos, neg], req, 7.0, config, cfg_parallel=True)
        self.assertTrue(torch.equal(serial, parallel))
        self.assertFalse(torch.equal(serial, neg + 7.0 * (pos - neg)))

    def test_dispatch_gathers_for_layered_and_preserves_legacy_fast_path(self):
        stage = DenoisingStage.__new__(DenoisingStage)
        batch = SimpleNamespace(
            do_classifier_free_guidance=True,
            cfg_normalization=0,
            guidance_rescale=0,
        )
        config = SimpleNamespace(
            get_classifier_free_guidance_scale=lambda batch, scale: scale,
            postprocess_cfg_noise=lambda batch, pred, cond: pred,
        )
        pos = torch.tensor([1.0], dtype=torch.bfloat16)
        neg = torch.tensor([0.1], dtype=torch.bfloat16)
        module = "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising"
        for same_order in [False, True]:
            with self.subTest(same_order=same_order):
                policy = CFGPolicy(parallel_uses_serial_arithmetic=same_order).build(
                    batch, {}, {}, {}
                )
                with (
                    patch(
                        f"{module}.get_classifier_free_guidance_world_size",
                        return_value=2,
                    ),
                    patch(
                        f"{module}.run_cfg_parallel", return_value=[pos, neg]
                    ) as gather,
                    patch(
                        f"{module}.run_two_branch_cfg_parallel", return_value=pos
                    ) as reduce,
                ):
                    result = stage._predict_noise_with_cfg(
                        current_model=None,
                        latent_model_input=pos,
                        timestep=torch.tensor(1),
                        batch=batch,
                        timestep_index=0,
                        attn_metadata=None,
                        target_dtype=torch.bfloat16,
                        current_guidance_scale=7.0,
                        cfg_policy=policy,
                        cfg_gate_state=None,
                        server_args=SimpleNamespace(
                            enable_cfg_parallel=True, pipeline_config=config
                        ),
                        guidance=None,
                        latents=pos,
                    )
                self.assertEqual(gather.call_count, int(same_order))
                self.assertEqual(reduce.call_count, int(not same_order))
                expected = neg + 7.0 * (pos - neg) if same_order else pos
                self.assertTrue(torch.equal(result, expected))


class TestQwenImageRequestConstants(unittest.TestCase):
    def test_cfg_branches_share_additional_t_cond(self):
        """Both branches pass one zero tensor, so the negative forward reuses
        the positive forward's modulation instead of recomputing it per block."""
        config = QwenImageLayeredPipelineConfig()
        prompt = [torch.zeros(1, 5, 8)]
        batch = Req(
            sampling_params=SamplingParams(prompt="test"),
            latents=torch.zeros(1, 4, 8),
            height=32,
            width=32,
            prompt_embeds=prompt,
            negative_prompt_embeds=prompt,
        )
        batch.img_shapes = [[(1, 1, 1), (1, 1, 1)]]
        freqs = (torch.zeros(2, 4), torch.zeros(5, 4))
        with (
            patch.object(
                QwenImageEditPlusPipelineConfig,
                "get_freqs_cis",
                return_value=(freqs, freqs),
            ),
            patch.object(qwen_config, "shard_rotary_emb_for_sp", lambda x: x),
        ):
            pos = config.prepare_pos_cond_kwargs(batch, "cpu", None, torch.bfloat16)
            neg = config.prepare_neg_cond_kwargs(batch, "cpu", None, torch.bfloat16)

        self.assertIs(pos["additional_t_cond"], neg["additional_t_cond"])
        torch.testing.assert_close(
            pos["additional_t_cond"], torch.tensor([0]), atol=0, rtol=0
        )
        timestep, hidden_states = torch.ones(1), torch.zeros(1, 4, 8)
        with torch.no_grad():
            pos_key, neg_key = (
                _qwen_modulation_cache_key(
                    timestep, kwargs["additional_t_cond"], hidden_states
                )
                for kwargs in (pos, neg)
            )
        self.assertTrue(pos_key.matches(neg_key))

    def test_vae_latent_stats_are_uploaded_once(self):
        """Edit encode and decode normalize with one upload per (device, dtype):
        a per-request upload stalls the host on the VAE encode it follows."""
        config = QwenImagePipelineConfig()
        ac = config.vae_config.arch_config
        # loaded from the checkpoint's VAE config
        ac.latents_mean = [0.25 * i - 2.0 for i in range(ac.z_dim)]
        ac.latents_std = [0.5 + 0.125 * i for i in range(ac.z_dim)]
        device = torch.device("cpu")
        first = config.get_decode_scale_and_shift(device, torch.bfloat16, None)
        second = config.get_decode_scale_and_shift(device, torch.bfloat16, None)
        std = torch.tensor(ac.latents_std).view(1, ac.z_dim, 1, 1, 1)
        mean = torch.tensor(ac.latents_mean).view(1, ac.z_dim, 1, 1, 1)
        torch.testing.assert_close(
            first[0], 1.0 / std.to(torch.bfloat16), atol=0, rtol=0
        )
        torch.testing.assert_close(first[1], mean.to(torch.bfloat16), atol=0, rtol=0)
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])

    def test_condition_latents_reuse_uploaded_latent_stats(self):
        stage = object.__new__(qwen_image_layered.QwenImageLayeredBeforeDenoisingStage)
        stage.latent_channels = 4
        stage.latent_stats = {}
        stage.vae = Mock()
        stage.vae.config.latents_mean = [0.1, -0.2, 0.3, -0.4]
        stage.vae.config.latents_std = [1.5, 2.5, 0.5, 3.0]
        stage.use_declared_component = Mock(return_value=nullcontext(stage.vae))
        latent = torch.randn(
            1, 4, 1, 2, 2, generator=torch.Generator().manual_seed(0)
        ).bfloat16()
        mean = torch.tensor(stage.vae.config.latents_mean).view(1, 4, 1, 1, 1)
        std = torch.tensor(stage.vae.config.latents_std).view(1, 4, 1, 1, 1)
        expected = (latent - mean.bfloat16()) / std.bfloat16()
        stats = []
        with (
            patch.object(qwen_image_layered, "retrieve_latents", return_value=latent),
            patch.object(
                qwen_image_layered, "align_tensor_to_module_dtype", lambda x, _: x
            ),
        ):
            for _ in range(2):
                actual = stage._encode_vae_image(torch.zeros(1, 3, 1, 4, 4), None)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                stats.append(stage.latent_stats[(latent.dtype, latent.device)])
        # the second request normalizes with the first request's device tensors
        self.assertIs(stats[0], stats[1])


if __name__ == "__main__":
    unittest.main()
