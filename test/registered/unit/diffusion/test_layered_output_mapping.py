import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sglang.multimodal_gen.configs.sample.qwenimage import (
    QwenImageLayeredSamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.runtime.entrypoints import diffusion_generator as dg
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.entrypoints.utils import map_request_outputs
from sglang.multimodal_gen.runtime.managers import gpu_worker
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# The logic is CPU-only; this runner supplies the diffusion runtime dependencies.
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


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


if __name__ == "__main__":
    unittest.main()
