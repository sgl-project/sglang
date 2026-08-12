# SPDX-License-Identifier: Apache-2.0
"""Latent frames are aligned to the sequence-parallel degree, not the GPU count.

Only sequence parallelism splits the frame axis across ranks. Tensor, data and
CFG parallelism all replicate it, so padding the clip for them changes the
requested length and the sampled result for no benefit -- 49 frames under CFG
parallel used to become 57.
"""

import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2_5 import LTX25PipelineConfig
from sglang.multimodal_gen.configs.sample.ltx_2_5 import LTX25SamplingParams


class TestFrameAlignmentFollowsSequenceParallelism(unittest.TestCase):
    def _adjusted_num_frames(self, num_frames, *, num_gpus, sp_degree):
        pipeline_config = LTX25PipelineConfig()
        params = LTX25SamplingParams()
        params.num_frames = num_frames
        server_args = SimpleNamespace(
            num_gpus=num_gpus,
            sp_degree=sp_degree,
            pipeline_config=pipeline_config,
            comfyui_mode=True,
        )
        params._adjust_visual_fields(server_args, pipeline_config)
        return params.num_frames

    def test_single_gpu_leaves_the_request_alone(self):
        self.assertEqual(
            self._adjusted_num_frames(49, num_gpus=1, sp_degree=1),
            49,
        )

    def test_cfg_parallel_does_not_pad(self):
        # 2 GPUs, but the split is over CFG branches: each rank sees every frame.
        # Padding here would make the same seed diverge from a single GPU.
        self.assertEqual(
            self._adjusted_num_frames(49, num_gpus=2, sp_degree=1),
            49,
        )

    def test_sequence_parallel_still_pads(self):
        # 49 frames is 7 latent frames, which 2 ranks cannot split evenly, so
        # this must round up to 8 latent frames == 57 frames.
        self.assertEqual(
            self._adjusted_num_frames(49, num_gpus=2, sp_degree=2),
            57,
        )

    def test_sequence_parallel_leaves_an_aligned_request_alone(self):
        # 57 frames is 8 latent frames, already divisible by 2.
        self.assertEqual(
            self._adjusted_num_frames(57, num_gpus=2, sp_degree=2),
            57,
        )

    def test_alignment_ignores_gpus_spent_on_tensor_parallelism(self):
        # 4 GPUs as tp=2 x sp=2. 41 frames is 6 latent frames: already divisible
        # by the sp degree of 2, so nothing moves. Aligning to the GPU count
        # would have rounded 6 up to 8 and stretched the clip to 57 frames.
        self.assertEqual(
            self._adjusted_num_frames(41, num_gpus=4, sp_degree=2),
            41,
        )

    def test_missing_sp_degree_is_treated_as_one(self):
        server_args = SimpleNamespace(
            num_gpus=2,
            pipeline_config=LTX25PipelineConfig(),
            comfyui_mode=True,
        )
        params = LTX25SamplingParams()
        params.num_frames = 49
        params._adjust_visual_fields(server_args, server_args.pipeline_config)
        self.assertEqual(params.num_frames, 49)


if __name__ == "__main__":
    unittest.main()
