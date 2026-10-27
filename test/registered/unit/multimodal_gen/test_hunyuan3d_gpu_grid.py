# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import torch

from sglang.multimodal_gen.runtime.models.vaes.hunyuan3d_vae import (
    VanillaVolumeDecoder,
    generate_dense_grid_points,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestHunyuan3DGPUGrid(CustomTestCase):
    def test_vanilla_volume_decoder_grid_matches_numpy_reference(self):
        bounds = [-1.7, 0.2, -3.1, 2.3, 4.7, 0.8]
        resolution = 2
        xyz_reference, grid_size, _ = generate_dense_grid_points(
            bbox_min=np.array(bounds[:3]),
            bbox_max=np.array(bounds[3:]),
            octree_resolution=resolution,
            indexing="ij",
        )

        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        for device in devices:
            for dtype in (torch.float32, torch.float16):
                with self.subTest(device=device, dtype=dtype):
                    expected_xyz = (
                        torch.from_numpy(xyz_reference)
                        .to(device=device, dtype=dtype)
                        .reshape(-1, 3)
                    )
                    captured_queries = []

                    def geo_decoder(queries, latents):
                        captured_queries.append(queries.clone())
                        return queries.sum(dim=-1)

                    latents = torch.zeros((1, 1, 1), device=device, dtype=dtype)
                    actual = VanillaVolumeDecoder()(
                        latents=latents,
                        geo_decoder=geo_decoder,
                        bounds=bounds,
                        num_chunks=10,
                        octree_resolution=resolution,
                        enable_pbar=False,
                    )

                    actual_xyz = torch.cat(captured_queries, dim=1)[0]
                    self.assertTrue(torch.equal(actual_xyz, expected_xyz))
                    expected_logits = (
                        expected_xyz.sum(dim=-1).view(1, *grid_size).float()
                    )
                    self.assertTrue(torch.equal(actual, expected_logits))


if __name__ == "__main__":
    unittest.main()
