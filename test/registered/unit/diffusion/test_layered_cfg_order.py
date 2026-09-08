import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageLayeredPipelineConfig,
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


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


if __name__ == "__main__":
    unittest.main()
