# SPDX-License-Identifier: Apache-2.0
"""Real ModelOpt calibration -> native fused PiGemma FP8 -> graph replay."""

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn
from transformers import GemmaConfig, SiglipVisionConfig

from sglang.multimodal_gen.configs.pipeline_configs.pi05 import Pi05PipelineConfig
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    global_force_attn_backend_context_manager,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vlas.pi05_core import PiGemmaDecoderLayer
from sglang.multimodal_gen.runtime.models.vlas.pi05_policy import (
    Pi05CheckpointManifest,
    Pi05PolicyModel,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.vla.pi05_quantization import (
    finalize_fp8_weights,
    replace_projections,
)
from sglang.multimodal_gen.tools.quantize_pi05_modelopt_fp8 import (
    export_state,
    make_quantization_config,
)
from sglang.srt.models.siglip import SiglipEncoderLayer
from sglang.srt.runtime_context import publish, restore_context, snapshot_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")


def tiny_core():
    config = GemmaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        hidden_activation="gelu_pytorch_tanh",
        dtype="bfloat16",
    )
    core = nn.Module()
    core.paligemma_with_expert = nn.Module()
    core.paligemma_with_expert.gemma_expert = nn.Module()
    expert = core.paligemma_with_expert.gemma_expert
    expert.model = nn.Module()
    with global_force_attn_backend_context_manager(AttentionBackendEnum.TORCH_SDPA):
        layer = PiGemmaDecoderLayer(config, 0).to(torch.bfloat16)
    expert.model.layers = nn.ModuleList([layer])
    core.paligemma_with_expert.paligemma = nn.Module()
    prefix = core.paligemma_with_expert.paligemma
    prefix.model = nn.Module()
    prefix.model.vision_tower = nn.Module()
    prefix.model.vision_tower.encoder = nn.Module()
    vision_config = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
    )
    prefix.model.vision_tower.encoder.layers = nn.ModuleList(
        [
            SiglipEncoderLayer(
                vision_config,
                qkv_backend="sdpa",
                flatten_batch=False,
                use_data_parallel=True,
            ).to(torch.bfloat16)
        ]
    )
    prefix.model.multi_modal_projector = nn.Module()
    prefix.model.multi_modal_projector.linear = nn.Linear(64, 64).bfloat16()
    core.action_in_proj = nn.Linear(64, 64).float()
    core.action_out_proj = nn.Linear(64, 32).float()
    with torch.no_grad():
        for parameter in core.parameters():
            parameter.uniform_(-0.1, 0.1)
    return core.cuda()


class TestPi05ModelOptFp8CUDA(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved_context = snapshot_context()
        publish(ServerArgs(model_path="dummy", tp_size=1), role="diffusion_gpu_worker")

    @classmethod
    def tearDownClass(cls):
        restore_context(cls.saved_context)

    def test_calibrated_checkpoint_loader_and_mutable_graph_input(self):
        import modelopt.torch.quantization as mtq

        torch.manual_seed(123)
        core = tiny_core()
        components = ["action_expert", "vision", "projector", "action_heads"]
        names = replace_projections(core, components, quantized=False)
        x = torch.randn(1, 10, 64, device="cuda", dtype=torch.bfloat16)
        cos = torch.ones(1, 10, 16, device="cuda", dtype=torch.bfloat16)
        sin = torch.zeros_like(cos)

        def run(model):
            layer = model.paligemma_with_expert.gemma_expert.model.layers[0]
            with set_forward_context(current_timestep=0, attn_metadata=None):
                prefix = model.paligemma_with_expert.paligemma.model
                hidden = model.action_in_proj(x.float()).bfloat16()
                hidden = prefix.vision_tower.encoder.layers[0](hidden, None, None)
                hidden = prefix.multi_modal_projector.linear(hidden)
                hidden = layer(hidden, position_embeddings=(cos, sin))
                return model.action_out_proj(hidden.float())

        self.assertEqual(len(names), 11)
        baseline = run(core)
        config = make_quantization_config(names)
        with torch.no_grad():
            mtq.quantize(core, config, forward_loop=run)
            fake = run(core)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            exported = export_state(core, names)
            save_file(exported, str(path))
            # Exercise the production loader, including to_empty's parameter lifecycle.
            policy = Pi05PolicyModel.__new__(Pi05PolicyModel)
            nn.Module.__init__(policy)
            policy.core_model = tiny_core()
            policy.config = Pi05PipelineConfig()
            policy.runtime_role = "all"
            policy.device = torch.device(
                "cpu"
            )  # CPU safetensors source -> CUDA targets.
            policy.manifest = Pi05CheckpointManifest(directory, [str(path)])
            policy._fp8_projection_names = replace_projections(
                policy.core_model,
                components,
                quantized=True,
            )
            policy._to_empty_preserve_buffers(
                policy.core_model, device=torch.device("cuda")
            )
            policy._load_weights()
            for name in names:
                loaded = policy.core_model.get_submodule(name)
                torch.testing.assert_close(
                    loaded.weight.float().cpu(),
                    exported[f"{name}.weight"].float(),
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    loaded.weight_scale.cpu(),
                    exported[f"{name}.weight_scale"],
                    atol=0,
                    rtol=0,
                )
                if loaded.bias is not None:
                    torch.testing.assert_close(
                        loaded.bias.cpu(),
                        exported[f"{name}.bias"].bfloat16(),
                        atol=0,
                        rtol=0,
                    )
            finalize_fp8_weights(policy.core_model, names)
        actual = run(policy.core_model)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(actual.isfinite().all())
        # Fake quant dequantizes to BF16 before GEMM; native FP8 GEMM
        # avoids that intermediate rounding. FP32 action-head fake GEMMs also
        # differ from native BF16 compute. Check the accumulated error,
        # and independently verify the loaded weight/scale contract exactly.
        torch.testing.assert_close(actual, fake, atol=0.02, rtol=0.05)
        self.assertLess((actual - baseline).abs().mean().item(), 0.01)
        torch.testing.assert_close(actual, baseline, atol=0.03, rtol=0.15)
        for _ in range(3):
            run(policy.core_model)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(policy.core_model)
        x.mul_(0.5)
        expected = run(policy.core_model)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
