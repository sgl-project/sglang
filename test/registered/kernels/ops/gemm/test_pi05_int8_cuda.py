# SPDX-License-Identifier: Apache-2.0
"""Pi0.5 online W8A8 INT8 projections: selection, caller interface, accuracy, graph replay."""

import unittest

import torch
from torch import nn
from transformers import GemmaConfig, SiglipVisionConfig

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    global_force_attn_backend_context_manager,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vlas.pi05_core import PiGemmaDecoderLayer
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.vla.pi05_int8 import (
    INT8_COMPONENTS,
    Pi05Int8Linear,
    quantize_projections_int8,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.srt.models.siglip import SiglipEncoderLayer
from sglang.srt.runtime_context import publish, restore_context, snapshot_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def tiny_core():
    config = GemmaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        hidden_activation="gelu_pytorch_tanh",
        dtype="bfloat16",
        use_adarms=True,
        adarms_cond_dim=64,
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
    vision_config = SiglipVisionConfig(hidden_size=64, intermediate_size=128, num_attention_heads=4)
    prefix.model.vision_tower.encoder.layers = nn.ModuleList(
        [
            SiglipEncoderLayer(
                vision_config, qkv_backend="sdpa", flatten_batch=False, use_data_parallel=True
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


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0), "needs sm80+"
)
class TestPi05Int8CUDA(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved_context = snapshot_context()
        publish(ServerArgs(model_path="dummy", tp_size=1), role="diffusion_gpu_worker")
        # Real SGLang (Row/Column)ParallelLinear layers read the TP group even at tp_size=1.
        if not model_parallel_is_initialized():
            ensure_distributed_env_defaults()
            maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    @classmethod
    def tearDownClass(cls):
        restore_context(cls.saved_context)

    def test_projections_keep_interface_accuracy_and_graph_replay(self):
        torch.manual_seed(0)
        core = tiny_core()
        x = torch.randn(1, 10, 64, device="cuda", dtype=torch.bfloat16)
        cos = torch.ones(1, 10, 16, device="cuda", dtype=torch.bfloat16)
        sin = torch.zeros_like(cos)
        cond = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16)

        def run(model):
            layer = model.paligemma_with_expert.gemma_expert.model.layers[0]
            with set_forward_context(current_timestep=0, attn_metadata=None):
                prefix = model.paligemma_with_expert.paligemma.model
                hidden = model.action_in_proj(x.float()).bfloat16()
                hidden = prefix.vision_tower.encoder.layers[0](hidden, None, None)
                hidden = prefix.multi_modal_projector.linear(hidden)
                hidden = layer(hidden, position_embeddings=(cos, sin), adarms_cond=cond)
                return model.action_out_proj(hidden.float())

        with torch.no_grad():
            baseline = run(core)
            names = quantize_projections_int8(core, list(INT8_COMPONENTS))
            # 4 action-expert + 4 SigLIP projections + projector; the FP32 action/time heads and
            # AdaRMS modulation dense layers must stay unquantized.
            self.assertEqual(len(names), 9)
            self.assertFalse(any(n.startswith("action_") or ".dense" in n for n in names))
            self.assertTrue(all(isinstance(core.get_submodule(n), Pi05Int8Linear) for n in names))
            actual = run(core)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertLess((actual - baseline).abs().mean().item(), 0.01)
        torch.testing.assert_close(actual, baseline, atol=0.03, rtol=0.15)
        with torch.no_grad():
            for _ in range(3):
                run(core)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = run(core)
            x.mul_(0.5)
            expected = run(core)
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
