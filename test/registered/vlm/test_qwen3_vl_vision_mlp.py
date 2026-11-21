import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

from sglang.srt.models import qwen3_vl
from sglang.srt.runtime_context import get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-small")


class TestQwen3VLVisionMLP(unittest.TestCase):
    @torch.inference_mode()
    def test_chunked_residual(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", mm_attention_backend="sdpa"), role="test"
        )
        config = Qwen3VLVisionConfig(
            depth=1,
            hidden_size=128,
            intermediate_size=256,
            num_heads=2,
            patch_size=4,
            temporal_patch_size=1,
            out_hidden_size=128,
            num_position_embeddings=256,
            spatial_merge_size=2,
            deepstack_visual_indexes=[],
        )
        with (
            torch.random.fork_rng(devices=[torch.cuda.current_device()]),
            get_parallel().override(
                tp_group=None,
                attn_tp_group=None,
                pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
            ),
        ):
            torch.manual_seed(49132)
            model = qwen3_vl.Qwen3VLMoeVisionModel(config, use_data_parallel=True)
            model = model.cuda().bfloat16().eval()
            for param in model.parameters():
                param.normal_(std=0.02)
            grid = [[1, 6, 6]] * 3
            pixels = torch.randn(108, 48, device="cuda", dtype=torch.bfloat16)
            rows = []
            hook = model.blocks[0].mlp.register_forward_pre_hook(
                lambda _, args: rows.append(args[0].size(0))
            )
            self.addCleanup(hook.remove)
            reference = model(pixels, grid)
            self.assertEqual(rows, [108])
            rows.clear()
            with patch.object(qwen3_vl, "_VISION_MLP_CHUNK_ROWS", 32):
                actual = model(pixels, grid)
            self.assertEqual(rows, [32, 32, 32, 12])
            torch.testing.assert_close(actual, reference, rtol=0.016, atol=0.002)


if __name__ == "__main__":
    unittest.main()
