import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.dits.hunyuanvideo import (
    HunyuanVideoArchConfig,
    HunyuanVideoConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention import layer
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPABackend
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import hunyuanvideo
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.test.test_utils import CustomTestCase

# the model's fused layers need a GPU on a GPU platform; elsewhere run on the CPU
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _CountedUploads(torch.Tensor):
    uploads = 0

    def to(self, *args, **kwargs):
        _CountedUploads.uploads += 1
        return super().to(*args, **kwargs).as_subclass(torch.Tensor)


class TestHunyuanVideoRopeUpload(CustomTestCase):
    @torch.no_grad()
    def test_rope_tables_uploaded_once_per_request(self):
        if not model_parallel_is_initialized():
            ensure_distributed_env_defaults()
            maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
        with ExitStack() as stack:
            for name, value in (
                ("get_attn_backend", SDPABackend),
                ("get_compute_dtype", torch.float32),
                ("_resolve_sp_attention_mode", ("ulysses", False)),
                ("_maybe_install_backend_autotune", None),
            ):
                stack.enter_context(patch.object(layer, name, return_value=value))
            # tiny shapes: run every custom op's reference path
            stack.enter_context(
                patch.object(CustomOp, "dispatch_forward", lambda op: op.forward_native)
            )
            torch.manual_seed(0)
            arch = HunyuanVideoArchConfig(
                num_attention_heads=2,
                attention_head_dim=16,
                num_layers=1,
                num_single_layers=1,
                num_refiner_layers=1,
                rope_axes_dim=(4, 6, 6),
                text_embed_dim=32,
                pooled_projection_dim=8,
                in_channels=4,
                out_channels=4,
            )
            model = hunyuanvideo.HunyuanVideoTransformer3DModel(
                HunyuanVideoConfig(arch_config=arch), hf_config={}
            ).to(_DEVICE)
            for parameter in model.parameters():
                parameter.normal_(std=0.05)
            latents = torch.randn(1, 4, 2, 4, 4, device=_DEVICE)
            prompt = torch.randn(1, 6, 32, device=_DEVICE)

            def run_steps(forward_batch):
                outputs = []
                for t in (900.0, 500.0):
                    with set_forward_context(0, None, forward_batch=forward_batch):
                        outputs.append(
                            model(
                                hidden_states=latents,
                                encoder_hidden_states=prompt,
                                timestep=torch.tensor([t], device=_DEVICE),
                                guidance=torch.tensor([6016.0], device=_DEVICE),
                            )
                        )
                return outputs

            # without a request every forward builds and uploads the tables itself
            per_forward = run_steps(None)
            rope = hunyuanvideo.get_rotary_pos_embed
            _CountedUploads.uploads = 0
            with patch.object(
                hunyuanvideo,
                "get_rotary_pos_embed",
                side_effect=lambda *args, **kwargs: tuple(
                    table.as_subclass(_CountedUploads)
                    for table in rope(*args, **kwargs)
                ),
            ):
                per_request = run_steps(
                    SimpleNamespace(
                        extra={}, enable_teacache=False, enable_spectrum=False
                    )
                )
            # cos and sin, once for the request's two steps
            self.assertEqual(_CountedUploads.uploads, 2)
            for actual, expected in zip(per_request, per_forward, strict=True):
                self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
