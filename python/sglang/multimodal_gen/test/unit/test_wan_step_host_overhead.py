import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.dits.longlive2 import (
    LongLive2ArchConfig,
    LongLive2VideoConfig,
)
from sglang.multimodal_gen.configs.models.dits.wanvideo import (
    WanVideoArchConfig,
    WanVideoConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention import layer
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPABackend
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import causal_wanvideo, wanvideo
from sglang.multimodal_gen.runtime.models.dits.longlive2 import (
    LongLive2Transformer3DModel,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.test.test_utils import CustomTestCase

# the models' fused layers need a GPU on a GPU platform; elsewhere run on the CPU
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _RopeRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.freqs = []
        self.modulation_dtypes = []

    def forward(self, hidden_states, encoder_hidden_states, temb, freqs_cis, **_):
        self.freqs.append(freqs_cis)
        self.modulation_dtypes.append(temb.dtype)
        return hidden_states


class _CountedUploads(torch.Tensor):
    uploads = 0

    def to(self, *args, **kwargs):
        _CountedUploads.uploads += 1
        return super().to(*args, **kwargs).as_subclass(torch.Tensor)


class TestCausalWanRopeUpload(CustomTestCase):
    @torch.no_grad()
    def test_rope_tables_uploaded_once_per_chunk_position(self):
        if not model_parallel_is_initialized():
            ensure_distributed_env_defaults()
            maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
        # a 256-wide hidden state keeps CUDA on the fused output norm, the only
        # one that takes the per-frame shift
        arch = LongLive2ArchConfig(
            num_attention_heads=2,
            attention_head_dim=128,
            in_channels=4,
            out_channels=4,
            text_dim=16,
            freq_dim=16,
            ffn_dim=32,
            num_layers=1,
        )
        with ExitStack() as stack:
            if _DEVICE.type == "cpu":
                # tiny shapes: run every custom op's reference path
                stack.enter_context(
                    patch.object(
                        CustomOp, "dispatch_forward", lambda op: op.forward_native
                    )
                )
            model = LongLive2Transformer3DModel(
                LongLive2VideoConfig(arch_config=arch), hf_config={}
            ).to(device=_DEVICE, dtype=torch.bfloat16)
        recorder = _RopeRecorder()
        model.blocks = nn.ModuleList([recorder])

        bf16 = dict(device=_DEVICE, dtype=torch.bfloat16)

        def run(start_frames, forward_batch):
            for start_frame in start_frames:
                with set_forward_context(0, None, forward_batch=forward_batch):
                    model(
                        hidden_states=torch.randn(1, 4, 2, 4, 4, **bf16),
                        encoder_hidden_states=torch.randn(1, 3, 16, **bf16),
                        timestep=torch.tensor([[500, 500]], device=_DEVICE),
                        kv_cache=[None],
                        crossattn_cache=[None],
                        start_frame=start_frame,
                    )

        # without a request every forward builds and uploads the tables itself
        run((0, 0, 2), None)
        per_forward = list(recorder.freqs)
        recorder.freqs.clear()
        rope = causal_wanvideo.get_rotary_pos_embed
        _CountedUploads.uploads = 0
        with patch.object(
            causal_wanvideo,
            "get_rotary_pos_embed",
            side_effect=lambda *args, **kwargs: tuple(
                table.as_subclass(_CountedUploads) for table in rope(*args, **kwargs)
            ),
        ):
            run((0, 0, 2), SimpleNamespace(extra={}))
        # cos and sin per chunk position: two steps share the first chunk's tables
        self.assertEqual(_CountedUploads.uploads, 4)
        for actual, expected in zip(recorder.freqs, per_forward, strict=True):
            for actual_table, expected_table in zip(actual, expected, strict=True):
                self.assertTrue(torch.equal(actual_table, expected_table))
        # the blocks add it to their fp32 table, so their temb.float() is a no-op
        self.assertEqual(set(recorder.modulation_dtypes), {torch.float32})


class TestWanBlockModulation(CustomTestCase):
    @torch.no_grad()
    def test_blocks_receive_fp32_modulation(self):
        if not model_parallel_is_initialized():
            ensure_distributed_env_defaults()
            maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
        with ExitStack() as stack:
            for name, value in (
                ("get_attn_backend", SDPABackend),
                ("get_compute_dtype", torch.bfloat16),
                ("_resolve_sp_attention_mode", ("ulysses", False)),
                ("_maybe_install_backend_autotune", None),
            ):
                stack.enter_context(patch.object(layer, name, return_value=value))
            # tiny shapes: run every custom op's reference path
            stack.enter_context(
                patch.object(CustomOp, "dispatch_forward", lambda op: op.forward_native)
            )
            stack.enter_context(
                set_forward_context(
                    0,
                    None,
                    forward_batch=SimpleNamespace(
                        enable_sequence_shard=False,
                        enable_teacache=False,
                        enable_spectrum=False,
                    ),
                )
            )
            # one production-sized head: the GPU RoPE kernel expects it
            arch = WanVideoArchConfig(
                num_attention_heads=1,
                attention_head_dim=128,
                in_channels=4,
                out_channels=4,
                text_dim=16,
                freq_dim=16,
                ffn_dim=64,
                num_layers=2,
            )
            model = wanvideo.WanTransformer3DModel(
                WanVideoConfig(arch_config=arch), hf_config={}
            ).to(device=_DEVICE, dtype=torch.bfloat16)
            modulation_dtypes = []
            for block in model.blocks:
                block.register_forward_pre_hook(
                    lambda module, args: modulation_dtypes.append(args[2].dtype)
                )
            bf16 = dict(device=_DEVICE, dtype=torch.bfloat16)
            model(
                hidden_states=torch.randn(1, 4, 2, 4, 4, **bf16),
                encoder_hidden_states=torch.randn(1, 5, 16, **bf16),
                timestep=torch.tensor([500.0], device=_DEVICE),
            )
        # each block's temb.float() is then a no-op instead of a cast kernel
        self.assertEqual(modulation_dtypes, [torch.float32] * arch.num_layers)


if __name__ == "__main__":
    unittest.main()
