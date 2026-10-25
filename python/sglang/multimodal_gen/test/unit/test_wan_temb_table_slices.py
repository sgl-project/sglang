import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
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
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import causal_wanvideo, wanvideo
from sglang.multimodal_gen.runtime.models.dits.longlive2 import (
    LongLive2Transformer3DModel,
)
from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    _eager_temb_table_slices,
    _wan_temb_table_slices,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.test.unit.host_sync_guard import (
    CountedUploads,
    reference_custom_ops,
    tiny_dit_runtime,
)

# the models' fused layers need a GPU on a GPU platform; elsewhere run on the CPU
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestWanTembTableSlices(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and current_platform.is_cuda(),
        "requires NVIDIA CUDA fusion",
    )
    def test_fused_matches_eager_and_is_contiguous(self):
        torch.manual_seed(0)
        for batch, seq, hidden in [(1, 517, 3072), (2, 64, 1536)]:
            temb = torch.randn(
                batch, seq, 6, hidden, device="cuda", dtype=torch.bfloat16
            )
            table = torch.randn(1, 6, hidden, device="cuda", dtype=torch.float32)
            reference = _eager_temb_table_slices(table, temb)
            fused = _wan_temb_table_slices(table, temb)
            self.assertEqual(len(fused), 6)
            for ref, out in zip(reference, fused):
                self.assertTrue(torch.equal(ref, out))
                self.assertTrue(out.is_contiguous())

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fp32_temb_also_supported(self):
        torch.manual_seed(1)
        temb = torch.randn(1, 33, 6, 512, device="cuda", dtype=torch.float32)
        table = torch.randn(1, 6, 512, device="cuda", dtype=torch.float32)
        reference = _eager_temb_table_slices(table, temb)
        fused = _wan_temb_table_slices(table, temb)
        for ref, out in zip(reference, fused):
            self.assertTrue(torch.equal(ref, out))

    def test_cpu_falls_back_to_eager(self):
        torch.manual_seed(2)
        temb = torch.randn(1, 9, 6, 64, dtype=torch.bfloat16)
        table = torch.randn(1, 6, 64, dtype=torch.float32)
        reference = _eager_temb_table_slices(table, temb)
        out = _wan_temb_table_slices(table, temb)
        for ref, got in zip(reference, out):
            self.assertTrue(torch.equal(ref, got))


class _RopeRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.freqs = []
        self.modulation_dtypes = []

    def forward(self, hidden_states, encoder_hidden_states, temb, freqs_cis, **_):
        self.freqs.append(freqs_cis)
        self.modulation_dtypes.append(temb.dtype)
        return hidden_states


@pytest.mark.usefixtures("single_process_model_parallel")
class TestWanStepHostOverhead(unittest.TestCase):
    @torch.no_grad()
    def test_causal_rope_tables_uploaded_once_per_chunk_position(self):
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
        with reference_custom_ops() if _DEVICE.type == "cpu" else nullcontext():
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
        CountedUploads.uploads = 0
        with patch.object(
            causal_wanvideo,
            "get_rotary_pos_embed",
            side_effect=lambda *args, **kwargs: tuple(
                table.as_subclass(CountedUploads) for table in rope(*args, **kwargs)
            ),
        ):
            run((0, 0, 2), SimpleNamespace(extra={}))
        # cos and sin per chunk position: two steps share the first chunk's tables
        self.assertEqual(CountedUploads.uploads, 4)
        torch.testing.assert_close(recorder.freqs, per_forward, rtol=0, atol=0)
        # the blocks add it to their fp32 table, so their temb.float() is a no-op
        self.assertEqual(set(recorder.modulation_dtypes), {torch.float32})

    @torch.no_grad()
    def test_blocks_receive_fp32_modulation(self):
        forward_batch = SimpleNamespace(
            enable_sequence_shard=False, enable_teacache=False, enable_spectrum=False
        )
        with (
            tiny_dit_runtime(torch.bfloat16),
            set_forward_context(0, None, forward_batch=forward_batch),
        ):
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
