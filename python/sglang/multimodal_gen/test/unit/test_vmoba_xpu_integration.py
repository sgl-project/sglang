# SPDX-License-Identifier: Apache-2.0

import math
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import addict
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.vmoba import (
    VideoMobaAttentionMetadata,
    VideoMobaAttentionMetadataBuilder,
    VMOBAAttentionBackend,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.xpu import XpuPlatform

XPU_ADAPTER = "sglang.multimodal_gen.csrc.attn.vmoba_attn.vmoba.xpu_flash_attn"
VMOBA_BACKEND = (
    "sglang.multimodal_gen.runtime.layers.attention.backends.vmoba."
    "VMOBAAttentionBackend"
)
SDPA_BACKEND = (
    "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"
)


class TestVMoBAXpuSelection(unittest.TestCase):
    def test_supported_dtypes_select_vmoba(self):
        adapter = ModuleType(XPU_ADAPTER)
        adapter.flash_attn_varlen_forward = Mock()
        with patch.dict("sys.modules", {XPU_ADAPTER: adapter}):
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(dtype=dtype):
                    self.assertEqual(
                        XpuPlatform.get_attn_backend_cls_str(
                            AttentionBackendEnum.VMOBA_ATTN, 128, dtype
                        ),
                        VMOBA_BACKEND,
                    )
        adapter.flash_attn_varlen_forward.assert_not_called()

    def test_float32_does_not_fall_back_to_dense(self):
        with self.assertRaisesRegex(ValueError, "float16 or bfloat16"):
            XpuPlatform.get_attn_backend_cls_str(
                AttentionBackendEnum.VMOBA_ATTN, 128, torch.float32
            )

    def test_missing_xpu_adapter_does_not_fall_back_to_dense(self):
        with patch.dict("sys.modules", {XPU_ADAPTER: None}):
            with self.assertRaisesRegex(ImportError, "XPU varlen") as error:
                XpuPlatform.get_attn_backend_cls_str(
                    AttentionBackendEnum.VMOBA_ATTN, 128, torch.bfloat16
                )
        self.assertIsInstance(error.exception.__cause__, ImportError)

    def test_explicit_sdpa_is_unchanged(self):
        self.assertEqual(
            XpuPlatform.get_attn_backend_cls_str(
                AttentionBackendEnum.TORCH_SDPA, 128, torch.bfloat16
            ),
            SDPA_BACKEND,
        )

    def test_default_float32_fallback_is_unchanged(self):
        self.assertEqual(
            XpuPlatform.get_attn_backend_cls_str(None, 128, torch.float32),
            SDPA_BACKEND,
        )


class TestVMoBADenoisingMetadata(unittest.TestCase):
    def test_builds_metadata_without_mutating_config(self):
        config = addict.Dict(
            temporal_chunk_size=1,
            temporal_topk=3,
            spatial_chunk_size=[2, 4],
            spatial_topk=3,
            st_chunk_size=[3, 2, 4],
            st_topk=3,
            moba_select_mode="topk",
            moba_threshold_type="query_head",
        )
        original = config.to_dict()
        server_args = SimpleNamespace(
            attention_backend_config=addict.Dict(moba_config=config),
            pipeline_config=SimpleNamespace(
                dit_config=SimpleNamespace(patch_size=(1, 2, 2))
            ),
        )
        batch = SimpleNamespace(raw_latent_shape=(1, 16, 21, 60, 104))
        stage = DenoisingStage.__new__(DenoisingStage)
        stage.attn_backend = VMOBAAttentionBackend
        builder = VideoMobaAttentionMetadataBuilder()
        with (
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "denoising.get_local_torch_device",
                return_value=torch.device("xpu:1"),
            ),
            patch.object(
                VideoMobaAttentionMetadataBuilder, "build", wraps=builder.build
            ) as build,
        ):
            for step in (0, 7):
                metadata = stage._build_attn_metadata(step, batch, server_args)
                self.assertIsInstance(metadata, VideoMobaAttentionMetadata)
                self.assertEqual(metadata.current_timestep, step)
                self.assertEqual(metadata.patch_resolution, [21, 30, 52])
                self.assertEqual(metadata.moba_select_mode, "topk")
                self.assertEqual(metadata.spatial_chunk_size, [2, 4])
                self.assertEqual(metadata.st_topk, 3)
                build.assert_called_with(
                    **original,
                    current_timestep=step,
                    raw_latent_shape=(21, 60, 104),
                    patch_size=(1, 2, 2),
                    device=torch.device("xpu:1"),
                )
        self.assertEqual(
            server_args.attention_backend_config.moba_config.to_dict(), original
        )


@unittest.skipUnless(torch.xpu.is_available(), "requires an Intel XPU")
class TestVMoBAXpuKernelContract(unittest.TestCase):
    def test_varlen_forward_populates_softmax_lse(self):
        """An sgl-kernel-xpu build left softmax_lse unwritten for multi-query
        varlen attention, so vmoba read zeros and weighted every block equally,
        corrupting the output with no error and no failing shape/dtype test.
        With zeroed q/k/v every logit is 0, so the logsumexp over n_kv keys is
        log(n_kv).
        """
        from sglang.multimodal_gen.csrc.attn.vmoba_attn.vmoba.xpu_flash_attn import (
            flash_attn_varlen_forward,
        )

        n_kv, heads, head_dim = 8, 4, 64
        zeros = torch.zeros(n_kv, heads, head_dim, dtype=torch.float16, device="xpu")
        cu = torch.tensor([0, n_kv], dtype=torch.int32, device="xpu")

        _, softmax_lse, _, _ = flash_attn_varlen_forward(
            q=zeros,
            k=zeros,
            v=zeros,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=n_kv,
            max_seqlen_k=n_kv,
            softmax_scale=1.0,
        )

        torch.testing.assert_close(
            softmax_lse,
            torch.full_like(softmax_lse, math.log(n_kv)),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_strided_kv_matches_contiguous_copies(self):
        """vmoba slices K/V out of a stacked [tokens, 2, heads, dim] buffer, so
        they reach the kernel row-strided. An earlier build read only the last
        stride and returned wrong values for those slices; the shim now relies
        on sgl-kernel-xpu#471 and passes them through without copying.
        """
        from sglang.multimodal_gen.csrc.attn.vmoba_attn.vmoba.xpu_flash_attn import (
            flash_attn_varlen_forward,
        )

        tokens, heads, head_dim = 64, 4, 64
        torch.manual_seed(0)
        stacked = torch.randn(
            tokens, 2, heads, head_dim, dtype=torch.float16, device="xpu"
        )
        q = torch.randn(tokens, heads, head_dim, dtype=torch.float16, device="xpu")
        cu = torch.tensor([0, tokens], dtype=torch.int32, device="xpu")

        def run(k, v):
            return flash_attn_varlen_forward(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=tokens,
                max_seqlen_k=tokens,
            )[:2]

        strided_out, strided_lse = run(stacked[:, 0], stacked[:, 1])
        packed_out, packed_lse = run(
            stacked[:, 0].contiguous(), stacked[:, 1].contiguous()
        )

        torch.testing.assert_close(strided_out, packed_out, atol=0, rtol=0)
        torch.testing.assert_close(strided_lse, packed_lse, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
