import importlib.util
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn import (
    SolAttnBackend,
    SolAttnImpl,
    _parse_layer_ranges,
)
from sglang.multimodal_gen.runtime.platforms.cuda import CudaPlatformBase
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum


class FakeCudaPlatform(CudaPlatformBase):
    is_sm120_device = False
    is_blackwell_device = False
    supports_flash_attention = True

    @classmethod
    def is_sm120(cls):
        return cls.is_sm120_device

    @classmethod
    def is_blackwell(cls):
        return cls.is_blackwell_device

    @classmethod
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        return cls.supports_flash_attention


class TestSolAttnBackend(unittest.TestCase):
    def test_enum_name(self):
        self.assertEqual(str(AttentionBackendEnum.SOL_ATTN), "sol_attn")
        self.assertTrue(AttentionBackendEnum.SOL_ATTN.is_sparse)

    def test_parse_layer_ranges(self):
        self.assertEqual(_parse_layer_ranges("0,1,3-5"), frozenset({0, 1, 3, 4, 5}))

    def test_backend_head_size(self):
        self.assertEqual(SolAttnBackend.get_supported_head_sizes(), [128])

    def test_dense_guard_uses_early_steps(self):
        impl = SolAttnImpl(
            num_heads=8,
            head_size=128,
            causal=False,
            softmax_scale=128**-0.5,
            prefix="blocks.5.attn",
        )
        ctx = MagicMock()
        ctx.current_timestep = 3
        server_args = MagicMock()
        server_args.attention_backend_config = {
            "dense_steps": 10,
            "dense_layers": "0,1",
        }
        with (
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn.get_global_server_args",
                return_value=server_args,
            ),
            patch(
                "sglang.multimodal_gen.runtime.managers.forward_context.get_forward_context",
                return_value=ctx,
            ),
        ):
            self.assertTrue(impl._should_use_dense())

    def test_sparse_layer_after_dense_guard(self):
        impl = SolAttnImpl(
            num_heads=8,
            head_size=128,
            causal=False,
            softmax_scale=128**-0.5,
            prefix="blocks.5.attn",
        )
        ctx = MagicMock()
        ctx.current_timestep = 20
        server_args = MagicMock()
        server_args.attention_backend_config = {
            "dense_steps": 10,
            "dense_layers": "0,1",
        }
        with (
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn.get_global_server_args",
                return_value=server_args,
            ),
            patch(
                "sglang.multimodal_gen.runtime.managers.forward_context.get_forward_context",
                return_value=ctx,
            ),
        ):
            self.assertFalse(impl._should_use_dense())

    def test_cuda_resolver(self):
        if importlib.util.find_spec("sol_attn") is None:
            self.skipTest("sol_attn package is not available")
        cls_str = FakeCudaPlatform.get_attn_backend_cls_str(
            selected_backend=AttentionBackendEnum.SOL_ATTN,
            head_size=128,
            dtype=torch.bfloat16,
        )
        self.assertTrue(cls_str.endswith("SolAttnBackend"))

    def test_supports_packed_varlen(self):
        self.assertTrue(SolAttnBackend.supports_packed_varlen())


if __name__ == "__main__":
    unittest.main()
