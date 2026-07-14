import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt import server_args as server_args_module
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.layers.attention.dsv4.nvfp4_k_cache import (
    DSV4_NVFP4_BYTES_PER_TOKEN,
    DSV4_NVFP4_NOPE_DIM,
    DSV4_NVFP4_PACKED_NOPE_BYTES,
    DSV4_NVFP4_ROPE_BYTES,
    DSV4_NVFP4_ROPE_DIM,
    DSV4_NVFP4_SCALE_BYTES,
    dequantize_dsv4_nvfp4_k_cache_paged,
    quantize_dsv4_nvfp4_k_cache_into,
)
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    SparsePrefillWorkspace,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    default_cuda_graph_config,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _dsv4_config():
    return SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])


class TestDSV4NVFP4HostUnit(unittest.TestCase):
    def test_layout_and_cpu_page_scatter_roundtrip(self):
        self.assertEqual(DSV4_NVFP4_PACKED_NOPE_BYTES, 224)
        self.assertEqual(DSV4_NVFP4_SCALE_BYTES, 28)
        self.assertEqual(DSV4_NVFP4_ROPE_BYTES, 128)
        self.assertEqual(DSV4_NVFP4_BYTES_PER_TOKEN, 380)

        page_size = 4
        cache = torch.full(
            (2, page_size * DSV4_NVFP4_BYTES_PER_TOKEN),
            0xA5,
            dtype=torch.uint8,
        )
        e2m1_values = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            dtype=torch.bfloat16,
        )
        nope = e2m1_values.repeat(2, DSV4_NVFP4_NOPE_DIM // 8)
        rope = torch.arange(2 * DSV4_NVFP4_ROPE_DIM, dtype=torch.bfloat16).reshape(
            2, DSV4_NVFP4_ROPE_DIM
        )
        cache_k = torch.cat((nope, rope), dim=-1)

        quantize_dsv4_nvfp4_k_cache_into(
            cache_k,
            cache,
            torch.tensor([5, 0], dtype=torch.int32),
            page_size=page_size,
            global_scale=1.0,
        )

        rows = cache.view(-1, DSV4_NVFP4_BYTES_PER_TOKEN)
        self.assertTrue(torch.all(rows[1:5] == 0xA5))
        out = dequantize_dsv4_nvfp4_k_cache_paged(
            cache,
            torch.tensor([5, -1, 0, 8], dtype=torch.int32),
            page_size=page_size,
            global_scale=1.0,
        )
        torch.testing.assert_close(out[0, 0], cache_k[0], rtol=0, atol=0)
        torch.testing.assert_close(out[2, 0], cache_k[1], rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(out[1]).item(), 0)
        self.assertEqual(torch.count_nonzero(out[3]).item(), 0)

    def test_sparse_fallback_compacts_swa_and_extra_prefixes(self):
        backend = object.__new__(DeepseekV4AttnBackend)
        backend.sparse_prefill_workspace = SparsePrefillWorkspace(torch.device("cpu"))
        backend.softmax_scale = 0.125
        backend.head_dim_v = 512

        page_size = 4
        cache = torch.zeros(
            (4, page_size * DSV4_NVFP4_BYTES_PER_TOKEN), dtype=torch.uint8
        )
        q = torch.zeros((2, 1, 8, 512), dtype=torch.bfloat16)
        swa_indices = torch.tensor(
            [[[0, 1, -1, -1]], [[2, 3, 4, -1]]], dtype=torch.int32
        )
        extra_indices = torch.tensor(
            [[[5, 6, -1, -1]], [[7, -1, -1, -1]]], dtype=torch.int32
        )
        pool = SimpleNamespace(
            swa_window_size=page_size,
            get_swa_nvfp4_global_scale=lambda _layer_id: torch.tensor([1.0]),
            get_extra_key_page_size=lambda _layer_id: page_size,
            get_extra_nvfp4_global_scale=lambda _layer_id: torch.tensor([1.0]),
        )
        expected_o = torch.zeros((2, 8, 512), dtype=torch.bfloat16)

        with patch(
            "sgl_kernel.flash_mla.flash_mla_sparse_fwd",
            return_value=(expected_o, torch.empty(0), torch.empty(0)),
        ) as sparse_fwd:
            actual = backend._forward_nvfp4_sparse(
                q=q,
                layer_id=0,
                token_to_kv_pool=pool,
                swa_k_cache=cache,
                swa_indices=swa_indices,
                swa_topk_lengths=torch.tensor([2, 3], dtype=torch.int32),
                extra_k_cache=cache,
                extra_indices=extra_indices,
                extra_topk_lengths=torch.tensor([2, 1], dtype=torch.int32),
                attn_sink=torch.zeros(8, dtype=torch.float32),
            )

        self.assertIs(actual, expected_o)
        kwargs = sparse_fwd.call_args.kwargs
        torch.testing.assert_close(
            kwargs["indices"].squeeze(1),
            torch.tensor(
                [[0, 1, 8, 9, 0, 0, 0, 0], [4, 5, 6, 12, 0, 0, 0, 0]],
                dtype=torch.int32,
            ),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            kwargs["topk_length"],
            torch.tensor([4, 4], dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        self.assertEqual(tuple(kwargs["kv"].shape), (16, 1, 512))

    def test_server_gate_accepts_only_sm90_nvfp4_page256(self):
        args = SimpleNamespace(
            kv_cache_dtype="fp4_e2m1",
            fp4_kv_cache_recipe="nvfp4",
            enable_hisparse=False,
            speculative_algorithm=None,
            enable_prefill_cp=False,
            page_size=256,
            disable_cuda_graph=False,
            disable_decode_cuda_graph=False,
            disable_prefill_cuda_graph=False,
            cuda_graph_config=default_cuda_graph_config(),
            get_model_config=lambda: SimpleNamespace(hf_config=_dsv4_config()),
        )
        with (
            patch.object(server_args_module, "is_cuda", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
        ):
            ServerArgs._handle_kv4_compatibility(args)
        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.DISABLED)
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)

        args.page_size = 128
        with (
            patch.object(server_args_module, "is_cuda", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            self.assertRaisesRegex(ValueError, "page-size=256"),
        ):
            ServerArgs._handle_kv4_compatibility(args)


if __name__ == "__main__":
    unittest.main()
