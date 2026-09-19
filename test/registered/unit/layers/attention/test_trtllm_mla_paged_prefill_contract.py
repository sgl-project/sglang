import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention import trtllm_mla_backend as mla
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_backend(dtype=torch.bfloat16, device="cpu", page_size=32):
    backend = object.__new__(mla.TRTLLMMLABackend)
    backend.backend = "trtllm-gen"
    backend.device = device
    backend.data_type = dtype
    backend.q_data_type = torch.bfloat16
    backend.kv_lora_rank = 512
    backend.qk_rope_head_dim = 64
    backend.qk_nope_head_dim = 128
    backend.kv_cache_dim = 576
    backend.num_q_heads = 16
    backend.page_size = page_size
    backend.max_context_len = 256
    backend.disable_chunked_prefix_cache = True
    backend.use_native_paged_prefill = True
    backend._kv_shard_pool = None
    backend._multi_ctas_kv_counter_buffer = torch.zeros(1, dtype=torch.uint8)
    backend.workspace_buffer = None
    return backend


def make_batch(prefix=(31, 64, 0), suffix=(3, 1, 7), device="cpu"):
    prefix_tensor = torch.tensor(prefix, dtype=torch.int32, device=device)
    suffix_tensor = torch.tensor(suffix, dtype=torch.int32, device=device)
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(prefix),
        req_pool_indices=torch.arange(len(prefix), device=device),
        seq_lens=prefix_tensor + suffix_tensor,
        extend_prefix_lens=prefix_tensor,
        extend_prefix_lens_cpu=list(prefix),
        extend_seq_lens=suffix_tensor,
        extend_seq_lens_cpu=list(suffix),
        out_cache_loc=None,
    )


def init_metadata(backend, batch, *, graph=None):
    with (
        patch.object(mla, "is_in_tc_piecewise_cuda_graph", return_value=graph == "pcg"),
        patch.object(mla, "is_in_breakable_cuda_graph", return_value=graph == "bcg"),
        patch.object(
            mla,
            "grow_multi_ctas_kv_counter_buffer_if_needed",
            side_effect=lambda buffer, *args: buffer,
        ),
    ):
        backend.init_forward_metadata(batch)


class TestPagedPrefill(CustomTestCase):
    def test_supported_configurations(self):
        backend = make_backend()
        parallel = SimpleNamespace(dcp_enabled=False, attn_cp_size=1)
        with (
            patch.object(mla, "get_parallel", return_value=parallel),
            patch.object(torch.cuda, "get_device_capability") as capability,
        ):
            for cc in ((10, 0), (10, 3), (10, 7)):
                capability.return_value = cc
                for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                    backend.data_type = dtype
                    self.assertTrue(backend._supports_native_paged_prefill())
            for cc in ((9, 0), (12, 0), (12, 1)):
                capability.return_value = cc
                self.assertFalse(backend._supports_native_paged_prefill())

            capability.return_value = (10, 7)
            for name, value in (
                ("backend", "cute-dsl"),
                ("q_data_type", torch.float16),
                ("data_type", torch.float8_e5m2),
                ("kv_lora_rank", 256),
                ("page_size", 16),
                ("num_q_heads", 96),
                ("_kv_shard_pool", object()),
            ):
                with self.subTest(name=name), patch.object(backend, name, value):
                    self.assertFalse(backend._supports_native_paged_prefill())
            with patch.object(parallel, "dcp_enabled", True):
                self.assertFalse(backend._supports_native_paged_prefill())
            with patch.object(parallel, "attn_cp_size", 2):
                self.assertFalse(backend._supports_native_paged_prefill())

            class OtherMLABackend(mla.TRTLLMMLABackend):
                pass

            other = object.__new__(OtherMLABackend)
            other.__dict__.update(backend.__dict__)
            self.assertFalse(other._supports_native_paged_prefill())

    def test_fp8_prefix_prefill_keeps_cache_storage(self):
        """Cached-prefix prefill must not expand the preallocated FP8 pool to BF16."""
        from sglang.srt.layers.attention import flashinfer_mla_backend as legacy

        backend = make_backend(torch.float8_e4m3fn)
        backend._create_block_kv_indices = Mock(
            return_value=torch.zeros(3, 8, dtype=torch.int32)
        )
        backend.indices_updater_prefill = SimpleNamespace(update=Mock())
        cache = torch.zeros(1024, 1, 576).to(torch.float8_e4m3fn)
        backend.token_to_kv_pool = SimpleNamespace(get_key_buffer=lambda _: cache)
        q = torch.randn(11, 16, 576, dtype=torch.bfloat16)
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=16,
            scaling=0.125,
            k_scale_float=3.0,
            head_dim=576,
            v_head_dim=512,
            logit_cap=0.0,
        )

        def check_cache(tensor):
            self.assertEqual(tensor.dtype, cache.dtype)
            self.assertEqual(tensor.data_ptr(), cache.data_ptr())

        def native_kernel(**kwargs):
            check_cache(kwargs["kv_cache"])
            self.assertEqual(kwargs["cum_seq_lens_q"].tolist(), [0, 3, 4, 11])
            self.assertEqual(kwargs["seq_lens"].tolist(), [34, 65, 7])
            self.assertEqual(kwargs["bmm1_scale"], layer.scaling)
            self.assertEqual(kwargs["bmm2_scale"], 1.0)
            return torch.zeros(11, 16, 512, dtype=torch.bfloat16)

        def legacy_kernel(q, q_rope, kv, k_rope, *, out):
            check_cache(kv)
            return out

        backend.prefill_wrapper_paged = SimpleNamespace(run=legacy_kernel)
        flashinfer = SimpleNamespace(
            decode=SimpleNamespace(trtllm_batch_decode_with_kv_cache_mla=native_kernel)
        )
        execution = SimpleNamespace(
            kernel=SimpleNamespace(flashinfer_mla_disable_ragged=False)
        )
        for graph in (None, "pcg", "bcg"):
            with (
                self.subTest(graph=graph),
                patch.object(mla, "flashinfer", flashinfer, create=True),
                patch.object(legacy, "get_exec", return_value=execution),
            ):
                backend.disable_chunked_prefix_cache = graph is None
                batch = make_batch()
                batch.seq_lens_sum = 106
                batch.spec_info = None
                batch.attn_dcp_metadata = None
                batch.attn_attend_prefix_cache = None
                init_metadata(backend, batch, graph=graph)
                out = backend.forward_extend(
                    q, None, None, layer, batch, save_kv_cache=False
                )
                self.assertEqual(out.shape, (11, 16 * 512))


if __name__ == "__main__":
    unittest.main()
