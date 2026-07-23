import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton

from sglang.kernels.ops.attention.dcp_kernels import (
    create_mla_kv_page_table_for_dcp,
)
from sglang.kernels.ops.kvcache.mla_buffer import set_mla_kv_buffer_triton
from sglang.srt.layers.attention import tokenspeed_mla_backend as backend_module
from sglang.srt.layers.attention import trtllm_mla_backend as trt_backend_module
from sglang.srt.layers.attention.tokenspeed_mla_backend import (
    TokenspeedMLABackend,
    _get_tokenspeed_workspace,
)
from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLADecodeMetadata
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-b200")


def _make_interleaved_req_to_token(lengths, dcp_size, physical_page_size):
    """Build #14194 virtual pages backed by non-contiguous physical pages."""
    virtual_page_size = dcp_size * physical_page_size
    req_to_token = torch.full((len(lengths), max(lengths)), -1, dtype=torch.int32)
    physical_pages_by_req = []
    next_physical_page = 1
    for req_idx, length in enumerate(lengths):
        num_pages = triton.cdiv(length, virtual_page_size)
        physical_pages = [
            next_physical_page + 2 * logical_page for logical_page in range(num_pages)
        ]
        physical_pages_by_req.append(physical_pages)
        next_physical_page += 2 * num_pages + 1
        for position in range(length):
            logical_page = position // virtual_page_size
            within_page = position % virtual_page_size
            req_to_token[req_idx, position] = (
                physical_pages[logical_page] * virtual_page_size + within_page
            )
    return req_to_token.cuda(), physical_pages_by_req


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestTokenspeedMLADCP(CustomTestCase):
    def test_workspace_accounts_for_dcp_gathered_heads(self):
        device = torch.device("cuda")
        expected_size = 148 * (4 * 8) * 8 * (512 + 1) * 4
        allocated = SimpleNamespace()
        resources = SimpleNamespace(buffers={})

        with (
            patch.object(
                backend_module,
                "get_parallel",
                return_value=SimpleNamespace(attn_dcp_size=8),
            ),
            patch.object(
                backend_module.tokenspeed_mla, "get_num_sm", return_value=148
            ),
            patch(
                "sglang.srt.runtime_context.get_resources", return_value=resources
            ),
            patch.object(
                backend_module.torch, "empty", return_value=allocated
            ) as empty,
        ):
            actual = _get_tokenspeed_workspace(
                device,
                num_heads=4,
                kv_lora_rank=512,
                max_q_len=8,
            )

        self.assertIs(actual, allocated)
        empty.assert_called_once_with(
            expected_size,
            dtype=torch.int8,
            device=device,
        )

    def test_prefill_normalizes_strided_mla_values(self):
        backend = object.__new__(TokenspeedMLABackend)
        query = torch.empty((4, 2, 192), dtype=torch.bfloat16, device="cuda")
        key = torch.empty_like(query)
        value_storage = torch.empty((4, 2, 256), dtype=torch.bfloat16, device="cuda")
        value = value_storage[..., 128:]
        self.assertFalse(value.is_contiguous())

        seq_lens = torch.tensor([4], dtype=torch.int32, device="cuda")
        cum_seq_lens = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
        expected = torch.empty((4, 2, 128), dtype=torch.bfloat16, device="cuda")

        with patch.object(
            backend_module.tokenspeed_mla,
            "tokenspeed_mla_prefill",
            return_value=expected,
        ) as prefill:
            actual = backend._run_prefill_kernel(
                q=query,
                k=key,
                v=value,
                layer=SimpleNamespace(scaling=192**-0.5),
                batch_size=1,
                cum_seq_lens_q=cum_seq_lens,
                max_q_len=4,
                seq_lens_kv=seq_lens,
                cum_seq_lens_kv=cum_seq_lens,
                max_kv_len=4,
                is_causal=True,
                return_lse=False,
                out_buffer=expected,
            )

        self.assertIs(actual, expected)
        kwargs = prefill.call_args.kwargs
        self.assertTrue(kwargs["query"].is_contiguous())
        self.assertTrue(kwargs["key"].is_contiguous())
        self.assertTrue(kwargs["value"].is_contiguous())

    def test_decode_forwards_native_cp_and_lse_arguments(self):
        backend = object.__new__(TokenspeedMLABackend)
        backend._tokenspeed_workspace = torch.empty(1, dtype=torch.int8, device="cuda")
        backend.kv_lora_rank = 512
        backend.qk_rope_head_dim = 64

        query = torch.empty((2, 1, 4, 576), dtype=torch.float8_e4m3fn, device="cuda")
        kv_cache = torch.empty(
            (8, 1, 64, 576), dtype=torch.float8_e4m3fn, device="cuda"
        )
        block_tables = torch.zeros((2, 2), dtype=torch.int32, device="cuda")
        local_seq_lens = torch.tensor([17, 16], dtype=torch.int64, device="cuda")
        global_seq_lens = torch.tensor([67, 64], dtype=torch.int64, device="cuda")
        expected_out = torch.empty((2, 1, 4, 512), dtype=torch.bfloat16, device="cuda")
        expected_lse = torch.empty((2, 1, 4), dtype=torch.float32, device="cuda")
        layer = SimpleNamespace(scaling=0.125, k_scale_float=1.0)

        with patch.object(
            backend_module.tokenspeed_mla,
            "tokenspeed_mla_decode",
            return_value=(expected_out, expected_lse),
        ) as decode:
            actual = backend._run_decode_kernel(
                query=query,
                kv_cache=kv_cache,
                block_tables=block_tables,
                seq_lens=local_seq_lens,
                max_seq_len=17,
                layer=layer,
                causal_seqs=global_seq_lens,
                cp_world=4,
                cp_rank=2,
                return_lse=True,
            )

        self.assertIs(actual[0], expected_out)
        self.assertIs(actual[1], expected_lse)
        kwargs = decode.call_args.kwargs
        self.assertTrue(kwargs["return_lse"])
        self.assertEqual(kwargs["cp_world"], 4)
        self.assertEqual(kwargs["cp_rank"], 2)
        self.assertIs(kwargs["causal_seqs"], global_seq_lens)
        self.assertEqual(kwargs["seq_lens"].dtype, torch.int32)

    def test_static_verify_metadata_splits_global_and_local_lengths(self):
        backend = object.__new__(TokenspeedMLABackend)
        backend.num_draft_tokens = 8
        metadata = TRTLLMMLADecodeMetadata(
            block_kv_indices=torch.full((3, 4), -1, dtype=torch.int32, device="cuda"),
            seq_lens_k=torch.zeros(3, dtype=torch.int32, device="cuda"),
            global_seq_lens_k=torch.zeros(3, dtype=torch.int32, device="cuda"),
        )
        backend.decode_cuda_graph_metadata = {3: metadata}
        prefix_lens = torch.tensor([10, 20, 30], dtype=torch.int32, device="cuda")
        req_pool_indices = torch.arange(3, dtype=torch.int32, device="cuda")
        parallel = SimpleNamespace(dcp_enabled=True, dcp_size=4, dcp_rank=2)

        with (
            patch.object(backend_module, "get_parallel", return_value=parallel),
            patch.object(backend, "_fill_dcp_block_kv_indices") as fill,
        ):
            backend._apply_cuda_graph_metadata(
                bs=3,
                req_pool_indices=req_pool_indices,
                seq_lens=prefix_lens,
                forward_mode=ForwardMode.TARGET_VERIFY,
            )

        expected_global = prefix_lens + 8
        expected_local = get_dcp_lens(expected_global, 4, 2).to(torch.int32)
        torch.testing.assert_close(metadata.global_seq_lens_k, expected_global)
        torch.testing.assert_close(metadata.seq_lens_k, expected_local)
        fill.assert_called_once()
        torch.testing.assert_close(fill.call_args.args[2], expected_local)

    def test_non_dcp_ragged_verify_preserves_padding_path(self):
        backend = object.__new__(TokenspeedMLABackend)
        backend.forward_prefill_metadata = None
        backend.data_type = torch.bfloat16
        backend.q_data_type = torch.bfloat16
        backend.page_size = 1
        backend.kv_cache_dim = 2
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda _: torch.empty(
                (8, 2), dtype=torch.bfloat16, device="cuda"
            )
        )
        metadata = TRTLLMMLADecodeMetadata(
            block_kv_indices=torch.zeros((2, 1), dtype=torch.int32, device="cuda"),
            seq_lens_k=torch.tensor([11, 22], dtype=torch.int32, device="cuda"),
            max_seq_len_k=32,
        )
        layout = SimpleNamespace(
            bs=2,
            verify_lens=torch.tensor([1, 2], dtype=torch.int32, device="cuda"),
            qo_indptr_device=torch.tensor([0, 1, 3], dtype=torch.int32, device="cuda"),
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=2,
            decode_trtllm_mla_metadata=metadata,
            spec_info=SimpleNamespace(draft_token_num=4, ragged_verify_layout=layout),
        )
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=1,
            head_dim=2,
            v_head_dim=1,
        )
        query = torch.empty((3, 2), dtype=torch.bfloat16, device="cuda")
        padded_query = torch.empty((2, 4, 1, 2), dtype=torch.bfloat16, device="cuda")
        dense_output = torch.empty((2, 4, 1, 1), dtype=torch.bfloat16, device="cuda")
        expected = torch.empty((3, 1, 1), dtype=torch.bfloat16, device="cuda")

        with (
            patch.object(
                trt_backend_module,
                "get_parallel",
                return_value=SimpleNamespace(dcp_enabled=False),
            ),
            patch.object(
                backend, "pad_draft_extend_query", return_value=padded_query
            ) as pad,
            patch.object(backend, "_run_decode_kernel", return_value=dense_output),
            patch.object(
                backend, "unpad_draft_extend_output", return_value=expected
            ) as unpad,
        ):
            actual = backend.forward_extend(
                query,
                None,
                None,
                layer,
                forward_batch,
                save_kv_cache=False,
            )

        self.assertEqual(pad.call_count, 1)
        self.assertEqual(unpad.call_count, 1)
        self.assertTrue(unpad.call_args.kwargs["zero_uncovered"])
        torch.testing.assert_close(actual, expected.view(3, 1))

    def test_local_page_table_maps_only_rank_owned_physical_pages(self):
        dcp_size = 4
        physical_page_size = 64
        lengths = (513, 1025)
        req_to_token, physical_pages_by_req = _make_interleaved_req_to_token(
            lengths, dcp_size, physical_page_size
        )
        req_pool_indices = torch.arange(len(lengths), dtype=torch.int32, device="cuda")
        global_seq_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")

        for rank in range(dcp_size):
            local_seq_lens = get_dcp_lens(global_seq_lens, dcp_size, rank)
            table_width = 8
            block_tables = torch.full(
                (len(lengths), table_width), -1, dtype=torch.int32, device="cuda"
            )
            pages_per_block = 16
            create_mla_kv_page_table_for_dcp[
                (len(lengths), triton.cdiv(table_width, pages_per_block))
            ](
                req_to_token,
                req_pool_indices,
                local_seq_lens,
                block_tables,
                req_to_token.stride(0),
                table_width,
                PHYSICAL_PAGE_SIZE=physical_page_size,
                DCP_SIZE=dcp_size,
                DCP_RANK=rank,
                PAGES_PER_BLOCK=pages_per_block,
            )

            for req_idx, local_len in enumerate(local_seq_lens.tolist()):
                num_local_pages = triton.cdiv(local_len, physical_page_size)
                expected = physical_pages_by_req[req_idx][:num_local_pages]
                self.assertEqual(
                    block_tables[req_idx, :num_local_pages].cpu().tolist(), expected
                )
                self.assertTrue(
                    torch.all(block_tables[req_idx, num_local_pages:] == -1)
                )

    def test_mla_cache_write_uses_exactly_one_dcp_shard(self):
        dcp_size = 4
        global_tokens = 512
        local_tokens = global_tokens // dcp_size
        loc = torch.arange(global_tokens, dtype=torch.int64, device="cuda")
        cache_k_nope = (
            torch.arange(1, global_tokens + 1, dtype=torch.float32, device="cuda")
            .to(torch.bfloat16)
            .view(global_tokens, 1, 1)
        )
        cache_k_rope = -cache_k_nope

        for rank in range(dcp_size):
            kv_buffer = torch.zeros(
                (local_tokens, 1, 2), dtype=torch.bfloat16, device="cuda"
            )
            parallel = SimpleNamespace(
                dcp_enabled=True,
                attn_dcp_size=dcp_size,
                attn_dcp_rank=rank,
            )
            with patch(
                "sglang.kernels.ops.kvcache.mla_buffer.get_parallel",
                return_value=parallel,
            ):
                set_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)

            expected = cache_k_nope[rank::dcp_size, 0, 0]
            self.assertEqual(kv_buffer.shape[0] * dcp_size, global_tokens)
            torch.testing.assert_close(kv_buffer[:, 0, 0], expected)
            torch.testing.assert_close(kv_buffer[:, 0, 1], -expected)


if __name__ == "__main__":
    unittest.main()
