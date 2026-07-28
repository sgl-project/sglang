import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import (
    tokenspeed_mla_backend as tokenspeed_backend_module,
)
from sglang.srt.layers.attention import trtllm_mla_backend as trtllm_backend_module
from sglang.srt.layers.attention.tokenspeed_mla_backend import (
    TokenspeedMLABackend,
)
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLADecodeMetadata,
)
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla as forward_mla_module,
)
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    _is_dcp_mla_decode_phase,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTRTLLMMLADCPVerify(CustomTestCase):
    def test_decode_metadata_tracks_global_lengths(self):
        global_seq_lens = torch.tensor([11, 22], dtype=torch.int32)
        metadata = TRTLLMMLADecodeMetadata(global_seq_lens_k=global_seq_lens)

        self.assertIs(metadata.global_seq_lens_k, global_seq_lens)

    @patch.object(trtllm_backend_module, "fixup_zero_kv_rows")
    @patch.object(trtllm_backend_module, "get_parallel")
    def test_target_verify_forwards_dcp_kernel_contract(
        self, get_parallel, fixup_zero_rows
    ):
        get_parallel.return_value = SimpleNamespace(
            dcp_enabled=True, dcp_size=4, dcp_rank=2
        )
        backend = object.__new__(TRTLLMMLABackend)
        backend.forward_prefill_metadata = None
        backend.forward_decode_metadata = None
        backend.data_type = torch.bfloat16
        backend.q_data_type = torch.bfloat16
        backend.page_size = 1
        backend.kv_cache_dim = 2
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda _: torch.empty((8, 2), dtype=torch.bfloat16)
        )

        local_seq_lens = torch.tensor([3, 6], dtype=torch.int32)
        global_seq_lens = torch.tensor([11, 22], dtype=torch.int32)
        metadata = SimpleNamespace(
            block_kv_indices=torch.zeros((2, 1), dtype=torch.int32),
            seq_lens_k=local_seq_lens,
            global_seq_lens_k=global_seq_lens,
            max_seq_len_k=6,
            batch_size=2,
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=2,
            decode_trtllm_mla_metadata=metadata,
            spec_info=SimpleNamespace(draft_token_num=2),
        )
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=1,
            head_dim=2,
            v_head_dim=1,
        )
        query = torch.empty((4, 2), dtype=torch.bfloat16)
        raw_output = torch.empty((2, 2, 1, 1), dtype=torch.bfloat16)
        raw_lse = torch.empty((2, 2, 1), dtype=torch.float32)

        with patch.object(
            backend, "_run_decode_kernel", return_value=(raw_output, raw_lse)
        ) as run_kernel:
            output, lse = backend.forward_extend(
                query,
                None,
                None,
                layer,
                forward_batch,
                save_kv_cache=False,
            )

        kwargs = run_kernel.call_args.kwargs
        self.assertIs(kwargs["seq_lens"], local_seq_lens)
        self.assertIs(kwargs["causal_seqs"], global_seq_lens)
        self.assertEqual(kwargs["cp_world"], 4)
        self.assertEqual(kwargs["cp_rank"], 2)
        self.assertTrue(kwargs["return_lse"])
        self.assertEqual(kwargs["max_seq_len"], 6)
        self.assertEqual(output.shape, (4, 1))
        self.assertEqual(lse.shape, (4, 1))
        fixup_zero_rows.assert_called_once()

    @patch.object(tokenspeed_backend_module, "get_parallel")
    def test_tokenspeed_target_verify_splits_global_and_local_lengths(
        self, get_parallel
    ):
        get_parallel.return_value = SimpleNamespace(
            dcp_enabled=True, dcp_size=4, dcp_rank=2
        )
        backend = object.__new__(TokenspeedMLABackend)
        backend.num_draft_tokens = 8
        metadata = SimpleNamespace(
            block_kv_indices=torch.full((3, 4), -1, dtype=torch.int32),
            seq_lens_k=torch.zeros(3, dtype=torch.int32),
            global_seq_lens_k=torch.zeros(3, dtype=torch.int32),
        )
        backend.decode_cuda_graph_metadata = {3: metadata}
        prefix_lens = torch.tensor([10, 20, 30], dtype=torch.int32)
        req_pool_indices = torch.arange(3, dtype=torch.int32)

        with patch.object(backend, "_fill_dcp_block_kv_indices") as fill:
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
        torch.testing.assert_close(fill.call_args.args[2], expected_local)

    @patch(
        "sglang.srt.models.deepseek_common.attention_forward_methods."
        "forward_mla.get_parallel"
    )
    def test_target_verify_is_a_dcp_mla_decode_phase(self, get_parallel):
        get_parallel.return_value = SimpleNamespace(dcp_enabled=True)

        with (
            patch.object(forward_mla_module, "_is_cuda", True),
            patch.object(
                forward_mla_module,
                "get_server_args",
                return_value=SimpleNamespace(
                    decode_attention_backend="tokenspeed_mla",
                    attention_backend=None,
                    speculative_algorithm="DSPARK",
                    speculative_attention_mode="decode",
                ),
            ),
        ):
            self.assertTrue(
                _is_dcp_mla_decode_phase(
                    SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
                )
            )
        self.assertTrue(
            _is_dcp_mla_decode_phase(SimpleNamespace(forward_mode=ForwardMode.DECODE))
        )
        self.assertFalse(
            _is_dcp_mla_decode_phase(SimpleNamespace(forward_mode=ForwardMode.EXTEND))
        )

    @patch(
        "sglang.srt.models.deepseek_common.attention_forward_methods."
        "forward_mla.get_parallel"
    )
    def test_target_verify_does_not_change_other_dcp_mla_backends(self, get_parallel):
        get_parallel.return_value = SimpleNamespace(dcp_enabled=True)
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)

        for is_cuda, backend in (
            (False, "tokenspeed_mla"),
            (True, "flashinfer_mla"),
        ):
            with (
                self.subTest(is_cuda=is_cuda, backend=backend),
                patch.object(forward_mla_module, "_is_cuda", is_cuda),
                patch.object(
                    forward_mla_module,
                    "get_server_args",
                    return_value=SimpleNamespace(
                        decode_attention_backend=backend,
                        attention_backend=None,
                        speculative_algorithm="DSPARK",
                        speculative_attention_mode="decode",
                    ),
                ),
            ):
                self.assertFalse(_is_dcp_mla_decode_phase(forward_batch))


if __name__ == "__main__":
    unittest.main()
