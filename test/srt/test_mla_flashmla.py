import unittest
from types import SimpleNamespace

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

Q_LEN = 1

# class MockModelRunner:
#     def __init__(
#         self,
#         kv_lora_rank,
#         qk_rope_head_dim,
#     ):
#         attention_arch = AttentionArch.MLA
#         self.device = "cuda"
#         self.dtype = torch.float16
#         context_len = 2048
#         self.model_config = type(
#             "ModelConfig",
#             (),
#             {
#                 "context_len": context_len,
#                 "attention_arch": attention_arch,
#             },
#         )
#         self.sliding_window_size = None

#         batch_size = 160
#         # Create a proper req_to_token_pool with the req_to_token attribute
#         self.req_to_token_pool = type(
#             "TokenPool",
#             (),
#             {
#                 # A typical max_bs * max_context_len for cuda graph decode
#                 "size": batch_size,
#                 # Add req_to_token attribute
#                 "req_to_token": torch.zeros(
#                     batch_size, context_len, dtype=torch.int32, device=self.device
#                 ),
#             },
#         )
#         self.page_size = 1
#         max_total_num_tokens = batch_size * context_len
#         self.token_to_kv_pool = MLATokenToKVPool(
#             size=max_total_num_tokens,
#             page_size=self.page_size,
#             dtype=self.dtype,
#             kv_lora_rank=kv_lora_rank,
#             qk_rope_head_dim=qk_rope_head_dim,
#             layer_num=1,  # only consider layer=1 for unit test
#             device=self.device,
#             enable_memory_saver=False,
#         )


# class MockReqToTokenPool:
#     def __init__(self, batch_size, seq_len, device):
#         self.req_to_token = (
#             torch.arange(batch_size * seq_len, device=device)
#             .reshape(batch_size, seq_len)
#             .to(torch.int32)
#         )

# @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
# class TestFlashMLABackend(CustomTestCase):
#     def setUp(self):
#         # Test parameters
#         self.batch_size = 2
#         self.seq_len = 360
#         self.num_heads = 2
#         self.device = "cuda"
#         self.dtype = torch.float16
#         self.kv_lora_rank = 512
#         self.q_lora_rank = 128
#         self.qk_rope_head_dim = 64
#         self.qk_head_dim = self.qk_rope_head_dim + self.kv_lora_rank
#         # Assume no rope scaling
#         self.scaling = self.qk_head_dim**-0.5
#         # Initialize model runner and backend
#         self._init_model_runner()
#         self.backend = FlashMLABackend(self.model_runner)
#         self.num_local_heads = 2
#         self.page_size = 64

#     def _init_model_runner(self):
#         self.model_runner = MockModelRunner(
#             kv_lora_rank=self.kv_lora_rank,
#             qk_rope_head_dim=self.qk_rope_head_dim,
#         )
#         self.backend = FlashAttentionBackend(self.model_runner)

#     def _create_attention_layer(self):
#         """Create attention layer for testing."""
#         self.attn_mqa = RadixAttention(
#             num_heads=self.num_local_heads,
#             head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
#             scaling=self.scaling,
#             num_kv_heads=1,
#             layer_id=0,
#             v_head_dim=self.kv_lora_rank,
#             prefix="attn_mqa",
#         )
#         return self.attn_mqa

#     def _run_reference_forward(
#         self, mode, q, k, v, layer, forward_batch, expected_shape
#     ):
#         """Run reference forward pass using native backend."""
#         if mode == ForwardMode.EXTEND:
#             output = self.ref_backend.forward_extend(q, k, v, layer, forward_batch)
#         else:  # ForwardMode.DECODE
#             output = self.ref_backend.forward_decode(q, k, v, layer, forward_batch)
#         return output.view(expected_shape)

#     def _verify_output(self, output, expected_shape):
#         """Verify output tensor shape, dtype, and values."""
#         self.assertEqual(
#             output.shape,
#             expected_shape,
#             f"Expected shape {expected_shape}, got {output.shape}",
#         )
#         self.assertEqual(output.dtype, self.dtype)
#         self.assertEqual(output.device.type, "cuda")
#         self.assertEqual(
#             torch.isnan(output).sum().item(), 0, "Output contains NaN values"
#         )

#     def _create_forward_batch(self, mode, q_len=None, prefix_len=0):
#         """Create a forward batch for testing based on mode and lengths."""
#         # Default to self.seq_len if not specified
#         q_len = q_len or self.seq_len

#         if mode == ForwardMode.EXTEND:
#             total_len = prefix_len + q_len
#             out_cache_start = prefix_len * self.batch_size
#             out_cache_end = total_len * self.batch_size

#             forward_batch = ForwardBatch(
#                 batch_size=self.batch_size,
#                 input_ids=torch.randint(
#                     0, 100, (self.batch_size, q_len), device=self.device
#                 ),
#                 out_cache_loc=torch.arange(
#                     out_cache_start, out_cache_end, device=self.device
#                 ),
#                 seq_lens_sum=self.batch_size * total_len,
#                 forward_mode=mode,
#                 req_pool_indices=torch.arange(self.batch_size, device=self.device),
#                 seq_lens=torch.tensor(
#                     [total_len] * self.batch_size, device=self.device
#                 ),
#                 seq_lens_cpu=torch.tensor([total_len] * self.batch_size, device="cpu"),
#                 extend_prefix_lens=torch.tensor(
#                     [prefix_len] * self.batch_size, device=self.device
#                 ),
#                 extend_prefix_lens_cpu=torch.tensor(
#                     [prefix_len] * self.batch_size, device="cpu"
#                 ),
#                 extend_seq_lens=torch.tensor(
#                     [q_len] * self.batch_size, device=self.device
#                 ),
#                 extend_seq_lens_cpu=torch.tensor(
#                     [q_len] * self.batch_size, device="cpu"
#                 ),
#                 attn_backend=self.backend,
#             )

#         else:  # ForwardMode.DECODE
#             decode_len = Q_LEN  # typically 1 for decode mode
#             total_len = self.seq_len + decode_len
#             out_cache_start = self.batch_size * self.seq_len
#             out_cache_end = self.batch_size * total_len

#             forward_batch = ForwardBatch(
#                 batch_size=self.batch_size,
#                 input_ids=torch.randint(
#                     0, 100, (self.batch_size, decode_len), device=self.device
#                 ),
#                 out_cache_loc=torch.arange(
#                     out_cache_start, out_cache_end, device=self.device
#                 ),
#                 seq_lens_sum=self.batch_size * total_len,
#                 forward_mode=mode,
#                 req_pool_indices=torch.arange(self.batch_size, device=self.device),
#                 seq_lens=torch.tensor(
#                     [total_len] * self.batch_size, device=self.device
#                 ),
#                 seq_lens_cpu=torch.tensor([total_len] * self.batch_size, device="cpu"),
#                 attn_backend=self.backend,
#             )

#         # Add token pool from model runner to forward batch
#         forward_batch.req_to_token_pool = self.model_runner.req_to_token_pool

#         # Add KV cache from model runner to forward batch
#         forward_batch.token_to_kv_pool = self.model_runner.token_to_kv_pool

#         return forward_batch

#     def _setup_kv_cache(self, forward_batch, layer, cache_len):
#         """Set up KV cache with prefix tokens."""
#         if cache_len <= 0:
#             return

#         # Create constant values for the prefix cache for easy debugging
#         latent_cache = torch.ones(
#             self.batch_size * cache_len,
#             1,  # latent cache has only one head in MQA
#             self.kv_lora_rank + self.qk_rope_head_dim,
#             dtype=self.dtype,
#             device=self.device,
#         )

#         # Set the prefix KV cache
#         forward_batch.token_to_kv_pool.set_kv_buffer(
#             layer,
#             torch.arange(self.batch_size * cache_len, device=self.device),
#             latent_cache,
#             None,
#         )

#     def _run_attention_test(self, mode, q_len, prefix_len=0):
#         """
#             Run an attention test with the specified parameters.
#         Args:
#             mode: ForwardMode.EXTEND or ForwardMode.DECODE
#             q_len: Length of the query sequence. For decode mode, q_len is 1.
#             prefix_len: Length of the prefix sequence for extend mode
#         """
#         layer = self._create_attention_layer()

#         # Create forward batch and set up
#         forward_batch = self._create_forward_batch(mode, q_len, prefix_len)

#         # Create q, kv_compressed for testing
#         q_shape = (self.batch_size * q_len, self.num_heads, self.qk_head_dim)
#         kv_shape = (self.batch_size * q_len, self.qk_head_dim)
#         q = torch.randn(q_shape, dtype=self.dtype, device=self.device)
#         kv_compressed = torch.randn(kv_shape, dtype=self.dtype, device=self.device)
#         # v is not used for mqa, all values passed in through k
#         k = kv_compressed.unsqueeze(1)
#         v = torch.randn((1), dtype=self.dtype, device=self.device)

#         self._setup_kv_cache(forward_batch, layer, prefix_len)

#         self.backend.init_forward_metadata(forward_batch)

#         expected_shape = (
#             self.batch_size * q_len,
#             self.num_heads * self.kv_lora_rank,
#         )

#         if mode == ForwardMode.EXTEND:
#             output = self.backend.forward_extend(q, k, v, layer, forward_batch)
#         else:
#             output = self.backend.forward_decode(q, k, v, layer, forward_batch)

#         self._verify_output(output, expected_shape)
#         return output

#     def test_forward_extend(self):
#         """Test the standard extend operation."""
#         self._run_attention_test(ForwardMode.EXTEND, q_len=self.seq_len)

#     def test_forward_decode(self):
#         """Test the decode operation with cached tokens."""
#         self._run_attention_test(ForwardMode.DECODE, q_len=1)

#     def test_forward_extend_with_prefix(self):
#         """Test extending from cached prefix tokens."""
#         prefix_len = self.seq_len // 2
#         extend_len = self.seq_len - prefix_len
#         self._run_attention_test(
#             ForwardMode.EXTEND, q_len=extend_len, prefix_len=prefix_len
#         )


class TestFlashMLAMTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "lmsys/sglang-ci-dsv3-test"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--trust-remote-code"]
        if torch.cuda.is_available() and torch.version.cuda:
            other_args.extend(
                [
                    "--cuda-graph-max-bs",
                    "4",
                    "--disable-radix",
                    "--enable-torch-compile",
                    "--torch-compile-max-bs",
                    "1",
                    "--speculative-algorithm",
                    "EAGLE",
                    "--speculative-draft",
                    "lmsys/sglang-ci-dsv3-test-NextN",
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                    "--attention-backend",
                    "flashmla",
                    "--disable-cuda-graph",
                ]
            )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(metrics)

        self.assertGreater(metrics["accuracy"], 0.60)

        server_info = requests.get(self.base_url + "/get_server_info")
        print(f"{server_info=}")
        avg_spec_accept_length = server_info.json()["avg_spec_accept_length"]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, 2.5)


if __name__ == "__main__":
    unittest.main()
