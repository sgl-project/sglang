import unittest

import torch

from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.test_utils import CustomTestCase

TEST_CASES = [
    # Sequence with same prefix lens
    {
        "batch_size": 3,
        "prefix_lens": [64, 64, 64],
        "max_chunk_capacity": 48,
        "prefix_chunk_len": 16,
        "num_prefix_chunks": 4,
        "prefix_chunk_starts": torch.tensor(
            [
                [0, 0, 0],
                [16, 16, 16],
                [32, 32, 32],
                [48, 48, 48],
            ],
            dtype=torch.int32,
        ),
        "prefix_chunk_seq_lens": torch.tensor(
            [
                [16, 16, 16],
                [16, 16, 16],
                [16, 16, 16],
                [16, 16, 16],
            ],
            dtype=torch.int32,
        ),
    },
    # Sequence with different prefix lens
    {
        "batch_size": 4,
        "prefix_lens": [16, 32, 48, 64],
        "max_chunk_capacity": 64,
        "prefix_chunk_len": 16,
        "num_prefix_chunks": 4,
        "prefix_chunk_starts": torch.tensor(
            [
                [0, 0, 0, 0],
                [16, 16, 16, 16],
                [32, 32, 32, 32],
                [48, 48, 48, 48],
            ],
            dtype=torch.int32,
        ),
        "prefix_chunk_seq_lens": torch.tensor(
            [
                [16, 16, 16, 16],
                [0, 16, 16, 16],
                [0, 0, 16, 16],
                [0, 0, 0, 16],
            ],
            dtype=torch.int32,
        ),
    },
    # Sequence with irregular shapes
    {
        "batch_size": 2,
        "prefix_lens": [1, 64],
        "max_chunk_capacity": 31,
        "prefix_chunk_len": 15,
        "num_prefix_chunks": 5,
        "prefix_chunk_starts": torch.tensor(
            [
                [0, 0],
                [15, 15],
                [30, 30],
                [45, 45],
                [60, 60],
            ],
            dtype=torch.int32,
        ),
        "prefix_chunk_seq_lens": torch.tensor(
            [
                [1, 15],
                [0, 15],
                [0, 15],
                [0, 15],
                [0, 4],
            ],
            dtype=torch.int32,
        ),
    },
]


class MockForwardBatch(ForwardBatch):
    def __init__(self, max_chunk_capacity: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_chunk_capacity = max_chunk_capacity

    def get_max_chunk_capacity(self):
        return self.max_chunk_capacity


class MockReqToTokenPool:
    def __init__(self, batch_size, seq_len, device):
        self.req_to_token = (
            torch.arange(batch_size * seq_len, device=device)
            .reshape(batch_size, seq_len)
            .to(torch.int32)
        )


# Test correctness of triton kernel for computing kv indices
def check_kv_indices(forward_batch):
    for i in range(forward_batch.num_prefix_chunks):
        computed_kv_indices = forward_batch.prefix_chunk_kv_indices[i]
        req_to_token = forward_batch.req_to_token_pool.req_to_token[
            : forward_batch.batch_size, :
        ]
        ref_kv_indices = torch.empty(
            forward_batch.prefix_chunk_num_tokens[i],
            dtype=torch.int32,
            device=computed_kv_indices.device,
        )
        running_ptr = 0
        for j in range(forward_batch.batch_size):
            seq_start = forward_batch.prefix_chunk_starts[i, j].item()
            seq_len = forward_batch.prefix_chunk_seq_lens[i, j].item()
            ref_kv_indices[running_ptr : running_ptr + seq_len].copy_(
                req_to_token[j, seq_start : seq_start + seq_len]
            )
            running_ptr += seq_len
        assert torch.allclose(computed_kv_indices, ref_kv_indices)


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestPrefixChunkInfo(CustomTestCase):
    def setUp(self):
        # Common test parameters
        self.num_local_heads = 128
        self.kv_lora_rank = 512
        self.qk_rope_head_dim = 64
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.extend_len = 64
        self.max_bs = 4
        self.max_seq_len = 128

        # req_to_token_pool
        self.req_to_token_pool = MockReqToTokenPool(
            self.max_bs,
            self.max_seq_len,
            self.device,
        )

        # token_to_kv_pool
        self.token_to_kv_pool = MLATokenToKVPool(
            size=self.max_bs * self.max_seq_len,
            page_size=1,  # only consider page=1 for unit test
            dtype=self.dtype,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            layer_num=1,  # only consider layer=1 for unit test
            device=self.device,
            enable_memory_saver=False,
        )

    def test_prefix_chunk_info(self):
        """Test the standard extend operation."""

        for test_case in TEST_CASES:
            print(
                f"Test case with batch_size={test_case['batch_size']}, prefix_lens={test_case['prefix_lens']}, max_chunk_capacity={test_case['max_chunk_capacity']}"
            )
            batch_size = test_case["batch_size"]
            prefix_lens_cpu = test_case["prefix_lens"]
            assert len(prefix_lens_cpu) == batch_size
            prefix_lens = torch.tensor(prefix_lens_cpu, device=self.device)
            max_chunk_capacity = test_case["max_chunk_capacity"]
            seq_lens_cpu = [
                self.extend_len + prefix_lens_cpu[i] for i in range(batch_size)
            ]
            seq_lens = torch.tensor(seq_lens_cpu, device=self.device)

            # Create forward batch
            # input_ids and out_cache_loc are dummy tensors in this test
            forward_batch = MockForwardBatch(
                max_chunk_capacity=max_chunk_capacity,
                batch_size=batch_size,
                input_ids=torch.randint(
                    0, 100, (batch_size, self.extend_len), device=self.device
                ),
                out_cache_loc=torch.arange(
                    self.max_bs * self.max_seq_len - batch_size * self.extend_len,
                    self.max_bs * self.max_seq_len,
                    device=self.device,
                ),
                seq_lens_sum=sum(seq_lens_cpu),
                forward_mode=ForwardMode.EXTEND,
                req_pool_indices=torch.arange(batch_size, device=self.device),
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                extend_prefix_lens=prefix_lens,
                extend_prefix_lens_cpu=prefix_lens_cpu,
            )
            forward_batch.req_to_token_pool = self.req_to_token_pool
            forward_batch.token_to_kv_pool = self.token_to_kv_pool

            forward_batch.prepare_chunked_prefix_cache_info(self.device)
            assert forward_batch.get_max_chunk_capacity() == max_chunk_capacity
            assert forward_batch.prefix_chunk_len == test_case["prefix_chunk_len"]
            assert forward_batch.num_prefix_chunks == test_case["num_prefix_chunks"]
            assert torch.allclose(
                forward_batch.prefix_chunk_starts,
                test_case["prefix_chunk_starts"].to(self.device),
            )
            assert torch.allclose(
                forward_batch.prefix_chunk_seq_lens,
                test_case["prefix_chunk_seq_lens"].to(self.device),
            )

            check_kv_indices(forward_batch)


if __name__ == "__main__":
    unittest.main()
