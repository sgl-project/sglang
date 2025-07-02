import torch
import numpy as np
import os
import unittest

from sglang.srt.utils import next_power_of_2
from sglang.srt.speculative.eagle_utils import assign_draft_cache_locs
from sglang.srt.mem_cache.memory_pool import copy_all_layer_kv_cache

class TestEagleUtils(unittest.TestCase):

    def test_assign_draft_cache_locs_single_seq(self):
        # Testing Setup: req_to_token starting from 4
        # 4,5,6,7,{8,9,10}, 8,9,10 is the last partial page, 3 tokens < page_size=4
        # next kv cache will be stored starting 10,11,12...
        if os.environ.get("TRITON_INTERPRET") == "1":
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        num_seqs = 1
        page_size = 4
        speculative_num_steps = 5
        topk = 8
        seq_lens_num = 7
        extend_lens_num = 61
        req_pool_indices = torch.arange(num_seqs, dtype=torch.int32, device=device)
        req_to_token = torch.zeros((num_seqs, 100), dtype=torch.int32, device=device)
        req_to_token[0, :seq_lens_num] = torch.tensor([4, 5, 6, 7, 8, 9, 10], device=device)
        seq_lens = torch.tensor([seq_lens_num], dtype=torch.int32, device=device)
        extend_lens = torch.tensor([extend_lens_num], dtype=torch.int32, device=device)
        num_new_pages_per_topk = torch.tensor([2], dtype=torch.int32, device=device)
        out_cache_loc = torch.arange(11, 11 + extend_lens_num, device=device)
        last_page_lens = torch.tensor([3], dtype=torch.int32, device=device)
        duplicate_cache_len = last_page_lens.sum() * (topk - 1)
        target_cache_loc = torch.zeros(duplicate_cache_len, dtype=torch.int32, device=device)
        source_cache_loc = torch.zeros(duplicate_cache_len, dtype=torch.int32, device=device)
        assign_draft_cache_locs[(num_seqs,)](
                req_pool_indices,
                req_to_token,
                seq_lens,
                extend_lens,
                num_new_pages_per_topk,
                out_cache_loc,
                source_cache_loc,
                target_cache_loc,
                req_to_token.shape[1],
                topk,
                speculative_num_steps,
                page_size,
                next_power_of_2(num_seqs),
                next_power_of_2(speculative_num_steps),
        )

        out_cache_loc = out_cache_loc[
            : num_seqs * topk * speculative_num_steps
        ]
        print(f"{out_cache_loc=}")
        expected_source_cache_loc=torch.tensor([8, 9, 10] * (topk - 1), device=device, dtype=torch.int32)
        assert torch.allclose(source_cache_loc, expected_source_cache_loc)
        data_ptrs = torch.zeros(2, 1, dtype=torch.uint64, device=device)
        k_cache = [torch.zeros((100,1,1), dtype=torch.float32, device=device)]
        v_cache = [torch.zeros((100,1,1), dtype=torch.float32, device=device)]
        k_cache[0][:11, 0, 0] = torch.tensor(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            dtype=torch.float32,
            device=device,
        )
        v_cache[0][:11, 0, 0] = torch.tensor(
            [-0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0],
            dtype=torch.float32,
            device=device,
        )
        data_ptrs[0,0] = k_cache[0].data_ptr()
        data_ptrs[1,0] = v_cache[0].data_ptr()

        data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in k_cache + v_cache
            ],
            device=device,
            dtype=torch.int64,
        )
        copy_all_layer_kv_cache[(len(data_ptrs),)](
            data_ptrs,
            data_strides,
            target_cache_loc,
            source_cache_loc,
            len(target_cache_loc),
            next_power_of_2(len(target_cache_loc)),
        )
        assert torch.allclose(k_cache[0][16:19, 0, 0], torch.tensor(
            [0.8, 0.9, 1.0],
            dtype=torch.float32,
            device=device,
        ))
        assert torch.allclose(v_cache[0][16:19, 0, 0], torch.tensor(
            [-0.8, -0.9, -1.0],
            dtype=torch.float32,
            device=device,
        ))


    def test_assign_draft_cache_locs_multi_seq(self):
        test_assign_draft_cache_locs_multi_seq()


if __name__ == "__main__":
    unittest.main()