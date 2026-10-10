"""Equivalence tests for the sync-free EAGLE verify and draft planning.

`fast_verify_plan` replaces FlashInfer's `BatchPrefillWithPagedKVCacheWrapper.plan`
in the EAGLE target-verify CUDA graph when SGLANG_ENABLE_SYNC_FREE_SPEC_PLAN is set,
and `draft_kv_indptr_host` replaces the multi-step draft's `.cpu()` of the kv_indptr
rows. Both must leave exactly the state the stock path leaves: the graph replays the
same kernels on whatever the plan wrote, so any difference in `_plan_info`, the
pinned plan buffer or a device buffer the graph reads changes the verify output.
"""

import unittest

import torch

from sglang.kernels.ops.speculative.cache_locs import generate_draft_decode_kv_indices
from sglang.srt.layers.attention.flashinfer_sync_free_plan import (
    draft_kv_indptr_host,
    eagle_verify_plan_host_inputs,
    fast_verify_plan,
)
from sglang.srt.utils import next_power_of_2
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

try:
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    _HAS_FLASHINFER = True
except ImportError:
    _HAS_FLASHINFER = False

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

# GQA shapes like a small hybrid model's full-attention layers; page_size 1.
NUM_QO_HEADS = 16
NUM_KV_HEADS = 4
HEAD_DIM = 128
DRAFT_TOKEN_NUM = 4
MAX_LEN = 3000
# FlashInferAttnBackend.get_cuda_graph_seq_len_fill_value() for padding rows.
FILL = 1
DTYPE = torch.bfloat16


def _plan_args(seq_lens, generator, pool, device):
    """The plan() arguments EagleVerifyInput.generate_attn_arg_prefill produces."""
    bs = len(seq_lens)
    kv_lens = torch.tensor(seq_lens, dtype=torch.int64) + DRAFT_TOKEN_NUM
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(kv_lens, 0)
    qo_indptr = torch.arange(
        0, (bs + 1) * DRAFT_TOKEN_NUM, DRAFT_TOKEN_NUM, dtype=torch.int32
    )
    kv_indices = torch.randperm(pool, generator=generator)[: int(kv_indptr[-1])]
    # Any mask layout must round-trip; random bits cover top-k > 1 trees too.
    mask_bits = int((DRAFT_TOKEN_NUM * kv_lens).sum())
    custom_mask = torch.randint(0, 2, (mask_bits,), generator=generator).bool()
    args = (
        qo_indptr.to(device),
        kv_indptr.to(device),
        kv_indices.int().to(device),
        torch.ones(bs, dtype=torch.int32, device=device),
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        1,
    )
    kwargs = dict(
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        custom_mask=custom_mask.to(device),
        non_blocking=True,
    )
    return args, kwargs


def _plan_state(wrapper, packed_mask_len, bs, num_kv_indices):
    return {
        "plan_info": torch.tensor(list(wrapper._plan_info), dtype=torch.int64),
        "pinned": wrapper._pin_memory_int_workspace_buffer.clone(),
        "int_workspace": wrapper._int_workspace_buffer.cpu(),
        "qo_indptr": wrapper._qo_indptr_buf.cpu(),
        "kv_indptr": wrapper._paged_kv_indptr_buf.cpu(),
        "last_page_len": wrapper._paged_kv_last_page_len_buf.cpu(),
        "kv_indices": wrapper._paged_kv_indices_buf[:num_kv_indices].cpu(),
        "kv_lens": wrapper._kv_lens_buffer[:bs].cpu(),
        "mask": wrapper._custom_mask_buf[:packed_mask_len].cpu(),
        "mask_indptr": wrapper._mask_indptr_buf.cpu(),
        "max_q_len": torch.tensor([wrapper._max_q_len]),
        "max_kv_len": torch.tensor([wrapper._max_kv_len]),
    }


@unittest.skipUnless(_HAS_FLASHINFER, "requires flashinfer")
class TestFastVerifyPlan(CustomTestCase):
    def setUp(self):
        self.device = "cuda"
        self.generator = torch.Generator().manual_seed(0)
        self.pool = 1 << 17
        self.k_cache = torch.randn(
            self.pool, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=self.device
        )
        self.v_cache = torch.randn(
            self.pool, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=self.device
        )
        self.workspace = torch.empty(
            256 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )

    def _captured_wrapper(self, bs):
        """A cuda-graph wrapper planned once with stock plan() and a captured run()."""
        num_tokens = bs * DRAFT_TOKEN_NUM
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace,
            "NHD",
            use_cuda_graph=True,
            backend="fa2",
            qo_indptr_buf=torch.zeros(bs + 1, dtype=torch.int32, device=self.device),
            paged_kv_indptr_buf=torch.zeros(
                bs + 1, dtype=torch.int32, device=self.device
            ),
            paged_kv_indices_buf=torch.zeros(
                bs * (MAX_LEN + DRAFT_TOKEN_NUM), dtype=torch.int32, device=self.device
            ),
            paged_kv_last_page_len_buf=torch.ones(
                bs, dtype=torch.int32, device=self.device
            ),
            custom_mask_buf=torch.zeros(
                num_tokens * (MAX_LEN + DRAFT_TOKEN_NUM),
                dtype=torch.uint8,
                device=self.device,
            ),
            mask_indptr_buf=torch.zeros(bs + 1, dtype=torch.int32, device=self.device),
        )
        args, kwargs = _plan_args([FILL] * bs, self.generator, self.pool, self.device)
        wrapper.plan(*args, **kwargs)
        q = torch.zeros(
            num_tokens, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=self.device
        )
        wrapper.run(q, (self.k_cache, self.v_cache))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = wrapper.run(q, (self.k_cache, self.v_cache))
        return wrapper, graph, q, out

    def _replay(self, wrapper, graph, out, plan_fn, host):
        wrapper._pin_memory_int_workspace_buffer.zero_()
        out.zero_()
        plan_fn()
        graph.replay()
        torch.cuda.synchronize()
        bs = len(host["kv_lens_host"])
        state = _plan_state(
            wrapper, host["packed_mask_len"], bs, int(host["kv_indptr_host"][-1])
        )
        return state, out.clone()

    def test_matches_stock_plan(self):
        for bs in (1, 3, 16):
            wrapper, graph, q, out = self._captured_wrapper(bs)
            for _ in range(4):
                # Trailing cuda-graph padding rows carry the fill length.
                pad = int(torch.randint(0, bs, (1,), generator=self.generator))
                seq_lens = [
                    *torch.randint(1, MAX_LEN, (bs - pad,), generator=self.generator),
                    *[FILL] * pad,
                ]
                seq_lens = [int(x) for x in seq_lens]
                args, kwargs = _plan_args(
                    seq_lens, self.generator, self.pool, self.device
                )
                host = eagle_verify_plan_host_inputs(
                    seq_lens_cpu=torch.tensor(seq_lens),
                    draft_token_num=DRAFT_TOKEN_NUM,
                    bs=bs,
                )
                q.copy_(torch.randn(q.shape, generator=self.generator).to(q))

                stock_state, stock_out = self._replay(
                    wrapper,
                    graph,
                    out,
                    lambda: BatchPrefillWithPagedKVCacheWrapper.plan(
                        wrapper, *args, **kwargs
                    ),
                    host,
                )
                fast_state, fast_out = self._replay(
                    wrapper,
                    graph,
                    out,
                    lambda: fast_verify_plan(wrapper, *args, **kwargs, **host),
                    host,
                )
                for name, value in stock_state.items():
                    self.assertTrue(
                        torch.equal(value, fast_state[name]), f"bs={bs}: {name}"
                    )
                self.assertTrue(
                    torch.equal(stock_out.view(torch.int16), fast_out.view(torch.int16))
                )

    def test_mismatched_lengths_change_the_plan(self):
        """Guards against a vacuous comparison: host inputs one token short for one
        request must leave a different plan state than stock plan()."""
        bs = 3
        wrapper, graph, _, out = self._captured_wrapper(bs)
        seq_lens = [700, 25, 1900]
        args, kwargs = _plan_args(seq_lens, self.generator, self.pool, self.device)
        host = eagle_verify_plan_host_inputs(
            seq_lens_cpu=torch.tensor(seq_lens), draft_token_num=DRAFT_TOKEN_NUM, bs=bs
        )
        wrong = eagle_verify_plan_host_inputs(
            seq_lens_cpu=torch.tensor([700, 24, 1900]),
            draft_token_num=DRAFT_TOKEN_NUM,
            bs=bs,
        )
        stock_state, _ = self._replay(
            wrapper,
            graph,
            out,
            lambda: BatchPrefillWithPagedKVCacheWrapper.plan(wrapper, *args, **kwargs),
            host,
        )
        wrong_state, _ = self._replay(
            wrapper,
            graph,
            out,
            lambda: fast_verify_plan(wrapper, *args, **kwargs, **wrong),
            host,
        )
        self.assertTrue(
            any(not torch.equal(v, wrong_state[k]) for k, v in stock_state.items())
        )

    def test_draft_kv_indptr_matches_kernel(self):
        for topk, steps, window, sink in (
            (1, 3, 0, 0),
            (2, 5, 0, 0),
            (1, 3, 512, 4),
            (2, 3, 512, 4),
        ):
            for _ in range(4):
                num_seqs = int(torch.randint(1, 33, (1,), generator=self.generator))
                pad = int(torch.randint(0, num_seqs, (1,), generator=self.generator))
                raw = num_seqs - pad
                seq_lens = torch.randint(
                    1, MAX_LEN, (num_seqs,), generator=self.generator
                )
                seq_lens[raw:] = FILL
                # Draft positions: committed lengths repeated topk times, zero on
                # padding rows (EAGLEDraftCudaGraphRunner zeroes its position buffer).
                positions = torch.zeros(num_seqs * topk, dtype=torch.int64)
                positions[: raw * topk] = seq_lens[:raw].repeat_interleave(topk)
                pool_len = MAX_LEN + 64
                req_to_token = torch.arange(
                    num_seqs * pool_len, dtype=torch.int32, device=self.device
                ).view(num_seqs, pool_len)
                rows = num_seqs * topk
                kv_indices = torch.zeros(
                    (steps, rows * (MAX_LEN + steps)),
                    dtype=torch.int32,
                    device=self.device,
                )
                kv_indptr = torch.zeros(
                    (steps, 32 * topk + 1), dtype=torch.int32, device=self.device
                )
                generate_draft_decode_kv_indices[(steps, num_seqs, topk)](
                    torch.arange(num_seqs, device=self.device),
                    req_to_token,
                    seq_lens.to(self.device),
                    kv_indices,
                    kv_indptr,
                    positions.to(self.device),
                    pool_len,
                    kv_indices.shape[1],
                    kv_indptr.shape[1],
                    next_power_of_2(num_seqs),
                    next_power_of_2(steps),
                    next_power_of_2(rows),
                    1,
                    window,
                    sink,
                )
                host = draft_kv_indptr_host(
                    seq_lens_cpu=seq_lens,
                    num_seqs=num_seqs,
                    num_padding=pad,
                    topk=topk,
                    num_steps=steps,
                    window_cap=window + sink if window > 0 else 0,
                )
                self.assertTrue(
                    torch.equal(kv_indptr[:, : rows + 1].cpu(), host),
                    f"topk={topk} steps={steps} window={window}",
                )


if __name__ == "__main__":
    unittest.main()
