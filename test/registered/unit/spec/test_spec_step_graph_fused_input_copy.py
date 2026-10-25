# SPDX-License-Identifier: Apache-2.0
"""``SGLANG_SPEC_STEP_GRAPH_FUSED_INPUT_COPY``: one launch for a step-graph replay's input copies.

``_copy_step_inputs`` copies the live step inputs into the graph's static
buffers. With the small-tensor copy it is one launch when every tensor
qualifies ("fused"), one launch for the small tensors plus a copy per large
one ("partial", batch sizes above 1), and per-tensor copies otherwise (a tensor
that does not qualify, or a live tensor aliasing a static buffer). The static
buffers hold the same bytes either way.
"""

import unittest

import torch

from sglang.kernels.ops.memory.small_copy import try_small_copy
from sglang.srt.speculative.eagle_step_cuda_graph_runner import _copy_step_inputs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")

HIDDEN = 6144
TOPK = 2048


def _random(shape, dtype, generator):
    """A tensor of random bit patterns (NaN / inf payloads included for floats)."""
    numel = 1
    for d in shape:
        numel *= d
    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    raw = torch.randint(
        0, 256, (nbytes,), dtype=torch.uint8, device="cuda", generator=generator
    )
    if dtype == torch.bool:
        raw = raw % 2
    return raw.view(dtype).view(shape)


def _step_inputs(bs, generator):
    """Live inputs shaped like an EAGLE topk-1 decode step's per-step tensors."""
    specs = [
        ((bs,), torch.int64),  # req_pool_indices
        ((bs,), torch.int64),  # seq_lens
        ((bs,), torch.int32),  # orig_seq_lens
        ((bs, 1), torch.int64),  # top-k index
        ((bs, TOPK), torch.int32),  # sparse-attention top-k seed rows
        ((bs,), torch.int64),  # verified ids
        ((bs, HIDDEN), torch.bfloat16),  # hidden states
        ((bs, 1), torch.float32),  # top-k probabilities
        ((bs, 1), torch.float32),  # temperatures
        ((bs,), torch.float32),  # top-p
        ((bs,), torch.int32),  # top-k
        ((bs,), torch.float32),  # min-p
        ((bs,), torch.bool),  # a per-request flag
    ]
    live = [_random(shape, dtype, generator) for shape, dtype in specs]
    static = [torch.empty_like(t) for t in live]
    return static, live


def _bytes(tensor):
    return tensor.contiguous().view(-1).view(torch.uint8)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestStepGraphFusedInputCopy(CustomTestCase):
    def test_fused_equals_per_tensor(self):
        generator = torch.Generator(device="cuda").manual_seed(1234)
        for bs in (1, 2, 4, 8):
            with self.subTest(bs=bs):
                static, live = _step_inputs(bs, generator)
                ref = [torch.empty_like(t) for t in live]
                self.assertEqual(_copy_step_inputs(ref, live, None), "per_tensor")
                # bs 1: every tensor fits the one-launch copy; larger batches
                # exceed its element limit with the hidden states / top-k rows.
                want = "fused" if bs == 1 else "partial"
                self.assertEqual(_copy_step_inputs(static, live, try_small_copy), want)
                torch.cuda.synchronize()
                for i, (r, dst, src) in enumerate(zip(ref, static, live)):
                    self.assertEqual(
                        (dst.shape, dst.dtype, dst.stride()),
                        (src.shape, src.dtype, src.stride()),
                    )
                    self.assertTrue(torch.equal(_bytes(r), _bytes(src)), f"ref {i}")
                    self.assertTrue(torch.equal(_bytes(dst), _bytes(src)), f"fused {i}")

    def test_repeated_replays_overwrite(self):
        generator = torch.Generator(device="cuda").manual_seed(7)
        static, live = _step_inputs(1, generator)
        for _ in range(3):
            _, fresh = _step_inputs(1, generator)
            for src, new in zip(live, fresh):
                src.copy_(new)
            self.assertEqual(_copy_step_inputs(static, live, try_small_copy), "fused")
            torch.cuda.synchronize()
            for dst, src in zip(static, live):
                self.assertTrue(torch.equal(_bytes(dst), _bytes(src)))

    def test_declines_to_per_tensor_copies(self):
        generator = torch.Generator(device="cuda").manual_seed(11)
        static, live = _step_inputs(1, generator)
        # A 3-d tensor does not qualify for the one-launch copy: every tensor
        # is then copied per tensor, with the same bytes.
        live.append(_random((1, 2, 3), torch.float32, generator))
        static.append(torch.empty_like(live[-1]))
        self.assertEqual(_copy_step_inputs(static, live, try_small_copy), "per_tensor")
        torch.cuda.synchronize()
        for dst, src in zip(static, live):
            self.assertTrue(torch.equal(_bytes(dst), _bytes(src)))

    def test_shared_memory_keeps_per_tensor_order(self):
        generator = torch.Generator(device="cuda").manual_seed(5)
        static, live = _step_inputs(2, generator)
        # A live tensor that is a view of a static buffer: the partial path
        # would reorder the copies, so the per-tensor copies run in list order
        # (static[0] receives static[1]'s bytes from before its own copy).
        live[0] = static[1].view(-1)[: live[0].numel()].view(torch.int64)
        want = [_bytes(t).clone() for t in live]
        self.assertEqual(_copy_step_inputs(static, live, try_small_copy), "per_tensor")
        torch.cuda.synchronize()
        for i, (dst, ref) in enumerate(zip(static, want)):
            self.assertTrue(torch.equal(_bytes(dst), ref), f"tensor {i}")


if __name__ == "__main__":
    unittest.main()
