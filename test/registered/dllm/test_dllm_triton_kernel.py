from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-test-large-1-gpu")

import unittest
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from sglang.srt.dllm.kernels.post_process import (
    dllm_post_process_fused,
    dllm_post_process_pytorch,
)
from sglang.test.test_utils import CustomTestCase

BLOCK_SIZE = 32
VOCAB_SIZE = 128000
MASK_ID = VOCAB_SIZE - 1
THRESHOLD = 0.95


@dataclass
class MockLogitsProcessorOutput:
    next_token_logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    full_logits: Optional[torch.Tensor] = None


@dataclass
class MockModelRunnerOutput:
    logits_output: MockLogitsProcessorOutput = None
    can_run_graph: bool = True


class MockForwardBatch:
    def __init__(self, input_ids: torch.Tensor, batch_size: int):
        self.input_ids = input_ids
        self.batch_size = batch_size


class MockModelRunner:

    def __init__(self, vocab_size: int, seed: int = 42):
        self.vocab_size = vocab_size
        self.call_count = 0
        self.seed = seed

    def forward(self, forward_batch, pp_proxy_tensors=None):
        self.call_count += 1
        seq_len = forward_batch.input_ids.shape[0]
        device = forward_batch.input_ids.device
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed + self.call_count)
        logits = torch.randn(
            seq_len,
            self.vocab_size,
            dtype=torch.float16,
            device=device,
            generator=gen,
        )
        logits_output = MockLogitsProcessorOutput(full_logits=logits)
        return MockModelRunnerOutput(logits_output=logits_output, can_run_graph=True)


def _run_loop_pytorch(
    model_runner, input_ids, batch_size, block_size, mask_id, threshold
):
    """PyTorch loop matching LowConfidence.run()."""
    import torch.nn.functional as F

    forward_batch = MockForwardBatch(input_ids, batch_size)
    mask_index = input_ids == mask_id
    if torch.sum(mask_index).item() == 0:
        model_runner.forward(forward_batch)
        return input_ids.clone()

    start_list = []
    for block_id in range(batch_size):
        s = block_id * block_size
        e = s + block_size
        block_mask = input_ids[s:e] == mask_id
        start_list.append(block_size - torch.sum(block_mask).item())

    for _ in range(block_size):
        out = model_runner.forward(forward_batch)
        full_logits = out.logits_output.full_logits
        for bid in range(batch_size):
            cs, ce = bid * block_size, (bid + 1) * block_size
            blk_ids = input_ids[cs:ce]
            blk_mask = blk_ids == mask_id
            if torch.sum(blk_mask).item() == 0:
                continue
            logits = full_logits[cs:ce]
            x = torch.argmax(logits, dim=-1)
            p = torch.squeeze(
                torch.gather(
                    F.softmax(logits, dim=-1),
                    dim=-1,
                    index=torch.unsqueeze(x, -1),
                ),
                -1,
            )
            x = torch.where(blk_mask, x, blk_ids)
            confidence = torch.where(blk_mask, p, -np.inf)
            transfer = confidence > threshold
            if transfer.sum().item() == 0:
                _, sel = torch.topk(confidence, k=1)
                transfer[sel] = True
            blk_ids[transfer] = x[transfer]

        mask_count = (input_ids == mask_id).sum().item()
        if mask_count == 0:
            break

    return input_ids.clone(), start_list


def _run_loop_triton(
    model_runner, input_ids, batch_size, block_size, mask_id, threshold
):
    """Triton loop matching LowConfidence.run()."""
    forward_batch = MockForwardBatch(input_ids, batch_size)
    mask_index = input_ids == mask_id
    if torch.sum(mask_index).item() == 0:
        model_runner.forward(forward_batch)
        return input_ids.clone()

    start_list = []
    for block_id in range(batch_size):
        s = block_id * block_size
        e = s + block_size
        block_mask = input_ids[s:e] == mask_id
        start_list.append(block_size - torch.sum(block_mask).item())

    for _ in range(block_size):
        out = model_runner.forward(forward_batch)
        full_logits = out.logits_output.full_logits
        for bid in range(batch_size):
            cs, ce = bid * block_size, (bid + 1) * block_size
            blk_ids = input_ids[cs:ce]
            logits = full_logits[cs:ce]
            dllm_post_process_fused(logits, blk_ids, mask_id, threshold)

        mask_count = (input_ids == mask_id).sum().item()
        if mask_count == 0:
            break

    return input_ids.clone(), start_list


class TestDllmTritonKernel(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda")

    def test_kernel_all_masked(self):
        """All positions masked — kernel must fill every slot."""
        torch.manual_seed(42)
        logits = torch.randn(
            BLOCK_SIZE, VOCAB_SIZE, dtype=torch.float16, device=self.device
        )
        ids_pt = torch.full(
            (BLOCK_SIZE,), MASK_ID, dtype=torch.int64, device=self.device
        )
        ids_tr = ids_pt.clone()

        out_pt, _, _, _ = dllm_post_process_pytorch(logits, ids_pt, MASK_ID, THRESHOLD)
        dllm_post_process_fused(logits, ids_tr, MASK_ID, THRESHOLD)

        self.assertTrue(torch.equal(out_pt, ids_tr))

    def test_kernel_no_masked(self):
        """No masked positions — kernel should be a no-op on ids."""
        torch.manual_seed(42)
        logits = torch.randn(
            BLOCK_SIZE, VOCAB_SIZE, dtype=torch.float16, device=self.device
        )
        ids = torch.randint(
            0, VOCAB_SIZE - 1, (BLOCK_SIZE,), dtype=torch.int64, device=self.device
        )
        ids_tr = ids.clone()

        out_pt, _, _, _ = dllm_post_process_pytorch(
            logits, ids.clone(), MASK_ID, THRESHOLD
        )
        dllm_post_process_fused(logits, ids_tr, MASK_ID, THRESHOLD)

        self.assertTrue(torch.equal(out_pt, ids_tr))

    def test_kernel_mixed(self):
        """Half masked, half real tokens."""
        torch.manual_seed(42)
        logits = torch.randn(
            BLOCK_SIZE, VOCAB_SIZE, dtype=torch.float16, device=self.device
        )
        ids_base = torch.randint(
            0, VOCAB_SIZE - 1, (BLOCK_SIZE,), dtype=torch.int64, device=self.device
        )
        mask_positions = torch.rand(BLOCK_SIZE, device=self.device) < 0.5
        ids_base[mask_positions] = MASK_ID

        ids_pt = ids_base.clone()
        ids_tr = ids_base.clone()

        out_pt, _, _, _ = dllm_post_process_pytorch(logits, ids_pt, MASK_ID, THRESHOLD)
        dllm_post_process_fused(logits, ids_tr, MASK_ID, THRESHOLD)

        self.assertTrue(torch.equal(out_pt, ids_tr))

    def test_kernel_fallback(self):
        """When no token exceeds threshold, fallback must accept exactly one."""
        torch.manual_seed(42)
        # Use uniform logits so no single token has high probability
        logits = torch.zeros(
            BLOCK_SIZE, VOCAB_SIZE, dtype=torch.float16, device=self.device
        )
        ids_pt = torch.full(
            (BLOCK_SIZE,), MASK_ID, dtype=torch.int64, device=self.device
        )
        ids_tr = ids_pt.clone()

        out_pt, _, _, _ = dllm_post_process_pytorch(logits, ids_pt, MASK_ID, THRESHOLD)
        dllm_post_process_fused(logits, ids_tr, MASK_ID, THRESHOLD)

        self.assertTrue(torch.equal(out_pt, ids_tr))

    def test_loop_single_batch(self):
        """Single-batch loop: Triton and PyTorch must converge identically."""
        torch.manual_seed(123)
        seq_len = BLOCK_SIZE
        base = torch.randint(0, VOCAB_SIZE - 1, (seq_len,), device=self.device)
        base[BLOCK_SIZE // 2 :] = MASK_ID

        ids_pt = base.clone()
        ids_tr = base.clone()
        runner_pt = MockModelRunner(VOCAB_SIZE, seed=42)
        runner_tr = MockModelRunner(VOCAB_SIZE, seed=42)

        res_pt = _run_loop_pytorch(runner_pt, ids_pt, 1, BLOCK_SIZE, MASK_ID, THRESHOLD)
        res_tr = _run_loop_triton(runner_tr, ids_tr, 1, BLOCK_SIZE, MASK_ID, THRESHOLD)

        out_pt, starts_pt = res_pt
        out_tr, starts_tr = res_tr
        self.assertTrue(torch.equal(out_pt, out_tr))
        self.assertEqual(starts_pt, starts_tr)

    def test_loop_multi_batch(self):
        """Multi-batch loop with varying mask patterns per block."""
        batch_size = 4
        seq_len = BLOCK_SIZE * batch_size

        torch.manual_seed(999)
        base = torch.randint(
            0, VOCAB_SIZE - 1, (seq_len,), dtype=torch.int64, device=self.device
        )
        for b in range(batch_size):
            num_masks = min((b + 1) * 4, BLOCK_SIZE)
            s = (b + 1) * BLOCK_SIZE - num_masks
            e = (b + 1) * BLOCK_SIZE
            base[s:e] = MASK_ID

        ids_pt = base.clone()
        ids_tr = base.clone()
        runner_pt = MockModelRunner(VOCAB_SIZE, seed=42)
        runner_tr = MockModelRunner(VOCAB_SIZE, seed=42)

        res_pt = _run_loop_pytorch(
            runner_pt, ids_pt, batch_size, BLOCK_SIZE, MASK_ID, THRESHOLD
        )
        res_tr = _run_loop_triton(
            runner_tr, ids_tr, batch_size, BLOCK_SIZE, MASK_ID, THRESHOLD
        )

        out_pt, starts_pt = res_pt
        out_tr, starts_tr = res_tr
        self.assertTrue(torch.equal(out_pt, out_tr))
        self.assertEqual(starts_pt, starts_tr)


if __name__ == "__main__":
    unittest.main()
