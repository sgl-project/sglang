import unittest

import torch

from sglang.srt.lora.triton_ops import (
    step_a_q_fwd,
    step_a_v_fwd,
    step_b_q_fwd,
    step_b_v_fwd,
)
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, suite="stage-b-test-1-gpu-small")


class TestKvBLoRAAbsorbed(CustomTestCase):
    # fp16 has a 10-bit mantissa; matrix accumulations over rank<=4 with
    # 0.2-scaled inputs stay well within 5e-3.
    DTYPE = torch.float16
    RTOL = 5e-3
    ATOL = 5e-3

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for Triton LoRA kernels")
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        self.dtype = self.DTYPE
        self.num_heads = 3
        self.qk_nope_head_dim = 5
        self.v_head_dim = 4
        self.kv_lora_rank = 6
        self.max_rank = 4
        self.lora_ranks = [4, 2, 0]
        self.scalings = [0.5, 1.25, 0.0]

    def _batch_info(
        self,
        seg_indptr,
        weight_indices,
        permutation=None,
        use_cuda_graph=False,
        bs=None,
    ):
        seg_indptr_t = torch.tensor(seg_indptr, dtype=torch.int32, device=self.device)
        weight_indices_t = torch.tensor(
            weight_indices, dtype=torch.int32, device=self.device
        )
        permutation_t = (
            torch.tensor(permutation, dtype=torch.int32, device=self.device)
            if permutation is not None
            else None
        )
        seg_lens = [
            seg_indptr[i + 1] - seg_indptr[i] for i in range(len(seg_indptr) - 1)
        ]
        return LoRABatchInfo(
            use_cuda_graph=use_cuda_graph,
            bs=bs if bs is not None else len(weight_indices),
            num_segments=len(weight_indices),
            seg_indptr=seg_indptr_t,
            weight_indices=weight_indices_t,
            lora_ranks=torch.tensor(
                self.lora_ranks, dtype=torch.int32, device=self.device
            ),
            scalings=torch.tensor(
                self.scalings, dtype=torch.float32, device=self.device
            ),
            max_len=max(seg_lens) if seg_lens else 0,
            seg_lens=None,
            permutation=permutation_t,
        )

    def _weights(self):
        full_k = self.qk_nope_head_dim + self.v_head_dim
        A = 0.2 * torch.randn(
            len(self.lora_ranks),
            self.max_rank,
            self.kv_lora_rank,
            device=self.device,
            dtype=self.dtype,
        )
        B = 0.2 * torch.randn(
            len(self.lora_ranks),
            self.num_heads * full_k,
            self.max_rank,
            device=self.device,
            dtype=self.dtype,
        )
        return A.contiguous(), B.contiguous()

    def _segment_rows(self, batch_info):
        seg_indptr = batch_info.seg_indptr.cpu().tolist()
        weight_indices = batch_info.weight_indices.cpu().tolist()
        permutation = (
            batch_info.permutation.cpu().tolist()
            if batch_info.permutation is not None
            else None
        )
        for seg_id, weight_index in enumerate(weight_indices):
            start = seg_indptr[seg_id]
            end = seg_indptr[seg_id + 1]
            rows = (
                permutation[start:end]
                if permutation is not None
                else list(range(start, end))
            )
            yield rows, weight_index

    def _reference_q(self, q_nope, A, B, base_output, batch_info):
        full_k = self.qk_nope_head_dim + self.v_head_dim
        q_lora_a = torch.zeros(
            q_nope.shape[0],
            self.num_heads,
            self.max_rank,
            device=self.device,
            dtype=torch.float32,
        )
        output = base_output.float().clone()
        for rows, weight_index in self._segment_rows(batch_info):
            rank = self.lora_ranks[weight_index]
            if rank == 0 or not rows:
                continue
            row_t = torch.tensor(rows, dtype=torch.long, device=self.device)
            scaling = self.scalings[weight_index]
            for head in range(self.num_heads):
                b_start = head * full_k
                B_k = B[
                    weight_index,
                    b_start : b_start + self.qk_nope_head_dim,
                    :rank,
                ].float()
                tmp = q_nope[row_t, head, :].float() @ B_k
                q_lora_a[row_t, head, :rank] = tmp
                tmp_for_b = tmp.to(self.dtype).float()
                output[row_t, head, :] += (
                    tmp_for_b @ A[weight_index, :rank, :].float()
                ) * scaling
        return q_lora_a.to(self.dtype), output.to(self.dtype)

    def _reference_v(self, attn_output, A, B, base_output, batch_info):
        full_k = self.qk_nope_head_dim + self.v_head_dim
        attn_lora_a = torch.zeros(
            attn_output.shape[0],
            self.num_heads,
            self.max_rank,
            device=self.device,
            dtype=torch.float32,
        )
        output = base_output.float().clone()
        for rows, weight_index in self._segment_rows(batch_info):
            rank = self.lora_ranks[weight_index]
            if rank == 0 or not rows:
                continue
            row_t = torch.tensor(rows, dtype=torch.long, device=self.device)
            scaling = self.scalings[weight_index]
            A_t = A[weight_index, :rank, :].float().T
            for head in range(self.num_heads):
                tmp = attn_output[row_t, head, :].float() @ A_t
                attn_lora_a[row_t, head, :rank] = tmp
                tmp_for_b = tmp.to(self.dtype).float()
                b_start = head * full_k + self.qk_nope_head_dim
                B_v = B[
                    weight_index,
                    b_start : b_start + self.v_head_dim,
                    :rank,
                ].float()
                output[row_t, head, :] += (tmp_for_b @ B_v.T) * scaling
        return attn_lora_a.to(self.dtype), output.to(self.dtype)

    def _assert_valid_intermediate_close(self, actual, expected, batch_info):
        for rows, weight_index in self._segment_rows(batch_info):
            rank = self.lora_ranks[weight_index]
            if rank == 0 or not rows:
                continue
            row_t = torch.tensor(rows, dtype=torch.long, device=self.device)
            torch.testing.assert_close(
                actual[row_t, :, :rank],
                expected[row_t, :, :rank],
                rtol=self.RTOL,
                atol=self.ATOL,
            )

    def _run_case(self, name, batch_info, num_tokens):
        A, B = self._weights()
        q_nope = 0.2 * torch.randn(
            num_tokens,
            self.num_heads,
            self.qk_nope_head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        attn_output = 0.2 * torch.randn(
            num_tokens,
            self.num_heads,
            self.kv_lora_rank,
            device=self.device,
            dtype=self.dtype,
        )
        base_q = 0.2 * torch.randn(
            num_tokens,
            self.num_heads,
            self.kv_lora_rank,
            device=self.device,
            dtype=self.dtype,
        )
        base_v = 0.2 * torch.randn(
            num_tokens,
            self.num_heads,
            self.v_head_dim,
            device=self.device,
            dtype=self.dtype,
        )

        expected_q_a, expected_q = self._reference_q(q_nope, A, B, base_q, batch_info)
        actual_q_a = step_a_q_fwd(
            q_nope,
            B,
            batch_info,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        self._assert_valid_intermediate_close(actual_q_a, expected_q_a, batch_info)
        actual_q = step_b_q_fwd(actual_q_a, A, batch_info, base_q.clone())
        torch.testing.assert_close(
            actual_q, expected_q, rtol=self.RTOL, atol=self.ATOL, msg=name
        )

        expected_v_a, expected_v = self._reference_v(
            attn_output, A, B, base_v, batch_info
        )
        actual_v_a = step_a_v_fwd(attn_output, A, batch_info)
        self._assert_valid_intermediate_close(actual_v_a, expected_v_a, batch_info)
        actual_v = step_b_v_fwd(
            actual_v_a,
            B,
            batch_info,
            base_v.clone(),
            self.qk_nope_head_dim,
            self.v_head_dim,
        )
        torch.testing.assert_close(
            actual_v, expected_v, rtol=self.RTOL, atol=self.ATOL, msg=name
        )

    def test_contiguous_decode_single_adapter(self):
        batch_info = self._batch_info(
            seg_indptr=[0, 1, 2, 3, 4],
            weight_indices=[0, 0, 0, 0],
        )
        self._run_case("contiguous decode single adapter", batch_info, num_tokens=4)

    def test_contiguous_prefill_mixed_with_rank_zero(self):
        batch_info = self._batch_info(
            seg_indptr=[0, 3, 5, 6],
            weight_indices=[0, 1, 2],
        )
        self._run_case("contiguous prefill mixed", batch_info, num_tokens=6)

    def test_csgmv_permutation_mixed_with_rank_zero(self):
        batch_info = self._batch_info(
            seg_indptr=[0, 3, 6, 7],
            weight_indices=[1, 0, 2],
            permutation=[0, 2, 5, 1, 3, 4, 6],
        )
        self._run_case("csgmv permutation mixed", batch_info, num_tokens=7)

    def test_cuda_graph_padded_segments(self):
        # use_cuda_graph=True + bs > num_segments: grid axis-2 is sized to bs,
        # so the extra programs (batch_id >= num_segments) must early-return
        # without touching weight_indices/seg_indptr out of range. Output for
        # the real segments must match the non-padded case.
        batch_info = self._batch_info(
            seg_indptr=[0, 2, 3],
            weight_indices=[0, 1],
            use_cuda_graph=True,
            bs=4,  # 2 padded slots beyond num_segments=2
        )
        self._run_case("cuda graph padded", batch_info, num_tokens=3)


class TestKvBLoRAAbsorbedBF16(TestKvBLoRAAbsorbed):
    """Production dtype coverage. bf16 has a 7-bit mantissa (~8x lower than
    fp16) so accumulations need a looser tolerance, but the kernels keep
    accumulators in fp32 internally so the relative error should still be
    bounded by 1e-2 at the small ranks/dims used here."""

    DTYPE = torch.bfloat16
    RTOL = 1.5e-2
    ATOL = 1.5e-2


if __name__ == "__main__":
    unittest.main()
