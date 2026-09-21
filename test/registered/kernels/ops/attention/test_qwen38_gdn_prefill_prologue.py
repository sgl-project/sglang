"""Exact parity tests for the Qwen3.8 TP1 GDN fused prefill producer."""

import itertools
import unittest

import torch

from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    QWEN38_GDN_HEAD_DIM,
    QWEN38_GDN_NUM_QK_HEADS,
    QWEN38_GDN_NUM_V_HEADS,
    QWEN38_GDN_QKV_DIM,
    causal_conv1d_fn,
    qwen38_gdn_prefill_prologue,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")


def _cu_seqlens(lengths: list[int]) -> torch.Tensor:
    return torch.tensor(
        [0, *itertools.accumulate(lengths)],
        dtype=torch.int32,
        device="cuda",
    )


def _make_inputs(
    lengths: list[int],
    *,
    has_initial_state: bool,
    projection_stride: bool = True,
    seed: int = 20260921,
) -> dict[str, torch.Tensor | list[int]]:
    torch.manual_seed(seed)
    tokens = sum(lengths)
    if projection_stride:
        qkvz = torch.randn(
            tokens,
            QWEN38_GDN_QKV_DIM + QWEN38_GDN_NUM_V_HEADS * QWEN38_GDN_HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        x = qkvz[:, :QWEN38_GDN_QKV_DIM].transpose(0, 1)
    else:
        x = torch.randn(
            tokens,
            QWEN38_GDN_QKV_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        ).transpose(0, 1)

    ba = torch.randn(
        tokens,
        2 * QWEN38_GDN_NUM_V_HEADS,
        dtype=torch.bfloat16,
        device="cuda",
    )
    raw_b = ba[:, :QWEN38_GDN_NUM_V_HEADS]
    raw_a = ba[:, QWEN38_GDN_NUM_V_HEADS:]
    weight = (
        torch.randn(
            QWEN38_GDN_QKV_DIM,
            4,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.125
    )
    bias = (
        torch.randn(
            QWEN38_GDN_QKV_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.125
    )
    num_sequences = len(lengths)
    num_slots = num_sequences + 3
    conv_state = torch.randn(
        num_slots,
        QWEN38_GDN_QKV_DIM,
        3,
        dtype=torch.bfloat16,
        device="cuda",
    )
    cache_indices = torch.arange(
        1,
        num_sequences + 1,
        dtype=torch.int32,
        device="cuda",
    )
    initial = torch.full(
        (num_sequences,),
        has_initial_state,
        dtype=torch.bool,
        device="cuda",
    )
    A_log = torch.randn(
        QWEN38_GDN_NUM_V_HEADS,
        dtype=torch.float32,
        device="cuda",
    )
    dt_bias = torch.randn(
        QWEN38_GDN_NUM_V_HEADS,
        dtype=torch.bfloat16,
        device="cuda",
    )
    return {
        "x": x,
        "weight": weight,
        "bias": bias,
        "conv_state": conv_state,
        "query_start_loc": _cu_seqlens(lengths),
        "seq_lens_cpu": lengths,
        "cache_indices": cache_indices,
        "has_initial_state": initial,
        "A_log": A_log,
        "a": raw_a,
        "b": raw_b,
        "dt_bias": dt_bias,
    }


def _legacy_prologue(
    inputs: dict[str, torch.Tensor | list[int]],
    conv_state: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    mixed_qkv = causal_conv1d_fn(
        inputs["x"],
        inputs["weight"],
        inputs["bias"],
        activation="silu",
        conv_states=conv_state,
        has_initial_state=inputs["has_initial_state"],
        cache_indices=inputs["cache_indices"],
        query_start_loc=inputs["query_start_loc"],
        seq_lens_cpu=inputs["seq_lens_cpu"],
    ).transpose(0, 1)
    tokens = mixed_qkv.shape[0]
    qk_dim = QWEN38_GDN_NUM_QK_HEADS * QWEN38_GDN_HEAD_DIM
    q_raw = (
        mixed_qkv[:, :qk_dim]
        .reshape(1, tokens, QWEN38_GDN_NUM_QK_HEADS, QWEN38_GDN_HEAD_DIM)
        .contiguous()
    )
    k_raw = (
        mixed_qkv[:, qk_dim : 2 * qk_dim]
        .reshape(1, tokens, QWEN38_GDN_NUM_QK_HEADS, QWEN38_GDN_HEAD_DIM)
        .contiguous()
    )
    v = (
        mixed_qkv[:, 2 * qk_dim :]
        .reshape(1, tokens, QWEN38_GDN_NUM_V_HEADS, QWEN38_GDN_HEAD_DIM)
        .contiguous()
    )
    q = l2norm_fwd(q_raw)
    k = l2norm_fwd(k_raw)
    g, beta = fused_gdn_gating(
        inputs["A_log"],
        inputs["a"],
        inputs["b"],
        inputs["dt_bias"],
    )
    return q_raw, k_raw, v, g, beta, q, k


@unittest.skipIf(
    not torch.cuda.is_available() or torch.version.hip is None,
    "Qwen3.8 fused prefill prologue requires ROCm",
)
class TestQwen38GDNPrefillPrologue(unittest.TestCase):
    def _assert_exact(
        self,
        name: str,
        candidate: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        self.assertEqual(candidate.shape, reference.shape, name)
        self.assertEqual(candidate.dtype, reference.dtype, name)
        bit_dtype = {
            torch.bfloat16: torch.int16,
            torch.float16: torch.int16,
            torch.float32: torch.int32,
            torch.float64: torch.int64,
        }.get(candidate.dtype)
        if bit_dtype is None:
            candidate_bits = candidate
            reference_bits = reference
        else:
            candidate_bits = candidate.contiguous().view(bit_dtype)
            reference_bits = reference.contiguous().view(bit_dtype)
        if torch.equal(candidate_bits, reference_bits):
            return
        mismatch = candidate_bits != reference_bits
        first_index = tuple(int(index) for index in mismatch.nonzero()[0].tolist())
        diff = (candidate.float() - reference.float()).abs()
        self.fail(
            f"{name} crossed an exact legacy boundary: "
            f"mismatches={int(mismatch.sum().item())}/{mismatch.numel()}, "
            f"first_index={first_index}, "
            f"candidate={candidate[first_index].item()}, "
            f"reference={reference[first_index].item()}, "
            f"candidate_bits={candidate_bits[first_index].item()}, "
            f"reference_bits={reference_bits[first_index].item()}, "
            f"max_abs={diff.max().item()}"
        )

    def _assert_prologue_exact(
        self,
        lengths: list[int],
        *,
        has_initial_state: bool,
        projection_stride: bool = True,
    ) -> None:
        inputs = _make_inputs(
            lengths,
            has_initial_state=has_initial_state,
            projection_stride=projection_stride,
        )
        ref_state = inputs["conv_state"].clone()
        fused_state = inputs["conv_state"].clone()
        expected = _legacy_prologue(inputs, ref_state)
        actual = qwen38_gdn_prefill_prologue(
            inputs["x"],
            inputs["weight"],
            inputs["bias"],
            fused_state,
            inputs["query_start_loc"],
            inputs["seq_lens_cpu"],
            inputs["cache_indices"],
            inputs["has_initial_state"],
            inputs["A_log"],
            inputs["a"],
            inputs["b"],
            inputs["dt_bias"],
        )
        for name, candidate, reference in zip(
            ("q_raw_post_conv", "k_raw_post_conv", "v", "g", "beta"),
            actual,
            expected[:5],
        ):
            self._assert_exact(name, candidate, reference)

        # Localize any future mismatch: raw producer parity is checked above,
        # while this applies the exact legacy L2Norm kernel independently.
        self._assert_exact("q_after_legacy_l2norm", l2norm_fwd(actual[0]), expected[5])
        self._assert_exact("k_after_legacy_l2norm", l2norm_fwd(actual[1]), expected[6])
        self._assert_exact("conv_state", fused_state, ref_state)

    def test_dense_and_packed_varlen_exact(self):
        cases = (
            [64],
            [65],
            [13, 17, 19, 23],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        )
        for lengths in cases:
            for has_initial_state in (False, True):
                with self.subTest(
                    lengths=lengths,
                    has_initial_state=has_initial_state,
                ):
                    self._assert_prologue_exact(
                        lengths,
                        has_initial_state=has_initial_state,
                    )

    def test_scheduler_chunk_tail_exact(self):
        self._assert_prologue_exact(
            [16384, 1],
            has_initial_state=True,
            projection_stride=False,
        )

    def test_signed_zero_and_softplus_threshold_exact(self):
        inputs = _make_inputs([8], has_initial_state=False)
        inputs["A_log"].zero_()
        inputs["dt_bias"].zero_()
        threshold_values = torch.tensor(
            [20.0, 20.125, -0.0, 0.0],
            dtype=torch.bfloat16,
            device="cuda",
        )
        inputs["a"][:, :4] = threshold_values
        inputs["b"][:, :4] = torch.tensor(
            [-0.0, 0.0, -20.0, 20.0],
            dtype=torch.bfloat16,
            device="cuda",
        )
        ref_state = inputs["conv_state"].clone()
        fused_state = inputs["conv_state"].clone()
        expected = _legacy_prologue(inputs, ref_state)
        actual = qwen38_gdn_prefill_prologue(
            inputs["x"],
            inputs["weight"],
            inputs["bias"],
            fused_state,
            inputs["query_start_loc"],
            inputs["seq_lens_cpu"],
            inputs["cache_indices"],
            inputs["has_initial_state"],
            inputs["A_log"],
            inputs["a"],
            inputs["b"],
            inputs["dt_bias"],
        )
        self._assert_exact("g_threshold", actual[3], expected[3])
        self._assert_exact("beta_signed_zero", actual[4], expected[4])

    def test_unchanged_chunk_scan_state_semantics(self):
        for has_initial_state in (False, True):
            with self.subTest(has_initial_state=has_initial_state):
                inputs = _make_inputs(
                    [31, 33, 64, 65],
                    has_initial_state=has_initial_state,
                )
                ref_conv_state = inputs["conv_state"].clone()
                fused_conv_state = inputs["conv_state"].clone()
                q_raw, k_raw, v, g, beta, _, _ = _legacy_prologue(
                    inputs,
                    ref_conv_state,
                )
                q_fused, k_fused, v_fused, g_fused, beta_fused = (
                    qwen38_gdn_prefill_prologue(
                        inputs["x"],
                        inputs["weight"],
                        inputs["bias"],
                        fused_conv_state,
                        inputs["query_start_loc"],
                        inputs["seq_lens_cpu"],
                        inputs["cache_indices"],
                        inputs["has_initial_state"],
                        inputs["A_log"],
                        inputs["a"],
                        inputs["b"],
                        inputs["dt_bias"],
                    )
                )
                num_slots = inputs["conv_state"].shape[0]
                ref_ssm = torch.randn(
                    num_slots,
                    QWEN38_GDN_NUM_V_HEADS,
                    QWEN38_GDN_HEAD_DIM,
                    QWEN38_GDN_HEAD_DIM,
                    dtype=torch.float32,
                    device="cuda",
                )
                fused_ssm = ref_ssm.clone()
                ref_out, _, _ = chunk_gated_delta_rule(
                    q_raw,
                    k_raw,
                    v,
                    g,
                    beta,
                    initial_state=ref_ssm,
                    initial_state_indices=inputs["cache_indices"],
                    cu_seqlens=inputs["query_start_loc"],
                    use_qk_l2norm_in_kernel=True,
                )
                fused_out, _, _ = chunk_gated_delta_rule(
                    q_fused,
                    k_fused,
                    v_fused,
                    g_fused,
                    beta_fused,
                    initial_state=fused_ssm,
                    initial_state_indices=inputs["cache_indices"],
                    cu_seqlens=inputs["query_start_loc"],
                    use_qk_l2norm_in_kernel=True,
                )
                self._assert_exact("scan_q_raw", q_fused, q_raw)
                self._assert_exact("scan_k_raw", k_fused, k_raw)
                self._assert_exact("scan_v", v_fused, v)
                self._assert_exact("scan_g", g_fused, g)
                self._assert_exact("scan_beta", beta_fused, beta)
                self._assert_exact("scan_output", fused_out, ref_out)
                self._assert_exact("scan_state", fused_ssm, ref_ssm)


if __name__ == "__main__":
    unittest.main()
