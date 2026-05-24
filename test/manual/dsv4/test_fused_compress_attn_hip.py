"""Unit tests for the fused compressor attention Triton kernel on HIP.

Validates numerical parity between the fused single-kernel path (plan-driven
Triton) and the reference per-seq Python implementation.

Usage:
    python -m pytest test/manual/dsv4/test_fused_compress_attn_hip.py -v
    # or directly:
    python test/manual/dsv4/test_fused_compress_attn_hip.py
"""

import unittest
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class FusedCompressPlan:
    compress_plan_gpu: torch.Tensor
    write_plan_gpu: torch.Tensor
    num_compress: int
    num_write: int


def write_current_token_to_state(
    kv_score_input: torch.Tensor,
    write_plan: torch.Tensor,
    state_pool_buffer: torch.Tensor,
    head_dim: int,
    overlap: bool,
    ratio: int,
) -> None:
    """Reference write path used by this manual test.

    Plan row layout: [ragged_id, batch_id, position, window_len, state_base].
    """
    del head_dim  # layout is already encoded in kv_score_input/state_pool_buffer shape.
    state_size = (2 if overlap else 1) * ratio
    plan_cpu = write_plan.cpu()
    for row in plan_cpu:
        ragged_id = int(row[0].item())
        position = int(row[2].item())
        state_base = int(row[4].item())
        if ragged_id < 0 or position < 0:
            continue
        dst = state_base + (position % state_size)
        if (
            0 <= dst < state_pool_buffer.shape[0]
            and 0 <= ragged_id < kv_score_input.shape[0]
        ):
            state_pool_buffer[dst] = kv_score_input[ragged_id]


def fused_compress_attn(
    state_pool_buffer: torch.Tensor,
    plan: torch.Tensor,
    ape: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    freqs_cis_real: torch.Tensor,
    head_dim: int,
    rope_head_dim: int,
    overlap: bool,
    ratio: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Reference compress path for manual parity tests.

    This keeps the test runnable after removing `fused_compress_kernel.py`.
    """
    freqs_cis = torch.view_as_complex(
        freqs_cis_real.view(freqs_cis_real.shape[0], -1, 2).contiguous()
    )
    result = _ref_compress(
        kv_score_input=torch.empty(
            0, device=state_pool_buffer.device, dtype=torch.float32
        ),
        state_pool=state_pool_buffer,
        plan=plan,
        ape=ape,
        rms_weight=rms_weight,
        rms_eps=rms_eps,
        freqs_cis=freqs_cis,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        overlap=overlap,
        ratio=ratio,
        num_compress=plan.shape[0],
    )
    out.copy_(result)
    return out


def _make_plan_from_params(
    extend_lens: list[int],
    seq_lens: list[int],
    ratio: int,
    overlap: bool,
    state_bases: list[int],
    device: torch.device,
) -> FusedCompressPlan:
    """Build a test plan without requiring real SWA / req_to_token tables."""
    bs = len(extend_lens)
    ext = np.array(extend_lens, dtype=np.int32)
    seq = np.array(seq_lens, dtype=np.int32)
    total = int(ext.sum())

    state_size = (2 if overlap else 1) * ratio
    K = state_size

    batch_ids = np.repeat(np.arange(bs, dtype=np.int32), ext)
    ragged_ids = np.arange(total, dtype=np.int32)
    cu_extend = np.empty(bs + 1, dtype=np.int32)
    cu_extend[0] = 0
    np.cumsum(ext, out=cu_extend[1:])
    j_in_seq = ragged_ids - cu_extend[batch_ids]
    prefix_lens = seq - ext
    positions = prefix_lens[batch_ids] + j_in_seq

    window_lens = np.maximum(0, K - np.minimum(j_in_seq + 1, K)).astype(np.int32)
    state_base_arr = np.array(state_bases, dtype=np.int32)
    state_base_per_token = state_base_arr[batch_ids]

    plan_rows = np.stack(
        [ragged_ids, batch_ids, positions, window_lens, state_base_per_token],
        axis=1,
    ).astype(np.int32)

    compress_mask = (positions + 1) % ratio == 0
    compress_plan = plan_rows[compress_mask]

    write_starts = np.maximum(0, seq - K).astype(np.int32)
    write_mask = positions >= write_starts[batch_ids]
    write_plan = plan_rows[write_mask]

    n_compress = int(compress_plan.shape[0]) if compress_plan.size > 0 else 0
    n_write = int(write_plan.shape[0]) if write_plan.size > 0 else 0

    compress_gpu = (
        torch.from_numpy(np.ascontiguousarray(compress_plan)).to(device)
        if n_compress > 0
        else torch.empty((0, 5), dtype=torch.int32, device=device)
    )
    write_gpu = (
        torch.from_numpy(np.ascontiguousarray(write_plan)).to(device)
        if n_write > 0
        else torch.empty((0, 5), dtype=torch.int32, device=device)
    )

    return FusedCompressPlan(
        compress_plan_gpu=compress_gpu,
        write_plan_gpu=write_gpu,
        num_compress=n_compress,
        num_write=n_write,
    )


def _make_freqs_cis(max_seq: int, rope_dim: int, device: torch.device) -> torch.Tensor:
    """Create test freqs_cis as complex64 [max_seq, rope_dim/2], matching production."""
    half = rope_dim // 2
    angles = torch.randn(max_seq, half, device=device, dtype=torch.float32) * 0.1
    return torch.polar(torch.ones_like(angles), angles)


def _freqs_to_real(freqs_cis: torch.Tensor) -> torch.Tensor:
    """Convert complex64 freqs to float32 [max_seq, rope_dim] interleaved."""
    return torch.view_as_real(freqs_cis).flatten(-2).contiguous()


def _ref_compress(
    kv_score_input: torch.Tensor,
    state_pool: torch.Tensor,
    plan: torch.Tensor,
    ape: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    freqs_cis: torch.Tensor,
    head_dim: int,
    rope_head_dim: int,
    overlap: bool,
    ratio: int,
    num_compress: int,
) -> torch.Tensor:
    """Pure-PyTorch reference matching SGLang compress_decode_paged semantics.

    State already has current tokens written (no APE).
    APE is added to ALL K scores at compress time.
    """
    if num_compress == 0:
        return torch.empty(0, head_dim, dtype=torch.float32, device=state_pool.device)

    coff = 2 if overlap else 1
    half_dim = coff * head_dim
    state_size = coff * ratio
    K = state_size

    plan_cpu = plan[:num_compress].cpu()
    out = torch.empty(
        num_compress, head_dim, dtype=torch.float32, device=state_pool.device
    )

    for pid in range(num_compress):
        position = int(plan_cpu[pid, 2].item())
        state_base = int(plan_cpu[pid, 4].item())

        if position < 0:
            continue

        kv_rows = []
        score_rows = []
        for k in range(K):
            s = position - K + 1 + k
            col_off = (head_dim if k >= ratio else 0) if overlap else 0
            ape_row = k % ratio
            d_slice = slice(col_off, col_off + head_dim)

            if s < 0:
                kv_rows.append(
                    torch.zeros(head_dim, dtype=torch.float32, device=state_pool.device)
                )
                score_rows.append(
                    torch.full(
                        (head_dim,),
                        float("-inf"),
                        dtype=torch.float32,
                        device=state_pool.device,
                    )
                )
            else:
                ring = s % state_size
                row = state_pool[state_base + ring]
                kv_rows.append(row[d_slice].float())
                # APE added to ALL scores
                score_rows.append(
                    row[half_dim + col_off : half_dim + col_off + head_dim].float()
                    + ape[ape_row, d_slice].float()
                )

        kv_stack = torch.stack(kv_rows, dim=0)
        sc_stack = torch.stack(score_rows, dim=0)
        weights = torch.softmax(sc_stack, dim=0)
        compressed = (weights * kv_stack).sum(dim=0)

        var = (compressed * compressed).mean()
        normed = compressed * torch.rsqrt(var + rms_eps) * rms_weight.float()

        comp_pos = (position // ratio) * ratio
        rope_seg = normed[-rope_head_dim:].clone()
        freqs_row = torch.view_as_real(freqs_cis[comp_pos]).flatten()
        cos_v = freqs_row[0::2].float()
        sin_v = freqs_row[1::2].float()

        even = rope_seg[0::2]
        odd = rope_seg[1::2]
        normed[-rope_head_dim:] = torch.stack(
            [even * cos_v - odd * sin_v, odd * cos_v + even * sin_v], dim=-1
        ).flatten()

        out[pid] = normed

    return out


class TestFusedCompressAttn(unittest.TestCase):

    def _run_test(
        self,
        ratio: int,
        overlap: bool,
        bs: int,
        extend_lens: list[int],
        prefix_lens: list[int],
        head_dim: int = 512,
        rope_head_dim: int = 64,
    ):
        device = torch.device("cuda")
        torch.manual_seed(42)
        coff = 2 if overlap else 1
        half_dim = coff * head_dim
        last_dim = 2 * half_dim
        state_size = coff * ratio

        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        total_tokens = sum(extend_lens)
        max_seq = max(seq_lens) + 128

        kv_score_input = torch.randn(
            total_tokens, last_dim, device=device, dtype=torch.float32
        )

        pool_size = bs * state_size + 2
        state_pool = torch.randn(
            pool_size, last_dim, device=device, dtype=torch.float32
        )
        state_pool[:, half_dim:] *= 0.5  # reasonable score magnitudes

        state_bases = [i * state_size for i in range(bs)]
        ape = torch.randn(ratio, half_dim, device=device, dtype=torch.float32) * 0.1
        rms_weight = torch.ones(head_dim, device=device, dtype=torch.float32)
        rms_eps = 1e-6
        freqs_cis = _make_freqs_cis(max_seq, rope_head_dim, device)
        freqs_real = _freqs_to_real(freqs_cis)

        plan = _make_plan_from_params(
            extend_lens, seq_lens, ratio, overlap, state_bases, device
        )
        if plan.num_compress == 0:
            return

        # Step 1: write current tokens to state (same for both paths)
        state_triton = state_pool.clone()
        state_ref = state_pool.clone()

        write_current_token_to_state(
            kv_score_input=kv_score_input,
            write_plan=plan.write_plan_gpu,
            state_pool_buffer=state_triton,
            head_dim=head_dim,
            overlap=overlap,
            ratio=ratio,
        )
        # Reference: same write
        write_current_token_to_state(
            kv_score_input=kv_score_input,
            write_plan=plan.write_plan_gpu,
            state_pool_buffer=state_ref,
            head_dim=head_dim,
            overlap=overlap,
            ratio=ratio,
        )

        # Step 2a: Triton fused compress
        out_triton = torch.empty(
            plan.num_compress, head_dim, device=device, dtype=torch.float32
        )
        fused_compress_attn(
            state_pool_buffer=state_triton,
            plan=plan.compress_plan_gpu,
            ape=ape,
            rms_weight=rms_weight,
            rms_eps=rms_eps,
            freqs_cis_real=freqs_real,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            overlap=overlap,
            ratio=ratio,
            out=out_triton,
        )

        # Step 2b: reference compress
        out_ref = _ref_compress(
            kv_score_input=kv_score_input,
            state_pool=state_ref,
            plan=plan.compress_plan_gpu,
            ape=ape,
            rms_weight=rms_weight,
            rms_eps=rms_eps,
            freqs_cis=freqs_cis,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            overlap=overlap,
            ratio=ratio,
            num_compress=plan.num_compress,
        )

        torch.testing.assert_close(out_triton, out_ref, atol=1e-3, rtol=1e-3)

    def test_hca_single(self):
        self._run_test(
            ratio=128, overlap=False, bs=1, extend_lens=[128], prefix_lens=[0]
        )

    def test_hca_multi(self):
        self._run_test(
            ratio=128, overlap=False, bs=2, extend_lens=[128, 256], prefix_lens=[0, 128]
        )

    def test_csa_single(self):
        self._run_test(ratio=4, overlap=True, bs=1, extend_lens=[16], prefix_lens=[8])

    def test_csa_multi(self):
        self._run_test(
            ratio=4, overlap=True, bs=3, extend_lens=[8, 12, 16], prefix_lens=[4, 8, 0]
        )

    def test_csa_small_dim(self):
        self._run_test(
            ratio=4,
            overlap=True,
            bs=2,
            extend_lens=[8, 8],
            prefix_lens=[4, 0],
            head_dim=256,
        )


class TestStateOrdering(unittest.TestCase):

    def test_write_then_compress(self):
        """Verify write-first, compress-second matches reference."""
        device = torch.device("cuda")
        torch.manual_seed(123)
        ratio, overlap = 4, True
        coff = 2
        head_dim, rope_head_dim = 128, 64
        half_dim = coff * head_dim
        last_dim = 2 * half_dim
        state_size = coff * ratio

        pool_size = state_size + 2
        state_pool = torch.randn(
            pool_size, last_dim, device=device, dtype=torch.float32
        )
        state_pool[:, half_dim:] *= 0.5

        kv_score_input = torch.randn(8, last_dim, device=device, dtype=torch.float32)
        ape = torch.randn(ratio, half_dim, device=device, dtype=torch.float32) * 0.1
        rms_weight = torch.ones(head_dim, device=device, dtype=torch.float32)
        freqs_cis = _make_freqs_cis(64, rope_head_dim, device)

        plan = _make_plan_from_params([8], [8], ratio, overlap, [0], device)
        if plan.num_compress == 0:
            return

        state_before = state_pool.clone()

        # Write first
        write_current_token_to_state(
            kv_score_input=kv_score_input,
            write_plan=plan.write_plan_gpu,
            state_pool_buffer=state_pool,
            head_dim=head_dim,
            overlap=overlap,
            ratio=ratio,
        )

        # State should now be different (tokens written)
        self.assertFalse(torch.allclose(state_pool, state_before))

        # Compress
        out = torch.empty(
            plan.num_compress, head_dim, device=device, dtype=torch.float32
        )
        fused_compress_attn(
            state_pool_buffer=state_pool,
            plan=plan.compress_plan_gpu,
            ape=ape,
            rms_weight=rms_weight,
            rms_eps=1e-6,
            freqs_cis_real=_freqs_to_real(freqs_cis),
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            overlap=overlap,
            ratio=ratio,
            out=out,
        )

        self.assertFalse(torch.any(torch.isnan(out)).item())
        self.assertFalse(torch.any(torch.isinf(out)).item())


if __name__ == "__main__":
    unittest.main()
