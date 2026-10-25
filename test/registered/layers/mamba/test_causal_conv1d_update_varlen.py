"""Packed (varlen) causal_conv1d_update for ragged speculative verify.

A packed call must match the dense call on the same tokens, each sequence
zero-padded to the draft width, bit for bit: the real output rows, the conv
state and the intermediate conv windows (both the dense per-step layout and
the deduplicated overlapping one). Sequences without a state slot or without
rows, and packed rows past the last sequence, touch no state. Also checks
CUDA-graph replay against eager.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-small")

if not torch.cuda.is_available():
    pytest.skip("packed causal_conv1d_update needs CUDA", allow_module_level=True)

from sglang.kernels.ops.mamba.causal_conv1d_triton import (  # noqa: E402
    causal_conv1d_update,
)

DEV = "cuda"
WIDTH = 4


def _window_buffers(num_rows, dim, draft_tokens, dtype, dedup):
    """Intermediate conv window as the mamba pool allocates it: the physical
    buffer plus the [rows, step, dim, width - 1] view the kernel writes."""
    win = WIDTH - 1
    if dedup:
        phys = torch.randn(num_rows, dim, draft_tokens + win - 1, device=DEV)
        phys = phys.to(dtype)
        view = phys.as_strided(
            (num_rows, draft_tokens, dim, win),
            (phys.stride(0), phys.stride(2), phys.stride(1), phys.stride(2)),
        )
        return phys, view
    phys = torch.randn(num_rows, draft_tokens, dim, win, device=DEV).to(dtype)
    return phys, phys


class _Case:
    def __init__(self, lens, slots, *, dim, draft_tokens, dtype, dedup, tail=0):
        self.lens = list(lens)
        self.slots = list(slots)
        self.dim = dim
        self.T = draft_tokens
        self.dtype = dtype
        self.dedup = dedup
        bs = len(self.lens)
        self.num_tokens = sum(self.lens) + tail
        num_slots = max(self.slots + [0]) + 2
        g = torch.Generator(device=DEV).manual_seed(1234 + bs + tail)
        self.x = torch.randn(
            self.num_tokens, dim, device=DEV, dtype=torch.float32, generator=g
        ).to(dtype)
        self.weight = torch.randn(
            dim, WIDTH, device=DEV, dtype=torch.float32, generator=g
        ).to(dtype)
        self.bias = torch.randn(dim, device=DEV, dtype=torch.float32, generator=g)
        self.bias = self.bias.to(dtype)
        self.conv_state = torch.randn(
            num_slots, dim, WIDTH - 1, device=DEV, dtype=torch.float32, generator=g
        ).to(dtype)
        # Scratch rows are positional (row i of the batch), plus a spare row.
        self.win_phys, self.win_view = _window_buffers(
            bs + 1, dim, draft_tokens, dtype, dedup
        )
        self.qsl = torch.zeros(bs + 1, device=DEV, dtype=torch.int32)
        self.qsl[1:] = torch.cumsum(torch.tensor(self.lens, device=DEV), 0)
        self.slot_t = torch.tensor(self.slots, device=DEV, dtype=torch.int32)
        self.row_t = torch.arange(bs, device=DEV, dtype=torch.int32)

    def run_packed(self, conv_state, win_view, qsl=None):
        return causal_conv1d_update(
            self.x,
            conv_state,
            self.weight,
            self.bias,
            "silu",
            conv_state_indices=self.slot_t,
            intermediate_conv_window=win_view,
            intermediate_state_indices=self.row_t,
            query_start_loc=self.qsl if qsl is None else qsl,
            max_seqlen=self.T,
        )

    def run_reference(self, conv_state, win_view):
        """The dense call (token-major x, like the GDN dense verify) on the
        same tokens, each sequence zero-padded to the draft width."""
        dense = self.x.new_zeros(len(self.lens), self.T, self.dim)
        start = 0
        for i, n in enumerate(self.lens):
            dense[i, :n] = self.x[start : start + n]
            start += n
        out = causal_conv1d_update(
            dense.transpose(1, 2),
            conv_state,
            self.weight,
            self.bias,
            "silu",
            conv_state_indices=self.slot_t,
            intermediate_conv_window=win_view,
            intermediate_state_indices=self.row_t,
        )
        return out.transpose(1, 2)


def _check(case: _Case):
    ref_state = case.conv_state.clone()
    ref_win_phys = case.win_phys.clone()
    ref_win_view = ref_win_phys.as_strided(case.win_view.shape, case.win_view.stride())
    ref_out = case.run_reference(ref_state, ref_win_view)
    # A sequence without rows exits before touching its slot or window, even
    # when it has a slot; the padded dense call would run it on zeros.
    for i, (n, slot) in enumerate(zip(case.lens, case.slots)):
        if n == 0 and slot >= 0:
            ref_state[slot] = case.conv_state[slot]
            ref_win_phys[i] = case.win_phys[i]

    state = case.conv_state.clone()
    win_phys = case.win_phys.clone()
    win_view = win_phys.as_strided(case.win_view.shape, case.win_view.stride())
    out = case.run_packed(state, win_view)

    assert out.shape == case.x.shape
    start = 0
    for i, (n, slot) in enumerate(zip(case.lens, case.slots)):
        if slot >= 0:
            torch.testing.assert_close(
                out[start : start + n], ref_out[i, :n], rtol=0, atol=0
            )
        start += n
    # Every state byte matches: rows without a slot or without tokens and the
    # tier leftovers wrote nothing.
    torch.testing.assert_close(state, ref_state, rtol=0, atol=0)
    torch.testing.assert_close(win_phys, ref_win_phys, rtol=0, atol=0)


LAYOUTS = {
    # name: (verify lens, mamba slots, tier leftovers)
    "all_full": ([8, 8, 8, 8], [3, 0, 5, 1], 0),
    "mixed": ([1, 8, 3, 5, 2, 7, 4, 6], [7, 2, 0, 5, 1, 6, 3, 4], 0),
    "zero_length_rows": ([3, 0, 5, 0, 0, 1], [2, -1, 4, -1, -1, 0], 0),
    "null_slots": ([4, 2, 8, 3, 3], [1, -1, 0, -1, -1], 0),
    "zero_length_with_slot": ([3, 0, 5, 0, 2], [2, 4, 0, -1, 1], 0),
    "tier_leftovers": ([2, 5, 8, 1], [0, 2, 1, 3], 13),
    "single_token_rows": ([1, 1, 1], [2, 0, 1], 0),
}


@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("dedup", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim", [768, 1000])
def test_packed_rows_match_padded_dense(layout, dedup, dtype, dim):
    lens, slots, tail = LAYOUTS[layout]
    _check(
        _Case(lens, slots, dim=dim, draft_tokens=8, dtype=dtype, dedup=dedup, tail=tail)
    )


@pytest.mark.parametrize("draft_tokens", [2, 4, 16])
def test_packed_rows_other_widths(draft_tokens):
    lens = [draft_tokens, 1, max(1, draft_tokens // 2), 0, draft_tokens - 1 or 1]
    slots = [0, 3, 1, -1, 2]
    _check(
        _Case(
            lens,
            slots,
            dim=512,
            draft_tokens=draft_tokens,
            dtype=torch.bfloat16,
            dedup=True,
            tail=3,
        )
    )


def test_cuda_graph_replay_matches_eager():
    """One capture serves every layout of the same row count."""
    T, dim = 8, 640
    capture_lens = [2] * 8
    slots = [5, 2, 7, 0, -1, 3, -1, 1]
    case = _Case(
        capture_lens,
        slots,
        dim=dim,
        draft_tokens=T,
        dtype=torch.bfloat16,
        dedup=True,
        tail=0,
    )
    qsl_buf = case.qsl.clone()
    state = case.conv_state.clone()
    phys = case.win_phys.clone()
    view = phys.as_strided(case.win_view.shape, case.win_view.stride())

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        case.run_packed(state, view, qsl_buf)  # warm up / compile
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_graph = case.run_packed(state, view, qsl_buf)

    for lens in ([1, 3, 0, 4, 0, 2, 1, 5], [8, 1, 1, 1, 0, 0, 2, 3], [0] * 8):
        assert sum(lens) <= case.num_tokens
        qsl = torch.zeros(len(lens) + 1, device=DEV, dtype=torch.int32)
        qsl[1:] = torch.cumsum(torch.tensor(lens, device=DEV), 0)

        eager_state = case.conv_state.clone()
        eager_phys = case.win_phys.clone()
        eager_view = eager_phys.as_strided(case.win_view.shape, case.win_view.stride())
        eager = case.run_packed(eager_state, eager_view, qsl)

        state.copy_(case.conv_state)
        phys.copy_(case.win_phys)
        qsl_buf.copy_(qsl)
        graph.replay()
        torch.cuda.synchronize()
        n = int(qsl[-1])
        real = torch.zeros(case.num_tokens, dtype=torch.bool, device=DEV)
        for i, (s, e) in enumerate(zip(qsl[:-1].tolist(), qsl[1:].tolist())):
            if slots[i] >= 0:
                real[s:e] = True
        torch.testing.assert_close(out_graph[real], eager[real], rtol=0, atol=0)
        torch.testing.assert_close(state, eager_state, rtol=0, atol=0)
        torch.testing.assert_close(phys, eager_phys, rtol=0, atol=0)
        assert n <= case.num_tokens


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
