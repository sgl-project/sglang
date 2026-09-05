"""
Regression tests for sglang issue #37983:
"[Bug] VisionTritonAttention drops MiMo-style window attention and sinks"

The Triton varlen prefill kernel (context_attention_fwd) historically had no
sliding-window or per-head-sink support, and VisionTritonAttention dropped the
window_size / s_aux kwargs its caller (VisionAttention) always passes.  Models
whose ViT relies on local-window attention plus learnable sinks (e.g. MiMo-VL)
were silently computed as full attention without sinks.

Attention with a sliding window and per-head sinks has an exact mathematical
definition, so a plain PyTorch reference implementation on random tensors is a
complete oracle: no checkpoint download, no server needed.

Oracle calibration chain:
  1. test_reference_matches_hand_computed_example -- CPU-only, exact numbers
  2. test_kernel_matches_reference_full_attention -- the kernel agrees with
     the reference on the subset it always computed correctly
  3. test_fa3_cross_check_reference -- optional cross-check on Hopper (sm90)
     where the FA3 vision backend provably honors window_size + sinks
"""

import math
import unittest

import torch

from sglang.kernels.ops.attention.prefill_attention import context_attention_fwd
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Vision attention window/sink semantics (triton kernel + backend integration)
register_cuda_ci(est_time=2, stage="extra-a", runner_config="1-gpu-large")
register_amd_ci(est_time=3, suite="stage-b-test-1-gpu-small-amd")

ATOL = 2e-2
RTOL = 2e-2  # bf16 inputs with fp32 accumulation


def ref_varlen_attention(
    q, k, v, cu_seqlens, sm_scale, window_size=(-1, -1), sinks=None, causal=False
):
    """Non-causal varlen attention with optional sliding window and per-head sinks.

    q:          [total_tokens, Hq, D]
    k, v:       [total_tokens, Hk, D]   (Hq % Hk == 0, GQA allowed)
    cu_seqlens: list of length B+1
    window_size: (left, right); -1 = unbounded on that side (flash-attn
                 convention: query i attends keys in [i - left, i + right])
    sinks:      [Hq] per-head logit appended to the softmax denominator
                without a value vector (dilutes real attention weights)

    out_i = sum_j exp(s_ij - m) v_j / ( sum_j exp(s_ij - m) + exp(sink_h - m) )
    """
    _, hq, _ = q.shape
    hk = k.shape[1]
    if hk != hq:
        rep = hq // hk
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    device = q.device
    wl, wr = window_size
    outs = []
    for b in range(len(cu_seqlens) - 1):
        s, e = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
        qb, kb, vb = q[s:e].float(), k[s:e].float(), v[s:e].float()
        t = e - s
        scores = torch.einsum("ihd,jhd->hij", qb, kb) * sm_scale  # [H, Tq, Tk]

        iq = torch.arange(t, device=device).unsqueeze(1)  # [T, 1] query idx
        ik = torch.arange(t, device=device).unsqueeze(0)  # [1, T] key idx
        if causal:
            scores = scores.masked_fill(ik > iq, float("-inf"))
        if wl != -1:
            scores = scores.masked_fill(ik < iq - wl, float("-inf"))
        if wr != -1:
            scores = scores.masked_fill(ik > iq + wr, float("-inf"))

        if sinks is not None:
            sink_col = sinks.detach().float().view(-1, 1, 1).expand(hq, t, 1)
            scores = torch.cat([scores, sink_col], dim=-1)  # [H, T, T+1]

        probs = torch.softmax(scores, dim=-1)
        if sinks is not None:
            probs = probs[..., :t]  # the sink has no value vector
        outs.append(torch.einsum("hij,jhd->ihd", probs, vb).to(q.dtype))
    return torch.cat(outs, dim=0)


def make_varlen_inputs(
    seq_lens,
    num_heads=8,
    num_kv_heads=None,
    head_dim=64,
    dtype=torch.bfloat16,
    seed=0,
):
    """Random varlen-packed q/k/v plus matching cu_seqlens / seq_lens."""
    num_kv_heads = num_kv_heads or num_heads
    total = sum(seq_lens)
    device = get_device()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    def rnd(shape, scale):
        return (
            (
                torch.randn(shape, generator=gen, device=device, dtype=torch.float32)
                * scale
            )
            .to(dtype)
            .contiguous()
        )

    q = rnd((total, num_heads, head_dim), 0.5)
    k = rnd((total, num_kv_heads, head_dim), 0.5)
    v = rnd((total, num_kv_heads, head_dim), 1.0)
    cu = torch.tensor(
        [0, *torch.cumsum(torch.tensor(seq_lens), 0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    seq_lens_t = cu[1:] - cu[:-1]
    return q, k, v, cu, seq_lens_t, max(seq_lens)


def rand_sinks(num_heads, scale=3.0, seed=1234):
    gen = torch.Generator(device=get_device())
    gen.manual_seed(seed)
    return torch.randn(num_heads, generator=gen, device=get_device()) * scale


requires_cuda = unittest.skipIf(
    not torch.cuda.is_available(), "requires an NVIDIA/ROCm GPU"
)


class TestTritonAttentionWindowSink(CustomTestCase):
    """window_size / sinks semantics of the varlen prefill kernel."""

    # ------------------------------------------------------------------
    # 1. Oracle calibration
    # ------------------------------------------------------------------
    def test_reference_matches_hand_computed_example(self):
        """Exact hand-computed case: 1 head, 3 tokens, D=2, sm_scale=1.

        Query 0 produces logits [1, 2, 3] against keys 0..2; sink logit = 3;
        values v0=[1,0], v1=[0,1], v2=[1,1].  Only row 0 is asserted.
        """
        q = torch.zeros(3, 1, 2)
        k = torch.zeros(3, 1, 2)
        v = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).view(3, 1, 2)
        q[0, 0, 0] = 1.0
        k[0, 0, 0] = 1.0
        k[1, 0, 0] = 2.0
        k[2, 0, 0] = 3.0
        cu = [0, 3]
        sink = torch.tensor([3.0])

        # anchor: elementary softmax([1,2,3]) = [0.090031, 0.244728, 0.665241]
        torch.testing.assert_close(
            torch.softmax(torch.tensor([1.0, 2.0, 3.0]), -1),
            torch.tensor([0.090031, 0.244728, 0.665241]),
            atol=1e-5,
            rtol=1e-5,
        )

        # full attention, no sink: softmax over [1, 2, 3]
        out = ref_varlen_attention(q, k, v, cu, 1.0)
        torch.testing.assert_close(
            out[0, 0], torch.tensor([0.755272, 0.909969]), atol=1e-5, rtol=1e-5
        )

        # sink only: softmax over [1, 2, 3, sink=3]; real weights diluted
        out = ref_varlen_attention(q, k, v, cu, 1.0, sinks=sink)
        torch.testing.assert_close(
            out[0, 0], torch.tensor([0.453551, 0.546449]), atol=2e-5, rtol=2e-5
        )

        # window (1, 1) + sink: query 0 attends keys {0, 1} plus the sink
        out = ref_varlen_attention(q, k, v, cu, 1.0, window_size=(1, 1), sinks=sink)
        torch.testing.assert_close(
            out[0, 0], torch.tensor([0.090031, 0.244728]), atol=1e-5, rtol=1e-5
        )

        # window (0, 0) + sink: query 0 attends only itself + sink,
        # i.e. softmax over [1, sink=3] -> w = 1/(1+e^2) = 0.119203
        out = ref_varlen_attention(q, k, v, cu, 1.0, window_size=(0, 0), sinks=sink)
        torch.testing.assert_close(
            out[0, 0], torch.tensor([0.119203, 0.0]), atol=1e-5, rtol=1e-5
        )

    @requires_cuda
    def test_kernel_matches_reference_full_attention(self):
        """Calibrate the oracle against the kernel on the subset it always
        computed correctly: no window, no sink (ragged varlen batch)."""
        q, k, v, cu, seq_lens, max_len = make_varlen_inputs([64, 37], seed=0)
        scale = 1.0 / math.sqrt(q.shape[-1])
        o = torch.empty_like(q)
        context_attention_fwd(
            q, k, v, o, cu, seq_lens, max_len, is_causal=False, sm_scale=scale
        )
        ref = ref_varlen_attention(q, k, v, cu.tolist(), scale)
        torch.testing.assert_close(o.float(), ref.float(), atol=ATOL, rtol=RTOL)

    @requires_cuda
    def test_kernel_matches_reference_full_attention_gqa(self):
        """Same calibration with GQA (Hq=8, Hk=4): the kernel maps q-head h to
        kv-head h // kv_group_num, matching repeat_interleave in the reference."""
        q, k, v, cu, seq_lens, max_len = make_varlen_inputs(
            [48, 64, 21], num_heads=8, num_kv_heads=4, seed=7
        )
        scale = 1.0 / math.sqrt(q.shape[-1])
        o = torch.empty_like(q)
        context_attention_fwd(
            q, k, v, o, cu, seq_lens, max_len, is_causal=False, sm_scale=scale
        )
        ref = ref_varlen_attention(q, k, v, cu.tolist(), scale)
        torch.testing.assert_close(o.float(), ref.float(), atol=ATOL, rtol=RTOL)

    # ------------------------------------------------------------------
    # 2. Kernel: window / sinks semantics
    # ------------------------------------------------------------------
    WINDOW = (16, 16)

    @requires_cuda
    def test_kernel_window_only(self):
        q, k, v, cu, seq_lens, max_len = make_varlen_inputs([64, 48], seed=1)
        scale = 1.0 / math.sqrt(q.shape[-1])
        o = torch.empty_like(q)
        context_attention_fwd(
            q,
            k,
            v,
            o,
            cu,
            seq_lens,
            max_len,
            is_causal=False,
            sm_scale=scale,
            window_size=self.WINDOW,
        )
        ref = ref_varlen_attention(q, k, v, cu.tolist(), scale, window_size=self.WINDOW)
        torch.testing.assert_close(o.float(), ref.float(), atol=ATOL, rtol=RTOL)

    @requires_cuda
    def test_kernel_sinks_only(self):
        q, k, v, cu, seq_lens, max_len = make_varlen_inputs([64, 48], seed=2)
        scale = 1.0 / math.sqrt(q.shape[-1])
        sinks = rand_sinks(q.shape[1])
        o = torch.empty_like(q)
        context_attention_fwd(
            q,
            k,
            v,
            o,
            cu,
            seq_lens,
            max_len,
            is_causal=False,
            sm_scale=scale,
            sinks=sinks,
        )
        ref = ref_varlen_attention(q, k, v, cu.tolist(), scale, sinks=sinks)
        torch.testing.assert_close(o.float(), ref.float(), atol=ATOL, rtol=RTOL)

    @requires_cuda
    def test_kernel_window_and_sinks(self):
        q, k, v, cu, seq_lens, max_len = make_varlen_inputs([64, 48], seed=3)
        scale = 1.0 / math.sqrt(q.shape[-1])
        sinks = rand_sinks(q.shape[1])
        o = torch.empty_like(q)
        context_attention_fwd(
            q,
            k,
            v,
            o,
            cu,
            seq_lens,
            max_len,
            is_causal=False,
            sm_scale=scale,
            window_size=self.WINDOW,
            sinks=sinks,
        )
        ref = ref_varlen_attention(
            q, k, v, cu.tolist(), scale, window_size=self.WINDOW, sinks=sinks
        )
        torch.testing.assert_close(o.float(), ref.float(), atol=ATOL, rtol=RTOL)

    @requires_cuda
    def test_kernel_window_and_sinks_varlen_gqa(self):
        """Ragged varlen + GQA + window + sinks all at once (no cross-sequence
        leakage, correct head grouping, correct mask and denominator)."""
        q, k, v, cu, seq_lens, max_len = make_varlen_inputs(
            [91, 37, 64], num_heads=8, num_kv_heads=4, seed=4
        )
        scale = 1.0 / math.sqrt(q.shape[-1])
        sinks = rand_sinks(q.shape[1], seed=5678)
        o = torch.empty_like(q)
        context_attention_fwd(
            q,
            k,
            v,
            o,
            cu,
            seq_lens,
            max_len,
            is_causal=False,
            sm_scale=scale,
            window_size=self.WINDOW,
            sinks=sinks,
        )
        ref = ref_varlen_attention(
            q, k, v, cu.tolist(), scale, window_size=self.WINDOW, sinks=sinks
        )
        torch.testing.assert_close(o.float(), ref.float(), atol=ATOL, rtol=RTOL)

    @requires_cuda
    def test_kernel_window_long_seq_multi_block(self):
        """Sequences longer than one KV block with small windows: many query
        rows have fully-masked KV blocks on the way to their window (the NaN
        trap for naive online-softmax windowing: exp(-inf - (-inf)) poisons
        every later block unless the running max is kept finite and the
        l == 0 rescale is guarded).

        Also covers one-sided and asymmetric windows (-1 = unbounded side).
        """
        q, k, v, cu, seq_lens, max_len = make_varlen_inputs([300, 257, 385], seed=11)
        scale = 1.0 / math.sqrt(q.shape[-1])
        sinks = rand_sinks(q.shape[1], seed=11)
        for window in [(1, 1), (8, 8), (0, 64), (64, -1), (-1, 8)]:
            with self.subTest(window=window):
                o = torch.empty_like(q)
                context_attention_fwd(
                    q,
                    k,
                    v,
                    o,
                    cu,
                    seq_lens,
                    max_len,
                    is_causal=False,
                    sm_scale=scale,
                    window_size=window,
                    sinks=sinks,
                )
                ref = ref_varlen_attention(
                    q, k, v, cu.tolist(), scale, window_size=window, sinks=sinks
                )
                self.assertFalse(
                    torch.isnan(o.float()).any(), f"NaN leaked with window {window}"
                )
                torch.testing.assert_close(
                    o.float(),
                    ref.float(),
                    atol=ATOL,
                    rtol=RTOL,
                    msg=f"mismatch for window {window}",
                )

    @requires_cuda
    def test_kernel_window_no_sinks_long_seq(self):
        """Same fully-masked-block shape without sinks, i.e. the path where a
        zero denominator can genuinely occur (no sink seeds l = 1)."""
        q, k, v, cu, seq_lens, max_len = make_varlen_inputs([300, 257], seed=12)
        scale = 1.0 / math.sqrt(q.shape[-1])
        for window in [(1, 1), (4, 4)]:
            with self.subTest(window=window):
                o = torch.empty_like(q)
                context_attention_fwd(
                    q,
                    k,
                    v,
                    o,
                    cu,
                    seq_lens,
                    max_len,
                    is_causal=False,
                    sm_scale=scale,
                    window_size=window,
                )
                ref = ref_varlen_attention(
                    q, k, v, cu.tolist(), scale, window_size=window
                )
                self.assertFalse(
                    torch.isnan(o.float()).any(), f"NaN leaked with window {window}"
                )
                torch.testing.assert_close(
                    o.float(),
                    ref.float(),
                    atol=ATOL,
                    rtol=RTOL,
                    msg=f"mismatch for window {window}",
                )

    @requires_cuda
    def test_kernel_causal_with_window(self):
        """Causal + window must compose: the window only further restricts the
        causal key range."""
        q, k, v, cu, seq_lens, max_len = make_varlen_inputs([128, 200], seed=13)
        scale = 1.0 / math.sqrt(q.shape[-1])
        window = (16, 16)
        o = torch.empty_like(q)
        context_attention_fwd(
            q,
            k,
            v,
            o,
            cu,
            seq_lens,
            max_len,
            is_causal=True,
            sm_scale=scale,
            window_size=window,
        )
        ref = ref_varlen_attention(
            q, k, v, cu.tolist(), scale, window_size=window, causal=True
        )
        torch.testing.assert_close(o.float(), ref.float(), atol=ATOL, rtol=RTOL)

    @requires_cuda
    def test_kernel_rejects_mismatched_sinks(self):
        q, k, v, cu, seq_lens, max_len = make_varlen_inputs([32], seed=14)
        scale = 1.0 / math.sqrt(q.shape[-1])
        o = torch.empty_like(q)
        with self.assertRaises(ValueError):
            context_attention_fwd(
                q,
                k,
                v,
                o,
                cu,
                seq_lens,
                max_len,
                is_causal=False,
                sm_scale=scale,
                sinks=torch.zeros(q.shape[1] + 1, device=q.device),
            )

    # ------------------------------------------------------------------
    # 3. Backend integration: VisionTritonAttention must forward the kwargs
    # ------------------------------------------------------------------
    @requires_cuda
    def test_backend_window_and_sinks_acceptance(self):
        """The issue #37983 reproducer.

        VisionAttention always calls its backend with window_size= and s_aux=
        kwargs.  A backend must either honor them or fail explicitly -- never
        silently compute full attention without sinks.
        """
        from sglang.srt.layers.attention.vision import VisionTritonAttention

        num_heads = 8
        q, k, v, cu, seq_lens, max_len = make_varlen_inputs(
            [64, 64], num_heads=num_heads, seed=5
        )
        scale = 1.0 / math.sqrt(q.shape[-1])
        sinks = rand_sinks(num_heads, seed=4321)
        window = (16, 16)

        backend = VisionTritonAttention(use_data_parallel=True)
        out = backend(
            q,
            k,
            v,
            cu_seqlens=cu,
            bsz=2,
            seq_len=64,
            softmax_scale=scale,
            max_seqlen=max_len,  # avoid a GPU->CPU sync in the backend
            window_size=window,
            s_aux=sinks,
        )
        ref = ref_varlen_attention(
            q, k, v, cu.tolist(), scale, window_size=window, sinks=sinks
        )
        torch.testing.assert_close(
            out.float(),
            ref.float(),
            atol=ATOL,
            rtol=RTOL,
            msg="VisionTritonAttention silently dropped window_size/s_aux "
            "(issue #37983): output equals full attention without sinks",
        )

    @requires_cuda
    def test_backend_kwargs_change_output(self):
        """Sensitivity detector: changing sinks (or the window) must change the
        output.  With the bug, all outputs were bit-identical because the
        kwargs were dropped."""
        from sglang.srt.layers.attention.vision import VisionTritonAttention

        num_heads = 8
        q, k, v, cu, seq_lens, max_len = make_varlen_inputs(
            [64, 64], num_heads=num_heads, seed=6
        )
        scale = 1.0 / math.sqrt(q.shape[-1])
        sinks_a = torch.zeros(num_heads, device=q.device)
        sinks_b = torch.full((num_heads,), 3.0, device=q.device)

        backend = VisionTritonAttention(use_data_parallel=True)

        def run(window, sinks):
            return backend(
                q,
                k,
                v,
                cu_seqlens=cu,
                bsz=2,
                seq_len=64,
                softmax_scale=scale,
                max_seqlen=max_len,
                window_size=window,
                s_aux=sinks,
            )

        out_a = run((16, 16), sinks_a)
        out_b = run((16, 16), sinks_b)
        out_c = run((-1, -1), sinks_a)

        self.assertFalse(
            torch.allclose(out_a, out_b, atol=1e-3),
            "different s_aux produced identical output: sinks are dropped "
            "(issue #37983)",
        )
        self.assertFalse(
            torch.allclose(out_a, out_c, atol=1e-3),
            "windowed and full attention produced identical output: window_size "
            "is dropped (issue #37983)",
        )

    # ------------------------------------------------------------------
    # 4. Optional cross-validation against the FA3 path (Hopper only)
    # ------------------------------------------------------------------
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
        "FA3 vision backend requires Hopper (sm90)",
    )
    def test_fa3_cross_check_reference(self):
        """Cross-validate the reference against VisionFlash3Attention, which
        provably forwards window_size and s_aux (sinks) to flash-attn."""
        try:
            from sglang.srt.layers.attention.vision import VisionFlash3Attention
        except Exception as e:
            self.skipTest(f"cannot import FA3 backend here: {e}")

        q, k, v, cu, seq_lens, max_len = make_varlen_inputs([64, 48], seed=8)
        scale = 1.0 / math.sqrt(q.shape[-1])
        window = (16, 16)
        sinks = rand_sinks(q.shape[1], seed=9)

        try:
            backend = VisionFlash3Attention(use_data_parallel=True)
        except Exception as e:
            self.skipTest(f"cannot construct FA3 backend here: {e}")

        try:
            out = backend(
                q,
                k,
                v,
                cu_seqlens=cu,
                bsz=2,
                seq_len=64,
                softmax_scale=scale,
                max_seqlen=max_len,
                window_size=window,
                s_aux=sinks,
            )
        except (NotImplementedError, RuntimeError, TypeError) as e:
            self.skipTest(f"FA3 sinks/window unavailable in this build: {e}")

        ref = ref_varlen_attention(
            q, k, v, cu.tolist(), scale, window_size=window, sinks=sinks
        )
        torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    unittest.main()
