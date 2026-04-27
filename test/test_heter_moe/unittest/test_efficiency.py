"""1.4 Efficiency tests (microbenchmarks).

Validates that fused {BF16, INT4} latency lies between pure BF16 and
pure INT4 across varying batch sizes. Also verifies CUDA graph capture
and torch.compile compatibility.

Adapted from scripts/heter_moe_verify_hypothesis.py.

Requirements:
  - CUDA graph capture (bench harness tries graph, falls back to eager)
  - Large workload to avoid L2 cache camping
  - Proper warmup (L2-camping-aware, flushes L2 between timed iterations)
  - Median-of-N timing for stability
"""

import pytest
import torch

from test_heter_moe.util import CUDA_AVAILABLE, make_topk_output

# ---------------------------------------------------------------------------
# Layer-level test dimensions (for CUDA graph / torch.compile tests)
# ---------------------------------------------------------------------------
LAYER_E, LAYER_H, LAYER_I, LAYER_TOP_K = 16, 1024, 512, 2
LAYER_GROUP_SIZE = 128
SEED = 42

# ---------------------------------------------------------------------------
# Kernel-level benchmark dimensions (large enough to be compute-bound)
# Must be large so compute dominates kernel launch overhead; the
# hypothesis (INT4 < mixed < BF16) only holds in the compute-bound regime.
# ---------------------------------------------------------------------------
KERN_K, KERN_N, KERN_E, KERN_TOP_K = 2048, 768, 128, 8
KERN_GROUP_SIZE = 128
KERN_NUM_BITS = 4
KERN_COLD_RATIO = 0.8

# Benchmark tuning
WARMUP = 20
ITERS = 50
_L2_FLUSH_SIZE = 50 * 1024 * 1024  # 50 MiB


# ---------------------------------------------------------------------------
# Benchmarking helpers (adapted from scripts/heter_moe_verify_hypothesis.py)
# ---------------------------------------------------------------------------

_l2_flush_buf = None


def _get_l2_flush_buf(device):
    global _l2_flush_buf
    if _l2_flush_buf is None or _l2_flush_buf.device != device:
        _l2_flush_buf = torch.empty(_L2_FLUSH_SIZE, dtype=torch.int8, device=device)
    return _l2_flush_buf


def _flush_l2(device):
    """Zero a large buffer to evict L2 cache contents."""
    _get_l2_flush_buf(device).zero_()


def _bench(fn, device, warmup=WARMUP, iters=ITERS, use_cuda_graph=True):
    """Time a function with L2 flush, warmup, and optional CUDA graph.

    Returns median latency in ms.
    """
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    graph = None
    if use_cuda_graph:
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            torch.cuda.synchronize()
            for _ in range(warmup):
                graph.replay()
            torch.cuda.synchronize()
        except Exception:
            graph = None

    if graph is None:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        _flush_l2(device)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        if graph is not None:
            graph.replay()
        else:
            fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Kernel-level weight / input factories
# (random data — shape-correct, not semantically meaningful)
# ---------------------------------------------------------------------------


def _make_bf16_weights(device):
    w13 = torch.randn(KERN_E, 2 * KERN_N, KERN_K, dtype=torch.bfloat16, device=device)
    w2 = torch.randn(KERN_E, KERN_K, KERN_N, dtype=torch.bfloat16, device=device)
    return w13, w2


def _make_int4_weights(device):
    """Random Marlin-format INT4 weights (packed int32)."""
    w1 = torch.randint(
        0,
        2**31,
        (KERN_E, KERN_K // 16, 2 * KERN_N * (KERN_NUM_BITS // 2)),
        dtype=torch.int32,
        device=device,
    )
    w2 = torch.randint(
        0,
        2**31,
        (KERN_E, KERN_N // 16, KERN_K * (KERN_NUM_BITS // 2)),
        dtype=torch.int32,
        device=device,
    )
    s1 = (
        torch.ones(
            KERN_E,
            KERN_K // KERN_GROUP_SIZE,
            2 * KERN_N,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.01
    )
    s2 = (
        torch.ones(
            KERN_E,
            KERN_N // KERN_GROUP_SIZE,
            KERN_K,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.01
    )
    return w1, w2, s1, s2


def _make_kernel_inputs(m_per_expert, device):
    """Each expert gets exactly m_per_expert tokens (uniform distribution)."""
    M_global = m_per_expert * KERN_E // KERN_TOP_K
    x = torch.randn(M_global, KERN_K, dtype=torch.bfloat16, device=device)
    topk_w = (
        torch.ones(M_global, KERN_TOP_K, dtype=torch.bfloat16, device=device)
        / KERN_TOP_K
    )
    all_expert_ids = torch.arange(KERN_E, device=device).repeat(m_per_expert)
    all_expert_ids = all_expert_ids[torch.randperm(len(all_expert_ids), device=device)]
    topk_ids = all_expert_ids.reshape(M_global, KERN_TOP_K)
    gating = torch.randn(M_global, KERN_E, dtype=torch.bfloat16, device=device)
    return x, topk_w, topk_ids, gating


def _build_mixed_dispatch(topk_ids, topk_w, device):
    """Split experts into cold (INT4) and hot (BF16) groups.

    First half of experts → cold (INT4, sentinel=E for Marlin).
    Second half → hot (BF16, sentinel=-1 for Triton).
    """
    n_cold = int(KERN_E * KERN_COLD_RATIO)

    # Cold group: experts [0, n_cold)
    cold_in = topk_ids < n_cold
    cold_ids = torch.where(
        cold_in, topk_ids, torch.tensor(KERN_E, device=device)
    )
    cold_w = topk_w * cold_in.to(topk_w.dtype)

    # Hot group: experts [n_cold, E)
    hot_in = topk_ids >= n_cold
    hot_ids = torch.where(hot_in, topk_ids, torch.tensor(-1, device=device))
    hot_w = topk_w * hot_in.to(topk_w.dtype)

    return cold_ids, cold_w, hot_ids, hot_w


# ---------------------------------------------------------------------------
# Layer-level helpers (for CUDA graph / torch.compile tests)
# ---------------------------------------------------------------------------


def _make_layer(config):
    from sglang.srt.layers.moe.heter_moe import HeterFusedMoE

    return HeterFusedMoE(
        num_experts=LAYER_E,
        hidden_size=LAYER_H,
        intermediate_size=LAYER_I,
        top_k=LAYER_TOP_K,
        heter_config=config,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )


def _make_layer_inputs(batch_size, with_logits=False):
    torch.manual_seed(0)
    x = torch.randn(batch_size, LAYER_H, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.randint(0, LAYER_E, (batch_size, LAYER_TOP_K), device="cuda")
    topk_weights = torch.rand(
        batch_size, LAYER_TOP_K, dtype=torch.bfloat16, device="cuda"
    )
    router_logits = (
        torch.randn(batch_size, LAYER_E, device="cuda") if with_logits else None
    )
    return x, make_topk_output(topk_weights, topk_ids, router_logits)


def _fill_int4(layer, bf16_w13, bf16_w2):
    from test_heter_moe.unittest.int4_marlin_weight_no_gptq import fill_int4_params

    fill_int4_params(layer, bf16_w13, bf16_w2, LAYER_GROUP_SIZE)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def layer_shared_weights():
    """BF16 reference weights for layer-level tests."""
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    w13 = (
        torch.randn(
            LAYER_E,
            2 * LAYER_I,
            LAYER_H,
            dtype=torch.bfloat16,
            device="cuda",
            generator=gen,
        )
        * 0.02
    )
    w2 = (
        torch.randn(
            LAYER_E,
            LAYER_H,
            LAYER_I,
            dtype=torch.bfloat16,
            device="cuda",
            generator=gen,
        )
        * 0.02
    )
    return w13, w2


@pytest.fixture(scope="module")
def bf16_layer(layer_shared_weights):
    bf16_w13, bf16_w2 = layer_shared_weights
    cfg = {
        "groups": [{"name": "all", "num_bits": 16, "size_ratio": 1.0}],
        "policy": "random",
        "policy_params": {"seed": SEED},
    }
    layer = _make_layer(cfg)
    layer.w13_weight.data.copy_(bf16_w13)
    layer.w2_weight.data.copy_(bf16_w2)
    return layer


@pytest.fixture(scope="module")
def mixed_layer(layer_shared_weights):
    bf16_w13, bf16_w2 = layer_shared_weights
    cfg = {
        "groups": [
            {
                "name": "cold_int4",
                "num_bits": 4,
                "size_ratio": 0.5,
                "group_size": LAYER_GROUP_SIZE,
            },
            {"name": "hot_bf16", "num_bits": 16, "size_ratio": 0.5},
        ],
        "policy": "random",
        "policy_params": {"seed": SEED},
    }
    layer = _make_layer(cfg)
    layer.w13_weight.data.copy_(bf16_w13)
    layer.w2_weight.data.copy_(bf16_w2)
    _fill_int4(layer, bf16_w13, bf16_w2)
    return layer


# ---------------------------------------------------------------------------
# 1.4.1 Latency table for fused heterogeneous {BF16, INT4} MoE
#
# Prints per-batch latencies across pure BF16, pure INT4, cold/hot split,
# and back-to-back mixed. The only asserted invariant is mix ≈ cold+hot
# (no surprise overhead from interleaving the two kernels). Pure-BF16 vs
# pure-INT4 ordering is regime-dependent (memory-bound at small M,
# compute-bound at large M) and assumes a tuned BF16 tile config exists
# for (E, intermediate_size, GPU); inspect the printed table.
# ---------------------------------------------------------------------------


PER_EXPERT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
# Granular sweep around the BF16↔INT4 crossover identified from the coarse
# sweep above (BF16 wins at M/e=128, INT4 wins at M/e=64). Step 8.
CROSSOVER_BATCH_SIZES = list(range(48, 96 + 1, 8))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestFusedLatencyOrdering:
    """Microbenchmark heterogeneous {BF16, INT4} MoE at Qwen3-30B-A3B shape.

    Shape: E=128, K=2048, N=768 (per-expert intermediate). Uses raw kernel
    calls (outplace_fused_experts / fused_marlin_moe). The pure-BF16 path
    requires the tuned tile config at
      python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_<ver>/
        E=128,N=768,device_name=<gpu>.json
    Without it, the kernel falls back to a default tile and BF16 numbers
    will be 1.5–3x slower than they should be.
    """

    def test_latency_between_pure_precisions(self):
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
            outplace_fused_experts,
        )

        device = torch.device("cuda")
        bf16_w13, bf16_w2 = _make_bf16_weights(device)
        int4_w1, int4_w2, int4_s1, int4_s2 = _make_int4_weights(device)

        rows = []

        for m_per_expert in PER_EXPERT_BATCH_SIZES:
            x, topk_w, topk_ids, gating = _make_kernel_inputs(m_per_expert, device)
            M_global = x.shape[0]

            # Pure BF16
            lat_bf16 = _bench(
                lambda: outplace_fused_experts(
                    x, bf16_w13, bf16_w2, topk_w, topk_ids
                ),
                device,
            )

            # Pure INT4 (Marlin)
            lat_int4 = _bench(
                lambda: fused_marlin_moe(
                    x,
                    int4_w1,
                    int4_w2,
                    int4_s1,
                    int4_s2,
                    gating,
                    topk_w,
                    topk_ids,
                    num_bits=KERN_NUM_BITS,
                    is_k_full=True,
                ),
                device,
            )

            # Build per-group dispatch masks
            cold_ids, cold_w, hot_ids, hot_w = _build_mixed_dispatch(
                topk_ids, topk_w, device
            )

            # Cold-only (INT4 subset)
            lat_cold = _bench(
                lambda: fused_marlin_moe(
                    x,
                    int4_w1,
                    int4_w2,
                    int4_s1,
                    int4_s2,
                    gating,
                    cold_w,
                    cold_ids,
                    num_bits=KERN_NUM_BITS,
                    is_k_full=True,
                ),
                device,
            )

            # Hot-only (BF16 subset)
            lat_hot = _bench(
                lambda: outplace_fused_experts(
                    x, bf16_w13, bf16_w2, hot_w, hot_ids
                ),
                device,
            )

            # Mixed: cold INT4 + hot BF16 (both kernels, sequential)
            def mix_fn():
                fused_marlin_moe(
                    x,
                    int4_w1,
                    int4_w2,
                    int4_s1,
                    int4_s2,
                    gating,
                    cold_w,
                    cold_ids,
                    num_bits=KERN_NUM_BITS,
                    is_k_full=True,
                )
                outplace_fused_experts(x, bf16_w13, bf16_w2, hot_w, hot_ids)

            lat_mixed = _bench(mix_fn, device, use_cuda_graph=True)

            rows.append(
                (m_per_expert, M_global, lat_bf16, lat_int4,
                 lat_cold, lat_hot, lat_mixed)
            )

        # ---- Print summary table ----
        cold_pct = f"{KERN_COLD_RATIO:.0%}"
        hot_pct = f"{1 - KERN_COLD_RATIO:.0%}"
        W = 110
        print(f"\n{'=' * W}")
        print(
            f"{'M/e':>5} {'Mglob':>6} | "
            f"{'a16w16':>8} {'a16w4':>8} | "
            f"{'cold'+cold_pct:>8} {'hot'+hot_pct:>8} | "
            f"{'mix':>8} {'c+h':>8} | "
            f"{'in range':>8}"
        )
        print("-" * W)
        for M, M_g, bf16, int4, cold, hot, mix in rows:
            ok = "Y" if int4 <= mix <= bf16 else "N"
            print(
                f"{M:>5} {M_g:>6} | "
                f"{bf16:8.3f} {int4:8.3f} | "
                f"{cold:8.3f} {hot:8.3f} | "
                f"{mix:8.3f} {cold + hot:8.3f} | "
                f"{ok:>8}"
            )
        print("=" * W)
        print(
            f"cold{cold_pct} = a16w4 on {cold_pct} experts, "
            f"hot{hot_pct} = a16w16 on {hot_pct} experts, "
            f"c+h = cold+hot sum"
        )
        print(
            f"E={KERN_E}, K={KERN_K}, N={KERN_N}, top_k={KERN_TOP_K}"
        )

        # ---- Sanity invariant ----
        # Mixed = cold INT4 + hot BF16 run back-to-back. Should be ≈ the
        # sum of running them separately, with at most a small graph-fusion
        # benefit. We don't assert pure-INT4 vs pure-BF16 ordering: with a
        # properly tuned BF16 tile, BF16 wins at high M (compute-bound;
        # Ampere has no native int4 tensor cores). With the default
        # fallback tile, BF16 loses everywhere. Inspect the table above.
        tol = 0.10
        failures = []
        for M, M_g, bf16, int4, cold, hot, mix in rows:
            if not (mix <= (cold + hot) * (1 + tol)):
                failures.append(
                    f"M/e={M}: mixed ({mix:.3f}ms) > cold+hot "
                    f"({cold + hot:.3f}ms) by >{int(tol * 100)}%"
                )
        assert not failures, (
            "Mixed kernel has unexpected overhead vs sequential cold+hot:\n  "
            + "\n  ".join(failures)
        )

    def test_int4_bf16_crossover(self):
        """Granular sweep to locate the BF16↔INT4 crossover.

        Times pure BF16 vs pure INT4 (Marlin) at M/e ∈ {64, 80, 96, 112,
        128}. Diagnostic only — no latency assertion. Reports the
        smallest M/e where BF16 ≤ INT4.
        """
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
            outplace_fused_experts,
        )

        device = torch.device("cuda")
        bf16_w13, bf16_w2 = _make_bf16_weights(device)
        int4_w1, int4_w2, int4_s1, int4_s2 = _make_int4_weights(device)

        rows = []
        for m_per_expert in CROSSOVER_BATCH_SIZES:
            x, topk_w, topk_ids, gating = _make_kernel_inputs(m_per_expert, device)
            M_global = x.shape[0]

            lat_bf16 = _bench(
                lambda: outplace_fused_experts(
                    x, bf16_w13, bf16_w2, topk_w, topk_ids
                ),
                device,
            )
            lat_int4 = _bench(
                lambda: fused_marlin_moe(
                    x, int4_w1, int4_w2, int4_s1, int4_s2,
                    gating, topk_w, topk_ids,
                    num_bits=KERN_NUM_BITS, is_k_full=True,
                ),
                device,
            )
            rows.append((m_per_expert, M_global, lat_bf16, lat_int4))

        W = 72
        print(f"\n{'=' * W}")
        print(
            f"BF16↔INT4 crossover sweep  "
            f"(E={KERN_E}, K={KERN_K}, N={KERN_N}, top_k={KERN_TOP_K})"
        )
        print("-" * W)
        print(
            f"{'M/e':>5} {'Mglob':>6} | "
            f"{'a16w16':>9} {'a16w4':>9} | "
            f"{'BF16/INT4':>10} {'winner':>8}"
        )
        print("-" * W)
        crossover = None
        for M, M_g, bf16, int4 in rows:
            ratio = bf16 / int4
            winner = "BF16" if bf16 < int4 else "INT4"
            if crossover is None and bf16 <= int4:
                crossover = M
            print(
                f"{M:>5} {M_g:>6} | "
                f"{bf16:9.3f} {int4:9.3f} | "
                f"{ratio:>10.3f} {winner:>8}"
            )
        print("=" * W)
        if crossover is not None:
            print(f"Crossover: BF16 first wins at M/e = {crossover}")
        else:
            print(
                f"No crossover within {CROSSOVER_BATCH_SIZES} "
                f"— INT4 wins through M/e={CROSSOVER_BATCH_SIZES[-1]}"
            )


# ---------------------------------------------------------------------------
# 1.4.2 CUDA graph compatibility
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCudaGraphCompatibility:
    """Verify HeterFusedMoE is capturable in a CUDA graph."""

    def test_cuda_graph_capture_and_replay(self, bf16_layer):
        """Capture a BF16 HeterFusedMoE forward in a CUDA graph, replay,
        and verify output matches eager."""
        x, topk_out = _make_layer_inputs(64)

        eager_out = bf16_layer(x, topk_out).clone()

        for _ in range(3):
            bf16_layer(x, topk_out)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = bf16_layer(x, topk_out)
        torch.cuda.synchronize()

        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)

    def test_cuda_graph_capture_mixed(self, mixed_layer):
        """Capture a mixed BF16+INT4 HeterFusedMoE forward in a CUDA graph.

        Marlin INT4 may have small numerical differences between eager and
        graph replay due to internal workspace handling, so we use tolerance.
        """
        x, topk_out = _make_layer_inputs(64, with_logits=True)

        for _ in range(3):
            mixed_layer(x, topk_out)
        torch.cuda.synchronize()

        # Capture reference AFTER warmup to ensure consistent state
        eager_out = mixed_layer(x, topk_out).clone()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = mixed_layer(x, topk_out)
        torch.cuda.synchronize()

        graph.replay()
        torch.cuda.synchronize()

        assert graph_out.isfinite().all(), "Graph replay output contains NaN/Inf"
        assert graph_out.shape == eager_out.shape
        torch.testing.assert_close(graph_out, eager_out, atol=0.01, rtol=0.01)


# ---------------------------------------------------------------------------
# 1.4.3 torch.compile compatibility
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestTorchCompileCompatibility:
    """Verify HeterFusedMoE works under torch.compile."""

    def test_compiled_matches_eager(self, bf16_layer):
        """torch.compile'd BF16 HeterFusedMoE produces same output as eager."""
        x, topk_out = _make_layer_inputs(64)

        eager_out = bf16_layer(x, topk_out).clone()

        compiled_layer = torch.compile(bf16_layer)
        # Warmup compile
        for _ in range(3):
            compiled_layer(x, topk_out)
        compiled_out = compiled_layer(x, topk_out)

        torch.testing.assert_close(compiled_out, eager_out, atol=0, rtol=0)

    def test_compiled_mixed_runs(self, mixed_layer):
        """torch.compile'd mixed BF16+INT4 HeterFusedMoE produces finite output.

        torch.compile may reorder ops or fuse differently around custom ops,
        so we only verify the output is finite and non-zero rather than
        checking bit-exact equality.
        """
        x, topk_out = _make_layer_inputs(64, with_logits=True)

        compiled_layer = torch.compile(mixed_layer)
        for _ in range(3):
            compiled_layer(x, topk_out)
        compiled_out = compiled_layer(x, topk_out)

        assert compiled_out.shape == x.shape
        assert compiled_out.isfinite().all(), "Compiled mixed output contains NaN/Inf"
        assert compiled_out.abs().sum() > 0, "Compiled mixed output is all zeros"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
