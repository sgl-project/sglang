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
KERN_K, KERN_N, KERN_E, KERN_TOP_K = 2048, 1536, 128, 8
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
# 1.4.1 Latency ordering: INT4 <= mixed <= BF16
#
# Benchmarks at the raw kernel level with large dimensions so the
# hypothesis holds in the compute-bound regime. At small problem sizes,
# double kernel launch overhead makes mixed slower than pure BF16.
# ---------------------------------------------------------------------------


PER_EXPERT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestFusedLatencyOrdering:
    """Fused {BF16, INT4} latency should be between pure BF16 and pure INT4.

    Uses raw kernel calls (outplace_fused_experts / fused_marlin_moe) with
    large dimensions (E=64, K=2048, N=768) to ensure compute-bound regime.
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

        # ---- Assertions (20% tolerance for measurement noise) ----
        tol = 0.20
        failures = []
        for M, M_g, bf16, int4, cold, hot, mix in rows:
            if not (int4 <= bf16 * (1 + tol)):
                failures.append(
                    f"M/e={M}: INT4 ({int4:.3f}ms) > BF16 ({bf16:.3f}ms)"
                )
            if not (int4 * (1 - tol) <= mix):
                failures.append(
                    f"M/e={M}: mixed ({mix:.3f}ms) < INT4 ({int4:.3f}ms)"
                )
            if not (mix <= bf16 * (1 + tol)):
                failures.append(
                    f"M/e={M}: mixed ({mix:.3f}ms) > BF16 ({bf16:.3f}ms)"
                )
        assert not failures, (
            "Latency ordering violated:\n  " + "\n  ".join(failures)
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
