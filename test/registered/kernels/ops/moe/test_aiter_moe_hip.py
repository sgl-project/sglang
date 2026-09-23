"""The one-launch MoE sorting, the ROCm decode router gate and the fused gate + sort must reproduce aiter's `moe_sorting` and `topk_gating` bit for bit."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=25, suite="stage-b-kernel-test-1-gpu-amd-mi35x")


try:
    from aiter.fused_moe import moe_sorting as aiter_moe_sorting
except ImportError:  # aiter is absent off ROCm
    aiter_moe_sorting = None

try:
    from aiter import topk_gating as aiter_topk_gating
except ImportError:  # aiter is absent off ROCm
    aiter_topk_gating = None


NUM_EXPERTS = 384
MODEL_DIM = 5120
HIDDEN = 5120
TOPK = 6
ROUTED_SCALING = 1.5


def _routing(num_tokens, topk, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    ids = torch.stack(
        [
            torch.randperm(NUM_EXPERTS, device=device, generator=g)[:topk]
            for _ in range(num_tokens)
        ]
    ).to(torch.int32)
    weights = torch.rand(num_tokens, topk, device=device, generator=g)
    return ids, weights


def _assert_same_sort(test, ref, out, block_size, num_valid_rows, num_tokens):
    """Compare the region aiter's GEMMs read (up to the padded total)."""
    test.assertTrue(torch.equal(ref[3], out[3]), "num_valid_ids")
    total = int(ref[3][0])
    ref_ids, out_ids = ref[0][:total], out[0][:total]
    tokens = ref_ids & 0xFFFFFF
    test.assertTrue(torch.equal(tokens, out_ids & 0xFFFFFF), "sorted tokens")
    racy_slot = (tokens >= num_valid_rows) & (tokens < num_tokens)
    test.assertTrue(
        torch.equal(
            torch.where(racy_slot, 0, ref_ids), torch.where(racy_slot, 0, out_ids)
        ),
        "sorted slots",
    )
    test.assertTrue(torch.equal(ref[1][:total], out[1][:total]), "sorted weights")
    test.assertTrue(
        torch.equal(ref[2][: total // block_size], out[2][: total // block_size]),
        "sorted expert ids",
    )
    test.assertEqual(ref[4].shape, out[4].shape)
    test.assertTrue(torch.equal(ref[4], out[4]), "moe_buf")


@unittest.skipUnless(is_hip() and aiter_moe_sorting is not None, "ROCm + aiter only")
class TestFusedAiterMoeSorting(CustomTestCase):
    def setUp(self):
        from sglang.kernels.ops.moe.aiter_moe_sorting_fused import (
            fused_aiter_moe_sorting,
            local_expert_ids_from_mask,
        )

        self.fused = fused_aiter_moe_sorting
        self.local_ids = local_expert_ids_from_mask
        self.device = torch.device("cuda")

    def _mask(self, num_local, rank):
        mask = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=self.device)
        mask[rank * num_local : (rank + 1) * num_local] = 1
        return mask

    def test_matches_aiter_expert_parallel(self):
        for num_local, topk in ((96, TOPK),):
            for rank in (0, NUM_EXPERTS // num_local - 1):
                mask = self._mask(num_local, rank)
                local_ids = self.local_ids(mask, NUM_EXPERTS, self.device)
                for num_tokens in (1, 64):
                    for block_size in (16, 64):
                        ids, weights = _routing(
                            num_tokens, topk, num_tokens, self.device
                        )
                        ref = aiter_moe_sorting(
                            ids,
                            weights,
                            NUM_EXPERTS,
                            MODEL_DIM,
                            torch.bfloat16,
                            block_size,
                            mask,
                            None,
                            0,
                            accumulate=True,
                        )
                        out = self.fused(
                            ids,
                            weights,
                            local_ids,
                            num_local,
                            NUM_EXPERTS,
                            MODEL_DIM,
                            torch.bfloat16,
                            block_size,
                            zero_moe_buf=True,
                        )
                        _assert_same_sort(
                            self, ref, out, block_size, num_tokens, num_tokens
                        )
                        self.assertEqual(int(out[4].abs().sum()), 0)

    def test_padded_rows_are_masked_in_place(self):
        mask = self._mask(96, 1)
        local_ids = self.local_ids(mask, NUM_EXPERTS, self.device)
        for num_tokens in (64,):
            for num_valid in (1, num_tokens // 2, num_tokens):
                ids, weights = _routing(
                    num_tokens, 6, 7 * num_tokens + num_valid, self.device
                )
                count = torch.tensor([num_valid], dtype=torch.int32, device=self.device)
                masked_ids, masked_weights = ids.clone(), weights.clone()
                masked_ids[num_valid:] = 0
                masked_weights[num_valid:] = 0.0
                ref = aiter_moe_sorting(
                    masked_ids,
                    masked_weights,
                    NUM_EXPERTS,
                    MODEL_DIM,
                    torch.bfloat16,
                    32,
                    mask,
                    None,
                    0,
                    accumulate=True,
                )
                out = self.fused(
                    ids,
                    weights,
                    local_ids,
                    96,
                    NUM_EXPERTS,
                    MODEL_DIM,
                    torch.bfloat16,
                    32,
                    zero_moe_buf=True,
                    num_token_non_padded=count,
                )
                _assert_same_sort(self, ref, out, 32, num_valid, num_tokens)
                # The fills happened in the same launch.
                self.assertTrue(torch.equal(ids, masked_ids))
                self.assertTrue(torch.equal(weights, masked_weights))


def _aiter_gate(logits, bias, topk, renorm, rsf):
    weights = torch.empty(
        logits.shape[0], topk, dtype=torch.float32, device=logits.device
    )
    ids = torch.empty(logits.shape[0], topk, dtype=torch.int32, device=logits.device)
    aiter_topk_gating(
        weights, ids, logits, bias, renorm, rsf, score_func="sqrtsoftplus"
    )
    return weights, ids


@unittest.skipUnless(
    is_hip() and is_gfx95_supported() and aiter_topk_gating is not None,
    "ROCm gfx95 + aiter only",
)
class TestRocmRouterGate(CustomTestCase):
    def setUp(self):
        from sglang.kernels.ops.moe.rocm_router_gate import (
            ROCM_ROUTER_MAX_TOKENS,
            rocm_router_gate,
            rocm_router_gemv_split_k,
            rocm_router_max_tokens,
            rocm_router_reduce_partials,
        )

        self.max_tokens = ROCM_ROUTER_MAX_TOKENS
        self.gate = rocm_router_gate
        self.gemv = rocm_router_gemv_split_k
        self.max_tokens_for = rocm_router_max_tokens
        self.reduce = rocm_router_reduce_partials
        self.device = torch.device("cuda")
        self.gen = torch.Generator(device=self.device).manual_seed(0)
        self.bias_bf16 = (
            torch.randn(NUM_EXPERTS, device=self.device, generator=self.gen) * 0.5
        ).to(torch.bfloat16)

    def _randn(self, *shape, scale=1.0):
        return torch.randn(*shape, device=self.device, generator=self.gen) * scale

    def _assert_same_gate(self, logits, bias, renorm=True, rsf=ROUTED_SCALING, msg=""):
        for topk in (TOPK,):
            ref_w, ref_i = _aiter_gate(logits, bias, topk, renorm, rsf)
            out_w, out_i = self.gate(logits, bias, topk, renorm, rsf)
            self.assertTrue(torch.equal(ref_i, out_i), f"ids {msg} topk {topk}")
            self.assertTrue(torch.equal(ref_w, out_w), f"weights {msg} topk {topk}")

    def test_gate_matches_aiter_on_ties(self):
        zero_bias = torch.zeros(NUM_EXPERTS, device=self.device, dtype=torch.bfloat16)
        for num_tokens in (512,):
            levels = torch.randint(
                0, 3, (num_tokens, NUM_EXPERTS), device=self.device, generator=self.gen
            ).float()
            self._assert_same_gate(levels, zero_bias, msg="3 levels")
            levels = torch.randint(
                0, 8, (num_tokens, NUM_EXPERTS), device=self.device, generator=self.gen
            ).float()
            self._assert_same_gate(levels - 3, self.bias_bf16, msg="8 levels")
            self._assert_same_gate(
                torch.zeros(num_tokens, NUM_EXPERTS, device=self.device),
                zero_bias,
                msg="all equal",
            )
            few = torch.full(
                (num_tokens, NUM_EXPERTS), float("-inf"), device=self.device
            )
            few[:, :3] = 1.0
            self._assert_same_gate(few, zero_bias, msg="3 finite experts")

    def test_gate_matches_aiter_non_finite(self):
        for num_tokens in (16,):
            logits = self._randn(num_tokens, NUM_EXPERTS, scale=3.0)
            logits[logits < 0] = float("-inf")
            self._assert_same_gate(logits, self.bias_bf16, msg="-inf")
            logits = self._randn(num_tokens, NUM_EXPERTS, scale=3.0)
            logits[:, ::7] = float("nan")
            self._assert_same_gate(logits, self.bias_bf16, msg="nan")
            logits = self._randn(num_tokens, NUM_EXPERTS, scale=3.0)
            logits[:, 5] = float("inf")
            self._assert_same_gate(logits, self.bias_bf16, msg="+inf")

    def test_gemv_accuracy_batch_invariance_and_repeatability(self):
        weight = (self._randn(NUM_EXPERTS, HIDDEN) * 0.02).to(torch.bfloat16)
        x = self._randn(self.max_tokens, HIDDEN).to(torch.bfloat16)
        ref = (x.double() @ weight.double().T).float()
        full = torch.empty(self.max_tokens, NUM_EXPERTS, device=self.device)
        self.reduce(self.gemv(x, weight), full)
        self.assertTrue(torch.allclose(full, ref, atol=2e-3, rtol=1e-4))
        for num_tokens in (1, 17, 64):
            rows = x[:num_tokens]
            out = torch.empty(num_tokens, NUM_EXPERTS, device=self.device)
            self.reduce(self.gemv(rows, weight), out)
            self.assertTrue(
                torch.equal(out, full[:num_tokens]), f"batch of {num_tokens}"
            )
            # The same row moved to another position of the batch.
            shifted = torch.roll(x, shifts=num_tokens, dims=0)
            out_shifted = torch.empty_like(full)
            self.reduce(self.gemv(shifted, weight), out_shifted)
            self.assertTrue(torch.equal(out_shifted, torch.roll(full, num_tokens, 0)))


@unittest.skipUnless(
    is_hip()
    and is_gfx95_supported()
    and aiter_topk_gating is not None
    and aiter_moe_sorting is not None,
    "ROCm gfx95 + aiter only",
)
class TestRocmRouterGateSort(CustomTestCase):
    """One launch = gate launch + sorting launch, bit for bit, and aiter for both."""

    def setUp(self):
        from sglang.kernels.ops.moe.aiter_moe_sorting_fused import (
            fused_aiter_moe_sorting,
            local_expert_ids_from_mask,
        )
        from sglang.kernels.ops.moe.rocm_router_gate import (
            rocm_router_gate,
            rocm_router_gemv_split_k,
            rocm_router_reduce_partials,
        )
        from sglang.kernels.ops.moe.rocm_router_gate_sort import (
            ROCM_GATE_SORT_MAX_TOKENS,
            _handoff_buffer,
            rocm_router_gate_sort,
        )

        self.max_tokens = ROCM_GATE_SORT_MAX_TOKENS
        self.fused = rocm_router_gate_sort
        self.handoff = _handoff_buffer
        self.gate = rocm_router_gate
        self.gemv = rocm_router_gemv_split_k
        self.reduce = rocm_router_reduce_partials
        self.sort = fused_aiter_moe_sorting
        self.local_ids = local_expert_ids_from_mask
        self.device = torch.device("cuda")
        self.gen = torch.Generator(device=self.device).manual_seed(0)
        self.bias = (
            torch.randn(NUM_EXPERTS, device=self.device, generator=self.gen) * 0.5
        ).to(torch.bfloat16)
        self.weight = (
            torch.randn(NUM_EXPERTS, HIDDEN, device=self.device, generator=self.gen)
            * 0.02
        ).to(torch.bfloat16)

    def _mask(self, num_local, rank):
        if num_local == NUM_EXPERTS:
            return None
        mask = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=self.device)
        mask[rank * num_local : (rank + 1) * num_local] = 1
        return mask

    def _inputs(self, num_tokens, ties):
        if ties:
            logits = torch.randint(
                0, 3, (num_tokens, NUM_EXPERTS), device=self.device, generator=self.gen
            ).float()
            return logits, None
        x = torch.randn(num_tokens, HIDDEN, device=self.device, generator=self.gen).to(
            torch.bfloat16
        )
        partials = self.gemv(x, self.weight)
        logits = torch.empty(num_tokens, NUM_EXPERTS, device=self.device)
        self.reduce(partials, logits)
        return logits, partials

    def _assert_gate_sort_matches_two_launches(
        self, num_tokens, topk, block_size, num_local, rank, pad, ties
    ):
        msg = f"M={num_tokens} topk={topk} block={block_size} local={num_local}/{rank} pad={pad} ties={ties}"
        mask = self._mask(num_local, rank)
        local_ids = self.local_ids(mask, NUM_EXPERTS, self.device)
        logits, partials = self._inputs(num_tokens, ties)
        count = (
            torch.tensor(
                [max(1, num_tokens - 1)], dtype=torch.int32, device=self.device
            )
            if pad
            else None
        )
        n_valid = int(count) if pad else num_tokens
        # reference: the two launches, and aiter itself
        ref_w, ref_i = self.gate(
            logits.clone(), self.bias, topk, True, ROUTED_SCALING, partials=partials
        )
        aiter_w, aiter_i = _aiter_gate(logits, self.bias, topk, True, ROUTED_SCALING)
        self.assertTrue(
            torch.equal(aiter_w, ref_w) and torch.equal(aiter_i, ref_i), msg
        )
        ref_sort = self.sort(
            ref_i,
            ref_w,
            local_ids,
            num_local,
            NUM_EXPERTS,
            MODEL_DIM,
            torch.bfloat16,
            block_size,
            mask is not None,
            num_token_non_padded=count,
        )
        masked_i, masked_w = aiter_i.clone(), aiter_w.clone()
        masked_i[n_valid:] = 0
        masked_w[n_valid:] = 0.0
        aiter_sort = aiter_moe_sorting(
            masked_i,
            masked_w,
            NUM_EXPERTS,
            MODEL_DIM,
            torch.bfloat16,
            block_size,
            mask,
            None,
            0,
            accumulate=mask is not None,
        )
        fused_logits = torch.empty_like(logits) if partials is not None else logits
        out = self.fused(
            fused_logits,
            self.bias,
            topk,
            True,
            ROUTED_SCALING,
            partials,
            local_ids,
            NUM_EXPERTS,
            MODEL_DIM,
            torch.bfloat16,
            block_size,
            mask is not None,
            num_token_non_padded=count,
        )
        self.assertTrue(torch.equal(out[1], masked_i), f"ids {msg}")
        self.assertTrue(torch.equal(out[0], masked_w), f"weights {msg}")
        if partials is not None:
            self.assertTrue(torch.equal(fused_logits, logits), f"logits {msg}")
        _assert_same_sort(self, ref_sort, out[2:], block_size, n_valid, num_tokens)
        _assert_same_sort(self, aiter_sort, out[2:], block_size, n_valid, num_tokens)
        self.assertEqual(int(self.handoff(self.device).abs().sum()), 0, msg)

    def test_matches_two_launches_and_aiter(self):
        for num_tokens in (1, self.max_tokens):
            for topk in (TOPK,):
                for block_size in (32,):
                    for num_local, rank in ((96, 3),):
                        for pad in (False, True):
                            for ties in (False, True):
                                self._assert_gate_sort_matches_two_launches(
                                    num_tokens,
                                    topk,
                                    block_size,
                                    num_local,
                                    rank,
                                    pad,
                                    ties,
                                )


@unittest.skipUnless(is_hip(), "requires HIP")
class TestRouterFp32(CustomTestCase):
    def setUp(self):
        from sglang.srt.models.deepseek_v2 import MoEGate
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        self.forward = MoEGate.forward

    def test_close_scores_and_mutable_graph(self):
        """BF16 output rounding must not collapse distinct expert scores."""
        # Exact BF16 operands produce 16 distinct scores near 1; rounding the
        # GEMM output to BF16 would collapse them before expert selection.
        weight = torch.zeros(384, 5120, device="cuda", dtype=torch.bfloat16)
        weight[:, 0] = 1
        weight[:16, 1] = torch.arange(16, device="cuda") / 4096
        gate = SimpleNamespace(
            weight=weight, is_deepseek_v4=True, tiny_router_gemm_max_tokens=0
        )
        for rows in (64, 512):
            with self.subTest(rows=rows):
                x = torch.zeros(rows, 5120, device="cuda", dtype=torch.bfloat16)
                x[:, :2] = 1
                self.forward(gate, x)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = self.forward(gate, x)
                for sign in (1, -1):
                    x[:, 1] = sign
                    graph.replay()
                    expected = (
                        1
                        + sign
                        * torch.arange(16, device="cuda", dtype=torch.float32)
                        / 4096
                    )
                    self.assertEqual(output.dtype, torch.float32)
                    torch.testing.assert_close(
                        output[:, :16], expected.expand(rows, -1), rtol=0, atol=0
                    )
                    self.assertEqual(torch.unique(output[0, :16]).numel(), 16)


D, TOPK, E = 5120, 6, 384


def _reference(x, shared, ids, mask, alpha):
    m = shared.shape[0]
    xs = x.view(m, TOPK, D).float()
    acc = torch.zeros_like(shared, dtype=torch.float32)
    for k in range(TOPK):
        v = xs[:, k]
        if mask is not None:
            v = torch.where((mask[ids[:, k]] != 0)[:, None], v, 0.0)
        acc = acc + v
    return (acc * alpha + shared.float()).to(shared.dtype)


def _inputs(m, seed, local_fraction=0.25):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m * TOPK, D, device="cuda", dtype=torch.bfloat16, generator=g)
    shared = torch.randn(m, D, device="cuda", dtype=torch.bfloat16, generator=g)
    ids = torch.randint(0, E, (m, TOPK), device="cuda", dtype=torch.int32, generator=g)
    mask = (torch.arange(E, device="cuda") < int(E * local_fraction)).to(torch.int32)
    return x, shared, ids, mask


@unittest.skipUnless(is_hip(), "the fused reduction is the ROCm path")
class TestMoeTopkReduceAdd(CustomTestCase):
    def setUp(self):
        from sglang.kernels.ops.moe.moe_reduce_add_hip import moe_topk_reduce_add

        self.reduce_add = moe_topk_reduce_add

    def test_matches_reference(self):
        for m in (1, 33):
            for alpha in (1.0, 2.5):
                x, shared, ids, mask = _inputs(m, m)
                for use_mask in (True, False):
                    out = torch.empty_like(shared)
                    self.reduce_add(
                        x,
                        shared,
                        out,
                        TOPK,
                        ids if use_mask else None,
                        mask if use_mask else None,
                        alpha=alpha,
                    )
                    ref = _reference(x, shared, ids, mask if use_mask else None, alpha)
                    self.assertTrue(torch.equal(out, ref), (m, alpha, use_mask))

    def test_repeatable_and_batch_invariant(self):
        x, shared, ids, mask = _inputs(300, 7)
        full = torch.empty_like(shared)
        self.reduce_add(x, shared, full, TOPK, ids, mask)
        for rows in ([0], list(range(0, 300, 7))):
            idx = torch.tensor(rows, device="cuda")
            sub = torch.empty(len(rows), D, device="cuda", dtype=torch.bfloat16)
            self.reduce_add(
                x.view(300, TOPK, D)[idx].reshape(-1, D).contiguous(),
                shared[idx].contiguous(),
                sub,
                TOPK,
                ids[idx].contiguous(),
                mask,
            )
            self.assertTrue(torch.equal(sub, full[idx]), rows)


if __name__ == "__main__":
    unittest.main()
