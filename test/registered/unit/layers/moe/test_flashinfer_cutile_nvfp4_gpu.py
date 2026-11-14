"""GPU checks for the FlashInfer cuTile NVFP4 W4A4 MoE runner (SM120/SM121 only).

The runner is compared with float32 references that mirror its arithmetic, not
with a W4A16 kernel: quantizing activations to NVFP4 changes the result by design.
"""

import unittest

import torch

from sglang.srt.layers.moe.moe_runner import flashinfer_cutile as cutile
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# No CI runner is SM12x today, so this file skips there; run it by hand on an
# RTX 50xx / RTX PRO Blackwell / DGX Spark machine.
register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="1-gpu-large")

ALPHA, BETA, LIMIT = 1.702, 1.0, 7.0
# MiniMax-M3.1 expert shape with fewer experts, to fit a 16 GB card.
NUM_EXPERTS, TOP_K, HIDDEN, INTERMEDIATE = 16, 4, 6144, 3072
TOKEN_COUNTS = (1, 64, 412, 4096, 8192, 8193)
E2M1 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)


def _random_layer(device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)

    def packed(rows, cols):
        return torch.randint(
            0,
            256,
            (NUM_EXPERTS, rows, cols // 2),
            generator=g,
            dtype=torch.uint8,
            device=device,
        )

    def block_scale(rows, cols):
        scale = torch.rand(NUM_EXPERTS, rows, cols // 16, generator=g, device=device)
        return (scale + 0.5).to(torch.float8_e4m3fn)

    def encode_scale(*shape):
        return torch.rand(*shape, generator=g, device=device) * 200 + 300

    return dict(
        w13_packed=packed(2 * INTERMEDIATE, HIDDEN),
        w13_block_scale=block_scale(2 * INTERMEDIATE, HIDDEN),
        # Gate and up scales differ on purpose: a runner that kept only one would
        # miss the reference.
        w13_weight_global_scale=encode_scale(NUM_EXPERTS, 2),
        w13_input_global_scale=torch.ones(NUM_EXPERTS, 2, device=device),
        w2_packed=packed(HIDDEN, INTERMEDIATE),
        w2_block_scale=block_scale(HIDDEN, INTERMEDIATE),
        w2_weight_global_scale=encode_scale(NUM_EXPERTS),
        w2_input_global_scale=torch.ones(NUM_EXPERTS, device=device),
    )


def _dequantize(packed, block_scale, encode_scale):
    table = torch.tensor(E2M1, device=packed.device)
    values = torch.stack(
        (table[(packed & 15).long()], table[(packed >> 4).long()]), dim=-1
    ).reshape(packed.shape[0], -1)
    return values * (block_scale.float() / encode_scale).repeat_interleave(16, dim=1)


def _nvfp4(v):
    """Dynamic NVFP4 of each row: E4M3 scale of amax / 6 per 16 elements, unit global."""
    thresholds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=v.device)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=v.device)
    groups = v.reshape(v.shape[0], -1, 16)
    scale = (groups.abs().amax(-1) / 6.0).to(torch.float8_e4m3fn).float()
    q = groups / scale.clamp(min=2.0**-9).unsqueeze(-1)
    code = (q.abs().unsqueeze(-1) > thresholds).sum(-1)
    sign = torch.where(q < 0, -1.0, 1.0)
    return (levels[code] * sign * scale.unsqueeze(-1)).reshape(v.shape)


def _reference(layer, x, ids, weights, *, nvfp4_activations, swap=False, clamp=True):
    gate_scale, up_scale = layer["w13_weight_global_scale"].unbind(1)
    xf = _nvfp4(x.float()) if nvfp4_activations else x.float()
    out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32, device=x.device)
    for e in range(NUM_EXPERTS):
        rows, slot = (ids == e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        gate_w = _dequantize(
            layer["w13_packed"][e, :INTERMEDIATE],
            layer["w13_block_scale"][e, :INTERMEDIATE],
            gate_scale[e],
        )
        up_w = _dequantize(
            layer["w13_packed"][e, INTERMEDIATE:],
            layer["w13_block_scale"][e, INTERMEDIATE:],
            up_scale[e],
        )
        down_w = _dequantize(
            layer["w2_packed"][e],
            layer["w2_block_scale"][e],
            layer["w2_weight_global_scale"][e],
        )
        if swap:
            gate_w, up_w = up_w, gate_w
        gate, up = xf[rows] @ gate_w.T, xf[rows] @ up_w.T
        if clamp:
            gate, up = gate.clamp(max=LIMIT), up.clamp(-LIMIT, LIMIT)
        act = (gate * torch.sigmoid(ALPHA * gate) * (up + BETA)).bfloat16().float()
        if nvfp4_activations:
            act = _nvfp4(act)
        out.index_add_(0, rows, weights[rows, slot].unsqueeze(1) * (act @ down_w.T))
    return out


def _rel(y, ref):
    return float((y.float() - ref).norm() / ref.norm())


def _same_bits(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


class TestFlashInferCutileNvfp4GPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        reason = cutile.cutile_nvfp4_platform_unsupported_reason()
        if reason is not None:
            raise unittest.SkipTest(f"cuTile NVFP4 MoE is unavailable: {reason}")
        cls.device = torch.device("cuda")
        cls.layer = _random_layer(cls.device)
        cls.quant_info = cutile.build_cutile_nvfp4_quant_info(
            **cls.layer,
            top_k=TOP_K,
            swiglu_alpha=ALPHA,
            swiglu_beta=BETA,
            swiglu_limit=LIMIT,
        )

    def _inputs(self, num_tokens, *, seed=0, skewed=False, std=0.5):
        g = torch.Generator().manual_seed(seed * 100003 + num_tokens)
        x = (torch.randn(num_tokens, HIDDEN, generator=g) * std).bfloat16()
        if skewed:
            # Every token routes to the same four experts: the other twelve get
            # no rows and four get all of them.
            ids = torch.arange(TOP_K).repeat(num_tokens, 1)
        else:
            ids = torch.stack(
                [
                    torch.randperm(NUM_EXPERTS, generator=g)[:TOP_K]
                    for _ in range(num_tokens)
                ]
            )
        weights = torch.rand(num_tokens, TOP_K, generator=g)
        return x.to(self.device), ids.int().to(self.device), weights.to(self.device)

    def _run(self, x, ids, weights, **config):
        dispatch = StandardDispatchOutput(
            hidden_states=x,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=weights, topk_ids=ids, router_logits=None
            ),
        )
        runner_config = MoeRunnerConfig(
            num_experts=NUM_EXPERTS, num_local_experts=NUM_EXPERTS, **config
        )
        return cutile.fused_experts_none_to_flashinfer_cutile(
            dispatch, self.quant_info, runner_config
        ).hidden_states

    def test_matches_the_nvfp4_activation_reference(self):
        for skewed in (False, True):
            for seed in (0, 1):
                x, ids, w = self._inputs(412, seed=seed, skewed=skewed)
                out = self._run(x, ids, w)
                self.assertEqual(out.dtype, torch.bfloat16)
                self.assertTrue(bool(torch.isfinite(out).all()))
                ref_w4a4 = _reference(self.layer, x, ids, w, nvfp4_activations=True)
                ref_bf16 = _reference(self.layer, x, ids, w, nvfp4_activations=False)
                err = _rel(out, ref_w4a4)
                # The issue measured 0.023 on a real layer; the reference rounds
                # in fp32 while the kernel accumulates on FP4 tensor cores.
                self.assertLess(err, 0.06, (skewed, seed))
                # Closer to its own arithmetic than to the bf16-activation one.
                self.assertLess(err, _rel(out, ref_bf16))
                swapped = _reference(
                    self.layer, x, ids, w, nvfp4_activations=True, swap=True
                )
                self.assertGreater(_rel(out, swapped), 5 * err, (skewed, seed))

    def test_swiglu_clamp_is_applied(self):
        # At the usual input scale no pre-activation reaches the limit and a
        # dropped clamp is invisible; this input saturates a share of them.
        x, ids, w = self._inputs(412, std=32.0)
        out = self._run(x, ids, w)
        clamped = _reference(self.layer, x, ids, w, nvfp4_activations=True)
        unclamped = _reference(
            self.layer, x, ids, w, nvfp4_activations=True, clamp=False
        )
        self.assertGreater(_rel(unclamped, clamped), 0.2, "input does not saturate")
        self.assertLess(_rel(out, clamped), 0.1)
        self.assertGreater(_rel(out, unclamped), 0.2)

    def test_repeated_calls_are_bit_identical(self):
        for num_tokens in TOKEN_COUNTS:
            x, ids, w = self._inputs(num_tokens)
            first = self._run(x, ids, w).clone()
            for _ in range(19):
                self.assertTrue(_same_bits(first, self._run(x, ids, w)), num_tokens)

    def test_cuda_graph_replay_matches_eager(self):
        for num_tokens in (1, 64, 412, 4096):
            x, ids, w = self._inputs(num_tokens)
            eager = self._run(x, ids, w).clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = self._run(x, ids, w)
            for _ in range(4):
                graph.replay()
                self.assertTrue(_same_bits(eager, captured), num_tokens)
            # New activations and routing through the captured buffers.
            x2, ids2, w2 = self._inputs(num_tokens, seed=7)
            expected = self._run(x2, ids2, w2).clone()
            x.copy_(x2)
            ids.copy_(ids2)
            w.copy_(w2)
            graph.replay()
            self.assertTrue(_same_bits(expected, captured), num_tokens)
            del graph

    def test_token_chunks_equal_separate_calls(self):
        bound = cutile.CUTILE_NVFP4_MAX_TOKENS_PER_CALL
        x, ids, w = self._inputs(bound + 1)
        whole = self._run(x, ids, w)
        head = self._run(x[:bound].contiguous(), ids[:bound], w[:bound])
        tail = self._run(x[bound:].contiguous(), ids[bound:], w[bound:])
        self.assertTrue(_same_bits(whole[:bound], head))
        self.assertTrue(_same_bits(whole[bound:], tail))

    def test_workspace_is_reused_and_bounded_past_the_bucket_boundary(self):
        bound = cutile.CUTILE_NVFP4_MAX_TOKENS_PER_CALL
        peaks = {}
        for num_tokens in (bound, bound + 1):
            x, ids, w = self._inputs(num_tokens)
            self._run(x, ids, w)
            torch.cuda.synchronize()
            held = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                self._run(x, ids, w)
            torch.cuda.synchronize()
            # Nothing new is retained by repeating a warmed call.
            self.assertEqual(torch.cuda.memory_allocated(), held, num_tokens)
            peaks[num_tokens] = torch.cuda.max_memory_allocated() - held
        # One extra token must not pay for the next power-of-two workspace: only
        # for its own output and the concatenated copy.
        output_bytes = (bound + 1) * HIDDEN * 2
        self.assertLess(peaks[bound + 1], peaks[bound] + 3 * output_bytes)

    def test_padded_rows_contribute_nothing(self):
        """CUDA-graph padding rows carry expert id -1: they must not read out of
        range or change the real rows."""
        x, ids, w = self._inputs(64)
        real = self._run(x[:48].contiguous(), ids[:48], w[:48]).clone()
        padded_ids = ids.clone()
        padded_ids[48:] = -1
        out = self._run(x, padded_ids, w)
        self.assertTrue(bool(torch.isfinite(out[:48]).all()))
        # A different batch can pick another tactic, so compare within tolerance.
        self.assertLess(_rel(out[:48], real.float()), 1e-2)
        self.assertTrue(bool((out[48:] == 0).all()))
        # One unrouted slot in a real row drops only that expert's term.
        partial_ids, partial_w = ids.clone(), w.clone()
        partial_ids[:, -1] = -1
        partial_w[:, -1] = 0.0
        ref_ids = ids.clone()
        ref_ids[:, -1] = 0
        expected = self._run(x, ref_ids, partial_w)
        self.assertTrue(_same_bits(self._run(x, partial_ids, w), expected))

    def test_routed_scaling_factor_is_applied_once(self):
        x, ids, w = self._inputs(64)
        base = self._run(x, ids, w).float()
        scaled = self._run(x, ids, w, routed_scaling_factor=2.0)
        self.assertTrue(_same_bits(scaled, (base * 2.0).bfloat16()))


if __name__ == "__main__":
    unittest.main()
