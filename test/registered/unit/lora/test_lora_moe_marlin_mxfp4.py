"""MXFP4 Marlin LoRA core and hooks against an independent FP32 reference."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.quant_ref_utils import break_fp4_bytes
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b", runner_config="4-gpu-b200")

E, HIDDEN, INTER, TOPK, RANK, TOKENS = 8, 1024, 512, 4, 16, 48
# Pool slot of each token: adapters in slots 1 and 2, slot 0 holds none (rank 0).
MIXED_SLOTS = [1] * 12 + [0] * 8 + [2] * 16 + [1] * 12


def _checkpoint_experts(gen):
    """Packed E2M1 experts and E8M0 scales in checkpoint layout."""

    def mxfp4(rows, cols):
        packed = torch.randint(
            0,
            256,
            (E, rows, cols // 2),
            dtype=torch.uint8,
            device="cuda",
            generator=gen,
        )
        scales = torch.randint(
            121,
            124,
            (E, rows, cols // 32),
            dtype=torch.uint8,
            device="cuda",
            generator=gen,
        )
        return packed, scales

    return (*mxfp4(2 * INTER, HIDDEN), *mxfp4(HIDDEN, INTER))


def _dequant(packed, scales):
    experts, rows, half = packed.shape
    values = break_fp4_bytes(packed.reshape(-1, half)).view(experts, rows, 2 * half)
    return values * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=-1)


def _adapter(gen, *, zero_b=False):
    """Per-expert (gate/up A, gate/up B, down A, down B), with pre-scaled B."""

    def randn(*shape, std):
        return torch.randn(*shape, device="cuda", generator=gen) * std

    b_std = 0.0 if zero_b else 0.1
    return tuple(
        t.to(torch.bfloat16)
        for t in (
            randn(E, 2 * RANK, HIDDEN, std=HIDDEN**-0.5),
            randn(E, 2 * INTER, RANK, std=b_std),
            randn(E, RANK, INTER, std=INTER**-0.5),
            randn(E, HIDDEN, RANK, std=b_std),
        )
    )


def _routing(gen):
    logits = torch.randn(TOKENS, E, device="cuda", generator=gen)
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), TOPK, dim=-1)
    return weights / weights.sum(dim=-1, keepdim=True), ids.to(torch.int32)


class _Experts:
    """Prepare checkpoint-format experts with the production Marlin repack."""

    def __init__(self, w13, w13_s, w2, w2_s, *, expert_map=None):
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

        num_experts, inter = w13.shape[0], w13.shape[1] // 2
        self.config = MoeRunnerConfig(
            num_experts=E,
            num_local_experts=num_experts,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=inter,
            top_k=TOPK,
            params_dtype=torch.bfloat16,
            inplace=False,
        )
        layer = torch.nn.Module()
        layer.hidden_size = HIDDEN
        layer.intermediate_size_per_partition = inter
        layer.moe_runner_config = self.config
        layer.dispatcher = SimpleNamespace(local_expert_mapping=None, num_experts=E)
        # For Marlin the FP8 method only views the packed bytes as int8.
        fp8_method = SimpleNamespace(process_weights_after_loading=lambda _: None)
        method = Mxfp4MarlinMoEMethod(fp8_method, "experts")
        method.runner = SimpleNamespace(config=self.config)
        with torch.device("cuda"):
            method.create_weights(layer, num_experts, HIDDEN, inter, torch.bfloat16)
        layer.w13_weight.data.copy_(w13.view(torch.int8))
        layer.w2_weight.data.copy_(w2.view(torch.int8))
        layer.w13_weight_scale_inv.data.copy_(w13_s.view(torch.float8_e8m0fnu))
        layer.w2_weight_scale_inv.data.copy_(w2_s.view(torch.float8_e8m0fnu))
        method.process_weights_after_loading(layer)
        # FusedMoEWithLoRA reads the quant info when it wraps the layer, before the EP
        # map exists, and sets the map every forward; the stock path reads both at apply.
        self.lora_quant_info = method.get_marlin_quant_info(layer)
        self.lora_quant_info.expert_map = expert_map
        layer.dispatcher.local_expert_mapping = expert_map
        self.base_quant_info = method.get_marlin_quant_info(layer)


def _dispatch(x, topk_weights, topk_ids):
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    # The stock path checks only the router logits' token count.
    router_logits = torch.zeros(x.shape[0], E, device=x.device)
    return StandardDispatchOutput(
        x, None, StandardTopKOutput(topk_weights, topk_ids, router_logits)
    )


def _run_stock(experts, x, topk_weights, topk_ids):
    """Run the stock Marlin fused function."""
    from sglang.srt.layers.moe.moe_runner.marlin import fused_experts_none_to_marlin

    return fused_experts_none_to_marlin(
        _dispatch(x, topk_weights, topk_ids), experts.base_quant_info, experts.config
    ).hidden_states


def _run_lora(experts, x, topk_weights, topk_ids, lora_info):
    """Run production LoRA hooks and the Marlin core."""
    from sglang.srt.lora.lora_moe_runner_marlin import MarlinLoraRunnerCore
    from sglang.srt.lora.lora_moe_runners import build_lora_hooks

    hooks = build_lora_hooks(x, lora_info, topk_ids)
    return (
        MarlinLoraRunnerCore(experts.config)
        .run_from_dispatch(
            _dispatch(x, topk_weights, topk_ids),
            experts.lora_quant_info,
            experts.config,
            hooks=hooks,
        )
        .hidden_states
    )


def _segments(token_slots):
    """Segment consecutive tokens by adapter slot."""
    starts = [0] + [
        i for i in range(1, len(token_slots)) if token_slots[i] != token_slots[i - 1]
    ]
    ends = starts[1:] + [len(token_slots)]
    seg_indptr = torch.tensor(starts + ends[-1:], dtype=torch.int32, device="cuda")
    req_to_lora = torch.tensor(
        [token_slots[i] for i in starts], dtype=torch.int32, device="cuda"
    )
    return seg_indptr, req_to_lora, max(e - s for s, e in zip(starts, ends))


def _lora_info(adapters, token_slots, *, virtual_experts=True):
    """Build pool-style LoRA buffers and batch metadata."""
    from sglang.srt.lora.backend.base_backend import _compute_moe_lora_info
    from sglang.srt.lora.lora_moe_runners import LoRAInfo

    buffers = [
        torch.stack([torch.zeros_like(adapters[0][i])] + [a[i] for a in adapters])
        for i in range(4)
    ]
    lora_ranks = torch.tensor(
        [0] + [RANK] * len(adapters), dtype=torch.int32, device="cuda"
    )
    seg_indptr, req_to_lora, max_len = _segments(token_slots)
    adapter_enabled, token_lora_mapping = _compute_moe_lora_info(
        len(token_slots), seg_indptr, lora_ranks, req_to_lora, None, None, max_len
    )
    return LoRAInfo(
        gate_up_lora_a_weights=buffers[0],
        gate_up_lora_b_weights=buffers[1],
        down_lora_a_weights=buffers[2],
        down_lora_b_weights=buffers[3],
        seg_indptr=seg_indptr,
        req_to_lora=req_to_lora,
        lora_ranks=lora_ranks,
        adapter_enabled=adapter_enabled,
        token_lora_mapping=token_lora_mapping,
        max_lora_rank=RANK,
        num_experts=E,
        has_active_lora=any(slot > 0 for slot in token_slots),
        lora_use_virtual_experts=virtual_experts,
    )


def _reference(x, topk_weights, topk_ids, w13, w2, token_slots, adapters):
    """FP32 oracle; negative expert IDs belong to other EP ranks."""
    xf = x.float()
    slots = torch.tensor(token_slots, device="cuda")
    out = torch.zeros_like(xf)
    inter = w13.shape[1] // 2
    for e in range(w13.shape[0]):
        token, k = (topk_ids == e).nonzero(as_tuple=True)
        h = xf[token]
        gate_up = h @ w13[e].T
        for slot, (a_gu, b_gu, _, _) in enumerate(adapters, start=1):
            rows = slots[token] == slot
            a, b = a_gu[e].float(), b_gu[e].float()
            gate_up[rows, :inter] += h[rows] @ a[:RANK].T @ b[:inter].T
            gate_up[rows, inter:] += h[rows] @ a[RANK:].T @ b[inter:].T
        act = torch.nn.functional.silu(gate_up[:, :inter]) * gate_up[:, inter:]
        down = act @ w2[e].T
        for slot, (_, _, a_d, b_d) in enumerate(adapters, start=1):
            rows = slots[token] == slot
            down[rows] += act[rows] @ a_d[e].float().T @ b_d[e].float().T
        out.index_add_(0, token, down * topk_weights[token, k, None])
    return out


def _rel_err(actual, expected):
    return ((actual.float() - expected).norm() / expected.norm()).item()


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9,
    "MXFP4 Marlin needs SM90+",
)
class TestMarlinMxfp4MoELoRA(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        gen = torch.Generator(device="cuda").manual_seed(0)
        cls.ckpt = _checkpoint_experts(gen)
        cls.w13, cls.w2 = _dequant(*cls.ckpt[:2]), _dequant(*cls.ckpt[2:])
        cls.experts = _Experts(*cls.ckpt)
        cls.x = torch.randn(TOKENS, HIDDEN, device="cuda", generator=gen).bfloat16()
        cls.topk_weights, cls.topk_ids = _routing(gen)
        cls.adapters = [_adapter(gen), _adapter(gen)]
        cls.zero_adapter = _adapter(gen, zero_b=True)
        cls.routing = (cls.x, cls.topk_weights, cls.topk_ids)
        cls.stock = _run_stock(cls.experts, *cls.routing)
        cls.base_ref = _reference(*cls.routing, cls.w13, cls.w2, [0] * TOKENS, [])

    def test_base_matches_the_stock_marlin_path_and_the_reference(self):
        self.assertLess(_rel_err(self.stock, self.base_ref), 2e-2)
        for virtual_experts in (True, False):
            with self.subTest(virtual_experts=virtual_experts):
                info = _lora_info(
                    self.adapters, [0] * TOKENS, virtual_experts=virtual_experts
                )
                out = _run_lora(self.experts, *self.routing, info)
                torch.testing.assert_close(out, self.stock, atol=0, rtol=0)

    def test_a_zero_adapter_leaves_the_base_output(self):
        for virtual_experts in (True, False):
            with self.subTest(virtual_experts=virtual_experts):
                info = _lora_info(
                    [self.zero_adapter], [1] * TOKENS, virtual_experts=virtual_experts
                )
                out = _run_lora(self.experts, *self.routing, info)
                torch.testing.assert_close(out, self.stock, atol=0, rtol=0)

    def test_mixed_adapters_match_the_reference(self):
        ref = _reference(*self.routing, self.w13, self.w2, MIXED_SLOTS, self.adapters)
        adapted = torch.tensor(MIXED_SLOTS, device="cuda") > 0
        for virtual_experts in (True, False):
            with self.subTest(virtual_experts=virtual_experts):
                info = _lora_info(
                    self.adapters, MIXED_SLOTS, virtual_experts=virtual_experts
                )
                out = _run_lora(self.experts, *self.routing, info)
                self.assertLess(_rel_err(out, ref), 2e-2)
                # The adapter's own effect, not just the base it rides on.
                self.assertLess(
                    _rel_err(
                        out[adapted] - self.stock[adapted],
                        ref[adapted] - self.base_ref[adapted],
                    ),
                    5e-2,
                )
                torch.testing.assert_close(
                    out[~adapted], self.stock[~adapted], atol=0, rtol=0
                )

    def test_expert_parallel_ranks_sum_to_the_whole_layer(self):
        ref = _reference(*self.routing, self.w13, self.w2, MIXED_SLOTS, self.adapters)
        ep_size = 2
        local_experts = E // ep_size
        total = torch.zeros_like(ref)
        for rank in range(ep_size):
            with self.subTest(ep_rank=rank):
                local = slice(rank * local_experts, (rank + 1) * local_experts)
                expert_map = torch.full((E,), -1, dtype=torch.int32, device="cuda")
                expert_map[local] = torch.arange(
                    local_experts, dtype=torch.int32, device="cuda"
                )
                experts = _Experts(
                    *(t[local] for t in self.ckpt), expert_map=expert_map
                )
                # The standard dispatcher's localization; -1 marks other ranks' experts.
                local_ids = expert_map[self.topk_ids.long()]
                adapters = [tuple(t[local] for t in a) for a in self.adapters]
                info = _lora_info(adapters, MIXED_SLOTS)
                out = _run_lora(experts, self.x, self.topk_weights, local_ids, info)
                rank_ref = _reference(
                    self.x,
                    self.topk_weights,
                    local_ids,
                    self.w13[local],
                    self.w2[local],
                    MIXED_SLOTS,
                    adapters,
                )
                self.assertLess(_rel_err(out, rank_ref), 2e-2)
                total += out.float()
        self.assertLess(_rel_err(total, ref), 2e-2)

    def test_moe_tensor_parallel_ranks_sum_to_the_whole_layer(self):
        from sglang.srt.lora.layers import FusedMoEWithLoRA

        ref = _reference(*self.routing, self.w13, self.w2, MIXED_SLOTS, self.adapters)
        tp_size = 2
        part = INTER // tp_size
        # The wrapper's slicing reads only the TP shard geometry.
        slicer = FusedMoEWithLoRA.__new__(FusedMoEWithLoRA)
        torch.nn.Module.__init__(slicer)
        slicer.tp_size = tp_size
        slicer.intermediate_size_per_partition = part
        slicer._uses_interleaved_gate_up = False
        slicer.base_layer = SimpleNamespace(
            moe_runner_config=SimpleNamespace(is_gated=True)
        )
        w13, w13_s, w2, w2_s = self.ckpt
        total = torch.zeros_like(ref)
        for rank in range(tp_size):
            with self.subTest(tp_rank=rank):
                gate = slice(rank * part, (rank + 1) * part)
                up = slice(INTER + rank * part, INTER + (rank + 1) * part)
                experts = _Experts(
                    torch.cat([w13[:, gate], w13[:, up]], dim=1),
                    torch.cat([w13_s[:, gate], w13_s[:, up]], dim=1),
                    w2[:, :, rank * part // 2 : (rank + 1) * part // 2],
                    w2_s[:, :, rank * part // 32 : (rank + 1) * part // 32],
                )
                adapters = [
                    (
                        a_gu,
                        slicer.slice_moe_lora_b_weights(b_gu, rank, "gate_up_proj_moe"),
                        slicer.slice_moe_lora_a_weights(a_d, rank, "down_proj_moe"),
                        b_d,
                    )
                    for a_gu, b_gu, a_d, b_d in self.adapters
                ]
                info = _lora_info(adapters, MIXED_SLOTS)
                total += _run_lora(experts, *self.routing, info).float()
        self.assertLess(_rel_err(total, ref), 2e-2)

    def test_cuda_graph_replay_matches_eager(self):
        from sglang.srt.lora.backend.base_backend import _compute_moe_lora_info
        from sglang.srt.model_executor.runner_utils import capture_mode

        gen = torch.Generator(device="cuda").manual_seed(1)
        other_x = torch.randn(TOKENS, HIDDEN, device="cuda", generator=gen).bfloat16()
        # The captured batch fixes the segment count; both batches have four.
        batches = [
            (*self.routing, MIXED_SLOTS),
            (other_x, *_routing(gen), [2] * 10 + [1] * 20 + [0] * 8 + [2] * 10),
        ]
        static_x, static_weights, static_ids = (t.clone() for t in self.routing)
        info = _lora_info(self.adapters, MIXED_SLOTS)
        # Compile the Triton kernels before capture.
        _run_lora(self.experts, static_x, static_weights, static_ids, info)
        graph = torch.cuda.CUDAGraph()
        with (
            mock.patch.object(capture_mode, "is_capture_mode", True),
            torch.cuda.graph(graph),
        ):
            static_out = _run_lora(
                self.experts, static_x, static_weights, static_ids, info
            )
        for x, weights, ids, slots in batches:
            with self.subTest(slots=slots[:: TOKENS // 4]):
                static_x.copy_(x)
                static_weights.copy_(weights)
                static_ids.copy_(ids)
                # Refresh the batch's adapter mapping in place, as graph replay does.
                seg_indptr, req_to_lora, max_len = _segments(slots)
                info.seg_indptr.copy_(seg_indptr)
                info.req_to_lora.copy_(req_to_lora)
                _compute_moe_lora_info(
                    TOKENS,
                    info.seg_indptr,
                    info.lora_ranks,
                    info.req_to_lora,
                    info.adapter_enabled,
                    info.token_lora_mapping,
                    max_len,
                )
                graph.replay()
                eager = _run_lora(
                    self.experts, x, weights, ids, _lora_info(self.adapters, slots)
                )
                # Allow rounding differences from atomic split-K accumulation.
                # The second replay batch changes inputs, routing, and adapter mapping.
                self.assertLess(_rel_err(static_out, eager.float()), 1e-3)


if __name__ == "__main__":
    unittest.main()
