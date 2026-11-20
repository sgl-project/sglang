# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dspine_components import (
    DSpineCodeBridge,
    score_dspine_transitions,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


class TestDSpineFusedBridge(unittest.TestCase):
    def test_fused_eager_and_graph_replay(self):
        torch.manual_seed(42)
        config = SimpleNamespace(
            hidden_size=512,
            vocab_size=64,
            num_hidden_layers=2,
            block_size=4,
            dflash_config={
                "selected_prefix_terminal": False,
                "draft_block_causal": True,
                "selected_prefix_feature_writes": True,
                "selected_prefix_feature_write_mode": "independent",
                "selected_prefix_feature_layers": [1, 2],
                "selected_prefix_code_bridge": {
                    "code_size": 128,
                    "rank": 128,
                    "eps": 1e-6,
                },
            },
        )
        bridge = DSpineCodeBridge(config, params_dtype=torch.bfloat16).cuda()
        for name, tensor in (
            dict(bridge.named_parameters()) | dict(bridge.named_buffers())
        ).items():
            value = (
                torch.ones_like(tensor)
                if name.endswith("version")
                else torch.randn_like(tensor) * 0.05
            )
            if name == "out.weight":
                value.zero_()
            bridge.load_weight(name, value)
        bridge.prepare()
        hidden = torch.randn(8, 512, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn_like(hidden)
        ids = torch.randint(0, 64, (8,), device="cuda")
        norm = SimpleNamespace(
            variance_epsilon=1e-6,
            weight=torch.ones(512, dtype=torch.bfloat16, device="cuda"),
        )
        for layer in (0, 1):
            with self.subTest(layer=layer):
                actual = bridge(hidden, residual, ids, layer, norm)
                reference = bridge._forward_eager(hidden, residual, ids, layer, norm)
                for a, b in zip(actual, reference):
                    torch.testing.assert_close(
                        a.float(), b.float(), atol=0.025, rtol=0.025
                    )

        # Compare R-space scoring to vLLM's explicit FP32 refined states.
        candidates = torch.randint(0, 64, (2, 3, 4), device="cuda")
        head = torch.randn(64, 512, dtype=torch.bfloat16, device="cuda") * 0.05
        sender_weight = torch.cat(
            (bridge.predecessor_gate.weight, bridge.prev.weight)
        ).float()
        kwargs = dict(
            codebook=bridge.codebook,
            pred_gate=bridge.predecessor_gate.weight.float(),
            prev=bridge.prev.weight.float(),
            gate_bias=bridge.gate.bias.float(),
            write=bridge.feature_write_weights[-1].float(),
            norm=norm.weight.float(),
            lm_head_weight=head,
            norm_eps=norm.variance_epsilon,
            sender_weight=sender_weight,
            normed_write=norm.weight.float()[:, None]
            * bridge.feature_write_weights[-1].float(),
        )

        def scores():
            _, _, raw, rms, packed = bridge(hidden, residual, ids, 1, norm)
            return score_dspine_transitions(
                candidates,
                raw.view(2, 4, 512)[:, 1:],
                rms.view(2, 4)[:, 1:],
                packed[:, :128].view(2, 4, 128)[:, 1:],
                ids.view(2, 4)[:, 0],
                **kwargs,
            )

        _, _, raw, rms, packed = bridge(hidden, residual, ids, 1, norm)
        senders = torch.cat(
            (ids.view(2, 4)[:, :1, None].expand(2, 1, 4), candidates[:, :-1]), dim=1
        )
        features = bridge.codebook[senders].float() * (128**0.5)
        query = packed[:, :128].view(2, 4, 128)[:, 1:].float()
        message = torch.sigmoid(
            query[:, :, None] + features @ kwargs["pred_gate"].T + kwargs["gate_bias"]
        ) * (features @ kwargs["prev"].T)
        refined = raw.view(2, 4, 512)[:, 1:, None].float() + (
            rms.view(2, 4)[:, 1:, None, None] * (message @ kwargs["write"].T)
        )
        normalized = (
            refined
            * torch.rsqrt(
                refined.square().mean(-1, keepdim=True) + norm.variance_epsilon
            )
            * norm.weight.float()
        )
        reference = torch.einsum(
            "bsjh,bsch->bsjc", normalized, head[candidates].float()
        )
        torch.testing.assert_close(scores(), reference, atol=5e-3, rtol=5e-3)
        for _ in range(3):
            scores()
        torch.cuda.synchronize()
        score_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(score_graph):
            captured_scores = scores()
        hidden.add_(0.125)
        score_graph.replay()
        torch.testing.assert_close(captured_scores, scores(), atol=5e-3, rtol=5e-3)

        # Warm up compilation and BLAS before capture.
        for _ in range(3):
            bridge(hidden, residual, ids, 1, norm)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = bridge(hidden, residual, ids, 1, norm)
        for _ in range(2):
            hidden.add_(0.125)
            graph.replay()
            reference = bridge._forward_eager(hidden, residual, ids, 1, norm)
            for a, b in zip(captured, reference):
                torch.testing.assert_close(a.float(), b.float(), atol=0.025, rtol=0.025)


if __name__ == "__main__":
    unittest.main()
