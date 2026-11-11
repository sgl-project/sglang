"""Manual SM120/SM121 checks for the opt-in cuTile NVFP4 adapter.

Run this file in an installed SGLang GPU development environment. Set
SGLANG_TEST_MINIMAX_LAYER30_DIR to a directory containing config.json,
model.safetensors.index.json and the layer-30 checkpoint shard to enable
the optional real-weight case. No model server or downloads are started.
"""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ALPHA, BETA, LIMIT = 1.702, 1.0, 7.0
E2M1 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)


def _dequantize(packed, block_scale, global_scale):
    lut = torch.tensor(E2M1, device=packed.device)
    values = torch.stack((lut[(packed & 15).long()], lut[(packed >> 4).long()]), -1)
    values = values.reshape(packed.shape[0], -1)
    return values * block_scale.float().repeat_interleave(16, dim=-1) / global_scale


def _quantize_weight(weight):
    from flashinfer import fp4_quantize

    global_scale = (448 * 6 / weight.abs().amax()).float().reshape(1)
    packed, scale = fp4_quantize(weight, global_scale, 16, False, False)
    scale = scale.view(torch.float8_e4m3fn).reshape(weight.shape[0], -1)
    return packed.view(torch.uint8), scale, global_scale.squeeze(0)


def _activation_reference(values):
    """Use the kernel's global scale of one, not per-token global scales."""
    from flashinfer import fp4_quantize

    values = values.bfloat16().contiguous()
    global_scale = torch.ones(1, device=values.device)
    packed, scale = fp4_quantize(values, global_scale, 16, False, False)
    scale = scale.view(torch.float8_e4m3fn).reshape(values.shape[0], -1)
    return _dequantize(packed.view(torch.uint8), scale, global_scale)


def _layer(parameters, *, top_k):
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

    layer = torch.nn.Module()
    for name, value in parameters.items():
        layer.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
    experts, rows, packed_hidden = layer.w13_weight_packed.shape
    layer.num_experts = layer.num_local_experts = experts
    layer.hidden_size = packed_hidden * 2
    layer.intermediate_size_per_partition = rows // 2
    layer.params_dtype = torch.bfloat16
    layer.moe_ep_size = layer.moe_tp_size = 1
    layer.moe_ep_rank = layer.moe_tp_rank = 0
    layer.moe_runner_config = MoeRunnerConfig(
        num_experts=experts,
        num_local_experts=experts,
        hidden_size=layer.hidden_size,
        intermediate_size_per_partition=layer.intermediate_size_per_partition,
        top_k=top_k,
        num_fused_shared_experts=0,
        params_dtype=torch.bfloat16,
        gemm1_alpha=ALPHA,
        gemm1_beta=BETA,
        gemm1_clamp_limit=LIMIT,
        gate_up_interleaved=False,
        layer=layer,
    )
    return layer


def _synthetic_layer(*, experts=4, hidden_size=256, intermediate_size=256, top_k=2):
    w13 = (
        torch.randn(
            experts,
            2 * intermediate_size,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 4
    )
    w2 = (
        torch.randn(
            experts, hidden_size, intermediate_size, device="cuda", dtype=torch.bfloat16
        )
        / 8
    )
    q13, s13, g13 = zip(*(_quantize_weight(weight) for weight in w13))
    q2, s2, g2 = zip(*(_quantize_weight(weight) for weight in w2))
    return _layer(
        dict(
            w13_weight_packed=torch.stack(q13),
            w13_weight_scale=torch.stack(s13),
            w13_weight_global_scale=torch.stack(g13)[:, None].repeat(1, 2),
            w13_input_global_scale=torch.ones(experts, 2, device="cuda"),
            w2_weight_packed=torch.stack(q2),
            w2_weight_scale=torch.stack(s2),
            w2_weight_global_scale=torch.stack(g2),
            w2_input_global_scale=torch.ones(experts, device="cuda"),
        ),
        top_k=top_k,
    )


def _direct_flashinfer(layer, *, capacity):
    from flashinfer.fused_moe import (
        BackendOptions,
        CuTileNvfp4Config,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        MoELayer,
        MoEWeightPack,
        QuantConfig,
        QuantFormat,
        RoutingConfig,
        SwiGLU,
    )

    cfg = layer.moe_runner_config
    gate, up = layer.w13_weight_packed.chunk(2, dim=1)
    gate_scale, up_scale = layer.w13_weight_scale.chunk(2, dim=1)
    activation = SwiGLU(alpha=ALPHA, beta=BETA, limit=LIMIT)
    view = CuTileNvfp4Config.prepare_weights(
        w1_fp4=torch.cat((up, gate), dim=1),
        w1_block_scale=torch.cat((up_scale, gate_scale), dim=1),
        w1_global_scale=layer.w13_weight_global_scale[:, [1, 0]].contiguous(),
        w2_fp4=layer.w2_weight_packed,
        w2_block_scale=layer.w2_weight_scale,
        w2_global_scale=layer.w2_weight_global_scale,
        num_local_experts=cfg.num_experts,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size_per_partition,
        activation=activation,
        source_format="compressed_tensors",
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    weights = MoEWeightPack()
    weights.prepare_for("cutile_nvfp4", view)
    config = MoEConfig(
        routing=RoutingConfig(num_experts=cfg.num_experts, top_k=cfg.top_k),
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
        experts=ExpertConfig(intermediate_size=cfg.intermediate_size_per_partition),
        activation=activation,
        backend=BackendOptions(candidates=(CuTileNvfp4Config(),)),
        execution=ExecutionConfig(tune_max_num_tokens=capacity),
    )
    return MoELayer(config=config, device=layer.w13_weight_packed.device), weights


def _reference(layer, x, ids, weights):
    xq = _activation_reference(x)
    output = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(layer.num_experts):
        token, choice = (ids == expert).nonzero(as_tuple=True)
        if token.numel() == 0:
            continue
        qg, qu = layer.w13_weight_packed[expert].chunk(2, dim=0)
        sg, su = layer.w13_weight_scale[expert].chunk(2, dim=0)
        gg, gu = layer.w13_weight_global_scale[expert]
        gate = (xq[token] @ _dequantize(qg, sg, gg).T).clamp(max=LIMIT)
        up = (xq[token] @ _dequantize(qu, su, gu).T).clamp(-LIMIT, LIMIT)
        hidden = _activation_reference(gate * torch.sigmoid(ALPHA * gate) * (up + BETA))
        down = _dequantize(
            layer.w2_weight_packed[expert],
            layer.w2_weight_scale[expert],
            layer.w2_weight_global_scale[expert],
        )
        output.index_add_(0, token, (hidden @ down.T) * weights[token, choice, None])
    return output


def _dispatch(layer, tokens):
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    config = layer.moe_runner_config
    x = torch.randn(tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(tokens * config.top_k, device="cuda", dtype=torch.int32)
    ids = ids.reshape(tokens, config.top_k) % config.num_experts
    # Deliberately not normalized: an adapter must preserve the caller's weights.
    weights = (
        torch.linspace(0.25, 1.75, config.top_k, device="cuda", dtype=torch.float32)
        .expand(tokens, -1)
        .contiguous()
    )
    return StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_ids=ids, topk_weights=weights, router_logits=None
        ),
    )


class TestFlashInferCuTileNvfp4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("requires an SM120/SM121 GPU")
        if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
            raise unittest.SkipTest("cuTile NVFP4 is restricted to SM120/SM121")
        from sglang.srt.layers.moe.moe_runner import flashinfer_cutile

        cls.adapter = flashinfer_cutile

    def setUp(self):
        torch.manual_seed(17)
        self.capacity = self.enterContext(
            patch.object(self.adapter, "_get_max_num_tokens", return_value=512)
        )

    def _run_case(self, layer, tokens):
        from flashinfer.fused_moe import MoEActivationPack
        from flashinfer.fused_moe.core import RoutingInputMode

        dispatch = _dispatch(layer, tokens)
        x = dispatch.hidden_states
        ids = dispatch.topk_output.topk_ids
        weights = dispatch.topk_output.topk_weights
        reference = _reference(layer, x, ids, weights)
        direct, direct_weights = _direct_flashinfer(layer, capacity=512)
        prepared = self.adapter.prepare_flashinfer_cutile_nvfp4_weights(
            layer, layer.moe_runner_config
        )
        runner = self.adapter.FlashInferCuTileRunnerCore(layer.moe_runner_config)

        def run():
            return runner.run_from_dispatch(
                dispatch, prepared, layer.moe_runner_config
            ).hidden_states

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            expected = direct(
                MoEActivationPack(
                    hidden_states_q=x,
                    hidden_states_scale=None,
                    topk_ids=ids,
                    topk_weights=weights,
                    routing_input_mode=RoutingInputMode.PackedPrecomputed,
                ),
                direct_weights,
            ).clone()
            actual = run().clone()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            relative = (actual.float() - reference).norm() / reference.norm().clamp_min(
                1e-8
            )
            self.assertLess(relative.item(), 0.15)
            repeats = [run().clone() for _ in range(4)]
            self.assertTrue(
                all(
                    torch.equal(actual.view(torch.int16), value.view(torch.int16))
                    for value in repeats
                )
            )
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = run()
            graph.replay()
            self.assertTrue(
                torch.equal(captured.view(torch.int16), actual.view(torch.int16))
            )
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(20):
                graph.replay()
            end.record()
            end.synchronize()
            print(
                f"tokens={tokens} relative_error={relative.item():.6f} "
                f"graph_mean_ms={start.elapsed_time(end) / 20:.6f}"
            )
        torch.cuda.current_stream().wait_stream(stream)

    def test_synthetic_eager_graph_and_repeatability(self):
        for tokens in (1, 64, 129):
            with self.subTest(tokens=tokens):
                self._run_case(_synthetic_layer(), tokens)

    def test_scheme_loading_and_apply_use_cutile(self):
        from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
        from sglang.srt.layers.quantization import minimax_m3_nvfp4

        layer = _synthetic_layer()
        dispatch = _dispatch(layer, 8)
        reference_runner = self.adapter.FlashInferCuTileRunnerCore(
            layer.moe_runner_config
        )
        reference_weights = self.adapter.prepare_flashinfer_cutile_nvfp4_weights(
            layer, layer.moe_runner_config
        )
        expected = reference_runner.run_from_dispatch(
            dispatch, reference_weights, layer.moe_runner_config
        ).hidden_states.clone()
        with (
            patch.object(
                minimax_m3_nvfp4,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.FLASHINFER_CUTILE,
            ),
            patch.object(
                minimax_m3_nvfp4,
                "get_moe_a2a_backend",
                return_value=MoeA2ABackend.NONE,
            ),
        ):
            scheme = minimax_m3_nvfp4.MiniMaxM3Nvfp4MoE()
            scheme.create_moe_runner(layer, layer.moe_runner_config)
            scheme.process_weights_after_loading(layer)
            actual = scheme.apply_weights(layer, dispatch).hidden_states
        self.assertTrue(scheme.use_flashinfer_cutile)
        self.assertFalse(scheme.use_marlin)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_tp_weight_slices_match_unsharded_layer(self):
        """Check TP slice arithmetic on one GPU; distributed all-reduce is separate."""
        full = _synthetic_layer(intermediate_size=256)
        dispatch = _dispatch(full, 64)
        cfg = full.moe_runner_config
        full_runner = self.adapter.FlashInferCuTileRunnerCore(cfg)
        full_weights = self.adapter.prepare_flashinfer_cutile_nvfp4_weights(full, cfg)
        expected = full_runner.run_from_dispatch(
            dispatch, full_weights, cfg
        ).hidden_states.clone()
        partition_size = cfg.intermediate_size_per_partition // 2
        runners, results = [], []
        for rank in range(2):
            start, end = rank * partition_size, (rank + 1) * partition_size
            rows = torch.cat(
                (
                    torch.arange(start, end, device="cuda"),
                    torch.arange(
                        start + cfg.intermediate_size_per_partition,
                        end + cfg.intermediate_size_per_partition,
                        device="cuda",
                    ),
                )
            )
            params = {
                name: value.detach().clone() for name, value in full.named_parameters()
            }
            params["w13_weight_packed"] = full.w13_weight_packed[:, rows].contiguous()
            params["w13_weight_scale"] = (
                full.w13_weight_scale.float()[:, rows]
                .to(torch.float8_e4m3fn)
                .contiguous()
            )
            params["w2_weight_packed"] = full.w2_weight_packed[
                :, :, start // 2 : end // 2
            ].contiguous()
            params["w2_weight_scale"] = full.w2_weight_scale[
                :, :, start // 16 : end // 16
            ].contiguous()
            part = _layer(params, top_k=cfg.top_k)
            part.moe_tp_size, part.moe_tp_rank = 2, rank
            part_cfg = part.moe_runner_config
            runner = self.adapter.FlashInferCuTileRunnerCore(part_cfg)
            weights = self.adapter.prepare_flashinfer_cutile_nvfp4_weights(
                part, part_cfg
            )
            runners.append(runner)
            results.append(
                runner.run_from_dispatch(dispatch, weights, part_cfg)
                .hidden_states.float()
                .clone()
            )
        combined = torch.stack(results).sum(dim=0)
        relative = (combined - expected.float()).norm() / expected.float().norm()
        self.assertLess(relative.item(), 0.02)

    @unittest.skipUnless(
        os.environ.get("SGLANG_TEST_CUTILE_LARGE_BATCH") == "1",
        "set SGLANG_TEST_CUTILE_LARGE_BATCH=1 for workspace boundary measurements",
    )
    def test_workspace_bucket_boundary(self):
        # Match the issue's local TP2 geometry; this is a single-device layer test.
        layer = _synthetic_layer(
            experts=16, hidden_size=6144, intermediate_size=1536, top_k=4
        )
        cfg = layer.moe_runner_config
        prepared = self.adapter.prepare_flashinfer_cutile_nvfp4_weights(layer, cfg)
        for tokens in (8192, 8193):
            with self.subTest(tokens=tokens):
                self.capacity.return_value = tokens
                runner = self.adapter.FlashInferCuTileRunnerCore(cfg)
                dispatch = _dispatch(layer, tokens)
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                output = runner.run_from_dispatch(dispatch, prepared, cfg).hidden_states
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - baseline
                self.assertEqual(output.shape, (tokens, cfg.hidden_size))
                self.assertTrue(torch.isfinite(output).all().item())
                print(
                    f"tokens={tokens} workspace_and_temporary_peak_mib={peak / 2**20:.2f}"
                )
                del output, dispatch, runner
                torch.cuda.empty_cache()

    @unittest.skipUnless(
        os.environ.get("SGLANG_TEST_MINIMAX_LAYER30_DIR"),
        "set SGLANG_TEST_MINIMAX_LAYER30_DIR to test checkpoint weights",
    )
    def test_real_layer30_checkpoint(self):
        from safetensors import safe_open

        directory = Path(os.environ["SGLANG_TEST_MINIMAX_LAYER30_DIR"])
        config = json.loads((directory / "config.json").read_text())
        config = config.get("text_config", config)
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        weight_map = index["weight_map"]
        suffix = "layers.30.block_sparse_moe.experts.0.w1.weight_packed"
        prefix = next(name for name in weight_map if suffix in name).rsplit(
            ".0.w1.", 1
        )[0]
        tensors = {}
        for expert in range(32):
            for projection in ("w1", "w2", "w3"):
                for field in ("weight_packed", "weight_scale", "weight_global_scale"):
                    name = f"{prefix}.{expert}.{projection}.{field}"
                    with safe_open(
                        directory / weight_map[name], framework="pt", device="cpu"
                    ) as handle:
                        tensors[expert, projection, field] = handle.get_tensor(name).to(
                            "cuda"
                        )

        def stack(projection, field):
            return torch.stack([tensors[e, projection, field] for e in range(32)])

        parameters = dict(
            w13_weight_packed=torch.cat(
                (stack("w1", "weight_packed"), stack("w3", "weight_packed")), dim=1
            ).view(torch.uint8),
            w13_weight_scale=torch.cat(
                (stack("w1", "weight_scale"), stack("w3", "weight_scale")), dim=1
            ),
            w13_weight_global_scale=torch.stack(
                (
                    stack("w1", "weight_global_scale").reshape(32),
                    stack("w3", "weight_global_scale").reshape(32),
                ),
                dim=1,
            ).float(),
            w13_input_global_scale=torch.ones(32, 2, device="cuda"),
            w2_weight_packed=stack("w2", "weight_packed").view(torch.uint8),
            w2_weight_scale=stack("w2", "weight_scale"),
            w2_weight_global_scale=stack("w2", "weight_global_scale")
            .reshape(32)
            .float(),
            w2_input_global_scale=torch.ones(32, device="cuda"),
        )
        layer = _layer(parameters, top_k=config["num_experts_per_tok"])
        self._run_case(layer, 64)


if __name__ == "__main__":
    unittest.main()
