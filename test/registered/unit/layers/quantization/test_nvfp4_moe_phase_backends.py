"""Mixed NVFP4 MoE runners match independently prepared per-phase controls."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_context, get_exec, get_flags, get_parallel
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.quant_ref_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    quantize_nvfp4_shard,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="4-gpu-b200")

E, H, INTERMEDIATE, TOPK, M = 8, 1024, 1024, 2, 32
PREFILL = "flashinfer_trtllm"
DECODE = "flashinfer_cutedsl"


@contextmanager
def _runtime(backend, w4a16, *, mixed=False, per_token=False):
    with (
        envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override(w4a16),
        envs.SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION.override(per_token),
        get_context().override_server_args(model_path="dummy"),
        get_exec().moe.override(
            moe_prefill_runner_backend=PREFILL if mixed else None,
            moe_decode_runner_backend=DECODE if mixed else None,
        ),
        get_flags().moe.override(
            runner_backend=MoeRunnerBackend(backend),
            a2a_backend=MoeA2ABackend.NONE,
        ),
        get_parallel().override(
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
            tp_size=1,
            tp_rank=0,
        ),
    ):
        yield


def _forward(layer, x, topk, mode):
    batch = SimpleNamespace(
        forward_mode=mode,
        global_forward_mode=None,
        global_moe_forward_phase=None,
        moe_num_token_non_padded=lambda: None,
    )
    with forward_context(ForwardContext(attn_backend=None), forward_batch=batch):
        out = layer.forward(x, topk)
    if isinstance(out, torch.Tensor):
        return out
    return out[0] if isinstance(out, tuple) else out.hidden_states


@unittest.skipIf(get_device_sm() < 100, "NVFP4 MoE backends require SM100+")
class TestNvFp4MoePhaseBackends(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist(master_port=29639)
        torch.set_default_device("cuda")

    @staticmethod
    def _make_layer():
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        torch.manual_seed(7)
        layer = FusedMoE(
            num_experts=E,
            hidden_size=H,
            intermediate_size=INTERMEDIATE,
            layer_id=0,
            top_k=TOPK,
            params_dtype=torch.bfloat16,
            quant_config=ModelOptFp4Config(
                is_checkpoint_nvfp4_serialized=True, group_size=16
            ),
            gate_up_interleaved=False,
        ).cuda()
        act_scale = 1.0 / (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)
        for expert in range(E):
            for shard in ("w1", "w3", "w2"):
                rows, cols = (
                    (INTERMEDIATE, H) if shard in ("w1", "w3") else (H, INTERMEDIATE)
                )
                weight = torch.randn(rows, cols, dtype=torch.bfloat16) / 10
                packed, block_scale, global_scale, _ = quantize_nvfp4_shard(weight)
                prefix = "w13" if shard in ("w1", "w3") else "w2"
                for suffix, value in (
                    ("weight", packed),
                    ("weight_scale", block_scale),
                    ("weight_scale_2", (1.0 / global_scale).clone()),
                    ("input_scale", torch.tensor(act_scale)),
                ):
                    name = f"{prefix}_{suffix}"
                    layer.weight_loader(
                        getattr(layer, name),
                        value,
                        name,
                        shard_id=shard,
                        expert_id=expert,
                    )
        layer.quant_method.process_weights_after_loading(layer)
        return layer

    def test_per_token_phase_matches_global_controls_and_cuda_graphs(self):
        from sglang.srt.layers.moe.topk import TopKConfig, select_experts

        for backend in ("flashinfer_cutedsl", "flashinfer_trtllm"):
            controls = {}
            torch.manual_seed(11)
            x = torch.randn(M, H, dtype=torch.bfloat16) / 10
            logits = torch.randn(M, E, dtype=torch.float32)
            with _runtime(backend, False):
                topk = select_experts(
                    hidden_states=x,
                    router_logits=logits,
                    topk_config=TopKConfig(top_k=TOPK, renormalize=True),
                )
            for per_token in (False, True):
                with _runtime(backend, False, per_token=per_token):
                    layer = self._make_layer()
                    controls[per_token] = _forward(
                        layer, x, topk, ForwardMode.EXTEND
                    ).clone()
            for selector in ("prefill", "decode"):
                with (
                    self.subTest(backend=backend, selector=selector),
                    _runtime(backend, False, per_token=selector),
                ):
                    layer = self._make_layer()
                    for mode, phase in (
                        (ForwardMode.EXTEND, "prefill"),
                        (ForwardMode.DECODE, "decode"),
                        (ForwardMode.EXTEND, "prefill"),
                    ):
                        expected = controls[selector == phase]
                        torch.testing.assert_close(
                            _forward(layer, x, topk, mode),
                            expected,
                            rtol=1e-3,
                            atol=1e-3,
                        )
                    captures = []
                    for mode, phase in (
                        (ForwardMode.EXTEND, "prefill"),
                        (ForwardMode.DECODE, "decode"),
                    ):
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(3):
                                _forward(layer, x, topk, mode)
                        torch.cuda.current_stream().wait_stream(stream)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            captured = _forward(layer, x, topk, mode)
                        captures.append((graph, captured, controls[selector == phase]))
                    for _ in range(2):
                        for graph, captured, expected in captures:
                            graph.replay()
                            torch.testing.assert_close(
                                captured, expected, rtol=1e-3, atol=1e-3
                            )

    def test_mixed_runners_match_controls_and_cuda_graphs(self):
        from sglang.srt.layers.moe.topk import TopKConfig, select_experts

        with _runtime(PREFILL, False):
            prefill = self._make_layer()
        with _runtime(DECODE, True):
            decode = self._make_layer()
        with _runtime(PREFILL, "decode", mixed=True):
            mixed = self._make_layer()
            decode_view = mixed._decode_moe_view
            self.assertIsNot(mixed.quant_method, decode_view.quant_method)
            for name in ("w13_weight", "w2_weight", "w13_weight_scale_2"):
                self.assertNotEqual(
                    getattr(mixed, name).data_ptr(),
                    getattr(decode_view, name).data_ptr(),
                    name,
                )

        torch.manual_seed(11)
        x = torch.randn(M, H, dtype=torch.bfloat16) / 10
        logits = torch.randn(M, E, dtype=torch.float32)
        # All runners consume the same routing decisions from the real top-k path.
        with _runtime(PREFILL, False):
            topk = select_experts(
                hidden_states=x,
                router_logits=logits,
                topk_config=TopKConfig(top_k=TOPK, renormalize=True),
            )
            prefill_ref = _forward(prefill, x, topk, ForwardMode.EXTEND).clone()
        with _runtime(DECODE, True):
            decode_ref = _forward(decode, x, topk, ForwardMode.DECODE).clone()

        modes = (
            (ForwardMode.EXTEND, prefill_ref),
            (ForwardMode.DECODE, decode_ref),
            (ForwardMode.EXTEND, prefill_ref),
            (ForwardMode.TARGET_VERIFY, decode_ref),
        )
        with _runtime(PREFILL, "decode", mixed=True):
            for mode, expected in modes:
                with self.subTest(mode=mode):
                    torch.testing.assert_close(
                        _forward(mixed, x, topk, mode), expected, rtol=1e-3, atol=1e-3
                    )

            # Capture the two execution paths separately, then alternate replays.
            # Changing the current phase must not change either captured graph.
            captures = []
            for mode, expected in modes[:2]:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        _forward(mixed, x, topk, mode)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = _forward(mixed, x, topk, mode)
                captures.append((graph, captured, expected))
            for _ in range(3):
                for graph, captured, expected in captures:
                    graph.replay()
                    torch.testing.assert_close(captured, expected, rtol=1e-3, atol=1e-3)

            with self.assertRaisesRegex(ValueError, "Reloading weights"):
                mixed.quant_method.process_weights_after_loading(mixed)


if __name__ == "__main__":
    unittest.main()
