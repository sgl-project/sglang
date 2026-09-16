"""Engram host lookup and MXFP8 projection must survive concurrent graph replay."""

import os
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx95 MXFP8 projection")
class TestEngramPrefetch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
            init_distributed_environment,
            initialize_model_parallel,
        )
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args()
        override.install()
        cls.addClassCleanup(override.restore)
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=torch.cuda.current_device(),
            distributed_init_method=f"file://{cls.directory.name}/group",
            backend="gloo",
        )
        cls.addClassCleanup(destroy_distributed_environment)
        initialize_model_parallel(tensor_model_parallel_size=1, backend="gloo")
        cls.addClassCleanup(destroy_model_parallel)

    def _make_engram(self, backend):
        from sglang.srt.environ import envs
        from sglang.srt.layers.engram import Engram, EngramLayout
        from sglang.srt.layers.quantization import fp8_utils
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        layout = EngramLayout(
            max_ngram_size=3,
            layer_ids=(14,),
            num_embeddings=(128,),
            primes=(((31, 37, 41, 43), (47, 53, 59, 61)),),
            n_heads=4,
            head_dim=128,
        )
        with (
            envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(True),
            envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.override("shared"),
            envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_PIN.override(True),
            patch.object(fp8_utils, "FP8_GEMM_RUNNER_BACKEND", backend),
            torch.device("cuda"),
        ):
            engram = Engram(
                SimpleNamespace(hidden_size=128, hc_mult=4, rms_norm_eps=1e-6),
                14,
                layout,
                Fp8Config(
                    is_checkpoint_fp8_serialized=True,
                    weight_block_size=[32, 32],
                    scale_fmt="ue8m0",
                ),
            )
        host = engram.embed.host_table
        self.addCleanup(os.close, host.fd)
        self.addCleanup(host.mm.close)
        self.addCleanup(torch.cuda.cudart().cudaHostUnregister, host.bytes.data_ptr())
        with torch.no_grad():
            engram.embed.weight.copy_(torch.randn(128, 128).to(torch.float8_e4m3fn))
            engram.embed.scale.view(torch.uint8).fill_(127)
            engram.wkv.weight.copy_(
                torch.randn_like(engram.wkv.weight, dtype=torch.bfloat16).to(
                    torch.float8_e4m3fn
                )
            )
            engram.wkv.weight_scale_inv.fill_(1.0)
            engram.wkv.quant_method.process_weights_after_loading(engram.wkv)
        return engram

    @torch.inference_mode()
    def test_prefetch_matches_serial_with_concurrent_projection(self):
        from sglang.srt.layers.quantization.fp8_utils import Fp8GemmRunnerBackend

        torch.manual_seed(314)
        for backend in (Fp8GemmRunnerBackend.AUTO, Fp8GemmRunnerBackend.TRITON):
            with self.subTest(backend=backend):
                engram = self._make_engram(backend)
                hidden = torch.randn(1, 4, 128, device="cuda", dtype=torch.bfloat16)
                ids = torch.randint(0, 128, (1, 8), device="cuda")
                other_ids = torch.randint(0, 128, (1, 8), device="cuda")
                stream = torch.cuda.Stream()

                def overlapped():
                    current = torch.cuda.current_stream()
                    stream.wait_stream(current)
                    with torch.cuda.stream(stream):
                        kv = engram.project(ids)
                    ids.record_stream(stream)
                    other_kv = engram.project(other_ids)
                    current.wait_stream(stream)
                    kv.record_stream(current)
                    return engram.apply_gate(hidden, kv), other_kv

                for _ in range(3):
                    overlapped()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output, other_output = overlapped()
                for _ in range(10):
                    hidden.normal_()
                    ids.random_(0, 128)
                    other_ids.random_(0, 128)
                    graph.replay()
                    torch.testing.assert_close(
                        output, engram(hidden, ids), rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        other_output, engram.project(other_ids), rtol=0, atol=0
                    )
                torch.cuda.synchronize()

    @torch.inference_mode()
    def test_vision_model_prefetch_preserves_image_rows_on_replay(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.quantization.fp8_utils import Fp8GemmRunnerBackend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models import deepseek_v4

        engram = self._make_engram(Fp8GemmRunnerBackend.AUTO)
        hidden = torch.randn(1, 4, 128, device="cuda", dtype=torch.bfloat16)
        ids = torch.randint(0, 128, (1, 1, 8), device="cuda")
        input_ids = torch.zeros(1, device="cuda", dtype=torch.int64)
        positions = torch.zeros_like(input_ids)
        layer = SimpleNamespace(
            engram=engram,
            hc_boundary_fused=False,
            forward_hc_pre_from_prev=lambda **kw: (kw["hidden_states"], None),
        )
        model = SimpleNamespace(
            pp_group=SimpleNamespace(world_size=1),
            config=SimpleNamespace(
                model_type="deepseek_v41", vision_n_layers=32, image_token_id=42
            ),
            engram_hasher=lambda *_: ids,
            engram_prefetch_stream=torch.cuda.Stream(),
            late_layer_start=None,
            start_layer=14,
            end_layer=15,
            layers=[None] * 14 + [layer],
        )
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        recorder = SimpleNamespace(with_current_layer=lambda _: nullcontext())

        def forward():
            return deepseek_v4.DeepseekV4Model._forward_layers_hc_pre_from_prev(
                model, positions, hidden, batch, input_ids, input_ids, False, []
            )[0]

        with (
            patch.object(deepseek_v4, "is_cp_active", return_value=False),
            patch.object(deepseek_v4, "check_cuda_graph_backend", return_value=False),
            patch.object(
                deepseek_v4,
                "get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
        ):
            for fused in (False, True):
                with (
                    self.subTest(fused=fused),
                    envs.SGLANG_OPT_HIP_FUSED_DECODE_GLUE.override(fused),
                ):
                    for _ in range(3):
                        forward()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = forward()
                    for token in (7, 42, 13, 42):
                        hidden.normal_()
                        ids.random_(0, 128)
                        input_ids.fill_(token)
                        graph.replay()
                        expected = hidden if token == 42 else engram(hidden, ids[:, 0])
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)
                    torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
