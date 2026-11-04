"""Single-GPU decode CUDA graphs cover more batch sizes than the GPU tier, when and
only when the graph and memory inputs are left at their defaults. The default plan
keeps the running limit and reserves for its graphs exactly as the matching explicit
flag would; sizing the pool after capture caps the default running batch at them.

The planner gates on the final configuration, so the cases resolve the whole
pipeline over a real config.json rather than calling the hook on a fixture.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    post_capture_kv_sizing_planned,
    resolution_result,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

# nvidia-smi memory.total on the single-GPU targets.
_A10G_MIB = 23028
_L40S_MIB = 46068
_H100_MIB = 81559

# Qwen3-8B's attention and vocabulary shape; two layers keep resolution cheap.
_QWEN3_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_hidden_layers": 2,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
}


class TestDecodeGraphPlan(CustomTestCase):
    def setUp(self):
        super().setUp()
        # Resolution writes environment variables that outlive the record.
        environment = dict(os.environ)

        def restore():
            os.environ.clear()
            os.environ.update(environment)

        self.addCleanup(restore)
        self.model_path = self._model_dir(_QWEN3_CONFIG)

    def _model_dir(self, config):
        path = tempfile.mkdtemp(prefix="decode_graph_plan_")
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        with open(os.path.join(path, "config.json"), "w") as handle:
            json.dump(config, handle)
        return path

    def _resolve(self, gpu_mem=_H100_MIB, *, model_path=None, launcher=None, **fields):
        """Resolve on a CUDA GPU of ``gpu_mem`` MiB; returns (decode graph config,
        running cap, post-capture KV sizing planned, mem_fraction_static)."""
        fields.setdefault("attention_backend", "triton")
        args = ServerArgs(
            model_path=model_path or self.model_path, device="cuda", **fields
        )
        if launcher:
            declare_resolution(args, "launcher", **launcher)
        with (
            override_platform(is_cuda=True, is_hip=False, is_npu=False, is_xpu=False),
            patch("sglang.srt.utils.is_cuda", return_value=True),
            patch(
                "sglang.srt.arg_groups.memory_hook.get_device_memory_capacity",
                return_value=gpu_mem,
            ),
        ):
            args.resolve_once()
            planned = post_capture_kv_sizing_planned(args)
        return (
            resolution_result(args, "cuda_graph_config").decode,
            resolution_result(args, "_decode_graph_running_cap"),
            planned,
            resolution_result(args, "mem_fraction_static"),
        )

    def _resolve_without_planner(self, gpu_mem=_H100_MIB, **kwargs):
        with envs.SGLANG_ENABLE_DECODE_GRAPH_PLANNER.override(False):
            return self._resolve(gpu_mem, **kwargs)

    def test_default_plan_per_gpu_tier(self):
        """Over 60 GiB graphs cover 512 requests, at 35-60 GiB 256; neither caps the
        running limit, and each reserves what the explicit graph flag reserves.
        Below 35 GiB the tier plan stays."""
        for gpu_mem, ceiling, graphs in ((_H100_MIB, 512, 52), (_L40S_MIB, 256, 36)):
            with self.subTest(gpu_mem=gpu_mem):
                decode, cap, planned, fraction = self._resolve(gpu_mem)
                self.assertEqual(
                    (decode.max_bs, len(decode.bs), cap, planned),
                    (ceiling, graphs, None, False),
                )
                explicit = self._resolve(gpu_mem, cuda_graph_max_bs_decode=ceiling)
                self.assertEqual(fraction, explicit[3])

        self.assertEqual(
            self._resolve(_A10G_MIB), self._resolve_without_planner(_A10G_MIB)
        )

    def test_explicit_graph_running_and_memory_flags_keep_their_meaning(self):
        h100_tier = self._resolve(mem_fraction_static=0.8)
        self.assertEqual(
            (h100_tier[0].max_bs, h100_tier[1], h100_tier[2], h100_tier[3]),
            (256, None, False, 0.8),
        )
        # A launcher's fraction is as deliberate as the operator's.
        launched = self._resolve(launcher={"mem_fraction_static": 0.7})
        self.assertEqual(
            (launched[0].max_bs, launched[1], launched[3]), (256, None, 0.7)
        )

        decode, cap, planned, _ = self._resolve(cuda_graph_max_bs_decode=64)
        self.assertEqual((decode.max_bs, cap, planned), (64, None, False))

        decode, cap, planned, _ = self._resolve(cuda_graph_bs_decode=[1, 2, 4])
        self.assertEqual((decode.bs, cap, planned), ([1, 2, 4], None, False))

        decode, cap, _, _ = self._resolve(disable_cuda_graph=True)
        self.assertEqual((decode.backend, cap), (Backend.DISABLED, None))

        # Only the full decode backend replays every bucket from one graph.
        decode, cap, _, _ = self._resolve(cuda_graph_backend_decode="breakable")
        self.assertEqual((decode.backend, decode.max_bs, cap), ("breakable", 256, None))

        # The graphs follow an explicit limit up to the ceiling; above it, decode
        # stays eager there and the reserve is the explicit graph flag's.
        decode, cap, planned, _ = self._resolve(_L40S_MIB, max_running_requests=200)
        self.assertEqual((decode.max_bs, cap, planned), (200, None, False))
        decode, cap, planned, fraction = self._resolve(max_running_requests=1024)
        self.assertEqual((decode.max_bs, cap, planned), (512, None, False))
        explicit = self._resolve(
            max_running_requests=1024, cuda_graph_max_bs_decode=512
        )
        self.assertEqual(fraction, explicit[3])

    def test_post_capture_sizing_bounds_decode_by_its_graphs(self):
        """With SGLANG_ENABLE_POST_CAPTURE_KV_SIZING, graphs cover 512 requests on
        any in-scope GPU and cap the default running batch there, so the runtime
        reserve has a bound; the fraction is the opt-in's post-capture floor."""
        with envs.SGLANG_ENABLE_POST_CAPTURE_KV_SIZING.override(True):
            for gpu_mem in (_L40S_MIB, _H100_MIB):
                with self.subTest(gpu_mem=gpu_mem):
                    decode, cap, planned, fraction = self._resolve(gpu_mem)
                    self.assertEqual(
                        (decode.max_bs, len(decode.bs), cap, planned),
                        (512, 52, 512, True),
                    )
                    opt_in = self._resolve(gpu_mem, cuda_graph_max_bs_decode=512)
                    self.assertEqual(fraction, opt_in[3])
            decode, cap, planned, _ = self._resolve(max_running_requests=200)
        self.assertEqual((decode.max_bs, cap, planned), (200, None, True))

    def test_graphs_disabled_after_the_memory_hook_keep_the_tier_plan(self):
        """torch_native disables CUDA graphs only after the memory heuristic ran; a
        plan made before that would charge the eager pool for graphs it never has."""
        plan = self._resolve(attention_backend="torch_native")
        self.assertEqual((plan[0].backend, plan[1]), (Backend.DISABLED, None))
        self.assertEqual(
            plan, self._resolve_without_planner(attention_backend="torch_native")
        )

    def test_out_of_scope_configs_keep_the_tier_plan(self):
        """dLLM decode graphs capture block_size tokens per request, and dLLM runs one
        request unless told otherwise; multi-GPU, unknown memory and the kill switch
        keep the tier plan as well."""
        sdar = self._model_dir({**_QWEN3_CONFIG, "architectures": ["SDARForCausalLM"]})
        for gpu_mem in (_L40S_MIB, _H100_MIB):
            with self.subTest(dllm_gpu_mem=gpu_mem):
                dllm = {"model_path": sdar, "dllm_algorithm": "LowConfidence"}
                self.assertEqual(
                    self._resolve(gpu_mem, **dllm),
                    self._resolve_without_planner(gpu_mem, **dllm),
                )

        decode, cap, planned, _ = self._resolve(tp_size=2)
        self.assertEqual((decode.max_bs, cap, planned), (256, None, False))
        decode, cap, planned, _ = self._resolve(gpu_mem=None)
        self.assertEqual((decode.max_bs, cap, planned), (160, None, False))
        self.assertEqual(
            self._resolve_without_planner(),
            self._resolve(cuda_graph_max_bs_decode=256),
        )


if __name__ == "__main__":
    unittest.main()
