"""Aux captures must include real TP collectives and survive later decoder work.

Uses production model loops, decoder forwards, communicators and CUDA norms.
Attention/MLP are deterministic small modules; this is a distributed regression
test, not a pretrained-model acceptance or performance benchmark.
"""

import importlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from torch import nn

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="2-gpu-large")

MODELS = [
    ("gpt_oss", "GptOssModel", "GptOssDecoderLayer"),
    ("glm4_moe", "Glm4MoeModel", "Glm4MoeDecoderLayer"),
    ("glm4_moe_lite", "Glm4MoeLiteModel", "Glm4MoeLiteDecoderLayer"),
    ("qwen3_vl_moe", "Qwen3MoeLLMModel", "Qwen3MoeDecoderLayer"),
    ("bailing_moe", "BailingMoEModel", "BailingMoEBlock"),
    ("bailing_moe_v3", "BailingMoELinearModel", "BailingMoELinearDecoderLayer"),
]


class Attention(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states.roll(1, -1) / 16


class MLP(nn.Module):
    def forward(self, hidden_states, forward_batch=None):
        from sglang.srt.runtime_context import get_parallel

        return hidden_states.roll(-1, -1) * ((get_parallel().tp_rank + 1) / 8)


def make_model(entry, dtype, hidden_dim):
    from sglang.srt.layers.communicator import (
        LayerCommunicator,
        LayerScatterModes,
        ScatterMode,
    )
    from sglang.srt.layers.layernorm import RMSNorm

    name, model_name, decoder_name = entry
    module = importlib.import_module("sglang.srt.models." + name)
    model_cls, decoder_cls = getattr(module, model_name), getattr(module, decoder_name)
    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.pp_group = NS(is_first_rank=True, is_last_rank=True)
    model.start_layer, model.end_layer = 0, 3
    model.first_k_dense_replace = 0
    model.layers_to_capture = [0, 1] if name == "bailing_moe_v3" else [1, 2]
    model.capture_aux_hidden_states = True
    model.hidden_size = hidden_dim
    model.use_hf_deepstack_order = False
    model.deepstack_embed_to_decoder_layer = range(3)
    model.norm = RMSNorm(hidden_dim, eps=1e-6).to(device="cuda", dtype=dtype)
    layers = []
    for index in range(3):
        layer = decoder_cls.__new__(decoder_cls)
        nn.Module.__init__(layer)
        layer.layer_id = index
        layer.attn_quant_format = ""
        layer.attention_type = 0
        layer.is_layer_sparse = True
        layer.use_mla = False
        layer.self_attn = Attention()
        layer.attention = Attention()
        layer.mlp = MLP()
        layer.input_layernorm = RMSNorm(hidden_dim, eps=1e-6).to(
            device="cuda", dtype=dtype
        )
        layer.post_attention_layernorm = RMSNorm(hidden_dim, eps=1e-6).to(
            device="cuda", dtype=dtype
        )
        layer.layer_scatter_modes = LayerScatterModes(
            layer_input_mode=ScatterMode.TP_ATTN_FULL,
            attn_mode=ScatterMode.TP_ATTN_FULL,
            mlp_mode=ScatterMode.FULL,
            middle_residual_mode=ScatterMode.TP_ATTN_FULL,
            layer_output_mode=ScatterMode.TP_ATTN_FULL,
        )
        layer.layer_communicator = LayerCommunicator(
            layer_scatter_modes=layer.layer_scatter_modes,
            input_layernorm=layer.input_layernorm,
            post_attention_layernorm=layer.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=index == 2,
        )
        layers.append(layer)
    model.layers = nn.ModuleList(layers)
    return model


def check_case(entry, dtype, rows, mode, fused, graph):
    from sglang.srt.runtime_context import get_exec

    with get_exec().comm.override(
        flashinfer_allreduce_fusion_backend="trtllm" if fused else None
    ):
        return _check_case(entry, dtype, rows, mode, fused, graph)


def _check_case(entry, dtype, rows, mode, fused, graph):
    import torch.distributed as dist

    from sglang.srt.distributed import get_moe_tp_group
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    model = make_model(entry, dtype, 2880)
    generator = torch.Generator(device="cuda").manual_seed(42)
    inputs = torch.randn(rows, 2880, generator=generator, device="cuda", dtype=dtype)
    positions = torch.arange(rows, device="cuda")
    batch = NS(
        input_ids=positions,
        forward_mode=mode,
        can_run_tbo=False,
        attn_cp_metadata=None,
        capture_hidden_mode=CaptureHiddenMode.FULL,
    )
    expected, markers, handles = [], [], []

    def hook(layer, args, kwargs):
        values = inspect.signature(layer.forward).bind(*args, **kwargs).arguments
        hidden, residual = values["hidden_states"], values.get("residual")
        pending = bool(getattr(hidden, "_sglang_needs_allreduce_fusion", False))
        reference = hidden.float().clone()
        if pending:
            dist.all_reduce(reference, group=get_moe_tp_group().device_group)
        if residual is not None:
            reference = reference + residual.float()
        expected.append(reference.to(dtype))
        markers.append(pending)

    for i in [1, 2]:
        handles.append(
            model.layers[i].register_forward_pre_hook(hook, with_kwargs=True)
        )

    def forward():
        arg = "inputs_embeds" if entry[0] == "bailing_moe_v3" else "input_embeds"
        return model(None, positions, batch, **{arg: inputs.clone()})

    final, captures = forward()
    for handle in handles:
        handle.remove()
    if graph:
        for _ in range(3):
            forward()
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph):
            final, captures = forward()
        cuda_graph.replay()
        cuda_graph.replay()
        torch.cuda.synchronize()

    saved_ids = model.layers_to_capture
    model.layers_to_capture = []
    target_without_capture = forward()
    model.layers_to_capture = saved_ids
    errors = [
        (actual.float() - reference.float()).abs().max().item()
        for actual, reference in zip(captures, expected, strict=True)
    ]
    atol = 0.025 if dtype == torch.bfloat16 else 0.003
    close = [
        torch.allclose(actual, reference, atol=atol, rtol=0.01)
        for actual, reference in zip(captures, expected, strict=True)
    ]
    return {
        "model": entry[0],
        "dtype": str(dtype),
        "rows": rows,
        "phase": mode.name,
        "fused": fused,
        "graph": graph,
        "pending": markers,
        "max_abs_errors": errors,
        "captures_correct": bool(close) and all(close),
        "target_unchanged": torch.equal(final, target_without_capture),
        "fusion_covered": all(markers) if fused else not any(markers),
    }


def worker(output):
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    set_global_server_args_for_scheduler(
        ServerArgs(
            model_path="dummy",
            tp_size=world_size,
            device="cuda",
            flashinfer_allreduce_fusion_backend="trtllm",
        )
    )
    set_custom_all_reduce(False)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method="env://",
        backend="nccl",
        timeout=120,
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    results = []
    with torch.inference_mode():
        for entry in MODELS:
            for dtype in [torch.float16, torch.bfloat16]:
                for rows, mode in [
                    (1, ForwardMode.EXTEND),
                    (8, ForwardMode.TARGET_VERIFY),
                ]:
                    for fused in [False, True] if world_size > 1 else [False]:
                        row = check_case(
                            entry, dtype, rows, mode, fused, graph=rows == 8
                        )
                        results.append(row)
                        Path(output, f"rank{rank}.json").write_text(
                            json.dumps(results, indent=2) + "\n"
                        )
                        if rank == 0:
                            print(json.dumps(row), flush=True)
    destroy_model_parallel()
    destroy_distributed_environment()


class TestSpeculativeCaptureReduction(CustomTestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA GPUs")
    def test_capture_matches_completed_layer_output(self):
        if torch.cuda.get_device_capability(0)[0] != 9:
            self.skipTest("this test exercises the Hopper trtllm fusion backend")
        with tempfile.TemporaryDirectory() as output:
            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc-per-node=2",
                    str(Path(__file__).resolve()),
                    "--worker",
                    output,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=240,
                check=False,
            )
            self.assertEqual(run.returncode, 0, run.stdout)
            for rank in range(2):
                rows = json.loads(Path(output, f"rank{rank}.json").read_text())
                self.assertEqual(len(rows), len(MODELS) * 8)
                for row in rows:
                    with self.subTest(rank=rank, case=row):
                        self.assertTrue(row["fusion_covered"])
                        self.assertTrue(row["target_unchanged"])
                        self.assertTrue(row["captures_correct"])


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker(sys.argv[sys.argv.index("--worker") + 1])
    else:
        unittest.main()
