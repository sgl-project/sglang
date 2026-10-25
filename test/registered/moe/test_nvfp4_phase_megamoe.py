"""Four-rank real FusedMoE parity for CuTe W4A4 prefill/MegaMoE W4A16 decode.

Each worker loads the same
canonical synthetic checkpoint, independently prepares both global controls,
and compares each mixed phase with its matching control. This is numerics and
capture validation, not a performance benchmark.
"""

import json
import os
import subprocess
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import set_dp_buffer_len, set_is_extend_in_batch
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_exec, get_flags
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.quant_ref_utils import quantize_nvfp4_shard
from sglang.test.test_utils import (
    CustomTestCase,
    find_available_port,
    publish_build_topology,
)

register_cuda_ci(est_time=90, stage="base-b", runner_config="4-gpu-b200")

EXPERTS, HIDDEN, INTERMEDIATE, TOPK, TOKENS = 32, 1024, 1024, 2, 32
PREFILL, DECODE = "flashinfer_cutedsl", "flashinfer_megamoe"


@contextmanager
def scope(backend, w4a16, per_token, *, mixed=False):
    a2a = "flashinfer_megamoe" if backend == DECODE else "flashinfer"
    with (
        envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override(w4a16),
        envs.SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION.override(per_token),
        get_exec().moe.override(
            moe_a2a_backend=a2a,
            flashinfer_a2a_dispatch_type="bf16",
            moe_prefill_runner_backend=PREFILL if mixed else None,
            moe_decode_runner_backend=DECODE if mixed else None,
        ),
        get_flags().moe.override(
            runner_backend=MoeRunnerBackend(backend),
            a2a_backend=MoeA2ABackend(a2a),
        ),
    ):
        yield


def make_layer():
    torch.manual_seed(7)
    layer = FusedMoE(
        num_experts=EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        layer_id=0,
        top_k=TOPK,
        params_dtype=torch.bfloat16,
        quant_config=ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True, group_size=16
        ),
        gate_up_interleaved=False,
    ).cuda()
    # Fixed global scale gives gate/up identical alpha, as MegaMoE requires.
    gs = torch.tensor(2688.0, device="cuda", dtype=torch.float32)
    for expert in range(EXPERTS):
        for shard in ("w1", "w3", "w2"):
            rows, cols = (
                (INTERMEDIATE, HIDDEN)
                if shard in ("w1", "w3")
                else (HIDDEN, INTERMEDIATE)
            )
            weight = torch.randn(rows, cols, dtype=torch.bfloat16) / 10
            packed, block_scale, global_scale, _ = quantize_nvfp4_shard(weight, gs)
            prefix = "w13" if shard in ("w1", "w3") else "w2"
            for suffix, value in (
                ("weight", packed),
                ("weight_scale", block_scale),
                ("weight_scale_2", (1.0 / global_scale).clone()),
                ("input_scale", torch.tensor(1.0)),
            ):
                name = f"{prefix}_{suffix}"
                layer.weight_loader(
                    getattr(layer, name), value, name, shard_id=shard, expert_id=expert
                )
    layer.quant_method.process_weights_after_loading(layer)
    return layer


def forward(layer, x, topk, mode, *, extend_comm=False):
    set_is_extend_in_batch(extend_comm)
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


def _run_worker():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_default_device("cuda")
    init_distributed_environment(
        world_size=world, rank=rank, local_rank=rank, backend="nccl"
    )
    publish_build_topology(
        world_rank=rank, tp_size=world, ep_size=world, attn_dp_size=world
    )
    initialize_model_parallel(backend="nccl")
    get_flags().dp.enabled = True
    set_dp_buffer_len(
        global_dp_buffer_len=world * TOKENS,
        local_dp_buffer_len=TOKENS,
        dp_max_padding=True,
        global_num_tokens=[TOKENS] * world,
    )
    torch.manual_seed(11 + rank)
    x = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16) / 10
    logits = torch.randn(TOKENS, EXPERTS, dtype=torch.float32)
    with scope(PREFILL, False, True):
        topk = select_experts(
            hidden_states=x,
            router_logits=logits,
            topk_config=TopKConfig(top_k=TOPK, renormalize=True),
        )
        prefill = make_layer()
        # Exercise actual eager prompt all-gatherv plus local expert reduce-scatter.
        ref_prefill = forward(
            prefill, x, topk, ForwardMode.EXTEND, extend_comm=True
        ).clone()
        # Capture uses the A2A path while preserving the prompt precision phase.
        ref_prefill_graph = forward(prefill, x, topk, ForwardMode.EXTEND).clone()
    with scope(DECODE, True, False):
        decode = make_layer()
        ref_decode = forward(decode, x, topk, ForwardMode.DECODE).clone()
    with scope(PREFILL, "decode", "prefill", mixed=True):
        mixed = make_layer()
        secondary = mixed._decode_moe_view
        for name in ("w13_weight", "w2_weight"):
            assert (
                getattr(mixed, name).data_ptr() != getattr(secondary, name).data_ptr()
            )
        cases = [
            ("prefill_eager", ForwardMode.EXTEND, True, ref_prefill),
            ("decode", ForwardMode.DECODE, False, ref_decode),
            ("prefill_a2a", ForwardMode.EXTEND, False, ref_prefill_graph),
            ("target_verify", ForwardMode.TARGET_VERIFY, False, ref_decode),
            ("prefill_eager_repeat", ForwardMode.EXTEND, True, ref_prefill),
        ]
        errors = {}
        for label, mode, extend_comm, expected in cases:
            actual = forward(mixed, x, topk, mode, extend_comm=extend_comm)
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
            errors[label] = (actual.float() - expected.float()).abs().max().item()
        captures = []
        for label, mode, expected in (
            ("prefill_graph", ForwardMode.EXTEND, ref_prefill_graph),
            ("decode_graph", ForwardMode.DECODE, ref_decode),
        ):
            for _ in range(3):
                forward(mixed, x, topk, mode)
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = forward(mixed, x, topk, mode)
            captures.append((label, graph, captured, expected))
        for _ in range(3):
            for label, graph, captured, expected in captures:
                graph.replay()
                torch.testing.assert_close(captured, expected, rtol=1e-3, atol=1e-3)
                errors[label] = (captured.float() - expected.float()).abs().max().item()
    print(
        json.dumps(
            {"rank": rank, "world": world, "max_abs_errors": errors, "status": "PASS"}
        ),
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()


@unittest.skipIf(
    torch.cuda.device_count() < 4
    or (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 10),
    "Requires four Blackwell GPUs connected by NVLink",
)
class TestNvFp4PhaseMegaMoe(CustomTestCase):
    def test_cutedsl_per_token_prefill_megamoe_w4a16_decode(self):
        env = {
            **os.environ,
            "SGLANG_FLASHINFER_MEGAMOE_MAX_TOKENS_PER_RANK": "128",
            "SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
            "SGLANG_MOE_NVFP4_DISPATCH": "0",
            "SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE": "0",
        }
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=1",
                "--nproc-per-node=4",
                "--master-addr=127.0.0.1",
                f"--master-port={find_available_port(29649)}",
                str(Path(__file__).resolve()),
                "--worker",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=900,
        )
        print(result.stdout)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count('"status": "PASS"'), 4, result.stdout)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _run_worker()
    else:
        unittest.main()
