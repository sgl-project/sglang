# SPDX-License-Identifier: Apache-2.0
"""Reproduce the FlashInfer MXFP8 MoE PDL CUDA-graph wedge.

This is intentionally a manual, single-GPU test because the known-bad path can
leave a CUDA kernel resident until its process is killed.  The CUDA workload is
isolated in a spawned child process so pytest can detect the stall and recover.

Run on an SM100 GPU with FlashInfer TRT-LLM MoE support:

    CUDA_VISIBLE_DEVICES=0 pytest -s \
      test/manual/layers/moe/test_flashinfer_trtllm_mxfp8_pdl_graph.py

Override SGLANG_MXFP8_PDL_TEST_TOKENS and SGLANG_MXFP8_PDL_TEST_REPLAYS
to sweep other capture shapes without editing this file.

The explicit PDL-off control must finish.  The policy run derives enable_pdl
from SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS.  With the old 8192-token default, the
7680-token case enables PDL and typically stops making progress during replay.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import statistics
import time
import traceback

import pytest
import torch

from sglang.kernels.ops.moe.pack_topk_ids import PackTopkIds
from sglang.srt.environ import envs
from sglang.srt.layers.moe.flashinfer_trtllm_moe import (
    trtllm_fp8_block_scale_routed_moe_out_wrapper,
)
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.utils.common import next_power_of_2

# MiniMax-M3 TP8-local routed-expert shape at the failing prefill capture size.
NUM_TOKENS = int(os.getenv("SGLANG_MXFP8_PDL_TEST_TOKENS", "7680"))
HIDDEN_SIZE = 6144
NUM_EXPERTS = 128
INTERMEDIATE_SIZE = 384
TOP_K = 4
NUM_LAYERS = 57
NUM_REPLAYS = int(os.getenv("SGLANG_MXFP8_PDL_TEST_REPLAYS", "1000"))
SYNC_EVERY = int(os.getenv("SGLANG_MXFP8_PDL_TEST_SYNC_EVERY", "10"))
PDL_MAX_TOKENS = envs.SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS.get()
STALL_TIMEOUT_S = 45
STARTUP_TIMEOUT_S = 900


def _shuffle_weights(w31, w31_scale, w2, w2_scale):
    from flashinfer import (
        reorder_rows_for_gated_act_gemm,
        shuffle_matrix_a,
        shuffle_matrix_sf_a,
    )

    tile_m = 128
    q31, s31, q2, s2 = [], [], [], []
    for expert in range(NUM_EXPERTS):
        q31_expert = reorder_rows_for_gated_act_gemm(w31[expert])
        s31_expert = reorder_rows_for_gated_act_gemm(w31_scale[expert])
        q31.append(shuffle_matrix_a(q31_expert.view(torch.uint8), tile_m))
        s31.append(shuffle_matrix_sf_a(s31_expert.view(torch.uint8), tile_m))
        q2.append(shuffle_matrix_a(w2[expert].view(torch.uint8), tile_m))
        s2.append(shuffle_matrix_sf_a(w2_scale[expert].view(torch.uint8), tile_m))
    return (
        torch.stack(q31).view(torch.float8_e4m3fn),
        torch.stack(s31),
        torch.stack(q2).view(torch.float8_e4m3fn),
        torch.stack(s2),
    )


@torch.inference_mode()
def _build_case():
    from flashinfer import mxfp8_quantize
    from flashinfer.fused_moe import Fp8QuantizationType
    from flashinfer.fused_moe.core import ActivationType

    torch.manual_seed(42)
    device = torch.device("cuda")

    hidden = (
        torch.randn(NUM_TOKENS, HIDDEN_SIZE, device=device, dtype=torch.bfloat16) / 8
    )
    token_ids = torch.arange(NUM_TOKENS, device=device).unsqueeze(1)
    expert_offsets = torch.arange(TOP_K, device=device).unsqueeze(0)
    topk_ids = ((token_ids + expert_offsets * (NUM_EXPERTS // TOP_K)) % NUM_EXPERTS).to(
        torch.int32
    )
    topk_weights = torch.full(
        (NUM_TOKENS, TOP_K),
        2.0 / TOP_K,
        device=device,
        dtype=torch.float32,
    )
    packed_topk_ids = PackTopkIds.execute(topk_ids, topk_weights)

    w31 = (
        torch.randn(
            NUM_EXPERTS,
            2 * INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )
        / HIDDEN_SIZE**0.5
    )
    w2 = (
        torch.randn(
            NUM_EXPERTS,
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )
        / INTERMEDIATE_SIZE**0.5
    )
    w31_q, w31_scale = mxfp8_quantize(w31.flatten(0, 1), False)
    w2_q, w2_scale = mxfp8_quantize(w2.flatten(0, 1), False)
    w31_q = w31_q.view_as(w31)
    w2_q = w2_q.view_as(w2)
    w31_scale = w31_scale.reshape(NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, -1)
    w2_scale = w2_scale.reshape(NUM_EXPERTS, HIDDEN_SIZE, -1)
    del w31, w2
    torch.cuda.empty_cache()

    w31_q, w31_scale, w2_q, w2_scale = _shuffle_weights(
        w31_q, w31_scale, w2_q, w2_scale
    )
    hidden_q, hidden_scale = mxfp8_quantize(hidden, False, backend="cute-dsl")

    # This models the separate shared-expert branch running on the primary
    # stream while routed MoE runs on the alternate stream.
    shared_w1 = (
        torch.randn(
            HIDDEN_SIZE,
            2 * INTERMEDIATE_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )
        / HIDDEN_SIZE**0.5
    )
    shared_w2 = (
        torch.randn(
            INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )
        / INTERMEDIATE_SIZE**0.5
    )
    shared_gate = torch.empty(
        NUM_TOKENS,
        2 * INTERMEDIATE_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    shared_out = torch.empty_like(hidden)
    routed_out = torch.empty_like(hidden)
    combined = torch.empty_like(hidden)

    def expert_param(value: float):
        return torch.full((NUM_EXPERTS,), value, device=device, dtype=torch.float32)

    moe_args = dict(
        topk_ids=packed_topk_ids,
        routing_bias=None,
        hidden_states=hidden_q,
        hidden_states_scale=hidden_scale.reshape(NUM_TOKENS, -1),
        gemm1_weights=w31_q,
        gemm1_weights_scale=w31_scale,
        gemm2_weights=w2_q,
        gemm2_weights_scale=w2_scale,
        output=routed_out,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
        routed_scaling_factor=2.0,
        routing_method_type=int(RoutingMethodType.MiniMax2),
        use_shuffled_weight=True,
        tune_max_num_tokens=next_power_of_2(NUM_TOKENS),
        fp8_quantization_type=int(Fp8QuantizationType.MxFp8),
        activation_type=ActivationType.Swiglu.value,
        gemm1_alpha=expert_param(1.702),
        gemm1_beta=expert_param(1.0),
        gemm1_clamp_limit=expert_param(7.0),
    )
    shared = (hidden, shared_w1, shared_w2, shared_gate, shared_out, combined)
    return moe_args, shared


@torch.inference_mode()
def _run_graph(case, enable_pdl: bool, label: str, report):
    moe_args, shared = case
    hidden, shared_w1, shared_w2, shared_gate, shared_out, combined = shared
    primary_stream = torch.cuda.Stream()
    routed_stream = torch.cuda.Stream()

    def pattern():
        current_stream = torch.cuda.current_stream()
        for _ in range(NUM_LAYERS):
            routed_stream.wait_stream(current_stream)
            torch.mm(hidden, shared_w1, out=shared_gate)
            torch.mm(
                shared_gate[:, :INTERMEDIATE_SIZE],
                shared_w2,
                out=shared_out,
            )
            with torch.cuda.stream(routed_stream):
                trtllm_fp8_block_scale_routed_moe_out_wrapper(
                    **moe_args, enable_pdl=enable_pdl
                )
            current_stream.wait_stream(routed_stream)
            torch.add(shared_out, moe_args["output"], out=combined)

    report((label, "warmup", 0))
    primary_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(primary_stream):
        for _ in range(3):
            pattern()
    torch.cuda.current_stream().wait_stream(primary_stream)
    torch.cuda.synchronize()

    report((label, "capture", 0))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=primary_stream):
        pattern()
    torch.cuda.synchronize()

    report((label, "replay", 0))
    chunk_times_us = []
    with torch.cuda.stream(primary_stream):
        chunk_start = torch.cuda.Event(enable_timing=True)
        chunk_end = torch.cuda.Event(enable_timing=True)
        chunk_start.record()
        for replay in range(1, NUM_REPLAYS + 1):
            graph.replay()
            if replay % SYNC_EVERY == 0:
                chunk_end.record()
                chunk_end.synchronize()
                latency_us = chunk_start.elapsed_time(chunk_end) * 1000 / SYNC_EVERY
                chunk_times_us.append(latency_us)
                report((label, "replay", (replay, round(latency_us, 3))))
                if replay != NUM_REPLAYS:
                    chunk_start.record()
    primary_stream.synchronize()
    stable_times = chunk_times_us[1:] or chunk_times_us
    report(
        (
            label,
            "done",
            {
                "replays": NUM_REPLAYS,
                "median_us": round(statistics.median(stable_times), 3),
            },
        )
    )


def _worker(messages):
    try:
        messages.put(("worker", "build", 0))
        case = _build_case()
        torch.cuda.synchronize()
        messages.put(("worker", "built", 0))
        _run_graph(case, False, "pdl_off_control", messages.put)
        _run_graph(
            case,
            NUM_TOKENS <= PDL_MAX_TOKENS,
            "production_policy",
            messages.put,
        )
    except BaseException:
        messages.put(("worker", "error", traceback.format_exc()))
        raise


def _stop(process: mp.Process):
    if process.is_alive():
        process.kill()
        process.join(timeout=30)


def test_flashinfer_trtllm_mxfp8_pdl_multistream_graph_replay():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("The MiniMax-M3 MXFP8 path requires SM100 or newer")

    ctx = mp.get_context("spawn")
    messages = ctx.Queue()
    process = ctx.Process(target=_worker, args=(messages,))
    process.start()

    history = []
    started = time.monotonic()
    last_progress = started
    replay_stage = None
    try:
        while process.is_alive():
            try:
                message = messages.get(timeout=1)
                history.append(message)
                print(message, flush=True)
                label, stage, _ = message
                if stage == "replay":
                    replay_stage = label
                    last_progress = time.monotonic()
                else:
                    replay_stage = None
            except queue.Empty:
                pass

            now = time.monotonic()
            if replay_stage is not None and now - last_progress > STALL_TIMEOUT_S:
                _stop(process)
                pytest.fail(
                    f"{replay_stage} CUDA graph replay made no progress for "
                    f"{STALL_TIMEOUT_S}s; last messages: {history[-8:]}"
                )
            if now - started > STARTUP_TIMEOUT_S:
                _stop(process)
                pytest.fail(
                    f"worker exceeded {STARTUP_TIMEOUT_S}s; "
                    f"last messages: {history[-8:]}"
                )

        process.join()
        while True:
            try:
                message = messages.get_nowait()
                history.append(message)
                print(message, flush=True)
            except queue.Empty:
                break
        assert process.exitcode == 0, (
            f"worker exited with code {process.exitcode}; "
            f"last messages: {history[-8:]}"
        )
    finally:
        _stop(process)
