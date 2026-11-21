"""Eight-rank H100 component checks, outside any serving measurement.

Run with torchrun --standalone --nproc-per-node=8 and the candidate PYTHONPATH.
DSV41_TEST_MODEL points at the unmodified checkpoint; DSV41_TEST_OUTPUT is a
private output directory. These are component checks, not quality equivalence.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.pop("SGLANG_RAGGED_VERIFY_MODE", None)

import torch
import torch.distributed as dist

from sglang.srt.distributed import parallel_state as ps
from sglang.srt.layers import dp_attention as dp
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def check_close(actual, expected):
    difference = actual.float() - expected.float()
    result = {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (
            difference.norm() / expected.float().norm().clamp_min(1e-20)
        ).item(),
    }
    # BF16 output can change GEMM tactic when all ranks' rows are coalesced.
    # Keep this explicit bound; do not weaken it after observing a failure.
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2**-6, atol=2**-4)
    return result


def main():
    rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 8
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (9, 0)
    model = Path(os.environ["DSV41_TEST_MODEL"])
    output = Path(os.environ["DSV41_TEST_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    args = ServerArgs(
        model_path=str(model),
        trust_remote_code=True,
        device="cuda",
        tp_size=8,
        attn_dp_size=8,
        ep_size=8,
        attn_cp_size=1,
        attention_backend="dsv4",
        moe_runner_backend="flashinfer_mxfp4",
        flashinfer_mxfp4_moe_precision="fp8",
        speculative_algorithm="DSPARK",
        speculative_dspark_block_size=5,
        enable_dp_lm_head=True,
    )
    publish(args, role="test", ranks=SpawnRanks(world_rank=rank, gpu_id=rank))
    ps.init_distributed_environment(
        world_size=8, rank=rank, local_rank=rank, backend="nccl", timeout=180
    )
    ps.initialize_model_parallel()
    dp.initialize_dp_attention(args)
    dp.set_is_extend_in_batch(False)
    group = get_parallel().tp_group
    results = []

    def record(item):
        results.append(item)
        with (output / f"rank-{rank}-journal.jsonl").open("a") as journal:
            journal.write(json.dumps(item) + "\n")

    # Exercise dynamic graph counts and ensure router-owned IDs are unchanged.
    batch = SimpleNamespace(
        dp_spec_prefill_coordination_applied=False,
        global_num_tokens=[0, 1, 2, 3, 4, 5, 6, 8],
        global_num_tokens_for_logprob=[0, 1, 2, 3, 4, 5, 6, 8],
        can_run_decode_cuda_graph=True,
        can_run_dp_draft_cuda_graph=True,
    )
    fb = SimpleNamespace(
        spec_info=SimpleNamespace(
            num_tokens_per_req=6, num_tokens_for_logprob_per_req=6
        ),
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_algorithm=SpeculativeAlgorithm.DSPARK,
        ragged_verify_layout=None,
    )
    ForwardBatch.init_mlp_sync_metadata(fb, batch, "cuda")
    counts = fb.moe_dp_valid_counts
    torch.testing.assert_close(
        counts, torch.tensor(batch.global_num_tokens, device="cuda") * 6
    )
    fb.global_num_tokens_gpu.fill_(48)
    fb.global_dp_buffer_len = 384
    fb.global_num_tokens_padded_cpu = [48] * 8
    fb.dp_padding_mode = dp.DpPaddingMode.MAX_LEN
    dp.set_dp_buffer_len_from_batch(fb)
    ids = (
        torch.arange(384 * 6, device="cuda", dtype=torch.int32)
        .view(384, 6)
        .remainder_(384)
    )
    original = ids.clone()
    masked = dp.copy_mask_dp_moe_padding(ids)
    for turn in range(3):
        valid = torch.arange(384, device="cuda") % 48 < counts.repeat_interleave(48)
        expected = torch.where(valid[:, None], ids, -1)
        torch.testing.assert_close(masked, expected, rtol=0, atol=0)
        torch.testing.assert_close(ids, original, rtol=0, atol=0)
        counts.copy_(counts.roll(1))
        masked = dp.copy_mask_dp_moe_padding(ids)
    with group.graph_capture() as capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture.stream):
            graph_masked = dp.copy_mask_dp_moe_padding(ids)
    for _ in range(3):
        counts.copy_(counts.roll(1))
        ids.add_(1)
        original = ids.clone()
        graph.replay()
        torch.cuda.synchronize()
        valid = torch.arange(384, device="cuda") % 48 < counts.repeat_interleave(48)
        torch.testing.assert_close(
            graph_masked, torch.where(valid[:, None], ids, -1), rtol=0, atol=0
        )
        torch.testing.assert_close(ids, original, rtol=0, atol=0)
    record({"component": "moe_pad_mask", "passed": True})

    # Exercise the actual EP8 Humming/CUTLASS consumer. Both calls use the
    # native preprocessor and identical tensors; only real-row metadata differs.
    from flashinfer.fused_moe import preprocess_moe_weights_for_sm90_mixed_gemm_humming

    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
        FlashInferCutlassMxfp4MoeQuantInfo,
        _fused_experts_flashinfer_mxfp4_cutlass,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    assert not envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get()
    generator = torch.Generator(device="cuda").manual_seed(66068407 + rank)
    hidden, intermediate, experts = 5120, 128, 48
    w13 = torch.randint(
        0,
        256,
        (experts, 2 * intermediate, hidden // 2),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden, intermediate // 2),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    sf13 = torch.full(
        (experts, 2 * intermediate, hidden // 32), 120, device="cuda", dtype=torch.uint8
    )
    sf2 = torch.full(
        (experts, hidden, intermediate // 32), 120, device="cuda", dtype=torch.uint8
    )
    w13, sf13, res13 = preprocess_moe_weights_for_sm90_mixed_gemm_humming(w13, sf13)
    w2, sf2, res2 = preprocess_moe_weights_for_sm90_mixed_gemm_humming(w2, sf2)
    quant = FlashInferCutlassMxfp4MoeQuantInfo(
        w13_weight=w13,
        w2_weight=w2,
        w13_weight_scale=sf13,
        w2_weight_scale=sf2,
        w13_humming_residual_scale=(res13 * 64).contiguous(),
        w2_humming_residual_scale=(res2 * 64).contiguous(),
        humming_fc2_act_scale=torch.ones((), device="cuda", dtype=torch.float32),
    )
    layer = SimpleNamespace(
        moe_tp_size=1, moe_tp_rank=0, moe_ep_size=8, moe_ep_rank=rank
    )
    config = MoeRunnerConfig(
        layer=layer,
        num_experts=384,
        num_local_experts=48,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        top_k=6,
    )
    x = (
        torch.randn(
            (384, hidden), device="cuda", dtype=torch.bfloat16, generator=generator
        )
        * 0.1
    )
    routes = (
        torch.arange(384, device="cuda")[:, None]
        + torch.arange(6, device="cuda")[None, :]
    ).remainder(48).to(torch.int32) + rank * 48
    original = routes.clone()
    dispatch = StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=torch.full(
                (384, 6), 1 / 6, device="cuda", dtype=torch.float32
            ),
            topk_ids=routes,
            router_logits=torch.empty((384, 384), device="cuda"),
        ),
    )
    dp.set_dp_buffer_len(384, 48, True, [48] * 8, moe_dp_valid_counts=None)
    reference = _fused_experts_flashinfer_mxfp4_cutlass(
        dispatch, quant, config
    ).hidden_states
    torch.cuda.synchronize()
    assert torch.isfinite(reference).all() and reference.abs().max() > 0
    counts = torch.tensor(
        [0, 6, 12, 18, 24, 30, 36, 48], device="cuda", dtype=torch.int32
    )
    dp.set_dp_buffer_len(384, 48, True, [48] * 8, moe_dp_valid_counts=counts)
    # Capture/replay the real changed runner, including native MoE finalization.
    eager = _fused_experts_flashinfer_mxfp4_cutlass(
        dispatch, quant, config
    ).hidden_states
    with group.graph_capture() as capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture.stream):
            captured = _fused_experts_flashinfer_mxfp4_cutlass(
                dispatch, quant, config
            ).hidden_states
    for turn in range(2):
        graph.replay()
        torch.cuda.synchronize()
        valid = torch.arange(384, device="cuda") % 48 < counts.repeat_interleave(48)
        torch.testing.assert_close(
            captured[valid], reference[valid], rtol=2**-6, atol=2**-8
        )
        torch.testing.assert_close(
            captured[~valid], torch.zeros_like(captured[~valid]), rtol=0, atol=0
        )
        torch.testing.assert_close(routes, original, rtol=0, atol=0)
        assert torch.isfinite(captured).all()
        if turn == 0:
            torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        counts.copy_(counts.roll(1))
    record({"component": "native_humming_mask_finalization", "passed": True})

    (output / f"rank-{rank}.json").write_text(
        json.dumps({"rank": rank, "passed": True, "results": results}, indent=2)
    )
    dist.barrier(group=group.cpu_group)
    if rank == 0:
        print(json.dumps({"passed": True, "ranks": 8}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
