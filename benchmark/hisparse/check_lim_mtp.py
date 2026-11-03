"""Validate chain MTP KV publication/attention on Ascend, without model weights.

Uses the real host allocator, manager, Copy-SFA and vendor SFA. Checks a padded
batch, rejected-token rewrites, draft-extend masks and short/long graph replay
against CPU attention. This is a component check, not model-quality evidence.
"""

import argparse
import json
import sys
from types import SimpleNamespace

import numpy as np
import torch
from bench_lim_framework import FrameworkCase

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.attention import (
    forward_sparsity_driven_kv_offload,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def run(device, capacity):
    torch.npu.set_device(device)
    torch.manual_seed(20261024)
    batch, width, heads = 4, 3, 8
    context = max(4096, capacity - 128)
    args = SimpleNamespace(
        batch=batch,
        context=context,
        tail=1,
        capacity=capacity,
        index_heads=32,
        query_heads=heads,
        layers=2,
    )
    # Build payloads on device. Large CPU torch allocations can start hundreds
    # of OpenMP workers on A3 hosts; the CPU oracle below uses bounded BLAS.
    fixture_context = context + 128
    rng = np.random.default_rng(20261024)
    host_values = rng.standard_normal((3, fixture_context, 1, 576), dtype=np.float32)
    expected_host_bits = (host_values.view(np.uint32) >> 16).astype(np.uint16)
    fixture = SimpleNamespace(
        context=fixture_context,
        key=torch.zeros(
            (2 * fixture_context // 128, 128, 1, 128),
            device="npu",
            dtype=torch.bfloat16,
        ),
        table=torch.arange(
            2 * fixture_context // 128, device="npu", dtype=torch.int32
        ).view(2, -1),
        host=torch.from_numpy(expected_host_bits.copy()).view(torch.bfloat16).to("npu"),
        q=torch.randn((2, heads, 512), device="npu", dtype=torch.bfloat16),
        rope=torch.randn((2, heads, 64), device="npu", dtype=torch.bfloat16),
    )
    print("MTP fixture ready", file=sys.stderr, flush=True)
    with (
        envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.override(True),
        envs.SGLANG_NPU_ENABLE_COPY_SFA.override(True),
    ):
        case = FrameworkCase(
            "lim", fixture, args, request_capacity=2, graph_batch_capacity=batch
        )
        case.backend.graph_mode = False
        draft = FrameworkCase(
            "lim",
            fixture,
            SimpleNamespace(**dict(vars(args), layers=1)),
            request_capacity=2,
            graph_batch_capacity=batch,
            allocator_kv=case.manager.paged_kv_cache,
        )
        assert draft.manager.layer_num == 1 and case.manager.layer_num == 2
        assert draft.manager.host_ptr_list[0] != case.manager.host_ptr_list[0]
        prefix = case.manager.host_kv_buffer[0][1, 0, 0, 0].item()
        draft.manager.host_kv_buffer[0][1, 0, 0, 0] = 99
        assert case.manager.host_kv_buffer[0][1, 0, 0, 0].item() == prefix
        print(
            "MTP shared allocator has distinct target/draft KV pools",
            file=sys.stderr,
            flush=True,
        )
        print("MTP manager ready", file=sys.stderr, flush=True)
        fb = case.batch
        fb.req_pool_indices.copy_(torch.tensor([2, 0, 1, 0], device="npu"))
        fb.forward_mode = ForwardMode.TARGET_VERIFY
        fb.spec_info = SimpleNamespace(
            num_tokens_per_req=width,
            num_accept_tokens=torch.tensor([1, 0, 2, 0], device="npu"),
        )
        fb.seq_lens.copy_(torch.tensor([context, 0, context, 0], device="npu"))
        fb.seq_lens_cpu = fb.seq_lens.cpu()
        fb.positions = (
            fb.seq_lens[:, None] + torch.arange(width, device="npu")[None]
        ).reshape(-1)
        fb.out_cache_loc = torch.arange(batch * width, device="npu")
        q = fixture.q.repeat(2, 1, 1)[:, None].expand(-1, width, -1, -1)
        q = q.reshape(-1, heads, 512)
        rope = fixture.rope.repeat(2, 1, 1)[:, None].expand(-1, width, -1, -1)
        rope = rope.reshape(-1, heads, 64)
        kv = torch.randn(batch * width, 1, 576, device="npu", dtype=torch.bfloat16)
        sources = torch.arange(2048, device="npu", dtype=torch.int32)
        sources = sources.expand(batch, width, -1).clone()
        sources[:, :, -2:] = fb.positions.view(batch, width, 1) - torch.tensor(
            [1, 0], device="npu"
        )

        def step():
            print("MTP forward begin", file=sys.stderr, flush=True)
            output = forward_sparsity_driven_kv_offload(
                case.backend,
                q,
                kv[..., :512],
                None,
                case.layer,
                fb,
                q_rope=rope,
                k_rope=kv[..., 512:],
                topk_indices=sources.view(batch * width, -1),
            )
            print("MTP forward submitted", file=sys.stderr, flush=True)
            return output

        def verify(output):
            print("MTP synchronize begin", file=sys.stderr, flush=True)
            torch.npu.synchronize()
            print("MTP synchronize complete", file=sys.stderr, flush=True)

            def fp32(tensor):
                bits = tensor.cpu().view(torch.uint16).numpy().astype(np.uint32)
                return (bits << 16).view(np.float32)

            if output.shape[0] > batch * width:
                np.testing.assert_array_equal(fp32(output[batch * width :]), 0)
            result = fp32(output[: batch * width]).reshape(batch, width, heads, 512)
            host_bits = case.manager.host_kv_buffer[0].view(torch.uint16).numpy()
            ids = sources.cpu().numpy()
            positions = (
                fb.positions[: batch * width].cpu().numpy().reshape(batch, width)
            )
            requests = fb.req_pool_indices.cpu().tolist()
            counts = fb.spec_info.num_accept_tokens.cpu().tolist()
            query, query_rope = fp32(q), fp32(rope)
            kv_bits = kv.cpu().view(torch.uint16).numpy()
            print("MTP CPU oracle begin", file=sys.stderr, flush=True)
            for row, request in enumerate(requests):
                for column in range(width):
                    live = request > 0 and (
                        not fb.forward_mode.is_draft_extend_v2() or column < counts[row]
                    )
                    if not live:
                        np.testing.assert_array_equal(result[row, column], 0)
                        continue
                    pos = int(positions[row, column])
                    expected_host_bits[request, pos] = kv_bits[row * width + column]
                    selected_ids = ids[row, column]
                    selected_ids = selected_ids[
                        (selected_ids >= 0) & (selected_ids <= pos)
                    ]
                    selected = (
                        expected_host_bits[request, selected_ids, 0].astype(np.uint32)
                        << 16
                    ).view(np.float32)
                    scores = (
                        query[row * width + column] @ selected[:, :512].T
                        + query_rope[row * width + column] @ selected[:, 512:].T
                    ) * case.layer.scaling
                    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
                    weights /= weights.sum(axis=-1, keepdims=True)
                    expected = weights @ selected[:, :512]
                    np.testing.assert_allclose(
                        result[row, column], expected, rtol=0.02, atol=0.002
                    )
                    # KV is published at the absolute position, including all
                    # draft columns of verify and only live extend columns.
                    np.testing.assert_array_equal(
                        host_bits[request, pos], kv_bits[row * width + column]
                    )
            print("MTP CPU oracle complete", file=sys.stderr, flush=True)

        verify(step())
        print("MTP eager attention verified", file=sys.stderr, flush=True)
        expected_path = "mtp_copy_sfa" if capacity == 2048 else "mtp_reference"
        assert case.manager.last_decode_path == expected_path
        kv.add_(0.5)  # Rewrite previously cached speculative positions.
        verify(step())
        fb.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        verify(step())
        print("MTP draft extend verified", file=sys.stderr, flush=True)
        full_sources = sources
        for short_topk in (1, 64):
            sources = full_sources[:, :, :short_topk].contiguous()
            verify(step())
        sources = full_sources
        print("MTP partial TopK verified", file=sys.stderr, flush=True)
        # TP appends token rows separately from request padding. Keep Q=3 and
        # K=2048 when 4*3 logical queries are padded to 16 execution rows.
        originals = q, rope, kv, fb.positions, fb.out_cache_loc
        q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, 4))
        rope = torch.nn.functional.pad(rope, (0, 0, 0, 0, 0, 4))
        kv = torch.nn.functional.pad(kv, (0, 0, 0, 0, 0, 4), value=99)
        fb.positions = torch.nn.functional.pad(fb.positions, (0, 4))
        fb.out_cache_loc = torch.nn.functional.pad(fb.out_cache_loc, (0, 4))
        for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2):
            fb.forward_mode = mode
            verify(step())
        q, rope, kv, fb.positions, fb.out_cache_loc = originals
        print("MTP TP token padding verified", file=sys.stderr, flush=True)
        # Graphs always use the variable-length selected-KV path. Replaying
        # short prefixes must not consume baked-in Copy-SFA length decisions.
        case.backend.graph_mode = True
        step()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            graph_output = step()
        graph.replay()
        verify(graph_output)
        print("MTP graph captured and verified", file=sys.stderr, flush=True)
        fb.seq_lens[0] = 32
        fb.positions[:width].copy_(torch.arange(32, 32 + width, device="npu"))
        sources[0, :, -2:] = fb.positions[:width, None] - torch.tensor(
            [1, 0], device="npu"
        )
        graph.replay()
        verify(graph_output)
        fb.req_pool_indices[0] = 0
        graph.replay()
        verify(graph_output)
        fb.req_pool_indices[0] = 2
        fb.seq_lens[0] = context
        fb.positions[:width].copy_(torch.arange(context, context + width, device="npu"))
        sources[0, :, -2:] = fb.positions[:width, None] - torch.tensor(
            [1, 0], device="npu"
        )
        graph.replay()
        verify(graph_output)
        return dict(
            capacity=capacity,
            batch=batch,
            query_width=width,
            eager_path=expected_path,
            reject_rewrite="pass",
            draft_extend="pass",
            partial_topk="pass",
            tp_token_padding="pass",
            distinct_target_draft_pools="pass",
            graph_replay_short_long_padding="pass",
            cpu_attention="pass",
            kv_bitwise="pass",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--capacity", type=int, choices=(2048, 4096, 8192), default=2048
    )
    args = parser.parse_args()
    print(json.dumps(run(args.device, args.capacity), indent=2))
