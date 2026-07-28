"""Distributed test for GLM-5.2 EAGLE-3 PP auxiliary hidden-state propagation.

This test runs with torchrun --nproc-per-node=2 to verify actual PP send/recv
of the packed auxiliary hidden state tensor.

Usage:
    # CPU/Gloo
    torchrun --standalone --nproc_per_node=2 \
        test/unittest/test_glm52_eagle3_pp_distributed.py \
        --backend gloo --device cpu --warmup 5 --iterations 50 \
        --output-json /tmp/glm52_pp_gloo.json

    # GPU/NCCL
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
        test/unittest/test_glm52_eagle3_pp_distributed.py \
        --backend nccl --device cuda --warmup 10 --iterations 100 \
        --output-json /tmp/glm52_pp_nccl.json
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed PP aux propagation test"
    )
    parser.add_argument("--backend", default="gloo", choices=["gloo", "nccl"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def get_layer_partition(rank, size, num_layers):
    base = num_layers // size
    remainder = num_layers % size
    start = sum(base + (1 if r >= size - remainder else 0) for r in range(rank))
    end = start + base + (1 if rank >= size - remainder else 0)
    return start, end


def run_test_aux_propagation(rank, size, device, iterations, warmup):
    """Test aux hidden states propagate correctly through PP send/recv."""
    from sglang.srt.speculative.glm52_eagle3_pp import (
        GLM52_EAGLE3_AUX_PP_KEY,
        allocate_packed_aux_buffer,
        build_layer_to_slot_map,
        build_slot_ownership_map,
        get_local_capture_layers,
        pack_aux_into_buffer,
        unpack_aux_from_buffer,
    )

    num_layers = 10
    hidden_size = 64
    global_capture_layers = [2, 5, 8]
    num_capture = len(global_capture_layers)
    num_tokens = 4

    layer_to_slot = build_layer_to_slot_map(global_capture_layers)
    slot_ownership = build_slot_ownership_map(
        global_capture_layers, size, num_layers
    )

    start, end = get_layer_partition(rank, size, num_layers)
    local_capture = get_local_capture_layers(global_capture_layers, start, end)

    all_features = {}
    for lid in global_capture_layers:
        all_features[lid] = torch.full(
            (num_tokens, hidden_size), float(lid * 10 + rank),
            dtype=torch.float32, device=device,
        )

    latencies = []

    for it in range(warmup + iterations):
        if rank == 0:
            packed = allocate_packed_aux_buffer(
                num_tokens, num_capture, hidden_size, torch.float32, device
            )
            if local_capture:
                feats = [all_features[lid] for lid in local_capture]
                pack_aux_into_buffer(packed, feats, local_capture, layer_to_slot)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            dist.send(packed, dst=1)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            if it >= warmup:
                latencies.append(t1 - t0)
        else:
            packed = allocate_packed_aux_buffer(
                num_tokens, num_capture, hidden_size, torch.float32, device
            )
            dist.recv(packed, src=0)

            if local_capture:
                feats = [all_features[lid] for lid in local_capture]
                pack_aux_into_buffer(packed, feats, local_capture, layer_to_slot)

            result = unpack_aux_from_buffer(
                packed, global_capture_layers, layer_to_slot,
                slot_ownership, local_capture,
                pp_rank=rank, pp_size=size,
            )

            for i, lid in enumerate(global_capture_layers):
                owner = slot_ownership[lid]
                expected_value = float(lid * 10 + owner)
                actual = result[i][0, 0].item()
                assert actual == expected_value, (
                    f"Layer {lid} (slot {i}, owner PP{owner}): "
                    f"expected {expected_value}, got {actual}"
                )

    return latencies


def run_test_verify_row_count(rank, size, device):
    """Test target verify row count (bs * verify_tokens) is preserved."""
    from sglang.srt.speculative.glm52_eagle3_pp import (
        allocate_packed_aux_buffer,
        build_layer_to_slot_map,
        build_slot_ownership_map,
        get_local_capture_layers,
        pack_aux_into_buffer,
        unpack_aux_from_buffer,
    )

    num_layers = 10
    hidden_size = 32
    global_capture_layers = [2, 5, 8]
    num_capture = len(global_capture_layers)
    bs = 4
    verify_tokens_per_req = 4
    num_tokens = bs * verify_tokens_per_req

    layer_to_slot = build_layer_to_slot_map(global_capture_layers)
    slot_ownership = build_slot_ownership_map(
        global_capture_layers, size, num_layers
    )

    start, end = get_layer_partition(rank, size, num_layers)
    local_capture = get_local_capture_layers(global_capture_layers, start, end)

    all_features = {}
    for lid in global_capture_layers:
        all_features[lid] = torch.randn(
            num_tokens, hidden_size, dtype=torch.float32, device=device,
        )

    if rank == 0:
        packed = allocate_packed_aux_buffer(
            num_tokens, num_capture, hidden_size, torch.float32, device
        )
        if local_capture:
            feats = [all_features[lid] for lid in local_capture]
            pack_aux_into_buffer(packed, feats, local_capture, layer_to_slot)
        dist.send(packed, dst=1)
    else:
        packed = allocate_packed_aux_buffer(
            num_tokens, num_capture, hidden_size, torch.float32, device
        )
        dist.recv(packed, src=0)
        if local_capture:
            feats = [all_features[lid] for lid in local_capture]
            pack_aux_into_buffer(packed, feats, local_capture, layer_to_slot)

        result = unpack_aux_from_buffer(
            packed, global_capture_layers, layer_to_slot,
            slot_ownership, local_capture,
            pp_rank=rank, pp_size=size,
        )

        for i, lid in enumerate(global_capture_layers):
            assert result[i].shape == (num_tokens, hidden_size), (
                f"Layer {lid}: expected shape ({num_tokens}, {hidden_size}), "
                f"got {result[i].shape}"
            )


def run_test_proxy_coexistence(rank, size, device):
    """Test DSA topk and aux hidden states coexist in PP proxy."""
    from sglang.srt.speculative.glm52_eagle3_pp import (
        GLM52_EAGLE3_AUX_PP_KEY,
        allocate_packed_aux_buffer,
        build_layer_to_slot_map,
        get_local_capture_layers,
        pack_aux_into_buffer,
    )

    num_layers = 10
    hidden_size = 32
    topk_size = 128
    global_capture_layers = [2, 5, 8]
    num_capture = len(global_capture_layers)
    num_tokens = 4

    layer_to_slot = build_layer_to_slot_map(global_capture_layers)
    start, end = get_layer_partition(rank, size, num_layers)
    local_capture = get_local_capture_layers(global_capture_layers, start, end)

    # Use deterministic fill values so rank 1 can verify content
    hidden_states = torch.full((num_tokens, hidden_size), 1.0, device=device)
    residual = torch.full((num_tokens, hidden_size), 2.0, device=device)
    topk_indices = torch.randint(
        0, 100, (num_tokens, topk_size), dtype=torch.int32, device=device
    )
    packed_aux = allocate_packed_aux_buffer(
        num_tokens, num_capture, hidden_size, torch.float32, device
    )

    if local_capture:
        feats = [torch.randn(num_tokens, hidden_size, device=device) for _ in local_capture]
        pack_aux_into_buffer(packed_aux, feats, local_capture, layer_to_slot)

    proxy_tensors = {
        "hidden_states": hidden_states,
        "residual": residual,
        "topk_indices": topk_indices,
        GLM52_EAGLE3_AUX_PP_KEY: packed_aux,
    }

    if rank == 0:
        for key, tensor in proxy_tensors.items():
            dist.send(tensor.contiguous(), dst=1)
    else:
        recv_proxy = {}
        for key in proxy_tensors:
            shape = proxy_tensors[key].shape
            dtype = proxy_tensors[key].dtype
            recv_proxy[key] = torch.zeros(shape, dtype=dtype, device=device)
            dist.recv(recv_proxy[key], src=0)

        assert "hidden_states" in recv_proxy
        assert "residual" in recv_proxy
        assert "topk_indices" in recv_proxy
        assert GLM52_EAGLE3_AUX_PP_KEY in recv_proxy
        assert recv_proxy["topk_indices"].shape == (num_tokens, topk_size)
        assert recv_proxy[GLM52_EAGLE3_AUX_PP_KEY].shape == (
            num_tokens, num_capture, hidden_size
        )
        # Verify shapes match for all received tensors
        assert recv_proxy["hidden_states"].shape == (num_tokens, hidden_size)
        assert recv_proxy["residual"].shape == (num_tokens, hidden_size)
        # Verify content: rank 0 sent known fill values, rank 1 checks
        assert torch.all(recv_proxy["hidden_states"] == 1.0)
        assert torch.all(recv_proxy["residual"] == 2.0)


def run_test_stale_row_safety(rank, size, device):
    """Test that no stale row survives a later smaller batch."""
    from sglang.srt.speculative.glm52_eagle3_pp import (
        allocate_packed_aux_buffer,
        build_layer_to_slot_map,
        build_slot_ownership_map,
        get_local_capture_layers,
        pack_aux_into_buffer,
        unpack_aux_from_buffer,
    )

    num_layers = 10
    hidden_size = 16
    global_capture_layers = [2, 5, 8]
    num_capture = len(global_capture_layers)
    layer_to_slot = build_layer_to_slot_map(global_capture_layers)
    slot_ownership = build_slot_ownership_map(
        global_capture_layers, size, num_layers
    )
    start, end = get_layer_partition(rank, size, num_layers)
    local_capture = get_local_capture_layers(global_capture_layers, start, end)

    # Round 1: 8 tokens
    num_tokens_large = 8
    if rank == 0:
        packed = allocate_packed_aux_buffer(
            num_tokens_large, num_capture, hidden_size, torch.float32, device
        )
        for lid in local_capture:
            slot = layer_to_slot[lid]
            packed[:, slot, :] = float(lid)
        dist.send(packed, dst=1)
    else:
        packed = allocate_packed_aux_buffer(
            num_tokens_large, num_capture, hidden_size, torch.float32, device
        )
        dist.recv(packed, src=0)

    # Round 2: 2 tokens — must not see stale data from round 1
    num_tokens_small = 2
    if rank == 0:
        packed_small = allocate_packed_aux_buffer(
            num_tokens_small, num_capture, hidden_size, torch.float32, device
        )
        for lid in local_capture:
            slot = layer_to_slot[lid]
            packed_small[:, slot, :] = float(lid * 100)
        dist.send(packed_small, dst=1)
    else:
        packed_small = allocate_packed_aux_buffer(
            num_tokens_small, num_capture, hidden_size, torch.float32, device
        )
        dist.recv(packed_small, src=0)
        # Fill local capture layers (as production code would)
        for lid in local_capture:
            slot = layer_to_slot[lid]
            packed_small[:, slot, :] = float(lid * 100)
        # Now verify all capture layers have correct values
        for lid in global_capture_layers:
            slot = layer_to_slot[lid]
            for row in range(num_tokens_small):
                assert packed_small[row, slot, 0].item() == float(lid * 100), (
                    f"Stale data detected: row {row}, slot {slot}, "
                    f"expected {float(lid * 100)}, "
                    f"got {packed_small[row, slot, 0].item()}"
                )


def main():
    args = parse_args()

    if args.device == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    size = dist.get_world_size()

    assert size == 2, f"This test requires exactly 2 ranks, got {size}"

    if rank == 0:
        print(f"Distributed test: backend={args.backend}, device={device}")
        print(f"  world_size={size}, rank={rank}")

    results = {"rank": rank, "backend": args.backend, "device": str(device)}

    try:
        # Test 1: Aux propagation with timing
        if rank == 0:
            print("\n=== Test 1: Aux Hidden State Propagation ===")
        latencies = run_test_aux_propagation(
            rank, size, device, args.iterations, args.warmup
        )
        if latencies:
            results["send_latency_us"] = {
                "min": min(latencies) * 1e6,
                "median": sorted(latencies)[len(latencies) // 2] * 1e6,
                "p95": sorted(latencies)[int(len(latencies) * 0.95)] * 1e6,
                "mean": sum(latencies) / len(latencies) * 1e6,
            }
        dist.barrier()
        if rank == 0:
            print("  PASSED")

        # Test 2: Verify row count
        if rank == 0:
            print("\n=== Test 2: Target Verify Row Count ===")
        run_test_verify_row_count(rank, size, device)
        dist.barrier()
        if rank == 0:
            print("  PASSED")

        # Test 3: Proxy coexistence
        if rank == 0:
            print("\n=== Test 3: DSA topk + Aux Coexistence ===")
        run_test_proxy_coexistence(rank, size, device)
        dist.barrier()
        if rank == 0:
            print("  PASSED")

        # Test 4: Stale row safety
        if rank == 0:
            print("\n=== Test 4: Stale Row Safety ===")
        run_test_stale_row_safety(rank, size, device)
        dist.barrier()
        if rank == 0:
            print("  PASSED")

        if rank == 0:
            print("\n=== All distributed tests PASSED ===")

    except Exception as e:
        print(f"RANK {rank} FAILED: {e}", file=sys.stderr)
        results["error"] = str(e)
        raise
    finally:
        dist.barrier()
        dist.destroy_process_group()

    if args.output_json and rank == 0:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output_json}")


if __name__ == "__main__":
    main()
