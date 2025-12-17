"""
Test deterministic custom RS+AG (reduce-scatter + all-gather) behavior with batch size invariance.

This test compares:
1. NCCL all_reduce (same batch size) - may be deterministic
2. NCCL all_reduce (different batch size) - typically NON-deterministic for bfloat16
3. Custom RS+AG (same batch size) - should be DETERMINISTIC
4. Custom RS+AG (different batch size) - should be DETERMINISTIC

Usage:
    # Default TP=8
    python test/test_rs_ag.py
    
    # Custom TP size
    python test/test_rs_ag.py --tp 2
"""

import multiprocessing as mp
import socket
import torch
import torch.distributed as dist


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def custom_rs_ag(inp_flat, world_size, chunk_size, custom_ar, ar_group, device):
    """Custom RS+AG implementation for deterministic all-reduce."""
    output_chunk = torch.empty(chunk_size, dtype=inp_flat.dtype, device=device)
    
    # Reduce-scatter
    if hasattr(custom_ar, '_reduce_scatter_tensor'):
        try:
            custom_ar._reduce_scatter_tensor(output_chunk, inp_flat)
        except:
            input_chunks = [inp_flat[i*chunk_size:(i+1)*chunk_size].clone() for i in range(world_size)]
            torch.distributed.reduce_scatter(output_chunk, input_chunks, group=ar_group)
    else:
        input_chunks = [inp_flat[i*chunk_size:(i+1)*chunk_size].clone() for i in range(world_size)]
        torch.distributed.reduce_scatter(output_chunk, input_chunks, group=ar_group)
    
    # All-gather
    total_size = chunk_size * world_size
    output_flat = torch.empty(total_size, dtype=inp_flat.dtype, device=device)
    output_chunks = [output_flat[i*chunk_size:(i+1)*chunk_size] for i in range(world_size)]
    torch.distributed.all_gather(output_chunks, output_chunk, group=ar_group)
    
    return output_flat


def worker(world_size, rank, port):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        from sglang.srt.distributed.device_communicators.custom_all_reduce import CustomAllreduce
        from torch.distributed import new_group
        
        # Create gloo group for custom AR (needed for RS+AG)
        dist.barrier()
        ar_group = new_group(backend="gloo")
        dist.barrier()
        
        custom_ar = CustomAllreduce(group=ar_group, device=device)
        
        if custom_ar is None or custom_ar.disabled:
            if rank == 0:
                print("✗ Custom AR not available or disabled")
            dist.destroy_process_group()
            return

        num_trials = 10
        BS = 50  # max batch size for varying batch test
        hidden_dim = 16384

        # Different seed per rank - each GPU has DIFFERENT input
        torch.manual_seed(42 + rank)

        # Create fixed inputs for all trials
        base_input = torch.randn(hidden_dim, dtype=torch.bfloat16, device=device)
        base_input_rand = torch.randn(hidden_dim, dtype=torch.bfloat16, device=device)

        dist.barrier()

        # =====================================================================
        # TEST 1: NCCL all_reduce (same batch size) - may be DETERMINISTIC
        # =====================================================================
        if rank == 0:
            print(f"\n{'='*70}")
            print("TEST 1: NCCL all_reduce (same batch size)")
            print(f"{'='*70}")
        dist.barrier()

        results_nccl_same = []
        for trial in range(num_trials):
            inp = base_input.clone()
            
            # Use default NCCL all_reduce
            dist.all_reduce(inp, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            
            checksum = inp.sum().item()
            first_vals = inp[:5].clone()
            results_nccl_same.append((checksum, first_vals))

            if rank == 0:
                print(f"  Trial {trial+1:2d}: sum={checksum:.6f}, first5={first_vals.tolist()}")

        nccl_same_deterministic = True
        if rank == 0:
            ref_sum, ref_vals = results_nccl_same[0]
            for i, (s, vals) in enumerate(results_nccl_same[1:], 1):
                # Check first5 values for determinism (more reliable than sum for bfloat16)
                if not torch.allclose(ref_vals, vals, rtol=1e-3):
                    nccl_same_deterministic = False
                    print(f"  Trial {i+1} DIFFERS! ref_first5={ref_vals.tolist()}, got={vals.tolist()}")

            if nccl_same_deterministic:
                print("  ✓ NCCL ALL_REDUCE (fixed BS): DETERMINISTIC")
            else:
                print("  ✗ NCCL ALL_REDUCE (fixed BS): NON-DETERMINISTIC")

        dist.barrier()

        # =====================================================================
        # TEST 2: NCCL all_reduce (different batch size) - typically NON-DETERMINISTIC
        # [a], [a, x], [a, x, x], ...
        # =====================================================================
        if rank == 0:
            print(f"\n{'='*70}")
            print("TEST 2: NCCL all_reduce (different batch size)")
            print("Batches: [a], [a,x], [a,x,x], ...")
            print(f"{'='*70}")
        dist.barrier()

        results_nccl_variant = []
        for bs in range(1, BS + 1):
            # Construct batch: [a, x, x, ...]
            batch = torch.stack([base_input] + [base_input_rand] * (bs - 1), dim=0)
            batch_flat = batch.view(-1)

            # Use default NCCL all_reduce
            dist.all_reduce(batch_flat, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()

            # Reshape back
            batch_out = batch_flat.view(bs, hidden_dim)

            # Only compare output corresponding to first request
            out_first_req = batch_out[0].clone()
            checksum = out_first_req.sum().item()
            first_vals = out_first_req[:5].clone()
            results_nccl_variant.append((bs, checksum, first_vals))

            if rank == 0:
                print(f"  Batch size {bs:2d}: sum={checksum:.6f}, first5={first_vals.tolist()}")

        nccl_variant_deterministic = True
        if rank == 0:
            _, ref_sum, ref_vals = results_nccl_variant[0]
            for bs, s, vals in results_nccl_variant[1:]:
                # Check first5 values for determinism (more reliable than sum for bfloat16)
                if not torch.allclose(ref_vals, vals, rtol=1e-3):
                    nccl_variant_deterministic = False

            if nccl_variant_deterministic:
                print("  ✓ NCCL ALL_REDUCE (variant BS): DETERMINISTIC")
            else:
                print("  ✗ NCCL ALL_REDUCE (variant BS): NON-DETERMINISTIC")

        dist.barrier()

        # =====================================================================
        # TEST 3: Custom RS+AG (same batch size) - should be DETERMINISTIC
        # =====================================================================
        if rank == 0:
            print(f"\n{'='*70}")
            print("TEST 3: Custom RS+AG (same batch size)")
            print(f"{'='*70}")
        dist.barrier()

        # Compute padded size for RS+AG
        original_size = hidden_dim
        total_size = hidden_dim
        if total_size % world_size != 0:
            total_size = ((total_size + world_size - 1) // world_size) * world_size
        chunk_size = total_size // world_size

        results_rsag_same = []
        for trial in range(num_trials):
            inp = base_input.clone()
            
            # Pad if needed
            if total_size != original_size:
                inp_flat = torch.cat([inp, torch.zeros(total_size - original_size, dtype=inp.dtype, device=device)])
            else:
                inp_flat = inp.clone()
            
            output_flat = custom_rs_ag(inp_flat, world_size, chunk_size, custom_ar, ar_group, device)
            
            # Remove padding
            if total_size != original_size:
                output_flat = output_flat[:original_size]
            
            torch.cuda.synchronize()
            
            checksum = output_flat.sum().item()
            first_vals = output_flat[:5].clone()
            results_rsag_same.append((checksum, first_vals))

            if rank == 0:
                print(f"  Trial {trial+1:2d}: sum={checksum:.6f}, first5={first_vals.tolist()}")

        rsag_same_deterministic = True
        if rank == 0:
            ref_sum, ref_vals = results_rsag_same[0]
            for i, (s, vals) in enumerate(results_rsag_same[1:], 1):
                # Check first5 values for determinism (more reliable than sum for bfloat16)
                if not torch.allclose(ref_vals, vals, rtol=1e-3):
                    rsag_same_deterministic = False
                    print(f"  Trial {i+1} DIFFERS! ref_first5={ref_vals.tolist()}, got={vals.tolist()}")

            if rsag_same_deterministic:
                print("  ✓ CUSTOM RS+AG (fixed BS): DETERMINISTIC (as expected)")
            else:
                print("  ✗ CUSTOM RS+AG (fixed BS): NON-DETERMINISTIC")

        dist.barrier()

        # =====================================================================
        # TEST 4: Custom RS+AG (different batch size) - should be DETERMINISTIC
        # [a], [a, x], [a, x, x], ...
        # =====================================================================
        if rank == 0:
            print(f"\n{'='*70}")
            print("TEST 4: Custom RS+AG (different batch size)")
            print("Batches: [a], [a,x], [a,x,x], ...")
            print(f"{'='*70}")
        dist.barrier()

        results_rsag_variant = []
        for bs in range(1, BS + 1):
            # Construct batch: [a, x, x, ...]
            batch = torch.stack([base_input] + [base_input_rand] * (bs - 1), dim=0)
            batch_flat = batch.view(-1)
            
            batch_size = batch_flat.numel()
            # Pad if needed
            if batch_size % world_size != 0:
                padded_size = ((batch_size + world_size - 1) // world_size) * world_size
                batch_flat = torch.cat([batch_flat, torch.zeros(padded_size - batch_size, dtype=batch.dtype, device=device)])
            else:
                padded_size = batch_size
            
            batch_chunk_size = padded_size // world_size
            output_flat = custom_rs_ag(batch_flat, world_size, batch_chunk_size, custom_ar, ar_group, device)
            
            # Remove padding
            if padded_size != batch_size:
                output_flat = output_flat[:batch_size]
            
            torch.cuda.synchronize()

            # Reshape back
            batch_out = output_flat.view(bs, hidden_dim)

            # Only compare output corresponding to first request
            out_first_req = batch_out[0].clone()
            checksum = out_first_req.sum().item()
            first_vals = out_first_req[:5].clone()
            results_rsag_variant.append((bs, checksum, first_vals))

            if rank == 0:
                print(f"  Batch size {bs:2d}: sum={checksum:.6f}, first5={first_vals.tolist()}")

        rsag_variant_deterministic = True
        if rank == 0:
            _, ref_sum, ref_vals = results_rsag_variant[0]
            for bs, s, vals in results_rsag_variant[1:]:
                # Check first5 values for determinism (more reliable than sum for bfloat16)
                if not torch.allclose(ref_vals, vals, rtol=1e-3):
                    rsag_variant_deterministic = False

            if rsag_variant_deterministic:
                print("  ✓ CUSTOM RS+AG (variant BS): DETERMINISTIC (as expected)")
            else:
                print("  ✗ CUSTOM RS+AG (variant BS): NON-DETERMINISTIC")

        # =====================================================================
        # SUMMARY
        # =====================================================================
        if rank == 0:
            print(f"\n{'='*70}")
            print("SUMMARY")
            print(f"{'='*70}")
            print(f"  NCCL all_reduce (fixed BS):   {'DETERMINISTIC' if nccl_same_deterministic else 'NON-DETERMINISTIC'}")
            print(f"  NCCL all_reduce (variant BS): {'DETERMINISTIC' if nccl_variant_deterministic else 'NON-DETERMINISTIC'}")
            print(f"  Custom RS+AG (fixed BS):      {'DETERMINISTIC' if rsag_same_deterministic else 'NON-DETERMINISTIC'}")
            print(f"  Custom RS+AG (variant BS):    {'DETERMINISTIC' if rsag_variant_deterministic else 'NON-DETERMINISTIC'}")
            print(f"{'='*70}")

        dist.barrier()
        dist.destroy_process_group()

    except Exception as e:
        if rank == 0:
            print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        dist.destroy_process_group()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallelism size (default: 8)")
    args = parser.parse_args()
    
    world_size = args.tp
    available_gpus = torch.cuda.device_count()

    print("=" * 70)
    print("Custom RS+AG Determinism Test")
    print("=" * 70)
    print(f"Available GPUs: {available_gpus}")
    print(f"Requested TP size: {world_size}")

    if available_gpus < world_size:
        print(f"WARNING: Only {available_gpus} GPUs available, using {available_gpus} instead")
        world_size = available_gpus

    if world_size < 2:
        print("ERROR: Need at least 2 GPUs for this test")
        return

    print(f"Using TP size: {world_size}")

    mp.set_start_method("spawn", force=True)
    port = get_open_port()

    procs = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(world_size, rank, port))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
