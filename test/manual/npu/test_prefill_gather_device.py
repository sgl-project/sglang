"""Manual fused physical-ID prefill gather correctness on an idle NPU."""

import argparse
import gc
import os
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm import (
    create_fused_shm_tensor as create_shm_tensor,
)
from sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm import (
    free_fused_shm as free_shm,
)

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_manager import (
    SparseKVCacheManager,
)
from sglang.srt.model_executor.forward_mode import ForwardMode

# Deliberately noncontiguous, with a physical alias across request rows.
MAP = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [17, 3, 25, 8, 30, 11, 15, 21],
    [3, 23, 9, 19, 6, 28, 12, 14],
    [7, 4, 5, 10, 13, 16, 18, 20],
]


def make_manager(dtype):
    m = SparseKVCacheManager.__new__(SparseKVCacheManager)
    m.start_layer, m.layer_num, m.size = 0, 2, 4
    m.max_context_len = m.logical_max_context_len = 8
    m.actual_request_rows, m.host_token_capacity = 4, 32
    m.head_num, m.head_dim, m.kv_lora_rank, m.qk_rope_head_dim = 1, 576, 512, 64
    m.device = torch.device("npu", torch.npu.current_device())
    m.store_dtype = dtype
    m.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.tensor(MAP, device=m.device)
    )
    m.budget = SimpleNamespace(prefill_workspace_tokens=32)
    m._prefill_workspace_stream = None
    m.prefill_packed = torch.empty((32, 1, 576), dtype=dtype, device=m.device)
    m.prefill_nope = torch.empty((32, 1, 512), dtype=dtype, device=m.device)
    m.prefill_rope = torch.empty((32, 1, 64), dtype=dtype, device=m.device)
    m.host_kv_buffer, m.dev_ptr_list = [], []
    for layer in range(2):
        host, _, ptr = create_shm_tensor(
            shape=(1, 32, 1, 576),
            dtype=dtype,
            device_id=torch.npu.current_device(),
            name=f"prefill_physical_{os.getpid()}_{layer}",
        )
        host[0].copy_(host_values(dtype, layer))
        m.host_kv_buffer.append(host)
        m.dev_ptr_list.append(ptr)
    return m


def host_values(dtype, layer):
    ids = torch.arange(32).view(-1, 1, 1)
    features = torch.arange(576).view(1, 1, -1)
    return (((ids * 17 + features * 3 + layer * 29) % 101 - 50).float() / 512).to(dtype)


def batch_for(m, ids, lengths):
    ids_cpu, lengths_cpu = torch.tensor(ids), torch.tensor(lengths)
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        req_pool_indices_cpu=ids_cpu,
        seq_lens_cpu=lengths_cpu,
        req_pool_indices=ids_cpu.to(m.device),
        seq_lens=lengths_cpu.to(m.device),
    )


def oracle(dtype, layer, ids, lengths):
    values = host_values(dtype, layer)
    return torch.cat(
        [values[torch.tensor(MAP[req][:length])] for req, length in zip(ids, lengths)]
    )


def report_memory(m):
    workspace_bytes = sum(
        t.numel() * t.element_size()
        for t in (m.prefill_packed, m.prefill_nope, m.prefill_rope)
    )
    print(
        f"workspace_bytes={workspace_bytes} allocated={torch.npu.memory_allocated()} reserved={torch.npu.memory_reserved()} peak_allocated={torch.npu.max_memory_allocated()} peak_reserved={torch.npu.max_memory_reserved()}",
        flush=True,
    )


def run(dtype):
    torch.npu.reset_peak_memory_stats()
    m = make_manager(dtype)
    report_memory(m)
    try:
        stream = torch.npu.current_stream(m.device)
        initial_batch = batch_for(m, [1], [2])
        other_stream = torch.npu.Stream(device=m.device)
        assert m.prepare_forward_kv_plan(initial_batch, other_stream) is None
        assert m.prepare_forward_kv_plan(initial_batch, None).stream == stream
        for ids, lengths in [([1, 2], [4, 3]), ([2, 1], [3, 4])]:
            batch = batch_for(m, ids, lengths)
            plan = m.prepare_forward_kv_plan(batch, stream)
            assert plan is not None
            for layer in range(2):
                nope, rope = m.get_forward_kv(layer, batch, stream, plan=plan)
                expected = oracle(dtype, layer, ids, lengths)
                torch.testing.assert_close(
                    nope.cpu(), expected[..., :512], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    rope.cpu(), expected[..., 512:], rtol=0, atol=0
                )
                print(f"PASS dtype={dtype} layer={layer} ids={ids}")
        batch = batch_for(m, [1, 2], [3, 3])
        batch.extend_prefix_lens_cpu = [2, 2]
        batch.extend_seq_lens_cpu = [1, 1]
        batch.out_cache_loc = torch.tensor([25, 9], device=m.device)
        current_cpu = torch.stack(
            (torch.full((1, 576), 2), torch.full((1, 576), 3))
        ).to(dtype)
        m.host_kv_buffer[0][0, 25] = current_cpu[0]
        m.host_kv_buffer[0][0, 9] = current_cpu[1]
        current = current_cpu.to(m.device)
        plan = m.prepare_forward_kv_plan(batch, stream)
        nope, rope = m.get_forward_kv(
            0,
            batch,
            stream,
            plan=plan,
            current=(current, None, torch.ones(2, dtype=torch.bool, device=m.device)),
        )
        expected = torch.cat(
            (
                m.host_kv_buffer[0][0, torch.tensor(MAP[1][:3])],
                m.host_kv_buffer[0][0, torch.tensor(MAP[2][:3])],
            )
        )
        torch.testing.assert_close(
            torch.cat((nope, rope), -1).cpu(), expected, rtol=0, atol=0
        )
        assert plan.current_mapping[2].cpu().tolist() == [True, True]
        print(f"PASS dtype={dtype} direct_current=true", flush=True)
    finally:
        torch.npu.synchronize()
        report_memory(m)
        del m
        gc.collect()
        free_shm(torch.npu.current_device())


def run_large(dtype):
    # Physical IDs are contiguous per request but request order is reversed.
    torch.npu.reset_peak_memory_stats()
    context, requests = 64000, 16
    capacity = (requests + 1) * context
    m = SparseKVCacheManager.__new__(SparseKVCacheManager)
    m.start_layer, m.layer_num, m.size = 0, 2, requests + 1
    m.actual_request_rows = requests + 1
    m.max_context_len = m.logical_max_context_len = context
    m.host_token_capacity = capacity
    m.head_num, m.head_dim, m.kv_lora_rank, m.qk_rope_head_dim = 1, 576, 512, 64
    m.device, m.store_dtype = torch.device("npu", torch.npu.current_device()), dtype
    m.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(capacity, device=m.device).reshape(
            requests + 1, context
        )
    )
    total = requests * context
    m.budget = SimpleNamespace(prefill_workspace_tokens=total)
    m._prefill_workspace_stream = None
    m.prefill_packed = torch.empty((total, 1, 576), dtype=dtype, device=m.device)
    m.prefill_nope = torch.empty((total, 1, 512), dtype=dtype, device=m.device)
    m.prefill_rope = torch.empty((total, 1, 64), dtype=dtype, device=m.device)
    m.host_kv_buffer, m.dev_ptr_list = [], []
    report_memory(m)
    try:
        for layer in range(2):
            host, _, ptr = create_shm_tensor(
                shape=(1, capacity, 1, 576),
                dtype=dtype,
                device_id=torch.npu.current_device(),
                name=f"prefill_large_{os.getpid()}_{layer}",
            )
            for req in range(1, requests + 1):
                host[0, req * context : (req + 1) * context].fill_(req + layer * 20)
            m.host_kv_buffer.append(host)
            m.dev_ptr_list.append(ptr)
        stream = torch.npu.current_stream(m.device)
        for ids in (list(range(1, requests + 1)), list(range(requests, 0, -1))):
            batch = batch_for(m, ids, [context] * requests)
            plan = m.prepare_forward_kv_plan(batch, stream)
            assert plan is not None
            for layer in range(2):
                nope, rope = m.get_forward_kv(layer, batch, stream, plan=plan)
                for idx, req in enumerate(ids):
                    for start in range(idx * context, (idx + 1) * context, 4096):
                        end = min(start + 4096, (idx + 1) * context)
                        assert bool((nope[start:end] == req + layer * 20).all().item())
                        assert bool((rope[start:end] == req + layer * 20).all().item())
                print(
                    f"PASS large dtype={dtype} layer={layer} batch=16 tokens=1024000",
                    flush=True,
                )
        # All logical rows are current. The prepopulated host contains the
        # same post-publication values; compare every D2D output against it.
        ids = list(range(1, requests + 1))
        batch = batch_for(m, ids, [context] * requests)
        batch.extend_prefix_lens_cpu = [0] * requests
        batch.extend_seq_lens_cpu = [context] * requests
        batch.out_cache_loc = torch.arange(context, capacity, device=m.device)
        valid = torch.ones(total, dtype=torch.bool, device=m.device)
        current = torch.empty((total, 1, 576), dtype=dtype, device=m.device)
        for layer in range(2):
            for index, req in enumerate(ids):
                current[index * context : (index + 1) * context].fill_(req + layer * 20)
            plan = m.prepare_forward_kv_plan(batch, stream)
            nope, rope = m.get_forward_kv(
                layer, batch, stream, plan=plan, current=(current, None, valid)
            )
            d2d_rows = int(plan.current_mapping[2].sum().item())
            host_rows = int(plan.current_mapping[3].sum().item())
            assert d2d_rows == total and host_rows == 0
            for index, req in enumerate(ids):
                for start in range(index * context, (index + 1) * context, 4096):
                    end = min(start + 4096, (index + 1) * context)
                    assert bool((nope[start:end] == req + layer * 20).all().item())
                    assert bool((rope[start:end] == req + layer * 20).all().item())
            print(
                f"PASS zero_prefix dtype={dtype} layer={layer} d2d_rows={d2d_rows} host_rows={host_rows} host_kernel_submitted=true",
                flush=True,
            )
            report_memory(m)
    finally:
        torch.npu.synchronize()
        report_memory(m)
        del m
        gc.collect()
        free_shm(torch.npu.current_device())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true")
    options = parser.parse_args()
    for dtype in (torch.bfloat16, torch.float16):
        (run_large if options.large else run)(dtype)
