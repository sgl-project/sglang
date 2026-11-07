"""Check same-stream SFA consumption before manager workspace reuse.

Run in an isolated idle-NPU process with matching torch_npu and kernel packages:
  python test/manual/npu/test_prefill_workspace_sfa_order.py
No layer/forward synchronization is inserted inside the candidate sequence.
This is a correctness check, not a serving performance benchmark.
"""

import gc
from types import SimpleNamespace

import torch
import torch_npu
from sgl_kernel_npu.sparsity_driven_kv_offload import free_shm
from test_prefill_gather_device import make_manager, oracle

from sglang.srt.model_executor.forward_mode import ForwardMode


def deterministic(shape, dtype, seed):
    count = 1
    for dimension in shape:
        count *= dimension
    value = torch.arange(count, dtype=torch.int64).reshape(shape)
    return (((value * 17 + seed * 29) % 101 - 50).float() / 512).to(dtype)


def run_sfa(q, qr, key, rope, topk, actual):
    return torch_npu.npu_sparse_flash_attention(
        query=q,
        key=key,
        value=key,
        query_rope=qr,
        key_rope=rope,
        sparse_indices=topk,
        scale_value=576**-0.5,
        actual_seq_lengths_query=actual,
        actual_seq_lengths_kv=actual,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="TND",
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=False,
    )[0]


def execute(dtype):
    manager = None
    try:
        manager = make_manager(dtype)
        device = manager.device
        forwards = [([1, 2], [4, 3]), ([2, 1], [3, 4])]
        total = 7
        q = deterministic((total, 8, 512), dtype, 101).to(device).contiguous()
        qr = deterministic((total, 8, 64), dtype, 211).to(device).contiguous()
        # Rebuild per-forward top-k because ragged request lengths change order.
        fixtures = []
        for ids, lengths in forwards:
            topk_cpu = torch.full((total, 1, 2048), -1, dtype=torch.int32)
            offset = 0
            for length in lengths:
                topk_cpu[offset : offset + length, 0, :length] = torch.arange(length)
                offset += length
            topk = topk_cpu.to(device).contiguous()
            actual = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(
                0, dtype=torch.int32
            )
            ids_cpu = torch.tensor(ids, dtype=torch.int64)
            lengths_cpu = torch.tensor(lengths, dtype=torch.int64)
            batch = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                req_pool_indices_cpu=ids_cpu,
                seq_lens_cpu=lengths_cpu,
                req_pool_indices=ids_cpu.to(device),
                seq_lens=lengths_cpu.to(device),
            )
            # Host rows are already populated with the values representing
            # completed publication. Current payloads use those same values.
            batch.extend_prefix_lens_cpu = [length - 1 for length in lengths]
            batch.extend_seq_lens_cpu = [1, 1]
            from test_prefill_gather_device import MAP

            batch.out_cache_loc = torch.tensor(
                [MAP[req][length - 1] for req, length in zip(ids, lengths)],
                dtype=torch.long,
                device=device,
            )
            oracle_by_layer = []
            current_by_layer = []
            for layer in range(2):
                packed = oracle(dtype, layer, ids, lengths)
                kn = packed[..., :512].contiguous().to(device)
                kr = packed[..., 512:].contiguous().to(device)
                oracle_by_layer.append((kn, kr))
                current_by_layer.append(
                    packed[[lengths[0] - 1, total - 1]].contiguous().to(device)
                )
            fixtures.append((batch, topk, actual, oracle_by_layer, current_by_layer))

        # Queue independent reference SFA first, then the complete candidate
        # sequence. No synchronization or CPU result read occurs between any
        # candidate gather and its following layer or forward.
        reference = []
        for _, topk, actual, oracle_layers, _ in fixtures:
            for kn, kr in oracle_layers:
                reference.append(run_sfa(q, qr, kn, kr, topk, actual))
        stream = torch.npu.current_stream(device)
        got, addresses, mappings = [], [], []
        valid = torch.ones(2, dtype=torch.bool, device=device)
        for batch, topk, actual, _, current_layers in fixtures:
            plan = manager.prepare_forward_kv_plan(batch, stream)
            for layer in range(2):
                kn, kr = manager.get_forward_kv(
                    layer,
                    batch,
                    stream,
                    plan=plan,
                    current=(current_layers[layer], None, valid),
                )
                addresses.append((kn.data_ptr(), kr.data_ptr()))
                mappings.append(plan.current_mapping)
                got.append(run_sfa(q, qr, kn, kr, topk, actual))
        torch.npu.synchronize()
        for mapping in mappings:
            assert int(mapping[2].sum().item()) == 2
            assert int(mapping[3].sum().item()) == total - 2
        if any(address != addresses[0] for address in addresses[1:]):
            raise AssertionError("workspace addresses changed across layers/forwards")
        if torch.equal(reference[0], reference[1]) or torch.equal(
            reference[0], reference[2]
        ):
            raise AssertionError(
                "reference fixture is not sensitive to layer/forward overwrite"
            )
        for index, (expected, actual) in enumerate(zip(reference, got)):
            torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
            diff = (actual.float() - expected.float()).abs().max().item()
            print(
                f"PASS dtype={dtype} output={index} "
                f"bitwise_equal={torch.equal(actual, expected)} max_abs_diff={diff} "
                "same_workspace=true",
                flush=True,
            )
    finally:
        torch.npu.synchronize()
        del manager
        gc.collect()
        free_shm(torch.npu.current_device())


if __name__ == "__main__":
    for dtype in (torch.bfloat16, torch.float16):
        execute(dtype)
