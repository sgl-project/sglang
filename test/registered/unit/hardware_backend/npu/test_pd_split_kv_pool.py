import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="base-a", runner_config="cpu")


def _make_pool():
    pool = NPUMLATokenToKVPool.__new__(NPUMLATokenToKVPool)
    pool.start_layer = 0
    pool.layer_num = 2
    pool.device = "cpu"
    pool.page_size = 128
    pool.cpu_offloading_chunk_size = 2
    pool.dcp_size = 1
    pool.is_draft_worker = False
    pool.use_combined_kv_buffer = False
    pool.dsa_kv_cache_store_fp8 = False
    pool.index_head_dim = 2
    pool.indexer_layer_id_to_slot = {0: 0, 1: 1}
    pool.k_buffer = torch.zeros(2, 2, 128, 1, 512, dtype=torch.bfloat16)
    pool.v_buffer = torch.zeros(2, 2, 128, 1, 64, dtype=torch.bfloat16)
    pool.index_k_buffer = torch.zeros(2, 2, 128, 1, 2, dtype=torch.bfloat16)
    pool.index_k_scale_buffer = None
    return pool


class TestPDSplitKVPool(CustomTestCase):
    def test_pool_allocation_follows_fused_role(self):
        from sglang.srt.environ import envs

        prefix = "sglang.srt.hardware_backend.npu.memory_pool_npu."
        for fused, mode, draft, native in (
            (True, "decode", False, False),
            (True, "null", False, False),
            (True, "prefill", False, True),
            (True, "decode", True, True),
            (False, "prefill", False, False),
            (False, "decode", True, False),
        ):
            with (
                self.subTest(fused=fused, mode=mode, draft=draft),
                envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.override(True),
                envs.SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA.override(fused),
                get_context().override_server_args(disaggregation_mode=mode),
                get_parallel().override(attn_dcp_rank=0),
                patch(prefix + "get_bool_env_var", return_value=False),
                patch.object(NPUMLATokenToKVPool, "_finalize_allocation_log"),
            ):
                pool = NPUMLATokenToKVPool(
                    size=128,
                    page_size=128,
                    dtype=torch.bfloat16,
                    kv_lora_rank=512,
                    qk_rope_head_dim=64,
                    layer_num=1,
                    device="cpu",
                    enable_memory_saver=False,
                    index_head_dim=128,
                    is_draft_worker=draft,
                )
                self.assertEqual(pool.k_buffer is not None, native)
                self.assertEqual(pool.v_buffer is not None, native)

    def test_native_combined_export_uses_upstream_layout(self):
        pool = _make_pool()
        pool.use_combined_kv_buffer = True
        pool.kv_buffer = torch.zeros(2, 2, 128, 1, 576, dtype=torch.bfloat16)

        ptrs, lengths, page_bytes = pool.get_contiguous_buf_infos()

        self.assertEqual(ptrs, [buffer.data_ptr() for buffer in pool.kv_buffer])
        self.assertEqual(lengths, [buffer.nbytes for buffer in pool.kv_buffer])
        self.assertEqual(page_bytes, [147456, 147456])

    def test_export_matches_prefill_page_order(self):
        prefill, decode = _make_pool(), _make_pool()
        k, v = list(decode.k_buffer), list(decode.v_buffer)
        decode.sparse_kv_manager = SimpleNamespace(
            host_pool=SimpleNamespace(buffers=(k, v))
        )
        decode.k_buffer = decode.v_buffer = None

        ptrs, lens, page_bytes = decode.get_contiguous_buf_infos()
        self.assertEqual(
            ptrs, [t.data_ptr() for t in k + v + list(decode.index_k_buffer)]
        )
        self.assertEqual((lens, page_bytes), prefill.get_contiguous_buf_infos()[1:])
        self.assertEqual(page_bytes, [131072] * 2 + [16384] * 2 + [512] * 2)

    def test_host_group_partitions_by_shared_memory(self):
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.host_pool import (
            _host_group,
        )

        singleton = SimpleNamespace(world_size=1)
        self.assertIs(_host_group(singleton), singleton)

        parent = SimpleNamespace(
            world_size=4,
            rank_in_group=0,
            ranks=[2, 4, 6, 8],
            cpu_group=object(),
            device_group=object(),
        )
        path = "sglang.srt.distributed.parallel_state.in_the_same_node_as"
        with (
            patch(path, return_value=[True] * 4),
            patch.object(torch.distributed, "new_group") as new_group,
        ):
            self.assertIs(_host_group(parent), parent)
            new_group.assert_not_called()

        masks = [[True, False, True, False], [False, True, False, True]]
        for rank in range(4):
            parent.rank_in_group = rank
            with (
                patch(path, side_effect=masks),
                patch.object(torch.distributed, "get_backend", return_value="hccl"),
                patch.object(
                    torch.distributed, "new_group", side_effect=["cpu", "device"]
                ) as create,
                patch.object(torch.distributed, "broadcast") as broadcast,
                patch.object(
                    torch.distributed, "broadcast_object_list"
                ) as broadcast_object,
            ):
                group = _host_group(parent)
                expected = [2, 6] if rank % 2 == 0 else [4, 8]
                self.assertEqual(group.ranks, expected)
                self.assertEqual(group.rank_in_group, rank // 2)
                self.assertEqual(create.call_args_list[0].kwargs["ranks"], expected)
                self.assertEqual(create.call_args_list[1].kwargs["backend"], "hccl")
                group.broadcast(torch.zeros(1), src=0)
                group.broadcast_object(None, src=0)
                self.assertEqual(broadcast.call_args.kwargs["src"], expected[0])
                self.assertEqual(broadcast_object.call_args.kwargs["src"], expected[0])

    def test_offload_writes_once_then_publishes_on_each_rank(self):
        """A peer must not overwrite the owner's published host KV."""
        path = Path(__file__).resolve().parents[5] / (
            "python/sglang/srt/hardware_backend/npu/"
            "sparsity_driven_kv_offload/fused_manager.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        manager = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "FusedSparseKVCacheManager"
        )
        method = next(
            n
            for n in manager.body
            if isinstance(n, ast.FunctionDef) and n.name == "offload_v2"
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                method,
            ],
            type_ignores=[],
        )
        ns = {"torch": torch}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
        src, dst = torch.tensor([0]), torch.tensor([1])
        valid, positions = torch.tensor([True]), torch.tensor([7])
        current = ("payload", "rows", "valid")
        kernels = ModuleType("sglang.kernels.ops.kvcache.npu_sparse_kv")
        kernels.prepare_sparse_kv_offload = lambda *args, **kwargs: (
            src,
            dst,
            positions,
            valid,
            current,
        )
        mtp = ModuleType(
            "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_lim"
        )
        mtp.sparse_query_width = lambda batch: 1
        shared = [torch.zeros(2, 1, w) for w in (2, 1)]
        events = []

        class Pool:
            def __init__(self, rank):
                self.rank = rank

            def copy_rows(self, sources, destinations, si, di, mask):
                for source, destination in zip(sources, destinations):
                    destination[di[mask]] = source[si[mask]]

            def publish(self):
                events.append(self.rank)

        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: False),
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([1]),
            positions=positions,
            out_cache_loc=dst,
        )
        with patch.dict(sys.modules, {kernels.__name__: kernels, mtp.__name__: mtp}):
            for rank, value in ((0, 3), (1, 99)):
                m = SimpleNamespace(
                    start_layer=0,
                    size=1,
                    logical_max_context_len=2,
                    host_token_capacity=2,
                    host_pool=Pool(rank),
                    host_k_buffer=[shared[0]],
                    host_v_buffer=[shared[1]],
                )
                result = ns["offload_v2"](
                    m,
                    torch.full((1, 1, 2), value, dtype=shared[0].dtype),
                    torch.full((1, 1, 1), value, dtype=shared[1].dtype),
                    SimpleNamespace(layer_id=0),
                    batch,
                )
                self.assertIs(result[0], current)
                self.assertIs(result[1], positions)
                self.assertIs(result[2], valid)
        self.assertEqual(events, [0, 1])
        for tensor in shared:
            self.assertTrue(torch.all(tensor[1] == 3))

    def test_shared_host_allocation_and_copy(self):
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.host_pool import (
            HostKVPool,
        )

        messages = []
        events = []

        class Group:
            world_size = 2

            def __init__(self, rank):
                self.rank_in_group = rank
                self.read_index = 0

            def broadcast_object(self, value):
                if self.rank_in_group == 0:
                    messages.append(value)
                    return value
                result = messages[self.read_index]
                self.read_index += 1
                return result

            def barrier(self):
                pass

            def broadcast(self, tensor, src):
                events.append((self.rank_in_group, src))

        class OffloadConfig:
            pass

        allocations = []

        def allocate(shape, dtype, pin_memory):
            result = torch.zeros(shape, dtype=dtype)
            allocations.append(result)
            return result

        def sparse_copy(src, dst, lengths, count, device):
            for source, destination in zip(src, dst):
                destination.reshape(-1, destination.shape[-2], destination.shape[-1])[
                    lengths[1][lengths[2]]
                ] = source.reshape(-1, source.shape[-2], source.shape[-1])[
                    lengths[0][lengths[2]]
                ]
            return 0

        offload = SimpleNamespace(
            OffloadConfig=OffloadConfig,
            Scene=SimpleNamespace(SHARED=object()),
            initialize=lambda config: 0,
            empty=allocate,
            sparse_copy=sparse_copy,
        )
        package = ModuleType("memfabric_hybrid")
        package.offload = offload
        row_module = ModuleType(
            "sgl_kernel_npu.sparsity_driven_kv_offload.row_metadata"
        )
        row_module.copy_rows_metadata = lambda s, d, si, di, valid: (
            s,
            d,
            (si, di, valid),
            None,
        )
        with (
            patch(
                "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.host_pool._host_group",
                side_effect=lambda group: group,
            ),
            patch.dict(
                sys.modules,
                {"memfabric_hybrid": package, row_module.__name__: row_module},
            ),
            patch.object(
                torch, "npu", SimpleNamespace(current_device=lambda: 0), create=True
            ),
        ):
            for page_size, head_num, widths, dtype in (
                (128, 1, (512, 64), torch.bfloat16),
                (4, 2, (3, 5), torch.float16),
            ):
                messages.clear()
                events.clear()
                allocations.clear()
                kwargs = dict(
                    tokens=page_size * 2,
                    layers=1,
                    page_size=page_size,
                    head_num=head_num,
                    widths=widths,
                    dtype=dtype,
                    device="cpu",
                )
                owner = HostKVPool(group=Group(0), **kwargs)
                peer = HostKVPool(group=Group(1), **kwargs)
                self.assertEqual(len(allocations), len(widths))
                self.assertEqual(
                    [c[0].data_ptr() for c in owner.buffers],
                    [c[0].data_ptr() for c in peer.buffers],
                )
                for component, width in zip(owner.buffers, widths):
                    self.assertEqual(
                        tuple(component[0].shape),
                        (2, page_size, head_num, width),
                    )
                payload = [
                    torch.full((1, head_num, w), 17, dtype=dtype) for w in widths
                ]
                owner.copy_rows(
                    payload,
                    [c[0] for c in owner.buffers],
                    torch.tensor([0]),
                    torch.tensor([3]),
                    torch.tensor([True]),
                )
                owner.publish()
                peer.publish()
                result = [torch.zeros((1, head_num, w), dtype=dtype) for w in widths]
                peer.copy_rows(
                    [c[0] for c in peer.buffers],
                    result,
                    torch.tensor([3]),
                    torch.tensor([0]),
                    torch.tensor([True]),
                )
                self.assertEqual(events, [(0, 0), (1, 0)])
                for tensor in result:
                    self.assertTrue(torch.all(tensor == 17))


if __name__ == "__main__":
    unittest.main()
