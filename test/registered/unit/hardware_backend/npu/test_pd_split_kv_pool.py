import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="base-a", runner_config="cpu")


def _make_pool():
    pool = NPUMLATokenToKVPool.__new__(NPUMLATokenToKVPool)
    pool.start_layer = 4
    pool.layer_num = 2
    pool.device = "cpu"
    pool.page_size = 128
    pool.cpu_offloading_chunk_size = 2
    pool.dcp_size = 1
    pool.is_draft_worker = False
    pool.use_combined_kv_buffer = False
    pool.dsa_kv_cache_store_c8 = False
    pool.index_head_dim = 2
    pool.indexer_layer_ids = (4, 5)
    pool.indexer_layer_id_to_slot = {4: 0, 5: 1}
    pool.k_buffer = torch.zeros(2, 2, 128, 1, 512, dtype=torch.bfloat16)
    pool.v_buffer = torch.zeros(2, 2, 128, 1, 64, dtype=torch.bfloat16)
    pool.index_k_buffer = torch.zeros(2, 2, 128, 1, 2, dtype=torch.bfloat16)
    pool.index_k_scale_buffer = None
    return pool


class TestPDSplitKVPool(CustomTestCase):
    def test_nonarch35_mla_builder_compacts_eligible_target_only(self):
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        owner_ids = (0, 1, 2, *range(6, 78, 4))
        hf_config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            index_topk=2048,
            index_head_dim=128,
            indexer_types=[
                "full" if layer_id in owner_ids else "shared" for layer_id in range(78)
            ],
        )

        def build_pool(*, start_layer, end_layer, num_layers, is_draft, eligible):
            configurator = object.__new__(KVCacheConfigurator)
            configurator.is_draft_worker = is_draft
            configurator.layer_info = SimpleNamespace(
                start_layer=start_layer,
                end_layer=end_layer,
                num_effective_layers=num_layers,
            )
            configurator.model_config = SimpleNamespace(
                hf_config=hf_config,
                index_head_dim=128,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
            )
            configurator.kv_cache_dtype = torch.bfloat16
            configurator.device = "cpu"
            configurator.server_args = SimpleNamespace(indexer_kv_cache_dtype=None)
            configurator.spec_algorithm = SimpleNamespace(is_dspark=lambda: False)
            configurator.modelslim_cache_layout = None
            with (
                patch(
                    "sglang.srt.layers.cp.utils.get_glm_dsa_cp_layer_shard_info",
                    return_value=(None, 1),
                ),
                patch(
                    "sglang.srt.mem_cache.kv_cache_configurator.get_parallel",
                    return_value=SimpleNamespace(attn_dcp_size=1),
                ),
                patch(
                    "sglang.srt.mem_cache.kv_cache_configurator.get_schedule",
                    return_value=SimpleNamespace(page_size=64),
                ),
                patch(
                    "sglang.srt.mem_cache.kv_cache_configurator.get_exec",
                    return_value=SimpleNamespace(
                        features=SimpleNamespace(enable_memory_saver=False)
                    ),
                ),
                patch(
                    "sglang.srt.hardware_backend.npu.memory_pool_npu.get_parallel",
                    return_value=SimpleNamespace(attn_dcp_size=1, attn_dcp_rank=0),
                ),
                patch(
                    "sglang.srt.hardware_backend.npu.utils.is_npu_arch35",
                    return_value=False,
                ),
                patch(
                    "sglang.srt.mem_cache.kv_cache_configurator._should_elide_dsa_index_k",
                    return_value=eligible,
                ),
            ):
                return configurator._build_ascend_mla_kv_pool(
                    max_total_num_tokens=128, max_running_requests=1, is_dsa_model=True
                )

        target = build_pool(
            start_layer=0,
            end_layer=78,
            num_layers=78,
            is_draft=False,
            eligible=True,
        )
        self.assertEqual(target.index_k_buffer.shape[0], 21)
        self.assertEqual(target.index_k_buffer.dtype, torch.bfloat16)
        self.assertEqual(target.index_k_buffer.nbytes, 21 * 3 * 64 * 128 * 2)
        self.assertEqual(target.get_index_k_buffer(74).shape[-1], 128)
        with self.assertRaises(KeyError):
            target.get_index_k_buffer(3)

        full_target = build_pool(
            start_layer=0,
            end_layer=78,
            num_layers=78,
            is_draft=False,
            eligible=False,
        )
        self.assertEqual(full_target.index_k_buffer.shape[0], 78)

        draft = build_pool(
            start_layer=78,
            end_layer=79,
            num_layers=1,
            is_draft=True,
            eligible=False,
        )
        self.assertEqual(draft.index_k_buffer.shape[0], 1)
        self.assertEqual(draft.index_k_buffer.nbytes, 1 * 3 * 64 * 128 * 2)
        self.assertEqual(draft.get_index_k_buffer(78).shape[-1], 128)

    def test_pool_allocation_follows_fused_role(self):
        from sglang.srt.environ import envs

        prefix = "sglang.srt.hardware_backend.npu.memory_pool_npu."
        api = ModuleType(
            "sgl_kernel_npu.operators.attention.fused_scatter_copy_sparse_flash_attention_split"
        )
        for fused, mode, draft, keeps_full_kv in (
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
                patch.object(
                    api, "validate_pd_split_copy_sfa_layout", create=True
                ) as validate,
                patch.dict(sys.modules, {api.__name__: api}),
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
                self.assertEqual(pool.k_buffer is not None, keeps_full_kv)
                self.assertEqual(pool.v_buffer is not None, keeps_full_kv)
                if fused and mode == "decode" and not draft:
                    validate.assert_called_once_with(512, 64, 128, torch.bfloat16)
                else:
                    validate.assert_not_called()

    def test_full_kv_export_uses_upstream_layout(self):
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
        decode.pd_host_pool = SimpleNamespace(buffers=(k, v))
        decode.k_buffer = decode.v_buffer = None

        ptrs, lens, page_bytes = decode.get_contiguous_buf_infos()
        self.assertEqual(
            ptrs, [t.data_ptr() for t in k + v + list(decode.index_k_buffer)]
        )
        self.assertEqual((lens, page_bytes), prefill.get_contiguous_buf_infos()[1:])
        self.assertEqual(page_bytes, [131072] * 2 + [16384] * 2 + [512] * 2)
        self.assertEqual(prefill.get_kv_layer_ids(), [4, 5] * 3)
        self.assertEqual(decode.get_kv_layer_ids(), prefill.get_kv_layer_ids())

    def test_packed_move_and_retract_preserve_payload_and_local_scales(self):
        pool = _make_pool()
        pool.dsa_kv_cache_store_c8 = True
        pool.size = 4
        pool.indexer_layer_ids = (5,)
        pool.indexer_layer_id_to_slot = {5: 0}
        key = torch.arange(8, dtype=torch.uint8).reshape(1, 1, 4, 1, 2)
        scale = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1)
        pool.index_k_buffer = key.view(torch.float8_e4m3fn)
        pool.index_k_scale_buffer = scale
        payloads = [
            torch.arange(4 * 656, dtype=torch.int8).reshape(1, 4, 1, 656) + layer
            for layer in range(2)
        ]

        def copy_rows(sources, destinations, src, dst, valid):
            (source,), (destination,) = sources, destinations
            destination.reshape(-1, 656)[dst[valid]] = source.reshape(-1, 656)[
                src[valid]
            ]

        pool.pd_host_pool = SimpleNamespace(
            rank=0, buffers=(payloads,), copy_rows=copy_rows, publish=Mock()
        )
        pool.invalidate_packed_request = Mock()
        source, target = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])
        state = [p.view(4, -1) for p in (*payloads, key, scale)]
        original = [s.clone() for s in state]
        with patch.object(
            torch, "npu", SimpleNamespace(synchronize=Mock()), create=True
        ):
            saved = pool.get_cpu_copy(source)
            self.assertEqual([len(chunks[0]) for chunks in saved], [1, 3])
            pool.move_kv_cache(target, source)
            for actual, before in zip(state, original, strict=True):
                torch.testing.assert_close(actual[target], before[source])
                actual.zero_()
            pool.load_cpu_copy(saved, source, req_pool_index=7)
            for actual, before in zip(state, original, strict=True):
                torch.testing.assert_close(actual[source], before[source])
        pool.invalidate_packed_request.assert_called_once_with(7)
        self.assertEqual(pool.pd_host_pool.publish.call_count, 2)

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
        offload_rows = (src, dst, positions, valid)
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
        )
        for rank, value in ((0, 3), (1, 99)):
            manager_state = SimpleNamespace(
                start_layer=0,
                host_pool=Pool(rank),
                host_k_buffer=[shared[0]],
                host_v_buffer=[shared[1]],
            )
            k = torch.full((1, 1, 2), value, dtype=shared[0].dtype)
            k_rope = torch.full((1, 1, 1), value + 1, dtype=shared[1].dtype)
            result = ns["offload_v2"](
                manager_state,
                k,
                k_rope,
                SimpleNamespace(layer_id=0),
                batch,
                offload_rows=offload_rows,
            )
            torch.testing.assert_close(result[0], torch.cat((k, k_rope), dim=-1))
            self.assertIs(result[1], positions)
            self.assertIs(result[2], valid)
        self.assertEqual(events, [0, 1])
        self.assertTrue(torch.all(shared[0][1] == 3))
        self.assertTrue(torch.all(shared[1][1] == 4))

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
