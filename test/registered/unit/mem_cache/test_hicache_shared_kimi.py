"""Kimi shared MLA construction keeps KDA state private and rolls back failures."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.pool_host import base as host_base
from sglang.srt.mem_cache.pool_host import mamba as host_mamba
from sglang.srt.mem_cache.pool_host.host_pool_decl import make_kv_pool_decl
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestSharedKimiBuilder(CustomTestCase):
    def _build(self, *, failure=None, shared=True):
        actions = []
        group = object()
        context = SimpleNamespace(
            is_shared=shared,
            is_dummy_rank=False,
            process_group=group,
        )

        def agree(error, *, operation):
            actions.append(("agree", operation, error is not None))
            if error is not None or failure == "peer_" + operation:
                raise RuntimeError("joint construction failure")

        context.synchronize_error = mock.Mock(side_effect=agree)

        def host(name, *, is_shared):
            return SimpleNamespace(
                allocator=SimpleNamespace(is_shared=is_shared),
                layer_num=2 if name == "kv" else 6,
                layout="page_first_direct",
                page_size=64 if name == "kv" else 1,
                device="cpu",
                size=1024,
                logical_size=1024,
                can_use_write_back_jit=False,
                destroy=mock.Mock(
                    side_effect=lambda: actions.append(("destroy", name))
                ),
            )

        kv_host = host("kv", is_shared=shared)
        state_host = host("state", is_shared=False)
        kv_pool = SimpleNamespace(layer_num=2, page_size=64)
        state_pool = SimpleNamespace(layer_num=6, size=8)
        params = SimpleNamespace(
            page_size=64,
            mtp_draft_device_pools=(),
            token_to_kv_pool_allocator=object(),
            tp_cache_group=group,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
            req_to_token_pool=SimpleNamespace(
                mamba_allocator=SimpleNamespace(alloc=mock.Mock(), free=mock.Mock())
            ),
        )
        memory = SimpleNamespace(
            hicache_size=0,
            hicache_ratio=2,
            hicache_mem_layout="page_first_direct",
            hicache_write_policy="write_through",
            hicache_io_backend="direct",
            hicache_host_memory_mode="cache",
        )
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(assembler, "get_memory", return_value=memory)
            )
            stack.enter_context(
                mock.patch.object(
                    assembler, "_get_allocator_type", return_value="default"
                )
            )
            stack.enter_context(
                mock.patch.object(
                    assembler, "_create_mla_dedup_context", return_value=context
                )
            )
            kv_factory = stack.enter_context(
                mock.patch.object(assembler, "build_kv_host_pool", return_value=kv_host)
            )
            state_factory = stack.enter_context(
                mock.patch.object(assembler, "MambaPoolHost", return_value=state_host)
            )
            controller_factory = stack.enter_context(
                mock.patch.object(assembler, "HybridCacheController")
            )
            if failure == "local_private":
                state_factory.side_effect = MemoryError("private state allocation")
            if failure == "local_controller":
                controller_factory.side_effect = RuntimeError("controller allocation")
            try:
                result = assembler.build_hybrid_mamba_stack(
                    params=params,
                    decls=(make_kv_pool_decl(kv_pool),),
                    mamba_pool=state_pool,
                    full_layer_mapping={3: 0, 7: 1},
                    mamba_layer_mapping={0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5},
                    load_cache_event=None,
                    storage_backend=None,
                    use_mla=True,
                )
                error = None
            except Exception as exc:
                result, error = None, exc
        return SimpleNamespace(
            result=result,
            error=error,
            actions=actions,
            context=context,
            group=group,
            kv_host=kv_host,
            state_host=state_host,
            kv_factory=kv_factory,
            state_factory=state_factory,
            controller_factory=controller_factory,
        )

    def test_three_kda_one_mla_layer_mapping_and_private_state(self):
        run = self._build()
        self.assertIsNone(run.error)
        group, _, _ = run.result
        self.assertIs(run.kv_factory.call_args.kwargs["shared_memory_group"], run.group)
        self.assertNotIn("shared_memory_group", run.state_factory.call_args.kwargs)
        self.assertFalse(group.get_pool(PoolName.MAMBA).allocator.is_shared)
        self.assertEqual(
            [group.get_entry(PoolName.KV).layer_mapper(i) for i in range(8)],
            [None, None, None, 0, None, None, None, 1],
        )
        self.assertEqual(
            [group.get_entry(PoolName.MAMBA).layer_mapper(i) for i in range(8)],
            [0, 1, 2, None, 3, 4, 5, None],
        )
        self.assertEqual(
            run.controller_factory.call_args.kwargs["transfer_layer_id_max"], 8
        )
        self.assertEqual(run.context.synchronize_error.call_count, 2)

    def test_local_private_allocation_failure_releases_shared_anchor(self):
        run = self._build(failure="local_private")
        self.assertIsInstance(run.error, RuntimeError)
        self.assertEqual(run.actions[-1], ("destroy", "kv"))
        run.controller_factory.assert_not_called()

    def test_peer_private_failure_releases_local_state_then_shared_anchor(self):
        run = self._build(failure="peer_private hybrid state construction")
        self.assertIsInstance(run.error, RuntimeError)
        self.assertEqual(run.actions[-2:], [("destroy", "state"), ("destroy", "kv")])
        run.controller_factory.assert_not_called()

    def test_controller_failure_is_joint_and_destroys_each_pool_once(self):
        for failure in ("local_controller", "peer_hybrid host stack construction"):
            with self.subTest(failure=failure):
                run = self._build(failure=failure)
                self.assertIsInstance(run.error, RuntimeError)
                self.assertEqual(
                    run.actions[-2:], [("destroy", "state"), ("destroy", "kv")]
                )
                run.kv_host.destroy.assert_called_once()
                run.state_host.destroy.assert_called_once()

    def test_broadcast_mode_has_no_shared_constructor_collectives(self):
        run = self._build(shared=False)
        self.assertIsNone(run.error)
        self.assertIsNone(run.kv_factory.call_args.kwargs["shared_memory_group"])
        run.context.synchronize_error.assert_not_called()


class TestPrivateKDAConstructionRollback(CustomTestCase):
    def _device_pool(self):
        return SimpleNamespace(
            size=8,
            device="cpu",
            num_mamba_layers=6,
            _slot_siblings=(),
            mamba_cache=SimpleNamespace(
                temporal=torch.zeros((6, 9, 2, 4, 4), dtype=torch.float32),
                conv=[torch.zeros((6, 9, 3, 32), dtype=torch.bfloat16)],
            ),
        )

    def _run_failure(self, *, fail_after_buffers):
        allocated, unregistered = [], []

        def allocate(dims, *, dtype, **kwargs):
            if not fail_after_buffers and allocated:
                raise MemoryError("second state allocation")
            value = torch.empty(dims, dtype=dtype)
            allocated.append(value)
            return value

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.dict(host_mamba.ALLOC_MEMORY_FUNCS, {"cpu": allocate})
            )
            stack.enter_context(
                mock.patch.object(
                    host_mamba, "host_memory_budget_bytes", return_value=2**30
                )
            )
            stack.enter_context(mock.patch.object(host_base, "_is_cuda", True))
            stack.enter_context(
                mock.patch.object(
                    host_base,
                    "_cuda_host_unregister",
                    side_effect=lambda value: unregistered.append(value),
                )
            )
            if fail_after_buffers:
                stack.enter_context(
                    mock.patch.object(
                        host_mamba.MambaPoolHost,
                        "clear",
                        side_effect=RuntimeError("metadata initialization"),
                    )
                )
            with self.assertRaises((MemoryError, RuntimeError)):
                host_mamba.MambaPoolHost(
                    self._device_pool(),
                    host_to_device_ratio=2,
                    host_size=0,
                    pin_memory=True,
                    layout="page_first_direct",
                )
        self.assertEqual(len(allocated), 2 if fail_after_buffers else 1)
        self.assertEqual([id(v) for v in unregistered], [id(v) for v in allocated])

    def test_later_buffer_failure_unpins_earlier_state(self):
        self._run_failure(fail_after_buffers=False)

    def test_metadata_failure_unpins_all_state_buffers(self):
        self._run_failure(fail_after_buffers=True)


if __name__ == "__main__":
    unittest.main()
