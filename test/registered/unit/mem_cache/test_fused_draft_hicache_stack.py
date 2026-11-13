"""A fused draft through HiCache, from the draft plan through the production
host assembler and registration to a backup and reload at newly translated slots.

A host pool of whole entries carries a fused draft's rows; a storage backend
that copies rows layer by layer gets them packed after the target's. Each case
asks the production code which host it builds rather than constructing one. On
CUDA the transfers run the real kernels; elsewhere the engine and the copy
kernel are stood in for.

    python -m pytest test/registered/unit/mem_cache/test_fused_draft_hicache_stack.py -v
"""

import tempfile
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.kvcache.hicache import _default_unroll, _tiles_across_lanes
from sglang.srt.managers.cache_controller import CacheOperation, LayerLoadingEvent
from sglang.srt.mem_cache import l2_transfer, unified_draft_pool
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.kv.decl import window_sub_pool_name
from sglang.srt.mem_cache.kv.layout.mla import DraftMLAView, MLAView
from sglang.srt.mem_cache.kv_cache_builder import maybe_register_hicache_draft
from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    DraftBand,
    FusedDraftPlacement,
    MLADraftRegion,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.unified import UnifiedPageEnvelopeHostPool
from sglang.srt.mem_cache.unified_draft_pool import bind_fused_draft
from sglang.srt.runtime_context import reset_context
from sglang.srt.speculative import base_spec_worker as spec
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

PS = 2
ON_GPU = torch.cuda.is_available()
DEVICE = "cuda" if ON_GPU else "cpu"
# Serving's layout and kernels on a GPU. The stand-ins below take the direct path,
# whose calls name every row, so the same assertions run on the CPU.
LAYOUT, IO_BACKEND = (
    ("page_first", "kernel") if ON_GPU else ("page_first_direct", "direct")
)
# Rows of 128 B: the kernel path refuses strided rows the JIT cannot tile.
HEAD_DIM = 64
KV_LORA_RANK, QK_ROPE_HEAD_DIM = 48, 16
# A window sub-pool is named by its window: the target's, and a band of the
# draft's own.
WINDOW = 32
TARGET_WINDOW = window_sub_pool_name(WINDOW)
DRAFT_WINDOW = window_sub_pool_name(16)
_HF = SimpleNamespace(architectures=["DraftForCausalLM"])
_NEXTN = SimpleNamespace(num_nextn_predict_layers=1, hf_config=_HF)
_NO_NEXTN = SimpleNamespace(num_nextn_predict_layers=None, hf_config=_HF)


def _dense_region(lane_num=1, **kw):
    return DenseDraftRegion(
        lane_num=lane_num,
        head_num=1,
        head_dim=HEAD_DIM,
        store_dtype=torch.float16,
        **kw,
    )


def _latent_region(lane_num=1, **kw):
    return MLADraftRegion(
        lane_num=lane_num,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        store_dtype=torch.float16,
        **kw,
    )


def _rows_of(view):
    if isinstance(view, (MLAView, DraftMLAView)):
        return list(view.kv_buffer)
    return [*view.k_buffer, *view.v_buffer]


def _fill(buffers, rows):
    for value, buffer in enumerate(buffers, 1):
        row = torch.arange(len(rows), dtype=buffer.dtype, device=buffer.device)
        buffer[rows] = (row * 16 + value).view(-1, *[1] * (buffer.dim() - 1))


def _page_slots(indices, page_size):
    return indices // page_size, indices % page_size


def _direct_pf_lf(*, src_ptrs, dst_ptrs, src_indices, dst_indices, layer_id, page_size):
    pages, offsets = _page_slots(src_indices, page_size)
    for src, dst in zip(src_ptrs, dst_ptrs, strict=True):
        dst[dst_indices] = src[pages, layer_id, offsets]


def _direct_lf_pf(*, src_ptrs, dst_ptrs, src_indices, dst_indices, page_size):
    pages, offsets = _page_slots(dst_indices, page_size)
    layers = len(src_ptrs) // len(dst_ptrs)
    for part, dst in enumerate(dst_ptrs):
        for layer in range(layers):
            src = src_ptrs[part * layers + layer]
            dst[pages, layer, offsets] = src[src_indices]


class _Event:
    def __init__(self, enable_timing=False):
        pass

    def record(self):
        pass

    def wait(self, stream=None):
        pass

    def synchronize(self):
        pass


class _Stream:
    def synchronize(self):
        pass


class _CpuDeviceModule:
    Event = _Event
    Stream = _Stream

    @staticmethod
    @contextmanager
    def stream(stream):
        yield


class _Stack:
    """The real pools, binding, plan, host assembly and registration for one
    configuration."""

    def __init__(
        self,
        *,
        region,
        mla_host=False,
        chain="mamba",
        host="full",
        algorithm="EAGLE",
        draft_config=_NEXTN,
        backend=None,
        new_bands=None,
        num_runners=1,
    ):
        counts = {host: [1] * num_runners}
        self.placement = FusedDraftPlacement.from_counts(
            counts=counts,
            regions={host: region},
            new_bands=new_bands,
            # The target's window sub-pool is named by its window, not its kind.
            hosted={host: "window"} if host == TARGET_WINDOW else None,
        )
        self.bundle = build_unified_bundle(
            UnifiedGeometry(
                chain=chain,
                full_layer_ids=(0,),
                swa_layer_ids=(1,) if chain == "swa" else (),
                window=WINDOW,
                state_layer_ids=(1,) if chain == "mamba" else (),
                mla=mla_host,
                head_dim=HEAD_DIM,
                swa_head_dim=HEAD_DIM,
                kv_lora_rank=KV_LORA_RANK,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                page_size=PS,
                fused_draft=self.placement,
                device=DEVICE,
            )
        )
        self.alloc = self.bundle.token_to_kv_pool_allocator
        self.storage = self.alloc.storage
        self.backend = backend
        self.drafts = tuple(self._bind(runner) for runner in range(num_runners))
        self.target_runner = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.from_string(algorithm),
            token_to_kv_pool=self.bundle.token_to_kv_pool,
            token_to_kv_pool_allocator=self.alloc,
            kv_cache_configurator=SimpleNamespace(),
        )
        self.draft_config = draft_config
        self.plan = self._plan()
        self.group, self.transfer_layer_id_max = self._assemble()

    def _bind(self, runner):
        with patch.object(
            unified_draft_pool, "draft_kv_layer_ids", return_value=[2 + runner]
        ):
            return bind_fused_draft(
                placement=self.placement,
                runner=runner,
                arena=self.alloc.arena,
                plan=self.storage.plan,
                owner=self.alloc.roles.owner,
                target_state=self.alloc.state,
                model=None,
                model_config=SimpleNamespace(
                    kv_lora_rank=KV_LORA_RANK, qk_rope_head_dim=QK_ROPE_HEAD_DIM
                ),
                req_to_token_pool=self.bundle.req_to_token_pool,
                page_size=PS,
            ).token_to_kv_pool

    def _plan(self):
        runners = tuple(
            SimpleNamespace(
                model_config=self.draft_config,
                token_to_kv_pool=draft,
                runner_ids=SimpleNamespace(is_fused_draft=True),
            )
            for draft in self.drafts
        )
        worker = SimpleNamespace(
            target_worker=SimpleNamespace(model_runner=self.target_runner),
            _draft_model_runners=lambda: runners,
        )
        memory = SimpleNamespace(
            enable_hierarchical_cache=True,
            enable_unified_cache_external_linker=False,
            hicache_storage_backend=self.backend,
        )
        with patch.object(spec, "get_memory", return_value=memory):
            return spec.BaseSpecWorker._build_hicache_draft_plan(worker)

    def _assemble(self):
        memory = SimpleNamespace(
            hicache_storage_backend=self.backend,
            hicache_size=0,
            hicache_ratio=1.0,
            hicache_mem_layout=LAYOUT,
            hicache_host_memory_mode="auto",
        )
        params = SimpleNamespace(
            page_size=PS,
            mtp_draft_device_pools=self.target_runner.mtp_draft_device_pools,
            mtp_draft_mamba_pools=self.target_runner.mtp_draft_mamba_pools,
        )
        seen = {}

        # The real assembly, without the storage threads a controller starts.
        def controller_without_threads(**kwargs):
            seen.update(kwargs)

        with (
            patch.object(assembler, "get_memory", return_value=memory),
            patch.object(
                assembler,
                "get_parallel",
                return_value=SimpleNamespace(dcp_enabled=False),
            ),
            patch.object(assembler, "_get_allocator_type", return_value="default"),
            patch.object(
                assembler, "_build_cache_controller", controller_without_threads
            ),
        ):
            result = assembler.build_unified_host_stack(
                storage=self.storage,
                params=params,
                cache=None,
                load_cache_event=None,
                storage_backend=self.backend,
            )
        return result.host_pool_group, seen["transfer_layer_id_max"]

    def register(self):
        """What serving calls once the cache is complete."""
        maybe_register_hicache_draft(
            tree_cache=SimpleNamespace(
                kv_storage=lambda: self.storage,
                cache_controller=SimpleNamespace(mem_pool_host=self.group),
            ),
            draft_plan=self.plan,
        )

    @property
    def host(self):
        return self.group.anchor_entry.host_pool

    @property
    def draft(self):
        return self.drafts[0]

    @property
    def draft_views(self):
        return tuple(
            draft.layer_router.view(draft.layer_router.sub_pools()[0].name)
            for draft in self.drafts
        )

    @property
    def draft_view(self):
        return self.draft_views[0]

    def _controller(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mem_pool_host = self.group
        controller.transfer_layer_id_max = self.transfer_layer_id_max
        controller.io_backend = IO_BACKEND
        controller.device = torch.device(DEVICE)
        return controller

    @staticmethod
    def _backup(controller, host_indices, device_indices, pool_transfers=None):
        # Through start_writing's own move: the I/O backend decides where indices live.
        op = CacheOperation(
            host_indices=host_indices,
            device_indices=device_indices,
            node_id=0,
            pool_transfers=pool_transfers,
        )
        return controller._l2_transfers(*controller._move_write_operation(op))

    @staticmethod
    def _load(controller, host_indices, device_indices, pool_transfers=None):
        # Through start_loading's own move.
        op = CacheOperation(
            host_indices=host_indices,
            device_indices=device_indices,
            node_id=0,
            pool_transfers=pool_transfers,
        )
        return controller._l2_load_transfers(*controller._move_op_indices(op))

    def buffers(self):
        rows = _rows_of(self.storage.view("full"))
        for view in self.draft_views:
            rows += _rows_of(view)
        return rows

    def round_trip(self, before_backup=None, after_backup=None):
        """Back a request's rows up through the controller's transfers, wipe the
        device, and load them at another request's slots: the draft's rows must
        arrive with the target's, at the rows those slots translate to, and nothing
        lands outside them."""
        controller = self._controller()
        src, dst = self.alloc.alloc(2 * PS), self.alloc.alloc(2 * PS)
        translate = self.storage.host_ids("full")
        src_rows, dst_rows = translate(src), translate(dst)
        assert not torch.equal(src_rows, src), "ids must translate"
        buffers = self.buffers()
        _fill(buffers, src_rows)
        if before_backup is not None:
            before_backup(src_rows)
        expected = [buffer[src_rows].clone() for buffer in buffers]
        host_indices = self.host.alloc(len(src))
        with self._transfer_context():
            engine = l2_transfer.L2TransferEngine(IO_BACKEND)
            engine.submit_device_to_host(self._backup(controller, host_indices, src))
            if ON_GPU:
                torch.cuda.synchronize()
            self.alloc.arena.raw.zero_()
            if after_backup is not None:
                after_backup(host_indices)
            engine.submit_host_to_device(
                self._load(controller, host_indices, dst),
                transfer_layer_id_max=self.transfer_layer_id_max,
            )
            if ON_GPU:
                torch.cuda.synchronize()
        self._assert_loaded(buffers, dst_rows, expected)

    def _assert_loaded(self, buffers, dst_rows, expected):
        for buffer, rows in zip(buffers, expected, strict=True):
            torch.testing.assert_close(buffer[dst_rows], rows, rtol=0, atol=0)
            outside = torch.ones(len(buffer), dtype=torch.bool, device=buffer.device)
            outside[dst_rows] = False
            assert not buffer[outside].any()

    def round_trip_side(self, name, before_backup=None):
        """The same through a window sub-pool's own transfer: its host pool, and
        the owner's token ids, which the window translates to its own rows."""
        controller = self._controller()
        pool = self.storage.plan.sub_pool(name).external.host_pool
        host = self.group.entry_map[pool].host_pool
        src, dst = self.alloc.alloc(2 * PS), self.alloc.alloc(2 * PS)
        translate = self.storage.host_ids(name)
        src_rows, dst_rows = translate(src), translate(dst)
        assert not torch.equal(src_rows, src), "ids must translate"
        views = [self.storage.view(name), *self.draft_views]
        buffers = [rows for view in views for rows in _rows_of(view)]
        assert buffers, "the sub-pool must hold rows"
        _fill(buffers, src_rows)
        if before_backup is not None:
            before_backup(src_rows)
        expected = [buffer[src_rows].clone() for buffer in buffers]
        host_indices = host.alloc(len(src))
        none = torch.empty(0, dtype=torch.int64, device=DEVICE)
        with self._transfer_context():
            engine = l2_transfer.L2TransferEngine(IO_BACKEND)
            engine.submit_device_to_host(
                self._backup(
                    controller,
                    none,
                    none,
                    [
                        PoolTransfer(
                            name=pool,
                            host_indices=host_indices,
                            device_indices=src,
                        )
                    ],
                )
            )
            if ON_GPU:
                torch.cuda.synchronize()
            self.alloc.arena.raw.zero_()
            engine.submit_host_to_device(
                self._load(
                    controller,
                    none,
                    none,
                    [
                        PoolTransfer(
                            name=pool,
                            host_indices=host_indices,
                            device_indices=dst,
                        )
                    ],
                ),
                transfer_layer_id_max=self.transfer_layer_id_max,
            )
            if ON_GPU:
                torch.cuda.synchronize()
        self._assert_loaded(buffers, dst_rows, expected)

    def round_trip_pages(self):
        """The same by whole pages, for rows that are not indexed by token id
        (replicas): the loaded pages hold the backed-up bytes, and no other does."""
        controller = self._controller()
        src, dst = self.alloc.alloc(2 * PS), self.alloc.alloc(2 * PS)
        translate = self.storage.host_ids("full")
        src_pages = translate(src)[::PS] // PS
        dst_pages = translate(dst)[::PS] // PS
        pages = self.storage.host_layout("full").get_page_envelope_buffer()
        pages[src_pages] = (
            torch.arange(pages[src_pages].numel(), device=pages.device)
            .remainder(251)
            .to(torch.uint8)
            .reshape(pages[src_pages].shape)
            + 1
        )
        expected = pages[src_pages].clone()
        host_indices = self.host.alloc(len(src))
        with self._transfer_context():
            engine = l2_transfer.L2TransferEngine(IO_BACKEND)
            engine.submit_device_to_host(self._backup(controller, host_indices, src))
            if ON_GPU:
                torch.cuda.synchronize()
            self.alloc.arena.raw.zero_()
            engine.submit_host_to_device(
                self._load(controller, host_indices, dst),
                transfer_layer_id_max=self.transfer_layer_id_max,
            )
            if ON_GPU:
                torch.cuda.synchronize()
        torch.testing.assert_close(pages[dst_pages], expected, rtol=0, atol=0)
        outside = torch.ones(len(pages), dtype=torch.bool, device=pages.device)
        outside[dst_pages] = False
        assert not pages[outside].any()

    @contextmanager
    def _transfer_context(self):
        if ON_GPU:
            yield
            return
        with (
            patch(
                "sglang.srt.mem_cache.pool_host.mha.transfer_kv_per_layer_direct_pf_lf",
                _direct_pf_lf,
                create=True,
            ),
            patch(
                "sglang.srt.mem_cache.pool_host.mha.transfer_kv_all_layer_direct_lf_pf",
                _direct_lf_pf,
                create=True,
            ),
            patch(
                "sglang.srt.mem_cache.pool_host.mla.transfer_kv_per_layer_direct_pf_lf",
                _direct_pf_lf,
                create=True,
            ),
            patch(
                "sglang.srt.mem_cache.pool_host.mla.transfer_kv_all_layer_direct_lf_pf",
                _direct_lf_pf,
                create=True,
            ),
            patch.object(l2_transfer, "device_module", _CpuDeviceModule),
            # The real kernels refuse rows that are not packed here and take the
            # JIT path on a GPU; the stand-ins above follow any row spacing.
            patch.object(HostKVCache, "_require_packed_device_rows", lambda *a: None),
        ):
            yield


class _StackCase(CustomTestCase):
    def setUp(self):
        setup_allocator_context()
        self.addCleanup(reset_context)
        if not ON_GPU:
            # Host pools pin their memory for the GPU's copies.
            for name in ("_cuda_host_register", "_cuda_host_unregister"):
                patcher = patch(f"sglang.srt.mem_cache.pool_host.common.{name}")
                patcher.start()
                self.addCleanup(patcher.stop)

    def _stack(self, **kw):
        stack = _Stack(**kw)
        self.addCleanup(lambda: [e.host_pool.destroy() for e in stack.group.entries])
        return stack

    def _assert_whole_entries(self, stack):
        self.assertEqual(stack.plan.mode, spec.HiCacheDraftMode.FUSED)
        self.assertIsInstance(stack.host, UnifiedPageEnvelopeHostPool)
        self.assertEqual(stack.target_runner.mtp_draft_device_pools, ())
        self.assertNotIn(PoolName.DRAFT, stack.group.entry_map)
        self.assertEqual(stack.group.anchor_entry.packed_draft_device_pools, ())
        stack.register()


class TestNextnHeadOnFullStateHost(_StackCase):
    """Qwen's and Ling's shapes: MHA and MLA rows on a full + state host."""

    CASES = ((False, _dense_region), (True, _latent_region))

    def test_whole_entries_carry_the_head_where_the_backend_keeps_them(self):
        for backend in (None, "file", "mori"):
            for mla, region in self.CASES:
                with self.subTest(backend=backend, mla=mla):
                    stack = self._stack(region=region(), mla_host=mla, backend=backend)
                    self._assert_whole_entries(stack)
                    stack.round_trip()

    def test_a_backend_that_copies_rows_gets_the_head_packed(self):
        for mla, region in self.CASES:
            with self.subTest(mla=mla):
                stack = self._stack(region=region(), mla_host=mla, backend="mooncake")
                self.assertEqual(stack.plan.mode, spec.HiCacheDraftMode.PACKED)
                self.assertEqual(
                    stack.target_runner.mtp_draft_device_pools, (stack.draft,)
                )
                self.assertIsInstance(
                    stack.host, MLATokenToKVPoolHost if mla else MHATokenToKVPoolHost
                )
                self.assertEqual(stack.host.layer_num, stack.host.target_layer_num + 1)
                self.assertEqual(
                    stack.group.anchor_entry.packed_draft_device_pools,
                    (stack.draft_view,),
                )
                stack.register()
                stack.round_trip()

    def test_a_packed_hosts_rows_take_the_jit_kernels(self):
        """The stand-ins follow any row spacing and a GPU's kernel path does not,
        so without this a narrower row fails on a GPU alone."""
        for mla, region in self.CASES:
            with self.subTest(mla=mla):
                host = self._stack(
                    region=region(), mla_host=mla, backend="mooncake"
                ).host
                dim = host.kv_cache_dim if mla else host.element_dim
                size = dim * host.dtype.itemsize
                self.assertTrue(_tiles_across_lanes(size, _default_unroll(size)))

    def test_what_a_row_backend_stores_holds_the_heads_rows(self):
        """The bytes Mooncake is pointed at for a page cover the packed layers."""
        for mla, region in self.CASES:
            with self.subTest(mla=mla):
                stack = self._stack(region=region(), mla_host=mla, backend="mooncake")
                indices = stack.host.alloc(PS)
                pointers, sizes = stack.host.get_page_buffer_meta(indices)
                self.assertEqual(len(pointers), len(sizes))
                self.assertEqual(sum(sizes), stack.host.get_size_per_token() * PS)
                per_layer = stack.host.get_size_per_token() // stack.host.layer_num
                self.assertEqual(sum(sizes), per_layer * 2 * PS)

    def test_an_envelope_never_receives_packed_rows(self):
        """The host assembly fails loudly if the plan and the layout disagree."""
        stack = self._stack(region=_dense_region())
        with self.assertRaisesRegex(ValueError, "already carry their draft parts"):
            stack.storage.host_layout("full").build(
                SimpleNamespace(), draft_buffer_infos=(object(),)
            )

    def test_several_runners_keep_their_own_rows(self):
        """One head per draft step: every runner's lane survives, by either route."""
        for mla, region in self.CASES:
            for backend in (None, "mooncake"):
                with self.subTest(mla=mla, backend=backend):
                    stack = self._stack(
                        region=region(lane_num=3),
                        mla_host=mla,
                        backend=backend,
                        num_runners=3,
                    )
                    if backend is None:
                        self._assert_whole_entries(stack)
                    else:
                        self.assertEqual(
                            stack.host.layer_num, stack.host.target_layer_num + 3
                        )
                    stack.round_trip()

    def test_a_file_store_keeps_the_rows(self):
        """Through the real file backend, by the path a controller takes: every
        target and draft row comes back from storage alone."""
        for mla, region in self.CASES:
            with self.subTest(mla=mla), tempfile.TemporaryDirectory() as directory:
                stack = self._stack(region=region(), mla_host=mla, backend="file")
                store = HiCacheFile(
                    HiCacheStorageConfig(
                        tp_rank=0,
                        tp_size=1,
                        pp_rank=0,
                        pp_size=1,
                        attn_cp_rank=0,
                        attn_cp_size=1,
                        is_mla_model=mla,
                        enable_storage_metrics=False,
                        is_page_first_layout=True,
                        model_name="model",
                    ),
                    file_path=directory,
                )
                store.register_mem_pool_host(stack.host)

                def through_storage(host_indices):
                    keys = [f"p{i}" for i in range(len(host_indices) // PS)]
                    pages = [
                        stack.host.get_data_page(host_indices[i * PS])
                        for i in range(len(keys))
                    ]
                    self.assertTrue(store.batch_set(keys, pages))
                    blank = stack.host.get_dummy_flat_data_page()
                    for i in range(len(keys)):
                        stack.host.set_from_flat_data_page(host_indices[i * PS], blank)
                    got = store.batch_get(
                        keys,
                        [stack.host.get_dummy_flat_data_page() for _ in keys],
                    )
                    for i, page in enumerate(got):
                        self.assertIsNotNone(page)
                        stack.host.set_from_flat_data_page(host_indices[i * PS], page)

                stack.round_trip(after_backup=through_storage)

    def test_only_token_rows_take_the_per_layer_layout(self):
        """The state sub-pool has one layout, and a page spanning several pool
        pages cannot be addressed by per-layer rows."""
        stack = self._stack(region=_dense_region(), backend="mooncake")
        state = stack.alloc.roles.state
        self.assertIs(
            type(stack.storage.packed_host_layout(state)),
            type(stack.storage.host_layout(state)),
        )
        band = stack.alloc.full_attn_allocator
        with patch.object(band, "pool_page_size", band.page_size * 2):
            with self.assertRaisesRegex(ValueError, "per-layer host rows"):
                stack.storage.packed_host_layout("full")

    def test_replicated_rows_stay_in_whole_entries(self):
        """Replicas price more bytes per entry, which an envelope copies as they are."""
        for mla, region in self.CASES:
            with self.subTest(mla=mla):
                stack = self._stack(region=region(token_replicas=2), mla_host=mla)
                self._assert_whole_entries(stack)
                stack.round_trip_pages()


class TestDflashFamilyKeepsWholeEntries(_StackCase):
    """DFLASH and DSPARK rows, MHA and MLA, on both kinds of host: never packed,
    never given a sidecar, whatever the backend, and a prefix-valid commit
    survives a reload."""

    SHAPES = (
        (False, _dense_region),
        (False, _latent_region),
        (True, _dense_region),
        (True, _latent_region),
    )

    def _commit_prefix(self, stack, mla_rows):
        view = stack.draft_view
        # The layer id the draft binds, through the route its worker commits by.
        layer = SimpleNamespace(layer_id=2)
        route = stack.draft.layer_router.layer(2)

        def commit(rows):
            loc_2d = rows.reshape(1, -1)
            # A window layer writes at its own sub-pool's ids.
            loc = KVWriteLoc(
                loc_2d,
                window_locs=None if route.owner else {route.route.sub_pool: loc_2d},
                physical=True,
            )
            commit_lens = torch.tensor(
                [loc_2d.shape[1] - 1], dtype=torch.int32, device=DEVICE
            )
            if mla_rows:
                latent = torch.full(
                    (loc_2d.shape[1], 1, KV_LORA_RANK),
                    3.0,
                    dtype=torch.float16,
                    device=DEVICE,
                )
                rope = torch.full(
                    (loc_2d.shape[1], 1, QK_ROPE_HEAD_DIM),
                    5.0,
                    dtype=torch.float16,
                    device=DEVICE,
                )
                route.store_mla_prefix(layer, loc, commit_lens, latent, rope)
                written = view.kv_buffer[0][rows]
                self.assertTrue(bool((written[:-1, 0, :KV_LORA_RANK] == 3).all()))
                self.assertTrue(bool((written[:-1, 0, KV_LORA_RANK:] == 5).all()))
                # Past commit_lens stays what it was.
                self.assertFalse(bool((written[-1] == 3).any()))
            else:
                k = torch.full(
                    (loc_2d.shape[1], 1, HEAD_DIM),
                    3.0,
                    dtype=torch.float16,
                    device=DEVICE,
                )
                route.store_prefix(layer, loc, commit_lens, k, k)
                written = view.k_buffer[0][rows]
                self.assertTrue(bool((written[:-1] == 3).all()))
                self.assertFalse(bool((written[-1] == 3).any()))

        return commit

    def test_rows_survive_a_reload_from_whole_entries(self):
        for algorithm in ("DFLASH", "DSPARK"):
            for mla_host, region in self.SHAPES:
                with self.subTest(
                    algorithm=algorithm, mla_host=mla_host, region=region
                ):
                    stack = self._stack(
                        region=region(),
                        mla_host=mla_host,
                        algorithm=algorithm,
                        draft_config=_NO_NEXTN,
                    )
                    self._assert_whole_entries(stack)
                    stack.round_trip(
                        self._commit_prefix(stack, region is _latent_region)
                    )

    def test_a_row_backend_does_not_pack_them(self):
        """Only the EAGLE family packs: these keep main's route, whose host pool
        then refuses the backend at registration or attach."""
        for algorithm in ("DFLASH", "DSPARK"):
            with self.subTest(algorithm=algorithm):
                stack = self._stack(
                    region=_dense_region(),
                    algorithm=algorithm,
                    draft_config=_NEXTN,
                    backend="mooncake",
                )
                self.assertEqual(stack.plan.mode, spec.HiCacheDraftMode.FUSED)
                self.assertEqual(stack.target_runner.mtp_draft_device_pools, ())
                self.assertIsInstance(stack.host, UnifiedPageEnvelopeHostPool)
                with self.assertRaisesRegex(ValueError, "requires separate host pools"):
                    stack._controller().attach_storage_backend("mooncake")

    def test_rows_survive_a_reload_from_a_window(self):
        """A draft in the target's window band and in a window band of its own,
        through the side pool's own transfer, with a prefix-valid commit."""
        for algorithm in ("DFLASH", "DSPARK"):
            for host, new_bands in (
                (TARGET_WINDOW, None),
                (DRAFT_WINDOW, {DRAFT_WINDOW: DraftBand(kind="window", window=16)}),
            ):
                with self.subTest(algorithm=algorithm, host=host):
                    stack = self._stack(
                        region=_dense_region(),
                        chain="swa",
                        host=host,
                        algorithm=algorithm,
                        draft_config=_NO_NEXTN,
                        new_bands=new_bands,
                    )
                    self.assertEqual(stack.plan.mode, spec.HiCacheDraftMode.FUSED)
                    self.assertEqual(stack.target_runner.mtp_draft_device_pools, ())
                    self.assertNotIn(PoolName.DRAFT, stack.group.entry_map)
                    stack.register()
                    stack.round_trip_side(host, self._commit_prefix(stack, False))


class TestEagleOnPageEnvelopes(_StackCase):
    """A hybrid sliding-window host's two ends share one envelope arena: a draft's
    rows in either, or in a band of the draft's own, back up with the page."""

    def _assert_pair(self, stack):
        self.assertEqual(stack.plan.mode, spec.HiCacheDraftMode.FUSED)
        self.assertEqual(stack.target_runner.mtp_draft_device_pools, ())
        for name in (PoolName.KV, TARGET_WINDOW):
            self.assertIsInstance(
                stack.group.entry_map[name].host_pool, UnifiedPageEnvelopeHostPool
            )
        self.assertNotIn(PoolName.DRAFT, stack.group.entry_map)
        stack.register()

    def test_a_draft_on_either_end_rides_in_the_envelopes(self):
        for algorithm in ("EAGLE", "EAGLE3"):
            for host in ("full", TARGET_WINDOW):
                with self.subTest(algorithm=algorithm, host=host):
                    stack = self._stack(
                        region=_dense_region(),
                        chain="swa",
                        host=host,
                        algorithm=algorithm,
                        draft_config=_NO_NEXTN,
                    )
                    self._assert_pair(stack)
                    if host == "full":
                        stack.round_trip()
                    else:
                        stack.round_trip_side(host)

    def test_a_draft_only_window_band_keeps_its_rows(self):
        stack = self._stack(
            region=_dense_region(),
            chain="swa",
            host=DRAFT_WINDOW,
            draft_config=_NO_NEXTN,
            new_bands={DRAFT_WINDOW: DraftBand(kind="window", window=16)},
        )
        self.assertEqual(stack.plan.mode, spec.HiCacheDraftMode.FUSED)
        self.assertEqual(stack.target_runner.mtp_draft_device_pools, ())
        self.assertNotIn(PoolName.DRAFT, stack.group.entry_map)
        stack.register()
        stack.round_trip_side(DRAFT_WINDOW)


@unittest.skipUnless(ON_GPU, "needs a CUDA stream and graph")
class TestFusedWaitOrdersTheGraph(CustomTestCase):
    """A reload still pending on its own stream, and a graph on another that reads
    what the reload writes. The wait alone decides which value the graph sees."""

    # Bounded, and long against the replay of a one-element copy.
    RELOAD_DELAY_CYCLES = 1_000_000_000

    def setUp(self):
        self.addCleanup(torch.cuda.synchronize)
        source = torch.full((1 << 12,), 7.0, device=DEVICE)
        self.target = torch.zeros_like(source)
        self.seen = torch.zeros(1, device=DEVICE)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.seen.copy_(self.target[-1:])
        self.reload = LayerLoadingEvent(2)
        self.reload_stream = torch.cuda.Stream()
        self.graph_stream = torch.cuda.Stream()
        with torch.cuda.stream(self.reload_stream):
            torch.cuda._sleep(self.RELOAD_DELAY_CYCLES)
            self.target.copy_(source, non_blocking=True)
            self.reload.complete(1)
        self.assertFalse(self.reload.finish_event.query())

    def _replay(self):
        replayed = torch.cuda.Event()
        with torch.cuda.stream(self.graph_stream):
            self.graph.replay()
            replayed.record()
        replayed.synchronize()

    def test_a_replayed_graph_reads_what_the_reload_wrote(self):
        worker = SimpleNamespace(
            hicache_draft_plan=spec.HiCacheDraftPlan(mode=spec.HiCacheDraftMode.FUSED),
            target_worker=SimpleNamespace(
                hicache_layer_transfer_counter=SimpleNamespace(
                    num_layers=2, events={3: self.reload}
                )
            ),
        )
        with torch.cuda.stream(self.graph_stream):
            spec.BaseSpecWorker._wait_for_fused_hicache_load(
                worker, SimpleNamespace(hicache_consumer_index=3)
            )
        self._replay()
        self.assertTrue(self.reload.finish_event.query())
        self.assertEqual(self.seen.item(), 7.0)

    def test_without_the_wait_the_same_graph_reads_the_old_rows(self):
        self._replay()
        pending_after_replay = not self.reload.finish_event.query()
        torch.cuda.synchronize()
        self.assertTrue(pending_after_replay)
        self.assertEqual(self.seen.item(), 0.0)
        self.assertEqual(self.target[-1].item(), 7.0)


if __name__ == "__main__":
    unittest.main()
