"""Draft KV through the cache controller's L2 path, with reused host slots.

``test_page_unified_host_pool.py`` moves draft rows by calling one host pool's
transfer arms directly, on host slots nothing was stored in before. What
serving does differs in two ways, and both are where a draft row can go wrong
without the target noticing:

* The controller decides which host slots and device rows each pool's transfer
  names. A draft KV pool beside the target's (DFLASH without its ring) has no
  slots of its own: its pages ride the KV pool's host slots and device rows.
  Packed draft layers (NEXTN-style MTP) are written with the target's pages
  and loaded back by one extra transfer per draft depth.
* Host slots are reused. Evicting a prompt from the host tier frees slot ids
  and leaves the bytes, so the next prompt's backup has to replace them in
  every pool that stores a page there.

So this assembles the stack the way the server does (staging, host pools,
controller, draft registration), fills the host tier, evicts prompt A, backs up
prompt B into A's slots, and loads B back into the device rows A was computed
on. The device is scrubbed in between, so every row read back is either B's,
A's (a stale host slot), or the scrub (a row no transfer wrote).
"""

import sys
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.cache_controller import HiCacheAck
from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.pool_host import HostPoolGroup
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_context, get_parallel, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import HiCacheDraftMode, HiCacheDraftPlan
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or is_npu()
    or is_xpu()
    or not (is_cuda() or is_hip()),
    reason="page_unified host pool tests require CUDA/ROCm.",
)

DEVICE = "cuda"
PAGE_SIZE = 16
DEVICE_PAGES = 12
# Half the device pool plus the host pool's padding page: 7 host pages.
HOST_RATIO = 0.5
TARGET_LAYERS = 4
# Prompt A is two tree nodes of two pages each.
A_NODE_PAGES = (2, 2)


def _mha_pool(layer_num, *, head_num=8, head_dim=32, dtype=torch.bfloat16):
    return MHATokenToKVPool(
        size=DEVICE_PAGES * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=dtype,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        device=DEVICE,
        enable_memory_saver=False,
        start_layer=0,
        end_layer=layer_num,
    )


def _mla_pool(layer_num):
    return MLATokenToKVPool(
        size=DEVICE_PAGES * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        kv_lora_rank=96,
        qk_rope_head_dim=32,
        layer_num=layer_num,
        device=DEVICE,
        enable_memory_saver=False,
        start_layer=0,
        end_layer=layer_num,
    )


class _TreeCacheStub:
    """What the assembler and the draft registration use of the cache tree."""

    def __init__(self):
        self.components = {ComponentType.FULL: SimpleNamespace()}
        self._enable_metrics_flag = False
        self.cache_controller = None
        self.host_pool_group = None
        self.sidecar_pool_specs: list[SidecarPoolSpec] = []

    def register_sidecar_pool(self, spec, entry=None):
        # As UnifiedRadixCache.register_sidecar_pool does.
        if entry is not None:
            self.cache_controller.register_host_pool_entry(entry)
        self.sidecar_pool_specs.append(spec)


@dataclass
class _Stack:
    controller: HybridCacheController
    allocator: PagedTokenToKVPoolAllocator
    # Every device KV buffer by name: one per pool, layer and component.
    buffers: dict[str, torch.Tensor]
    sidecar_specs: list[SidecarPoolSpec]

    @property
    def group(self) -> HostPoolGroup:
        return self.controller.mem_pool_host

    def sidecar_transfers(self) -> list[PoolTransfer] | None:
        """The side-pool transfers the tree hands the controller with a node:
        named, with their indices left to be bound to the KV pool's."""
        return [
            PoolTransfer(
                name=spec.pool_name,
                hit_policy=spec.hit_policy,
                indices_from_pool=spec.indices_from_pool,
            )
            for spec in self.sidecar_specs
        ] or None


def _device_buffers(pools):
    buffers = {}
    for label, pool in pools.items():
        for layer in range(pool.layer_num):
            if isinstance(pool, MLATokenToKVPool):
                buffers[f"{label} layer {layer} latent KV"] = pool.kv_buffer[layer]
            else:
                buffers[f"{label} layer {layer} K"] = pool.k_buffer[layer]
                buffers[f"{label} layer {layer} V"] = pool.v_buffer[layer]
    return buffers


@pytest.fixture
def build_stack():
    """Assemble device pools, host pools and a controller the way the server
    does, without model weights and without a storage backend (so the
    controller starts no worker thread)."""
    stacks = []

    def build(*, target, packed_drafts=(), sidecar_draft=None, **flags):
        server_args = ServerArgs(
            model_path="test",
            enable_hierarchical_cache=True,
            page_size=PAGE_SIZE,
            hicache_ratio=HOST_RATIO,
            hicache_mem_layout="page_unified",
            hicache_io_backend="kernel",
            **flags,
        )
        get_context().set_server_args(server_args)

        # Scheduler start-up order: the write-back staging first, which the
        # host pools then take over.
        draft_plan = None
        if sidecar_draft is not None:
            draft_plan = HiCacheDraftPlan(
                mode=HiCacheDraftMode.SIDECAR, device_pools=(sidecar_draft,)
            )
        kv_cache_builder.prepare_hicache_staging(
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    token_to_kv_pool=target, mtp_draft_device_pools=packed_drafts
                )
            ),
            draft_plan=draft_plan,
        )
        allocator = PagedTokenToKVPoolAllocator(
            target.size, PAGE_SIZE, target.dtype, DEVICE, target, need_sort=False
        )
        tree_cache = _TreeCacheStub()
        assembler.attach_hybrid_pool_to_unified_cache(
            tree_cache,
            CacheInitParams(
                disable=False,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=allocator,
                page_size=PAGE_SIZE,
                mtp_draft_device_pools=packed_drafts,
            ),
            server_args,
            load_cache_event=threading.Event(),
        )
        controller = tree_cache.cache_controller
        stacks.append(controller.mem_pool_host)
        assert type(controller) is HybridCacheController
        assert not controller.enable_storage
        if draft_plan is not None:
            # As kv_cache_builder.maybe_register_hicache_draft does.
            specs, entries = assembler.build_hicache_draft_sidecars(
                draft_device_pools=draft_plan.device_pools, tree_cache=tree_cache
            )
            for spec, entry in zip(specs, entries, strict=True):
                tree_cache.register_sidecar_pool(spec, entry)
        # The scheduler fences both copy directions behind the forward stream;
        # here the forward is the test filling device rows.
        controller.write_fence_stream = torch.cuda.current_stream()
        controller.load_fence_stream = torch.cuda.current_stream()

        pools = {"target": target}
        pools.update({f"draft {i}": pool for i, pool in enumerate(packed_drafts)})
        if sidecar_draft is not None:
            pools["draft"] = sidecar_draft
        return _Stack(
            controller=controller,
            allocator=allocator,
            buffers=_device_buffers(pools),
            sidecar_specs=tree_cache.sidecar_pool_specs,
        )

    reset_context()
    try:
        yield build
    finally:
        torch.cuda.synchronize()
        for group in stacks:
            group.destroy()
        reset_context()


def _random_rows(buffer, num_rows, generator):
    """One random row per token: values every KV dtype here holds exactly."""
    shape = (num_rows, *buffer.shape[1:])
    return torch.randint(0, 128, shape, generator=generator, dtype=torch.int32).to(
        device=DEVICE, dtype=buffer.dtype
    )


class _Prompt:
    """A prompt's KV: its own random row per token in every layer of every
    pool, cut into tree nodes of whole pages."""

    def __init__(self, name, node_pages, first_node_id, buffers, seed):
        self.name = name
        self.node_tokens = [pages * PAGE_SIZE for pages in node_pages]
        self.tokens = sum(self.node_tokens)
        self.node_ids = [first_node_id + i for i in range(len(node_pages))]
        generator = torch.Generator().manual_seed(seed)
        self.rows = {
            key: _random_rows(buffer, self.tokens, generator)
            for key, buffer in buffers.items()
        }
        self.device_rows = None
        self.host_nodes = None

    def compute(self, stack):
        """Take device rows and write the prompt's KV there, as a forward would."""
        self.device_rows = stack.allocator.alloc(self.tokens)
        assert self.device_rows is not None
        for key, buffer in stack.buffers.items():
            buffer[self.device_rows] = self.rows[key]

    @property
    def host_slots(self):
        return torch.cat(self.host_nodes)


def _assert_ack(stack, ack: HiCacheAck, node_ids, tokens):
    """The controller reports this transfer's nodes and tokens, no more."""
    assert ack.node_ids == node_ids
    assert ack.num_tokens == tokens
    # Pools that ride the KV pool's slots are counted in bytes, not in tokens.
    assert ack.num_tokens_by_pool == {PoolName.KV.value: tokens}
    assert ack.num_bytes == tokens * sum(
        entry.host_pool.size_per_token for entry in stack.group.entries
    )


def _backup(stack, prompt):
    """Back the prompt up node by node and flush once, as the tree does."""
    controller = stack.controller
    prompt.host_nodes = []
    for node_id, rows in zip(
        prompt.node_ids, prompt.device_rows.split(prompt.node_tokens), strict=True
    ):
        host = controller.write(
            rows, node_id=node_id, extra_pools=stack.sidecar_transfers(), flush=False
        )
        assert host is not None, f"prompt {prompt.name}: the host tier is full"
        prompt.host_nodes.append(host)
    # A pool beside the KV pool hands out no slots of its own: its page of a
    # node sits at the node's KV host slots, read from the node's device rows.
    for operation, host in zip(controller.write_queue, prompt.host_nodes, strict=True):
        transfers = operation.pool_transfers or []
        assert [t.name for t in transfers] == [s.pool_name for s in stack.sidecar_specs]
        for transfer in transfers:
            assert torch.equal(transfer.host_indices, host)
            assert torch.equal(transfer.device_indices, operation.device_indices)
    controller.start_writing()
    (ack,) = controller.ack_write_queue
    controller.ack_write_queue.clear()
    ack.finish_event.synchronize()
    _assert_ack(stack, ack, prompt.node_ids, prompt.tokens)


def _load(stack, node_ids, host_nodes):
    """Load nodes back into device rows the allocator hands out; one flush."""
    controller = stack.controller
    device_rows = []
    for node_id, host in zip(node_ids, host_nodes, strict=True):
        rows = controller.load(
            host, node_id=node_id, extra_pools=stack.sidecar_transfers()
        )
        assert rows is not None, "no device rows to load into"
        device_rows.append(rows)
    assert controller.start_loading() >= 0
    (ack,) = controller.ack_load_queue
    controller.ack_load_queue.clear()
    ack.finish_event.synchronize()
    device_rows = torch.cat(device_rows)
    _assert_ack(stack, ack, node_ids, len(device_rows))
    return device_rows


def _tokens_holding(got, rows):
    """Token positions of ``got`` whose row is, bit for bit, any row of ``rows``."""
    same = (got[:, None] == rows[None]).flatten(2).all(2).any(1)
    return same.nonzero().flatten().tolist()


def _assert_restored(stack, device_rows, expected, what, stale):
    """Every row of every layer of every pool is the expected token's row."""
    for key, buffer in stack.buffers.items():
        got = buffer[device_rows]
        for owner, rows in stale.items():
            holding = _tokens_holding(got, rows[key])
            assert not holding, f"{key}: tokens {holding} of {what} hold {owner}"
        wrong = (got != expected[key]).flatten(1).any(1).nonzero().flatten().tolist()
        assert not wrong, f"{key}: tokens {wrong} of {what} differ"


def _host_pages(stack):
    """Every page block of every host pool, by pool and page id."""
    pages = {}
    for entry in stack.group.entries:
        host_pool = entry.host_pool
        for page in range(host_pool.size // PAGE_SIZE):
            block = host_pool.get_data_page(page * PAGE_SIZE, flat=True)
            pages[entry.name, page] = block.clone()
    return pages


def _assert_unified_plan_cuts_the_host_pools(stack):
    """The pools moved here are the ones the unified key plan names: it cuts
    each of them into chunks that tile the page block, a draft pool beside
    the KV pool into its own head groups."""
    controller = stack.controller
    with get_parallel().override(tp_rank=0, attn_tp_rank=0, attn_dcp_rank=0, pp_rank=0):
        config = controller._generate_storage_config("test/model")
    anchor = stack.group.anchor_entry
    layout = anchor.host_pool.page_unified_layout
    # Per head group: the target's layers, then the packed draft layer if any.
    chunks_per_group = 2 if anchor.packed_draft_device_pools else 1
    assert len(config.unified_suffixes) == len(config.unified_chunk_spans)
    assert len(config.unified_chunk_spans) == layout.head_group_num * chunks_per_group
    assert sum(size for _, size in config.unified_chunk_spans) == layout.bytes_per_page
    assert set(config.unified_pool_chunks) == {s.pool_name for s in stack.sidecar_specs}
    for name, chunks in config.unified_pool_chunks.items():
        layout = stack.group.get_pool(name).page_unified_layout
        assert len(chunks.suffixes) == len(chunks.spans) == layout.head_group_num
        assert sum(size for _, size in chunks.spans) == layout.bytes_per_page


def _reuse_host_slots(stack, b_node_pages):
    """Prompt A, evicted; prompt B in A's host slots and A's device rows."""
    group, allocator, buffers = stack.group, stack.allocator, stack.buffers
    host_pages = group.size // PAGE_SIZE
    for entry in group.entries:
        assert entry.host_pool.layout == "page_unified"
        assert entry.host_pool.size == group.size

    a = _Prompt("A", A_NODE_PAGES, 10, buffers, seed=20261201)
    # Another prompt that stays in the host tier and takes the rest of it, so
    # that the only slots B can get are the ones A leaves.
    filler = _Prompt("C", (host_pages - sum(A_NODE_PAGES),), 20, buffers, seed=20261202)
    b = _Prompt("B", b_node_pages, 30, buffers, seed=20261203)
    assert filler.tokens > 0 and b.tokens <= a.tokens

    for prompt in (a, filler):
        prompt.compute(stack)
        _backup(stack, prompt)
    assert group.available_size() == 0

    # Evict A from the host tier node by node, the last node first, through
    # the call the tree's FULL component frees a node's host value with.
    for host in reversed(a.host_nodes):
        group.free(host, pool=PoolName.KV)
    assert group.available_size() == a.tokens
    host_before_b = _host_pages(stack)

    # B is computed on device rows of its own and backed up into A's slots.
    b.compute(stack)
    assert not set(b.device_rows.tolist()) & set(a.device_rows.tolist())
    _backup(stack, b)
    a_slots, b_slots = set(a.host_slots.tolist()), set(b.host_slots.tolist())
    assert b_slots & a_slots, "B reused none of A's host slots"
    assert b_slots <= a_slots
    # And not in ascending order, as A's were: a transfer that sorted its
    # host slots, or took them for the device rows' order, shows.
    assert b.host_slots.tolist() != sorted(b_slots), "B's host slots are ascending"
    stale_slots = a_slots - b_slots
    assert len(stale_slots) == a.tokens - b.tokens == group.available_size()

    # B's backup replaced the page block of each slot it took, in every host
    # pool, and wrote nowhere else: the filler's pages and the page of A that
    # was not reused are byte for byte what they were.
    b_pages = {slot // PAGE_SIZE for slot in b_slots}
    for (name, page), block in _host_pages(stack).items():
        rewritten = not torch.equal(block, host_before_b[name, page])
        assert rewritten == (page in b_pages), (
            f"host pool {name}, page {page}: "
            + ("rewritten by" if rewritten else "not rewritten by")
            + f" B's backup into pages {sorted(b_pages)}"
        )

    # The device evicts all three prompts and other requests overwrite every
    # row. They hold all rows but the ones A was computed on.
    for prompt in (a, filler, b):
        allocator.free(prompt.device_rows)
    generator = torch.Generator().manual_seed(20261204)
    scrub = {
        key: _random_rows(buffer, buffer.shape[0], generator)
        for key, buffer in buffers.items()
    }
    for key, buffer in buffers.items():
        buffer.copy_(scrub[key])
    assert allocator.alloc(allocator.available_size()) is not None
    allocator.free(a.device_rows)

    # Load B back: into rows A was computed on, from slots A was stored in.
    b_rows = _load(stack, b.node_ids, b.host_nodes)
    assert len(b_rows) == b.tokens
    assert set(b_rows.tolist()) <= set(a.device_rows.tolist())
    stale = {"prompt A": a.rows, "the scrub": scrub, "prompt C": filler.rows}
    _assert_restored(stack, b_rows, b.rows, "prompt B", stale)
    # Nothing but B's rows was written, and no row of A reached the device.
    for key, buffer in buffers.items():
        other = torch.ones(buffer.shape[0], dtype=torch.bool, device=DEVICE)
        other[b_rows] = False
        assert torch.equal(buffer[other], scrub[key][other]), key
        assert not _tokens_holding(buffer, a.rows[key]), key

    # The prompt that stayed in the host tier is intact after its neighbours
    # were freed and rewritten; it comes back into the rows B was computed on.
    allocator.free(b.device_rows)
    filler_rows = _load(stack, filler.node_ids, filler.host_nodes)
    assert set(filler_rows.tolist()) <= set(b.device_rows.tolist())
    stale = {"prompt A": a.rows, "the scrub": scrub, "prompt B": b.rows}
    _assert_restored(stack, filler_rows, filler.rows, "prompt C", stale)
    stale = {"prompt A": a.rows, "the scrub": scrub, "prompt C": filler.rows}
    _assert_restored(stack, b_rows, b.rows, "prompt B", stale)

    if not stale_slots:
        return
    # Control for the checks above: the slots B did not take are free and
    # still hold A's page in every pool, so a transfer that named them would
    # have put A's rows on the device. Name them on purpose.
    assert not group.anchor_entry.host_pool.slot_used[sorted(stale_slots)].any()
    positions = [i for i, s in enumerate(a.host_slots.tolist()) if s in stale_slots]
    stale_rows = _load(stack, [a.node_ids[0]], [a.host_slots[positions]])
    expected = {key: rows[positions] for key, rows in a.rows.items()}
    _assert_restored(stack, stale_rows, expected, "A's stale page", {})


@pytest.mark.parametrize(
    "b_node_pages", [(2, 2), (1, 2)], ids=["as_many_pages", "fewer_pages"]
)
@pytest.mark.parametrize("head_partition", [1, 2, None])
def test_draft_pool_beside_kv_follows_it_into_reused_host_slots(
    build_stack, head_partition, b_node_pages
):
    """DFLASH without its ring: a draft KV pool of its own geometry beside the
    target's, each cut into the unified head groups (or, without the unified
    key scheme, into one)."""
    scheme = {}
    if head_partition is not None:
        scheme = dict(
            hicache_storage_key_scheme="unified",
            hicache_storage_head_partition=head_partition,
            hicache_storage_draft_id="org/dflash@main",
            speculative_algorithm="DFLASH",
        )
    stack = build_stack(
        target=_mha_pool(TARGET_LAYERS),
        sidecar_draft=_mha_pool(2, head_num=4, head_dim=64, dtype=torch.float16),
        **scheme,
    )
    assert [spec.pool_name for spec in stack.sidecar_specs] == [PoolName.DRAFT]
    assert [spec.indices_from_pool for spec in stack.sidecar_specs] == [PoolName.KV]
    groups = {
        entry.name: entry.host_pool.page_unified_layout.head_group_num
        for entry in stack.group.entries
    }
    if head_partition is None:
        assert groups == {PoolName.KV: 1, PoolName.DRAFT: 1}
    else:
        assert groups == {
            PoolName.KV: 8 // head_partition,
            PoolName.DRAFT: 4 // head_partition,
        }
        _assert_unified_plan_cuts_the_host_pools(stack)

    _reuse_host_slots(stack, b_node_pages)


@pytest.mark.parametrize(
    "b_node_pages", [(2, 2), (1, 2)], ids=["as_many_pages", "fewer_pages"]
)
@pytest.mark.parametrize(
    "attention,draft_num,unified",
    [
        ("mha", 1, True),
        # The head partition alone decides the head groups a host pool is cut
        # into. The key plan stores one packed draft layer and refuses three
        # at start-up, so this pins the L2 geometry only.
        ("mha", 3, True),
        ("mha", 3, False),
        ("mla", 1, True),
        ("mla", 3, False),
    ],
)
def test_packed_draft_layers_follow_the_page_into_reused_host_slots(
    build_stack, attention, draft_num, unified, b_node_pages
):
    """NEXTN-style MTP: one-layer draft pools packed behind the target's
    layers of the page block, each depth with rows of its own."""
    make_pool = _mha_pool if attention == "mha" else _mla_pool
    scheme = {}
    if unified:
        scheme = dict(
            hicache_storage_key_scheme="unified",
            hicache_storage_head_partition=2 if attention == "mha" else None,
            hicache_storage_draft_id="bundled",
            speculative_algorithm="EAGLE",
        )
    stack = build_stack(
        target=make_pool(TARGET_LAYERS),
        packed_drafts=tuple(make_pool(1) for _ in range(draft_num)),
        **scheme,
    )
    assert not stack.sidecar_specs
    (entry,) = stack.group.entries
    host_pool = entry.host_pool
    assert len(entry.packed_draft_device_pools) == draft_num
    assert host_pool.target_layer_num == TARGET_LAYERS
    assert host_pool.layer_num == TARGET_LAYERS + draft_num
    assert host_pool.page_unified_layout.head_group_num == (
        4 if unified and attention == "mha" else 1
    )
    if unified and draft_num == 1:
        _assert_unified_plan_cuts_the_host_pools(stack)

    _reuse_host_slots(stack, b_node_pages)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
