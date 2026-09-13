"""End-to-end KV *content* checks for UnifiedRadixCache + HiCache, on CPU.

Every other unit test in this directory asserts bookkeeping: which node is in
which leaf set, what a lock_ref is, how many transfers were issued. None asserts
what a served request actually depends on -- that the KV slots a match hands
back hold the bytes that were cached under those token ids. A cache can keep
every counter consistent, pass ``sanity_check``, and still serve one request
another request's KV; that reaches the client as a plausible completion or an
immediate stop token, never as an error, so there is nothing in a log to find.

This file runs the real stack -- ``UnifiedRadixCache``, ``HiCacheController``,
the host pool, real D->H and H->D copies -- over fixed workloads, and checks
every matched prefix byte for byte. Each KV slot is stamped with its token id,
so a mismatch names the position, the layer, and the token that should be there.

The two fixture knobs that decide what gets covered:

* ``deferred_dma`` holds each copy until the fixture releases it, instead of
  letting it land at submit time. It is what makes an ack-before-copy bug
  visible: acking a write-back whose D->H has not run corrupts 150 prefixes in
  this file's workload with it on, and none with it off.
* the workload shape decides chain length. A conversation that only grows is one
  radix node, and its load-back is a single step, so ``branching_corpus`` is
  what puts several evicted nodes on one root path and makes
  ``split_full_load_back_spec`` do anything.

What is faked: accelerator streams and events, ``transfer_kv_direct`` (a CUDA op
restated in torch as the indexed copy it is), the host-memory budget check, and
``cudaHostRegister``. Admission, load-back, eviction, write-back, and the
controller's queues, acks and layer counter are all the shipped code.

Two things stay out of reach here: the unaligned page tail (the CPU fixture has
no ``alloc_extend``, so sequences are page-aligned) and anything that needs two
TP ranks.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import contextlib
import random
import unittest
import unittest.mock as mock
from typing import Optional

import msgspec
import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InitLoadBackParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName, SidecarPoolSpec
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import build_pool_entry
from sglang.srt.mem_cache.l2_transfer import TransferCompletion
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase

FULL = ComponentType.FULL

# fp32 KV so a token id round-trips through the pools exactly.
KV_DTYPE = torch.float32
HEAD_NUM = 1
HEAD_DIM = 4
LAYER_NUM = 2
# Admission reserves decode headroom it never spends here. Charging it is what
# keeps the batch from over-committing the pool, so it has to be realistic
# rather than 1 -- at 1 the reservation is too small to hold anything back.
MAX_NEW_TOKENS = 8


# ---------------------------------------------------------------------------
# CPU stand-ins for the accelerator primitives the transfer path uses
# ---------------------------------------------------------------------------


class _FakeEvent:
    """A completion flag, not a timestamp.

    ``pending`` events carry the transfer itself: nothing is copied until the
    event completes, which is what makes the fixture able to hold a DMA in
    flight while the tree keeps mutating around it.
    """

    def __init__(self, enable_timing: bool = False):
        self.enable_timing = enable_timing
        self.work = None
        self.done = True

    def arm(self, work) -> "_FakeEvent":
        self.work = work
        self.done = False
        return self

    def complete(self) -> None:
        if self.done:
            return
        self.done = True
        if self.work is not None:
            self.work()
            self.work = None

    def record(self, *args, **kwargs):
        pass

    def synchronize(self):
        self.complete()

    def query(self):
        return self.done

    def wait(self, *args, **kwargs):
        pass

    def elapsed_time(self, other):
        return 0.0


class _FakeStream:
    def __init__(self, *args, **kwargs):
        pass

    def wait_event(self, *args, **kwargs):
        pass

    def synchronize(self):
        pass


class _FakeDeviceModule:
    Event = _FakeEvent
    Stream = _FakeStream

    @staticmethod
    @contextlib.contextmanager
    def stream(stream):
        yield

    @staticmethod
    def current_stream():
        return _FakeStream()

    @staticmethod
    def synchronize():
        pass


class DeferredTransferEngine:
    """``L2TransferEngine`` with the copies held until the fixture releases them.

    The shipped engine issues each copy onto an accelerator stream and returns;
    the data lands later. Running the copy inline on CPU would hide every bug
    that needs a transfer to still be reading a host slot, or writing a device
    slot, while the tree hands that slot to someone else -- which is most of
    what the write-back and load-back pins exist to prevent.

    One queue per direction, completed in order, as a stream would.
    """

    def __init__(self, io_backend: str):
        self.io_backend = io_backend
        self.host_to_device: list[_FakeEvent] = []
        self.device_to_host: list[_FakeEvent] = []

    def submit_device_to_host(self, transfers):
        def work():
            for transfer in transfers:
                transfer.host_pool.backup_from_device_all_layer(
                    transfer.device_pool,
                    transfer.host_indices,
                    transfer.device_indices,
                    self.io_backend,
                )

        finish = _FakeEvent().arm(work)
        self.device_to_host.append(finish)
        return TransferCompletion(_FakeEvent(), finish, False)

    def submit_host_to_device(
        self, transfers, *, layer_num, start_event=None, on_layer_done=None
    ):
        primary = transfers[0] if transfers else None

        def work():
            for layer_id in range(layer_num):
                for transfer in transfers:
                    local_layer_id = (
                        transfer.layer_mapper(layer_id)
                        if transfer.layer_mapper is not None
                        else layer_id
                    )
                    if local_layer_id is None or (
                        transfer is not primary
                        and transfer.layer_mapper is None
                        and layer_id >= transfer.host_pool.layer_num
                    ):
                        continue
                    transfer.host_pool.load_to_device_per_layer(
                        transfer.device_pool,
                        transfer.host_indices,
                        transfer.device_indices,
                        local_layer_id,
                        self.io_backend,
                        is_draft=transfer.is_draft,
                    )
                if on_layer_done is not None:
                    on_layer_done(layer_id)

        finish = _FakeEvent().arm(work)
        self.host_to_device.append(finish)
        return TransferCompletion(_FakeEvent(), finish, False)

    def _release(self, queue: list, count: Optional[int]) -> int:
        take = len(queue) if count is None else min(count, len(queue))
        for event in queue[:take]:
            event.complete()
        del queue[:take]
        # Anything the cache force-synchronized is already done; drop it so the
        # queues stay a picture of what is still in flight.
        queue[:] = [event for event in queue if not event.done]
        return take

    def release_host_to_device(self, count: Optional[int] = None) -> int:
        return self._release(self.host_to_device, count)

    def release_device_to_host(self, count: Optional[int] = None) -> int:
        return self._release(self.device_to_host, count)

    @property
    def in_flight(self) -> int:
        return sum(
            1
            for queue in (self.device_to_host, self.host_to_device)
            for event in queue
            if not event.done
        )


def _transfer_kv_direct(src_layers, dst_layers, src_indices, dst_indices, page_size):
    """torch restatement of ``sgl_kernel.kvcacheio.transfer_kv_direct``.

    The kernel coalesces runs of consecutive indices and copies one src buffer
    into one dst buffer per layer; ``page_size`` only guards divisibility. The
    observable result is an indexed copy.
    """
    assert len(src_layers) == len(dst_layers)
    assert src_indices.numel() == dst_indices.numel()
    assert page_size > 0 and src_indices.numel() % page_size == 0
    for src, dst in zip(src_layers, dst_layers):
        dst[dst_indices] = src[src_indices].to(dst.dtype)


def cpu_hicache_patches():
    """Patches that make the real HiCache stack constructible and runnable on CPU."""
    import sglang.srt.managers.cache_controller as cache_controller
    import sglang.srt.mem_cache.l2_transfer as l2_transfer
    import sglang.srt.mem_cache.pool_host.base as pool_host_base
    import sglang.srt.mem_cache.pool_host.common as pool_host_common
    import sglang.srt.mem_cache.pool_host.mha as pool_host_mha

    fake_device = _FakeDeviceModule()
    return [
        mock.patch.object(cache_controller, "device_module", fake_device),
        mock.patch.object(l2_transfer, "device_module", fake_device),
        mock.patch.object(l2_transfer, "_timing_events_supported", lambda: False),
        # psutil reports the whole box; a shared login node can read as negative
        # free memory long before the fixture's few MB matter.
        mock.patch.object(pool_host_base, "host_memory_budget_bytes", lambda: 1 << 34),
        mock.patch.object(
            pool_host_common,
            "_cuda_host_register",
            lambda buf, granularity_bytes=None: None,
        ),
        mock.patch.object(pool_host_common, "_cuda_host_unregister", lambda buf: None),
        mock.patch.object(
            pool_host_mha, "transfer_kv_direct", _transfer_kv_direct, create=True
        ),
    ]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


class HiCacheFixture:
    """A real UnifiedRadixCache with a real HiCache controller, on CPU.

    ``device_size`` and ``host_ratio`` are the two knobs that create pressure:
    small device pool forces L1 eviction (and, under write_back, the D->H write
    that funds it), small host pool forces host reclaim.
    """

    def __init__(
        self,
        stack: contextlib.ExitStack,
        *,
        device_size: int = 96,
        host_ratio: float = 3.0,
        page_size: int = 1,
        serialize_load_back: bool = False,
        allow_subagent_keepalive: bool = False,
        write_policy: str = "write_back",
        deferred_dma: bool = False,
    ):
        server_args = ServerArgs(
            model_path="dummy",
            page_size=page_size,
            hicache_io_backend="direct",
            hicache_mem_layout="layer_first",
            hicache_write_policy=write_policy,
            hicache_host_memory_mode="cache",
            hicache_ratio=host_ratio,
            hicache_serialize_load_back=serialize_load_back,
        )
        set_global_server_args_for_scheduler(server_args)

        self.page_size = page_size
        self.kv_pool = MHATokenToKVPool(
            size=device_size,
            page_size=page_size,
            dtype=KV_DTYPE,
            head_num=HEAD_NUM,
            head_dim=HEAD_DIM,
            layer_num=LAYER_NUM,
            device="cpu",
            enable_memory_saver=False,
        )
        allocator_cls = (
            TokenToKVPoolAllocator if page_size == 1 else PagedTokenToKVPoolAllocator
        )
        allocator_kwargs = {} if page_size == 1 else {"page_size": page_size}
        self.allocator = allocator_cls(
            size=device_size,
            dtype=KV_DTYPE,
            device="cpu",
            kvcache=self.kv_pool,
            need_sort=False,
            **allocator_kwargs,
        )
        self.req_pool = ReqToTokenPool(
            size=64, max_context_len=4096, device="cpu", enable_memory_saver=False
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.req_pool,
            token_to_kv_pool_allocator=self.allocator,
            page_size=page_size,
            eviction_policy="lru",
            tree_components=(FULL,),
            allow_subagent_keepalive=allow_subagent_keepalive,
        )
        self.cache = UnifiedRadixCache(params)
        for patch in cpu_hicache_patches():
            stack.enter_context(patch)
        self.cache.init_hicache(server_args, params)
        stack.callback(self.cache.release_host_resources)
        self.engine = None
        if deferred_dma:
            self.engine = DeferredTransferEngine(
                self.cache.cache_controller.l2_transfer_engine.io_backend
            )
            self.cache.cache_controller.l2_transfer_engine = self.engine
        # Cache every node in both directions; the fixture is about placement,
        # not about the size heuristics.
        self.cache.write_through_threshold = 1
        self.cache.load_back_threshold = 1
        self.device_size = device_size
        self.host_pool = self.cache.cache_controller.mem_pool_host
        # Pools reserve slot 0 as padding, so read the free counts rather than
        # assuming they equal the configured sizes.
        self.sidecar = None
        self.device_capacity = self.allocator.available_size()
        self.host_capacity = self.host_pool.available_size()
        self._next_rid = 0

    # -- KV fingerprints ----------------------------------------------------

    def stamp(self, indices: torch.Tensor, tokens) -> None:
        """Write each token id into every layer of its own KV slot."""
        stamp = torch.tensor(tokens, dtype=KV_DTYPE).view(-1, 1, 1)
        for layer in range(LAYER_NUM):
            self.kv_pool.k_buffer[layer][indices] = stamp.expand(-1, HEAD_NUM, HEAD_DIM)
            self.kv_pool.v_buffer[layer][indices] = -stamp.expand(
                -1, HEAD_NUM, HEAD_DIM
            )

    def read_stamps(self, indices: torch.Tensor, layer: int = 0) -> list[int]:
        if indices.numel() == 0:
            return []
        return self.kv_pool.k_buffer[layer][indices][:, 0, 0].to(torch.int64).tolist()

    def check_stamps(self, indices: torch.Tensor, tokens) -> Optional[str]:
        """None if every layer of every slot holds its token's stamp."""
        for layer in range(LAYER_NUM):
            k = self.read_stamps(indices, layer)
            if k != list(tokens):
                return f"k_buffer layer {layer}: expected {list(tokens)}, got {k}"
            v = (
                self.kv_pool.v_buffer[layer][indices][:, 0, 0].to(torch.int64).tolist()
                if indices.numel()
                else []
            )
            if v != [-t for t in tokens]:
                return (
                    f"v_buffer layer {layer}: expected {[-t for t in tokens]}, got {v}"
                )
        return None

    # -- KV-derived sidecar pool --------------------------------------------

    def attach_sidecar(self) -> None:
        """Register a second pool whose indices come from the KV pool.

        This is the shape NSA/DSA models run with: their stack registers
        ``SidecarPoolSpec(pool_name=INDEXER, indices_from_pool=KV)``, and the
        controller resolves such a transfer's indices from the operation it
        travels on. Nothing else in the fixture has one, and a cache that moves
        Full KV correctly can still leave the sidecar behind.
        """
        device_pool = MHATokenToKVPool(
            size=self.device_size,
            page_size=self.page_size,
            dtype=KV_DTYPE,
            head_num=HEAD_NUM,
            head_dim=HEAD_DIM,
            layer_num=LAYER_NUM,
            device="cpu",
            enable_memory_saver=False,
        )
        host_pool = MHATokenToKVPoolHost(
            device_pool=device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=self.page_size,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
        )
        self.cache.register_sidecar_pool(
            SidecarPoolSpec(pool_name=PoolName.INDEXER, indices_from_pool=PoolName.KV),
            build_pool_entry(
                name=PoolName.INDEXER,
                host_pool=host_pool,
                device_pool=device_pool,
                layer_mapping={i: i for i in range(LAYER_NUM)},
                transfer_layer_num=LAYER_NUM,
            ),
        )
        self.sidecar = device_pool

    def stamp_sidecar(self, indices: torch.Tensor, tokens) -> None:
        stamp = torch.tensor(tokens, dtype=KV_DTYPE).view(-1, 1, 1)
        for layer in range(LAYER_NUM):
            self.sidecar.k_buffer[layer][indices] = stamp.expand(-1, HEAD_NUM, HEAD_DIM)

    def check_sidecar(self, indices: torch.Tensor, tokens) -> Optional[str]:
        for layer in range(LAYER_NUM):
            got = (
                self.sidecar.k_buffer[layer][indices][:, 0, 0].to(torch.int64).tolist()
                if indices.numel()
                else []
            )
            if got != list(tokens):
                wrong = sum(1 for a, b in zip(got, tokens) if a != b)
                return (
                    f"sidecar layer {layer}: {wrong}/{len(got)} slots hold "
                    f"another request's tokens"
                )
        return None

    # -- plumbing -----------------------------------------------------------

    def make_req(self, tokens, *, session_id: str = "") -> Req:
        req = Req(
            rid=str(self._next_rid),
            origin_input_text="",
            origin_input_ids=list(tokens),
            sampling_params=SamplingParams(
                temperature=0, max_new_tokens=MAX_NEW_TOKENS
            ),
            session_id=session_id,
        )
        self._next_rid += 1
        req.output_ids = []
        req.parent_session_id = ""
        self.req_pool.alloc([req])
        return req

    def release_host_to_device(self, count: Optional[int] = None) -> int:
        """Let queued H->D transfers land. No-op with an inline engine."""
        return 0 if self.engine is None else self.engine.release_host_to_device(count)

    def release_device_to_host(self, count: Optional[int] = None) -> int:
        """Let queued D->H write-backs land. No-op with an inline engine."""
        return 0 if self.engine is None else self.engine.release_device_to_host(count)

    def alloc_with_eviction(self, need: int) -> Optional[torch.Tensor]:
        """``alloc_token_slots``: evict only the shortfall, then allocate."""
        available = self.allocator.available_size()
        if available < need:
            self.cache.evict(EvictParams(num_tokens=need - available))
        return self.allocator.alloc(need)

    def alloc_extend(self, req: Req, extend_len: int) -> Optional[torch.Tensor]:
        """``PagedTokenToKVPoolAllocator.alloc_extend``, in torch.

        The shipped one is a Triton kernel, so a CPU fixture has to restate it:
        finish the prefix's partial page from ``last_loc + 1``, then take whole
        pages for the rest. Restating it is what buys the page_size > 1 shape --
        an unaligned prefix, which is the only case where
        ``cache_protected_len`` differs from ``len(prefix_indices)``.
        """
        page_size = self.page_size
        if page_size == 1:
            return self.alloc_with_eviction(extend_len)

        prefix_len = req.extend_range.start
        parts: list[torch.Tensor] = []
        remaining = extend_len
        partial = prefix_len % page_size
        if partial and remaining:
            last_loc = int(
                self.req_pool.req_to_token[req.kv.req_pool_idx, prefix_len - 1]
            )
            take = min(page_size - partial, remaining)
            parts.append(torch.arange(last_loc + 1, last_loc + 1 + take))
            remaining -= take
        if remaining:
            # Whole pages; the last page's unused tail stays with the request
            # until the page is released.
            pages = self.alloc_with_eviction(-(-remaining // page_size) * page_size)
            if pages is None:
                return None
            parts.append(pages[:remaining])
        return torch.cat(parts) if parts else torch.empty(0, dtype=torch.int64)

    @property
    def budget(self) -> int:
        """What ``PrefillAdder.rem_total_tokens`` starts from."""
        return self.allocator.available_size() + self.cache.evictable_size()

    # -- conservation -------------------------------------------------------

    def _tree_nodes(self):
        stack = [self.cache.tree_core.root_node]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if node is not self.cache.tree_core.root_node:
                yield node

    def slot_census(self) -> tuple[int, int]:
        """(device, host) slots the tree currently owns."""
        device = host = 0
        for node in self._tree_nodes():
            data = node.component_data[FULL]
            if data.value is not None:
                device += len(data.value)
            if data.host_value is not None:
                host += len(data.host_value)
        return device, host

    def conservation_error(self, request_held: int = 0) -> Optional[str]:
        """Every slot is owned by a tree node, held by a live request, or free.

        A load-back that overwrites a node's device indices, or drops a
        ``host_value`` it still owns, strands the old slots: they are in no node
        and in no free list, and the pool shrinks until allocation fails.

        ``request_held`` covers a chunked request's prefilled tail, which is in
        ``req_to_token`` but not yet in the tree.
        """
        device_owned, host_owned = self.slot_census()
        device_total = device_owned + request_held + self.allocator.available_size()
        if device_total != self.device_capacity:
            return (
                f"device slots: {device_owned} owned + {request_held} in flight + "
                f"{self.allocator.available_size()} free = {device_total}, "
                f"expected {self.device_capacity}"
            )
        host_total = host_owned + self.host_pool.available_size()
        if host_total != self.host_capacity:
            return (
                f"host slots: {host_owned} owned + "
                f"{self.host_pool.available_size()} free = {host_total}, "
                f"expected {self.host_capacity}"
            )
        return None


# ---------------------------------------------------------------------------
# A scheduler loop, in the order scheduler.py runs it
# ---------------------------------------------------------------------------


class StepResult(msgspec.Struct):
    admitted: list[str] = []
    deferred: int = 0
    prefix_lens: dict[str, int] = {}
    host_hits: dict[str, int] = {}
    corruption: list[str] = []
    chunked: int = 0

    @property
    def total_host_hit(self) -> int:
        return sum(self.host_hits.values())


class MiniScheduler:
    """The cache-facing half of the scheduler loop, over a fixed arrival trace.

    Only the calls that reach the cache are modelled, in the order
    ``get_next_batch_to_run`` -> ``get_new_batch_prefill`` -> forward ->
    ``process_batch_result`` makes them:

        release the previous step's write-backs, check_hicache_events
        stash the in-flight chunk         cache_unfinished_req(chunked=True)
        admit the chunk                   add_chunked_req (no load-back)
        admit new requests                match_prefix -> init_load_back -> lock
        ready_to_load_host_cache          the batched H->D
        allocate, evicting the shortfall
        forward                           read the prefix, write the extension
        completion                        cache_finished_req

    ``max_prefill_tokens`` is the served ``--max-prefill-tokens``: a request
    longer than it is prefilled over several steps, and between chunks its
    partial KV goes into the tree and its prefix is re-matched. Its prefix has
    to stay intact across every one of those steps, which is a longer exposure
    than a single-shot request ever gets.
    """

    def __init__(
        self,
        fx: HiCacheFixture,
        *,
        max_prefill_tokens: Optional[int] = None,
        keepalive: bool = False,
    ):
        self.fx = fx
        self.max_prefill_tokens = max_prefill_tokens or 1 << 30
        self.keepalive = keepalive
        self.waiting: list[Req] = []
        self.chunked_req: Optional[Req] = None

    # -- arrival -------------------------------------------------------------

    def submit(self, tokens, *, session_id="", parent_session_id="") -> None:
        req = self.fx.make_req(tokens, session_id=session_id)
        req.parent_session_id = parent_session_id
        self.waiting.append(req)

    # -- one iteration -------------------------------------------------------

    def step(self) -> StepResult:
        fx, cache = self.fx, self.fx.cache
        fx.release_device_to_host()
        cache.check_hicache_events()

        result = StepResult()
        self._stash_previous_chunk()

        can_run: list[Req] = []
        committed = 0
        if self.chunked_req is not None:
            committed += self._admit_chunk(self.chunked_req, can_run, result)

        # A truncated request spends the batch's whole chunk budget, so it is
        # the last one admitted -- scheduler.py asserts one chunked request at
        # a time.
        while self.waiting and self.chunked_req is None:
            req = self.waiting[0]
            if committed + len(req.origin_input_ids) + MAX_NEW_TOKENS > (
                fx.device_capacity
            ):
                result.deferred += 1
                break
            self.waiting.pop(0)
            committed += self._admit(req, can_run, result)

        # The batched H->D for everything admitted above. Serialized load-back
        # has already drained its own transfers, so this finds an empty queue.
        consumer_index = cache.ready_to_load_host_cache()
        self._allocate(can_run)
        # Only the forward waits on the load. Admission and allocation above ran
        # with it in flight, which is the window the load-back pins must cover.
        fx.release_host_to_device()
        self._forward(can_run, result, consumer_index)
        self._complete(can_run)
        return result

    # -- the pieces ----------------------------------------------------------

    def _stash_previous_chunk(self) -> None:
        """scheduler.get_next_batch_to_run: cache the chunk that just ran."""
        req = self.chunked_req
        if req is None or req.extend_range.end <= len(req.prefix_indices):
            return
        self.fx.cache.cache_unfinished_req(req, chunked=True)

    def _admit_chunk(self, req: Req, can_run: list[Req], result: StepResult) -> int:
        """PrefillAdder.add_chunked_req. No match, no load-back: the chunk
        carries the prefix_indices cache_unfinished_req left on it."""
        req.init_next_round_input()
        prefix_len = len(req.prefix_indices)
        remaining = len(req.full_untruncated_fill_ids) - prefix_len
        take = min(remaining, self.max_prefill_tokens)
        req.set_extend_range(prefix_len, prefix_len + take)
        can_run.append(req)
        result.admitted.append(req.rid)
        result.prefix_lens[req.rid] = prefix_len
        result.host_hits.setdefault(req.rid, 0)
        result.chunked += 1
        if take == remaining:
            self.chunked_req = None
        return req.extend_range.end

    def _admit(self, req: Req, can_run: list[Req], result: StepResult) -> int:
        """PrefillAdder.add_one_req, down to the calls that touch the cache."""
        cache = self.fx.cache
        if self.keepalive and req.parent_session_id:
            # What a subagent request does to its parent on arrival.
            cache.bump_session_keepalive(req.parent_session_id)

        req.init_next_round_input(cache)
        result.host_hits[req.rid] = req.host_hit_length

        guard = cache.inc_lock_ref(req.last_node)
        guard_node = req.last_node
        if req.needs_host_load_back():
            new_indices, req.last_node = cache.init_load_back(
                InitLoadBackParams(
                    best_match_node=req.best_match_node,
                    host_hit_length=req.host_hit_length,
                    req=req,
                )
            )
            req.prefix_indices = torch.cat([req.prefix_indices, new_indices])
            req.kv.cache_protected_len = len(req.prefix_indices)
        # _req_inc_lock_ref, then release the guard.
        cache.inc_lock_ref(req.last_node)
        req.skip_lock_node_ids = {}
        cache.dec_lock_ref(guard_node, guard.to_dec_params())

        prefix_len = len(req.prefix_indices)
        take = min(len(req.origin_input_ids) - prefix_len, self.max_prefill_tokens)
        req.set_extend_range(prefix_len, prefix_len + take)
        if req.extend_range.end < len(req.origin_input_ids):
            assert self.chunked_req is None, "one chunked request at a time"
            self.chunked_req = req
            result.chunked += 1

        can_run.append(req)
        result.admitted.append(req.rid)
        result.prefix_lens[req.rid] = prefix_len
        return req.extend_range.end

    def _allocate(self, can_run: list[Req]) -> None:
        """alloc_for_extend: evict the shortfall, then write req_to_token."""
        fx = self.fx
        for req in can_run:
            extend = fx.alloc_extend(req, req.extend_range.length)
            assert extend is not None, (
                f"req {req.rid} could not allocate {req.extend_range.length} "
                f"slots (available {fx.allocator.available_size()}, "
                f"evictable {fx.cache.evictable_size()})"
            )
            req.out_cache_loc = extend
            prefix_len = len(req.prefix_indices)
            fx.req_pool.write(
                (req.kv.req_pool_idx, slice(0, prefix_len)), req.prefix_indices
            )
            fx.req_pool.write(
                (req.kv.req_pool_idx, slice(prefix_len, req.extend_range.end)), extend
            )

    def _forward(
        self, can_run: list[Req], result: StepResult, consumer_index: int
    ) -> None:
        """Read what the attention would read, then write this chunk's KV."""
        fx = self.fx
        for req in can_run:
            seen = fx.req_pool.req_to_token[
                req.kv.req_pool_idx, : req.extend_range.start
            ]
            expected = req.origin_input_ids[: req.extend_range.start]
            problem = fx.check_stamps(seen, expected)
            if problem is not None:
                result.corruption.append(
                    f"req {req.rid} (prefix {req.extend_range.start}, host hit "
                    f"{result.host_hits.get(req.rid, 0)}, chunk "
                    f"[{req.extend_range.start}:{req.extend_range.end}), "
                    f"consumer {consumer_index}): {problem}"
                )
            if fx.sidecar is not None:
                problem = fx.check_sidecar(seen, expected)
                if problem is not None:
                    result.corruption.append(
                        f"req {req.rid} (prefix {req.extend_range.start}, host "
                        f"hit {result.host_hits.get(req.rid, 0)}): {problem}"
                    )
            new_tokens = req.origin_input_ids[
                req.extend_range.start : req.extend_range.end
            ]
            fx.stamp(req.out_cache_loc, new_tokens)
            if fx.sidecar is not None:
                fx.stamp_sidecar(req.out_cache_loc, new_tokens)

    def request_held_slots(self) -> int:
        """Device slots a chunked request holds beyond what the tree owns.

        Page-rounded: the allocator hands out whole pages, so the tail of the
        request's last page is held by it even though no token uses it yet.
        """
        req = self.chunked_req
        if req is None or req.extend_range is None:
            return 0
        page_size = self.fx.page_size
        held = -(-req.extend_range.end // page_size) * page_size
        return max(0, held - req.kv.cache_protected_len)

    def _complete(self, can_run: list[Req]) -> None:
        for req in can_run:
            if req is self.chunked_req:
                continue
            self.fx.cache.cache_finished_req(
                req, kv_len_to_handle=len(req.origin_input_ids)
            )
            self.fx.req_pool.free(req)


def run_step(fx: HiCacheFixture, batch: list[list[int]]) -> StepResult:
    """One unchunked step, for tests that do not need the loop's state."""
    scheduler = MiniScheduler(fx)
    for tokens in batch:
        scheduler.submit(tokens)
    return scheduler.step()


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------


def branching_corpus(*, depth: int, fanout: int, seg: int) -> list[list[int]]:
    """Leaves of a prefix tree: a shared root segment, then `depth` levels that
    branch `fanout` ways, each adding `seg` tokens.

    Chain length is the whole point. A radix node is created by divergence, so a
    single conversation that only grows is one node and its load-back is one
    step -- the per-node walk never runs. Branching is what puts several evicted
    nodes on one root path, which is the shape ``split_full_load_back_spec``
    exists to split. Every token id is globally unique, so a KV stamp identifies
    which branch and which level wrote it.
    """
    leaves: list[list[int]] = []

    def walk(prefix: list[int], level: int, path: list[int]) -> None:
        if level == depth:
            leaves.append(prefix)
            return
        for branch in range(fanout):
            base = 10_000 * (level + 1) + 1_000 * branch + 100 * len(path)
            walk(prefix + list(range(base, base + seg)), level + 1, path + [branch])

    walk(list(range(1, seg + 1)), 0, [])
    return leaves


def replay(
    corpus: list[list[int]], *, steps: int, per_step: int, seed: int
) -> list[list[list[int]]]:
    """Fixed, seeded reuse order over a corpus. Reuse order is what schedules
    eviction, so it is an input, not a source of flakiness -- the same seed
    replays the same tree, evictions and load-backs on every run."""
    rng = random.Random(seed)
    return [[list(rng.choice(corpus)) for _ in range(per_step)] for _ in range(steps)]


def growing_sessions(
    *, sessions: int, turns: int, base_len: int, turn_len: int, per_step: int
) -> list[list[list[int]]]:
    """Conversations resumed round-robin, each turn appending to its own prefix.

    More sessions than fit on device is the point: a session nobody touched this
    step is unlocked, so it is what eviction takes, and its next turn matches on
    the host tier.
    """
    convs = {
        s: list(range(100_000 * (s + 1), 100_000 * (s + 1) + base_len))
        for s in range(sessions)
    }
    counters = {s: 0 for s in range(sessions)}
    steps: list[list[list[int]]] = []
    cursor = 0
    for _ in range(turns * sessions // per_step):
        step = []
        for _ in range(per_step):
            session = cursor % sessions
            cursor += 1
            step.append(list(convs[session]))
            counters[session] += 1
            start = 100_000 * (session + 1) + 1_000 * counters[session]
            convs[session] = convs[session] + list(range(start, start + turn_len))
        steps.append(step)
    return steps


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class HiCacheKVIntegrityTest(CustomTestCase):
    """A matched prefix must hold the KV it was cached under.

    Nothing else in this directory checks the bytes. A load-back that commits
    the wrong device indices, or reads a host slot that has been handed to
    another node, keeps every counter consistent and every sanity_check green
    while serving one request another request's KV -- which reaches the client
    as a plausible completion or an immediate stop token, never as an error.
    """

    def _drive(self, steps, *, sidecar: bool = False, **fixture_kwargs):
        """Run a workload; return (corruption, host-hit tokens, load-backs)."""
        with contextlib.ExitStack() as stack:
            fixture = HiCacheFixture(stack, **fixture_kwargs)
            if sidecar:
                fixture.attach_sidecar()
            chain_lengths: list[int] = []
            split = fixture.cache.tree_core.split_full_load_back_spec

            def counting_split(kv_xfer):
                out = split(kv_xfer)
                chain_lengths.append(len(out))
                return out

            fixture.cache.tree_core.split_full_load_back_spec = counting_split

            corruption, host_hits = [], 0
            for step in steps:
                result = run_step(fixture, step)
                corruption.extend(result.corruption)
                host_hits += result.total_host_hit
                leak = fixture.conservation_error()
                self.assertIsNone(leak, leak)
                fixture.cache.sanity_check()
            return corruption, host_hits, chain_lengths

    def _assert_intact(self, steps, *, sidecar: bool = False, **fixture_kwargs):
        corruption, host_hits, chains = self._drive(
            steps, sidecar=sidecar, **fixture_kwargs
        )
        self.assertGreater(
            host_hits,
            0,
            "the workload never hit the host tier, so it never exercised "
            "load-back; shrink device_size or lengthen the workload",
        )
        self.assertEqual(
            corruption,
            [],
            f"{len(corruption)} corrupted prefixes:\n" + "\n".join(corruption[:5]),
        )
        return chains

    # -- growing conversations ----------------------------------------------

    def test_growing_sessions_batched(self):
        steps = growing_sessions(
            sessions=12, turns=8, base_len=16, turn_len=8, per_step=2
        )
        self._assert_intact(steps, device_size=96, host_ratio=2.5)

    def test_growing_sessions_serialized(self):
        steps = growing_sessions(
            sessions=12, turns=8, base_len=16, turn_len=8, per_step=2
        )
        self._assert_intact(
            steps, device_size=96, host_ratio=2.5, serialize_load_back=True
        )

    # -- branching corpus, which is what makes chains longer than one node ---

    def test_branching_corpus_batched(self):
        steps = replay(
            branching_corpus(depth=6, fanout=2, seg=8),
            steps=120,
            per_step=2,
            seed=20260908,
        )
        self._assert_intact(steps, device_size=128, host_ratio=1.6)

    def test_branching_corpus_serialized(self):
        steps = replay(
            branching_corpus(depth=6, fanout=2, seg=8),
            steps=120,
            per_step=2,
            seed=20260908,
        )
        chains = self._assert_intact(
            steps, device_size=128, host_ratio=1.6, serialize_load_back=True
        )
        self.assertGreater(
            max(chains, default=0),
            2,
            "no load-back split into more than two steps, so the per-node walk "
            "was never really exercised",
        )

    def test_branching_corpus_serialized_deferred_dma(self):
        """Same replay with the copies held in flight past admission, eviction
        and allocation, so a transfer is only correct if the tokens it named
        still belong to it when it lands.

        Distinct from the inline runs above, not a duplicate of them: an ack
        that fires before its D->H copy has run corrupts 150 prefixes in this
        workload with deferral on and none with it off, because inline copies
        make the ack and the copy the same instant.
        """
        steps = replay(
            branching_corpus(depth=6, fanout=2, seg=8),
            steps=120,
            per_step=2,
            seed=20260908,
        )
        self._assert_intact(
            steps,
            device_size=128,
            host_ratio=1.6,
            serialize_load_back=True,
            deferred_dma=True,
            sidecar=True,
        )

    def test_branching_corpus_batched_deferred_dma(self):
        steps = replay(
            branching_corpus(depth=6, fanout=2, seg=8),
            steps=120,
            per_step=2,
            seed=20260908,
        )
        self._assert_intact(steps, device_size=128, host_ratio=1.6, deferred_dma=True)

    # -- paged --------------------------------------------------------------

    def test_paged_branching_corpus_serialized(self):
        """page_size > 1 is the served configuration; the radix key, the host
        pool and the allocator are all page-indexed there."""
        steps = replay(
            branching_corpus(depth=5, fanout=2, seg=8),
            steps=80,
            per_step=2,
            seed=20260908,
        )
        self._assert_intact(
            steps,
            device_size=256,
            host_ratio=1.6,
            page_size=4,
            serialize_load_back=True,
            sidecar=True,
        )

    def test_paged_branching_corpus_batched(self):
        steps = replay(
            branching_corpus(depth=5, fanout=2, seg=8),
            steps=80,
            per_step=2,
            seed=20260908,
        )
        self._assert_intact(steps, device_size=256, host_ratio=1.6, page_size=4)

    def test_the_stamp_check_reports_a_planted_mismatch(self):
        """Keep the detector honest.

        Every case in this file is an assertion that ``check_stamps`` found
        nothing. If a refactor ever left it comparing something vacuous -- an
        empty slice, a list against itself -- the whole file would go green and
        stay green. So plant a corrupted slot and require it to be named.
        """
        with contextlib.ExitStack() as stack:
            fixture = HiCacheFixture(stack, device_size=32, host_ratio=2.0)
            indices = fixture.allocator.alloc(4)
            tokens = [11, 22, 33, 44]
            fixture.stamp(indices, tokens)
            self.assertIsNone(fixture.check_stamps(indices, tokens))

            fixture.stamp(indices[2:3], [99])
            problem = fixture.check_stamps(indices, tokens)
            self.assertIsNotNone(problem, "a corrupted slot went unreported")
            self.assertIn("99", problem)

            # A layer the workload never reads must still be checked: a
            # per-layer transfer bug can leave layer 0 right and layer 1 wrong.
            fixture.stamp(indices[2:3], [33])
            fixture.kv_pool.k_buffer[LAYER_NUM - 1][indices[1]] = 0
            self.assertIsNotNone(fixture.check_stamps(indices, tokens))

    # -- KV-derived sidecar pools -------------------------------------------

    def test_serialized_load_back_restores_a_kv_derived_sidecar(self):
        """A KV-derived sidecar must be restored for the whole chain, not just
        the node the last step happened to load.

        Such a sidecar (NSA/DSA register one as ``INDEXER``) carries no indices;
        the controller resolves them from the operation it rides on. Attaching
        one built for the whole chain to the final per-node step restores it for
        that node's tokens only, and every earlier node of the chain keeps
        whatever its freshly allocated slots held -- another request's data. The
        Full KV is correct throughout, so nothing in the tree looks wrong; the
        model just attends with someone else's index state.
        """
        steps = replay(
            branching_corpus(depth=6, fanout=2, seg=8),
            steps=120,
            per_step=2,
            seed=20260908,
        )
        for serialize in (False, True):
            with self.subTest(serialize_load_back=serialize):
                with contextlib.ExitStack() as stack:
                    fixture = HiCacheFixture(
                        stack,
                        device_size=128,
                        host_ratio=1.6,
                        serialize_load_back=serialize,
                    )
                    fixture.attach_sidecar()
                    chains: list[int] = []
                    split = fixture.cache.tree_core.split_full_load_back_spec

                    def counting_split(kv_xfer):
                        out = split(kv_xfer)
                        chains.append(len(out))
                        return out

                    fixture.cache.tree_core.split_full_load_back_spec = counting_split
                    scheduler = MiniScheduler(fixture)
                    corruption = []
                    for step in steps:
                        for tokens in step:
                            scheduler.submit(tokens)
                        corruption.extend(scheduler.step().corruption)
                    if serialize:
                        self.assertGreater(
                            max(chains, default=0),
                            1,
                            "no load-back split into more than one step, so the "
                            "sidecar could not have been left behind either way",
                        )
                    self.assertEqual(
                        corruption,
                        [],
                        f"{len(corruption)} requests read a stale prefix:\n"
                        + "\n".join(corruption[:5]),
                    )


class SerializedLoadBackControllerTest(CustomTestCase):
    """The serialized path's contract with the real cache controller.

    ``test_serialized_load_back.py`` covers the same code against a fake
    controller and a stubbed drain. These run it against the shipped
    ``HiCacheController`` -- its queues, its acks, its layer counter.
    """

    STEPS = replay(
        branching_corpus(depth=5, fanout=2, seg=8),
        steps=80,
        per_step=2,
        seed=20260908,
    )

    def test_drain_leaves_no_queued_transfer_for_the_batch(self):
        """Draining inside admission is what lets the batch's
        ``ready_to_load_host_cache`` return -1 and the forward skip the layer
        wait. A transfer left queued would be submitted with no consumer, and a
        left-over ack would be miscounted by the next ``loading_check``."""
        with contextlib.ExitStack() as stack:
            fixture = HiCacheFixture(
                stack, device_size=128, host_ratio=1.6, serialize_load_back=True
            )
            controller = fixture.cache.cache_controller
            saw_load_back = False
            for step in self.STEPS:
                result = run_step(fixture, step)
                saw_load_back = saw_load_back or bool(result.total_host_hit)
                self.assertEqual(
                    controller.load_queue, [], "a load-back transfer stayed queued"
                )
                self.assertEqual(
                    controller.ack_load_queue, [], "a load-back ack stayed unconsumed"
                )
            self.assertTrue(saw_load_back, "workload never exercised load-back")

    def test_per_node_drains_do_not_trip_the_producer_ring(self):
        """``LayerDoneCounter`` has three slots and ``update_producer`` asserts
        the slot it wraps onto has finished. Per-node draining issues one
        producer per node instead of one per batch, so it laps the ring far
        sooner than the batched path ever does."""
        with contextlib.ExitStack() as stack:
            fixture = HiCacheFixture(
                stack, device_size=128, host_ratio=1.6, serialize_load_back=True
            )
            counter = fixture.cache.cache_controller.layer_done_counter
            producers = 0
            update = counter.update_producer

            def counting_update():
                nonlocal producers
                producers += 1
                return update()

            counter.update_producer = counting_update
            for step in self.STEPS:
                run_step(fixture, step)
            self.assertGreater(
                producers,
                counter.num_counters,
                "the run never lapped the producer ring, so the assert it "
                "guards was never reached",
            )

    def test_serialized_load_back_does_not_reuse_less_than_the_batched_path(self):
        """Splitting the load-back exists to raise reuse, not to trade it away.

        Loading a chain as one transfer demands device room for the whole chain
        at the moment none of it is a reclaimable host duplicate, so the
        write-back that eviction cascades into can only be funded by destroying
        a sole host copy -- someone else's cached prefix. Per-node steps make
        each node a duplicate at its own ack. On a fixed replay the effect is
        measurable: the serialized run must reuse at least as many prefix tokens
        as the batched one.
        """
        reused = {}
        for serialize in (False, True):
            with contextlib.ExitStack() as stack:
                fixture = HiCacheFixture(
                    stack,
                    device_size=128,
                    host_ratio=1.6,
                    serialize_load_back=serialize,
                )
                total = 0
                for step in self.STEPS:
                    result = run_step(fixture, step)
                    self.assertEqual(result.corruption, [])
                    total += sum(result.prefix_lens.values())
                reused[serialize] = total
        self.assertGreater(reused[False], 0, "the control run reused nothing")
        self.assertGreaterEqual(
            reused[True],
            reused[False],
            f"serialized load-back reused {reused[True]} prefix tokens against "
            f"{reused[False]} for the batched path; the split is supposed to "
            f"protect host copies, not cost them",
        )


if __name__ == "__main__":
    unittest.main()
