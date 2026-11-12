from array import array

from sglang.srt.disaggregation.kv_events import (
    BlockRemoved,
    BlockStored,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
)
from sglang.srt.mem_cache.events import KvEventLoraNames
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.test.kits.unified_radix_cache.fixtures import (
    drop_hicache_atexit_pin,
    match_prefix,
    write_backup,
)


class LoraReq:
    """The request fields KvEventLoraNames reads."""

    def __init__(self, extra_key, lora_id="lora-id"):
        self.extra_key = extra_key
        self.lora_id = lora_id


def make_lora_names(**names_by_extra_key):
    """A name table plus the requests that keep its entries alive."""
    table = KvEventLoraNames()
    reqs = [LoraReq(key) for key in names_by_extra_key]
    for req in reqs:
        table.register(req, names_by_extra_key[req.extra_key])
    return table, reqs


class UnifiedRadixCacheKVEventsFixtures:
    def _insert(
        self,
        cache,
        allocator,
        tokens,
        *,
        extra_key=None,
        cache_salt=None,
        session_id=None,
    ):
        key = RadixKey(array("q", tokens), extra_key=extra_key, cache_salt=cache_salt)
        value = allocator.alloc(len(tokens))
        self.assertIsNotNone(value)
        return cache.insert(
            InsertParams(
                key=key,
                value=value[: len(key)],
                session_id=session_id,
            )
        )

    def _stored_events(self, cache, medium=None):
        events = [e for e in cache.take_events() if isinstance(e, BlockStored)]
        if medium is not None:
            events = [e for e in events if e.medium == medium]
        return events

    def _removed_events(self, cache, medium=None):
        events = [e for e in cache.take_events() if isinstance(e, BlockRemoved)]
        if medium is not None:
            events = [e for e in events if e.medium == medium]
        return events

    def _event_hashes(self, events):
        return [block_hash for event in events for block_hash in event.block_hashes]

    def _leaf_for(self, cache, tokens):
        match = match_prefix(cache, tokens)
        leaf = match.last_device_node
        self.assertNotEqual(leaf, cache.root_node_handle())
        return leaf

    def _init_hicache(self, cache, *, write_policy: str = "write_through"):
        # Production config: kernel IO backend + page_first layout.
        # Pinned host pools (kernel GPU DMA requires them) track their registered
        # pointers and unregister on destroy/GC, so fixtures sharing a process
        # cannot collide on recycled address ranges.
        server_args = ServerArgs(
            model_path="dummy",
            page_size=self.cfg.page_size,
            hicache_io_backend="kernel",
            hicache_write_policy=write_policy,
        )
        set_global_server_args_for_scheduler(server_args)
        cache.init_hicache(server_args, cache.cache_init_params)
        self.addCleanup(drop_hicache_atexit_pin, cache)
        self.addCleanup(cache.release_host_resources)
        cache.write_through_threshold = 1 << 30
        cache.load_back_threshold = 0

    def _backup_node(self, cache, node):
        backed_up = write_backup(cache, node, write_back=True)
        self.assertGreater(backed_up, 0)
        cache.writing_check(write_back=True)

    def _load_back_node(self, cache, node):
        loaded = cache.load_back(node)
        self.assertTrue(loaded)
        producer_id = cache.ready_to_load_host_cache()
        self.assertNotEqual(producer_id, -1)
        for ack in list(cache.cache_controller.ack_load_queue):
            ack.finish_event.synchronize()
        cache.loading_check()
