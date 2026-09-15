import hashlib
import heapq
import logging
import os
import pickle
import threading
import time
from functools import partial
from typing import List, Optional

import torch
import yaml

from sglang.srt.distributed.communication_tags import P2PTag
from sglang.srt.managers.eic_cache_controller import (
    EICCacheController,
    get_content_hash,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.eic_memory_pool import (
    EICDeepSeekV4TokenToKVPoolHost,
    EICMHATokenToKVPoolHost,
    EICMLATokenToKVPoolHost,
    EICNSATokenToKVPoolHost,
    MemoryStateInt,
    get_eic_config_file_path,
)
from sglang.srt.mem_cache.eic_pp_reconcile import eic_pp_unsupported_reason
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool as NSATokenToKVPool
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class EICHiRadixCacheBuilder:
    @staticmethod
    def build(
        params: CacheInitParams,
        server_args: ServerArgs,
    ):
        if server_args.disable_eic_shared:
            return EICHiRadixCache(
                params,
                server_args,
            )
        else:
            return EICPagedHiRadixCache(
                params,
                server_args,
            )


def mha_pool_get_flat_data(self: MHATokenToKVPool, indices: torch.Tensor):
    flatten = torch.stack(
        [
            torch.stack([self.k_buffer[i][indices] for i in range(self.layer_num)]),
            torch.stack([self.v_buffer[i][indices] for i in range(self.layer_num)]),
        ]
    )
    return flatten


def mha_pool_transfer(
    self: MHATokenToKVPool, indices: torch.Tensor, flat_data: torch.Tensor
):
    flat_data = flat_data.to(device=self.device, non_blocking=False)
    k_data, v_data = flat_data[0], flat_data[1]
    for i in range(self.layer_num):
        self.k_buffer[i][indices] = k_data[i]
        self.v_buffer[i][indices] = v_data[i]


def mla_pool_get_flat_data(self: MLATokenToKVPool, indices: torch.Tensor):
    return torch.stack([self.kv_buffer[i][indices] for i in range(self.layer_num)])


def mla_pool_transfer(
    self: MLATokenToKVPool, indices: torch.Tensor, flat_data: torch.Tensor
):
    flat_data = flat_data.to(device=self.device, non_blocking=False)
    for i in range(self.layer_num):
        self.kv_buffer[i][indices] = flat_data[i]


def nsa_pool_get_flat_data(self: NSATokenToKVPool, indices: torch.Tensor):
    num_pages = len(indices) // self.page_size

    # Gather MLA (num_tokens, 1, kv_cache_dim) -> (layer_num, num_tokens, kv_cache_dim)
    mla_data = torch.stack([self.kv_buffer[i][indices] for i in range(self.layer_num)])
    # (layer_num, num_pages, mla_page_bytes)
    mla_bytes = mla_data.view(self.layer_num, num_pages, -1).view(torch.uint8)

    # Gather NSA
    page_indices = indices.reshape(-1, self.page_size)[:, 0] // self.page_size
    # (layer_num, num_pages, 8448)
    nsa_packed = torch.stack(
        [self.index_k_with_scale_buffer[i][page_indices] for i in range(self.layer_num)]
    )

    combined = torch.cat(
        [mla_bytes, nsa_packed], dim=-1
    )  # (layer_num, num_pages, final_dim)
    return combined


def nsa_pool_transfer(
    self: NSATokenToKVPool,
    indices: torch.Tensor,
    flat_data: torch.Tensor,
):
    flat_data = flat_data.to(device=self.device, non_blocking=False)
    # flat_data: (layer_num, num_pages, final_dim)
    num_pages = len(indices) // self.page_size
    mla_page_bytes = self.kv_cache_dim * self.store_dtype.itemsize * self.page_size
    page_indices = indices.reshape(-1, self.page_size)[:, 0] // self.page_size

    for i in range(self.layer_num):
        layer_data = flat_data[i]  # (num_pages, final_dim)

        # Split
        mla_bytes = layer_data[:, :mla_page_bytes]
        nsa_bytes = layer_data[:, mla_page_bytes:]

        # Write MLA
        mla_part = (
            mla_bytes.reshape(-1)
            .view(self.store_dtype)
            .reshape(num_pages * self.page_size, 1, self.kv_cache_dim)
        )
        self.kv_buffer[i][indices] = mla_part

        # Write NSA
        self.index_k_with_scale_buffer[i][page_indices] = nsa_bytes


class EICHiRadixCache(RadixCache):

    # PP verdict protocol knobs (see the __init__ comment block).
    _KIND_SPAN, _KIND_FINAL = 0, 1
    _VERDICT_CAP = 64  # verdict rows per DAG round; overflow waits a round
    _EPOCH_CAP = 65536  # rid->epoch entries; eviction horizon >> stale lifetime
    _TOMBSTONE_TTL = 4096  # rounds a released rid keeps dropping stragglers
    _GC_AGE = 8192  # rounds before orphaned rank0 report entries are dropped

    def __init__(
        self,
        params: CacheInitParams,
        server_args: ServerArgs,
    ):
        self.tp_group = params.tp_cache_group
        self.tp_size = self.tp_group.size()
        self.rank = self.tp_group.rank()
        # deploy_key embeds pp_rank, so each PP stage load-backs independently;
        # loading_check reconciles the admitted length across stages.
        self.pp_group = params.pp_cache_group
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.work_list = []
        # --- Cross-PP load-back reconciliation (two-phase verdict protocol) ---
        # Each PP stage load-backs from its own EIC namespace (deploy_key embeds
        # pp_rank), so per-stage loaded amounts -- and, transitively, radix trees
        # and allocator headroom -- would diverge, breaking SPMD scheduling
        # (divergent extend_input_len => main_norm_rope assert at cp>1).
        # Two GPU-proven constraints shape the protocol: this gloo build's p2p is
        # effectively synchronous, and PP rank0 is the pipeline-output receiver,
        # so rank0 must NEVER post a receive in the scheduler thread; the only
        # safe p2p shape is the rank0-isend-down DAG at the lockstep site.
        # Hence: reports go UP through the default-PG TCPStore (non-collective
        # KV RPCs to a separate daemon -- not a p2p receive), verdicts come DOWN
        # on the DAG in a fixed tensor, so every stage applies the same decision
        # in the same scheduling loop (admission timing = batch composition).
        # Two phases keep every tree/allocator mutation PP-uniform:
        #   SPAN  = min over stages of (device_match + host_hit): fixes the
        #           allocation BEFORE any load is kicked (quota never exceeds the
        #           local host chain, so the alloc is a pure function of SPAN);
        #   FINAL = min over stages of (device_match + actually loaded): fixes
        #           the admitted prefix; the [FINAL, SPAN) tail is freed at a
        #           verdict boundary on every stage in the same loop.
        # Wire key = (rid_hash, epoch); the epoch kills stale-report pairing
        # when a client reuses a rid after an abort.
        self.ongoing_load_admit = {}  # rid -> per-req reconciliation state
        self._admit_verdict = {}  # rid_hash -> FINAL admit len, consumed by the gate
        self._h_rid = {}  # rid_hash -> rid, resolves incoming verdicts
        self._rid_epoch = {}  # rid -> gate-entry ordinal; FIFO-capped, never reset
        self._report_outbox = {}  # stage>0, tp0: (h, epoch, kind) -> report payload
        self._span_reports = {}  # rank0 tp0: (h, epoch) -> ({stage: (d, hh)}, born)
        self._load_reports = {}  # rank0 tp0: (h, epoch) -> ({stage: d+loaded}, born)
        self._await_load = (
            {}
        )  # rank0 tp0: (h, epoch) -> (span, loaders, {stage: d}, born)
        self._verdict_outbox = []  # rank0 tp0: formed verdicts awaiting a DAG slot
        self._tombstone = {}  # rid_hash -> expiry round; released rids drop stragglers
        self._pub_seq = 0  # stage>0: next store seq; bumps ONLY after a good set
        self._next_seq = {}  # rank0: stage -> next store seq to drain
        self._round = 0  # loading_check counter (tombstone/GC clock)
        self._store_handle = None  # lazy default-PG TCPStore (tp0 lanes only)
        self._loadback_rid = {}  # load-back node_id -> rid (maps an EIC ack to its req)
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()
        self.sliding_window_size = params.sliding_window_size
        self.load_cache_event = threading.Event()
        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = EICMHATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                "cpu",
                params.page_size,
                self.rank,
                extra_info=self.get_extra_info(params, server_args),
            )
            self.kv_cache.get_flat_data = partial(mha_pool_get_flat_data, self.kv_cache)
            self.kv_cache.transfer = partial(mha_pool_transfer, self.kv_cache)
        elif isinstance(self.kv_cache, NSATokenToKVPool):
            self.token_to_kv_pool_host = EICNSATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                "cpu",
                params.page_size,
                self.rank,
                extra_info=self.get_extra_info(params, server_args),
            )
            self.kv_cache.get_flat_data = partial(nsa_pool_get_flat_data, self.kv_cache)
            self.kv_cache.transfer = partial(nsa_pool_transfer, self.kv_cache)
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = EICMLATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                "cpu",
                params.page_size,
                self.rank,
                extra_info=self.get_extra_info(params, server_args),
            )
            self.kv_cache.get_flat_data = partial(mla_pool_get_flat_data, self.kv_cache)
            self.kv_cache.transfer = partial(mla_pool_transfer, self.kv_cache)
        elif isinstance(self.kv_cache, DeepSeekV4TokenToKVPool):
            self.token_to_kv_pool_host = EICDeepSeekV4TokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                "cpu",
                params.page_size,
                self.rank,
                extra_info=self.get_extra_info(params, server_args),
                params=params,
                server_args=server_args,
                load_cache_event=self.load_cache_event,
            )
        else:
            raise ValueError(
                "HiRadixCache only supports MHA, MLA, NSA and DeepSeek V4 yet"
            )

        self.cache_controller = EICCacheController(
            params.token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            params.page_size,
            tp_group=params.tp_cache_group,
            load_cache_event=self.load_cache_event,
            write_policy=server_args.hicache_write_policy,
            server_args=server_args,
        )

        # record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        # EIC is native to this cache (no pluggable L3 HiCacheStorage backend);
        # is_fully_idle() uses this flag to skip ongoing_prefetch/backup checks.
        self.enable_storage = False
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 3
        )
        self.load_back_threshold = 10
        # Headroom a load-back must leave for the next chunked-prefill extend;
        # both inputs are frozen for this object's lifetime.
        self.load_back_reserve = (
            server_args.chunked_prefill_size
            if server_args.chunked_prefill_size and server_args.chunked_prefill_size > 0
            else self.page_size
        )
        if self.pp_size > 1:
            # The verdict protocol assumes lockstep candidate iteration and
            # lockstep releases. These knobs break that: cache-aware policies
            # re-match every waiting req per loop from per-stage trees (and
            # clobber frozen matches), storage prefetch skips the gate per stage
            # asynchronously, dp-attention lanes have private waiting queues,
            # and the waiting timeout releases by per-stage clocks.
            if server_args.hicache_storage_backend:
                raise ValueError(
                    "EIC under PP is incompatible with hicache_storage_backend"
                )
            reason = eic_pp_unsupported_reason(server_args)
            if reason is not None:
                raise ValueError(f"EIC host load-back under PP: {reason}")
        super().__init__(params)

        self.save_decode_cache = True
        config_file = get_eic_config_file_path()
        if os.path.exists(config_file):
            with open(config_file, "r") as fin:
                config = yaml.safe_load(fin)
            self.init_hyper_params(config)

    def get_extra_info(self, params: CacheInitParams, server_args: ServerArgs):
        # TODO update when sglang support pp
        extra_info = {
            "model_path": server_args.model_path,
            "world_size": self.tp_size,
            "tp_rank": self.rank,
            "framework": "sglang",
            "pp_rank": params.pp_rank,
            "pp_size": params.pp_size,
        }
        return extra_info

    def init_hyper_params(self, config: dict):
        self.save_decode_cache = config.get("save_decode_cache", True)
        logger.info(
            f"EICHiRadixCache save_decode_cache set to {self.save_decode_cache}"
        )
        self.load_back_threshold = config.get("load_back_threshold", 10)
        logger.info(
            f"EICHiRadixCache load_back_threshold set to {self.load_back_threshold}"
        )

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        self.ongoing_load_back = {}
        self.ongoing_load_admit = {}
        self.ongoing_write_through = {}
        self._admit_verdict = {}
        self._h_rid = {}
        self._loadback_rid = {}
        self._report_outbox = {}
        self._span_reports = {}
        self._load_reports = {}
        self._await_load = {}
        self._verdict_outbox = []
        # Comm-layer state deliberately survives reset: seq counters must stay
        # monotonic (a pre-reset store batch may still be in flight), epochs
        # must never reuse, tombstones expire by round. Flush is idle-gated, so
        # everything cleared above can only reference dead rids.
        super().reset()

    def get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def write_backup(self, node: TreeNode, write_back=False):
        logger.debug(f"write backup for node {node.id}")
        if node.evicted:
            return 0
        if not write_back and (
            node.parent != self.root_node and not node.parent.backuped
        ):
            return 0
        host_indices = self.cache_controller.write(
            device_indices=node.value,
            priority=-self.get_height(node),
            node_id=node.id,
        )
        if host_indices is None:
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                priority=-self.get_height(node),
                node_id=node.id,
            )
        if host_indices is not None:
            node.host_value = host_indices
            self.ongoing_write_through[node.id] = node
            if not write_back:
                self.inc_lock_ref(node)
        else:
            return 0

        return len(host_indices)

    def inc_hit_count(self, node: TreeNode, chunked: bool = False):
        if self.cache_controller.write_policy == "write_back" or chunked:
            return
        node.hit_count += 1
        if not node.backuped:
            if node.hit_count >= self.write_through_threshold:
                self.write_backup(node)
                node.hit_count = 0

    def _backup_unbacked_path(self, node: TreeNode) -> int:
        if self.cache_controller.write_policy != "write_through":
            return 0

        path = []
        while node is not None and node != self.root_node:
            path.append(node)
            node = node.parent

        written = 0
        for path_node in reversed(path):
            if path_node.evicted or path_node.backuped:
                continue
            written += self.write_backup(path_node)
        return written

    def cache_finished_req(
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int
    ):
        """Cache a finished request and make EIC's remote prefix contiguous."""
        if self.disable_finished_insert:
            is_insert = False

        kv_committed_len = kv_len_to_handle
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.kv.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : len(token_ids)
        ]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        key_len = len(radix_key)
        values = kv_indices[:key_len].to(dtype=torch.int64, copy=True)

        if is_insert:
            priority = getattr(req, "priority", 0) or 0
            result = self.insert(
                InsertParams(key=radix_key, value=values, priority=priority)
            )
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.kv.cache_protected_len : result.prefix_len]
            )

            match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
            self._backup_unbacked_path(match_result.last_device_node)
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.kv.cache_protected_len : key_len]
            )
            if req.last_node is not None:
                self._backup_unbacked_path(req.last_node)

        self.token_to_kv_pool_allocator.free(kv_indices[key_len:])

        # Backstop SWA slots not released by per-step eviction or FULL cleanup.
        if self.sliding_window_size is not None:
            if hasattr(self.token_to_kv_pool_allocator, "free_swa"):
                _swa_lo = req.kv.swa_evicted_seqlen
                _swa_hi = kv_committed_len
                _prefix_len = len(req.prefix_indices)
                # The EIC-loaded prefix's full indices live in req.prefix_indices,
                # not req_to_token (which stays 0 for the loaded portion). Use
                # prefix_indices for the in-prefix slice so the in-window SWA is
                # actually freed; req_to_token would map to 0 and be dropped.
                if _swa_hi <= _prefix_len:
                    _swa_slots = req.prefix_indices[_swa_lo:_swa_hi]
                elif _swa_lo < _prefix_len:
                    _swa_slots = torch.cat(
                        [
                            req.prefix_indices[_swa_lo:_prefix_len],
                            self.req_to_token_pool.req_to_token[
                                req.kv.req_pool_idx, _prefix_len:_swa_hi
                            ],
                        ]
                    )
                else:
                    _swa_slots = self.req_to_token_pool.req_to_token[
                        req.kv.req_pool_idx, _swa_lo:_swa_hi
                    ]
                self.token_to_kv_pool_allocator.free_swa(_swa_slots)

        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        if self.ongoing_load_back:
            key = RadixKey(req.get_fill_ids(), req.extra_key, is_bigram=self.is_eagle)
            match = self.match_prefix(MatchPrefixParams(key=key))
            if self.prefix_loading(match.last_device_node):
                # Inserting now would swap this req's own KV for slots still
                # landing from EIC; keep it private, as ChunkCache does, until
                # the load settles.
                req.prefix_indices = self.req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, : len(req.get_fill_ids())
                ].to(dtype=torch.int64, copy=True)
                return
        super().cache_unfinished_req(req, chunked=chunked)
        if req.last_node is not None:
            self._backup_unbacked_path(req.last_node)

    def get_tp_result(self, flag):
        if isinstance(flag, bool):
            flag = [flag]
        if self.tp_size <= 1:
            return flag
        # synchronize the result across TP workers
        temp = [0 if x else 1 for x in flag]
        temp_tensor = torch.tensor(temp, dtype=torch.int64, device="cpu")
        torch.distributed.all_reduce(
            temp_tensor, op=torch.distributed.ReduceOp.SUM, group=self.tp_group
        )
        result_list = temp_tensor.tolist()
        result = []
        for i in range(len(result_list)):
            result.append(result_list[i] == 0)
        return result

    def writing_check(
        self,
        write_back=None,
        finish_count: Optional[int] = None,
        blocking: bool = False,
    ):
        # finish_count is the base class's PP lockstep hint; EIC reconciles the
        # ack count itself via the tp_group MIN below, so it is ignored here.
        if write_back is None:
            write_back = self.cache_controller.write_policy == "write_back"
        if len(self.ongoing_write_through) == 0:
            return
        write_check_start_time = time.perf_counter()
        if write_back and blocking:
            while (
                len(self.ongoing_write_through)
                != self.cache_controller.ack_write_queue.qsize()
            ):
                if time.perf_counter() - write_check_start_time > 30.0:
                    logger.error(
                        "writing_check barrier timed out after 30s: "
                        f"ongoing={len(self.ongoing_write_through)} "
                        f"acked={self.cache_controller.ack_write_queue.qsize()}; "
                        "EIC write thread likely stalled, proceeding with partial acks"
                    )
                    break
                time.sleep(0.01)
        queue_size = torch.tensor(
            self.cache_controller.ack_write_queue.qsize(), dtype=torch.int
        )
        # may skip synchronize queue_size for write
        if torch.distributed.get_world_size(group=self.tp_group) > 1:
            # synchrnoize TP workers to make the same update to radix cache
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        ack_list = []
        flags = []
        for _ in range(queue_size.item()):
            ack_id, success = self.cache_controller.ack_write_queue.get_nowait()
            ack_list.append(ack_id)
            flags.append(success)
        for ack_id, success in zip(ack_list, flags):
            if not success:
                node = self.ongoing_write_through[ack_id]
                if isinstance(self, EICPagedHiRadixCache):
                    node.host_value = None
                elif node.host_value is not None:
                    if (
                        self.cache_controller.mem_pool_host.get_state(node.host_value)
                        != MemoryStateInt.IDLE
                    ):
                        self.cache_controller.mem_pool_host.free(node.host_value)
                    node.host_value = None
            if not write_back:
                self.dec_lock_ref(self.ongoing_write_through[ack_id])
            # clear the reference
            del self.ongoing_write_through[ack_id]
        cost_time = time.perf_counter() - write_check_start_time
        if cost_time > 1:
            logger.warning(
                f"writing check cost {cost_time:.3f} seconds, "
                f"queue size {queue_size.item()}"
            )

    @property
    def _pp_active(self):
        return self.pp_size > 1 and self.pp_group is not None

    def _pp_bcast_from_first(self, tensor, tag=P2PTag.HIRADIX_PP_SYNC):
        # PP0-authoritative broadcast down the pipeline via non-blocking isend
        # (blocking/collective PP ops deadlock the out-of-phase pipeline). Each
        # logical stream passes its OWN tag so independent bcasts (num_ready vs the
        # admission verdict) never share a FIFO slot and can't cross-match.
        if not self._pp_active:
            return
        if self.pp_rank > 0:
            torch.distributed.recv(
                tensor, group_src=self.pp_rank - 1, group=self.pp_group, tag=tag
            )
        if self.pp_rank + 1 < self.pp_size:
            copied = tensor.clone()
            self.work_list.append(
                torch.distributed.isend(
                    copied, group_dst=self.pp_rank + 1, group=self.pp_group, tag=tag
                )
            )

    def _drain_async_work(self):
        for work in self.work_list:
            work.wait()
        self.work_list.clear()

    def _reduce_min(self, tensor):
        # CP/TP is synchronous -> a real MIN; PP is out-of-phase -> PP0-authoritative.
        if self.tp_size > 1:
            torch.distributed.all_reduce(
                tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
        self._pp_bcast_from_first(tensor)

    def loading_check(self, finish_count: Optional[int] = None):
        # The lockstep heartbeat of the verdict protocol: called exactly once per
        # get_new_batch_prefill on every rank, before any early return, so the
        # k-th VERDICT bcast pairs with the k-th recv on every stage and a
        # verdict lands at the same batch-forming call everywhere.
        self._round += 1
        self._drain_local_acks()
        if not self._pp_active:
            return
        if self.rank == 0:
            # tp0 speaks for its stage (reports are TP-uniform: complete_token
            # is MIN-reduced inside the cache controller before the ack).
            if self.pp_rank == 0:
                self._drain_peer_reports()
                self._form_verdicts()
            else:
                self._publish_reports()
        buf = torch.zeros(1 + self._VERDICT_CAP * 4, dtype=torch.int64, device="cpu")
        if self.pp_rank == 0:
            if self.rank == 0:
                n = min(len(self._verdict_outbox), self._VERDICT_CAP)
                buf[0] = n
                if n:
                    buf[1 : 1 + n * 4] = torch.tensor(
                        [x for row in self._verdict_outbox[:n] for x in row],
                        dtype=torch.int64,
                    )
                    del self._verdict_outbox[:n]
            if self.tp_size > 1:
                # Equalize the tp0-only poll result across this stage's lanes
                # BEFORE it forks into the per-lane DAGs. Unconditional: the
                # trigger is tp0-local, a skipped collective wedges the stage.
                torch.distributed.broadcast(buf, group=self.tp_group, group_src=0)
        self._pp_bcast_from_first(buf, tag=P2PTag.HIRADIX_PP_VERDICT)
        # The packed tensor is the single source of truth on EVERY rank
        # (including rank0): overflow verdicts wait in the outbox, so cap
        # overflow can never desync which loop a req admits in.
        self._apply_verdicts(buf)
        if self._round % 1024 == 0:
            self._sweep()

    def _rid_hash(self, rid):
        # Stable 56-bit id (fits a positive int64); hash() is per-process salted so
        # not PP-uniform. ponytail: collision-free for a batch-sized in-flight set.
        return int.from_bytes(
            hashlib.blake2b(rid.encode(), digest_size=7).digest(), "big"
        )

    @property
    def _store(self):
        # The default process group's TCPStore: non-collective KV RPCs served by
        # a separate daemon thread -- polling it is NOT a p2p receive, so the
        # rank0-must-never-receive constraint does not apply. Exceptions must
        # propagate: a swallowed set with a bumped seq is a permanent hole the
        # drain side polls forever (and a dead store means a dead job anyway).
        if self._store_handle is None:
            from torch.distributed import distributed_c10d

            self._store_handle = distributed_c10d._get_default_store()
        return self._store_handle

    def _bump_epoch(self, rid):
        # Per-rid gate-entry ordinal; never reused within a process, so a stale
        # in-flight report of a released incarnation can never pair with a fresh
        # one (client-supplied rids may repeat after an abort). Re-insertion
        # keeps the FIFO cap tracking recency.
        epoch = self._rid_epoch.pop(rid, 0) + 1
        self._rid_epoch[rid] = epoch
        while len(self._rid_epoch) > self._EPOCH_CAP:
            self._rid_epoch.pop(next(iter(self._rid_epoch)))
        return epoch

    def _queue_report(self, st, kind, value):
        if self.pp_size <= 1 or self.rank != 0:
            return
        key = (st["h"], st["epoch"])
        if self.pp_rank == 0:
            # rank0's own column feeds the decision tables directly.
            table = (
                self._span_reports if kind == self._KIND_SPAN else self._load_reports
            )
            table.setdefault(key, ({}, self._round))[0][0] = value
        else:
            self._report_outbox[(st["h"], st["epoch"], kind)] = value

    def _publish_reports(self):
        if not self._report_outbox:
            return
        payload = pickle.dumps(self._report_outbox)
        self._store.set(f"eiclb/{self.pp_rank}/{self._pub_seq}", payload)
        self._pub_seq += 1  # only after a successful set (no holes, ever)
        self._report_outbox = {}

    def _drain_peer_reports(self):
        for stage in range(1, self.pp_size):
            seq = self._next_seq.get(stage, 0)
            while self._store.check([f"eiclb/{stage}/{seq}"]):
                key = f"eiclb/{stage}/{seq}"
                batch = pickle.loads(self._store.get(key))
                self._store.delete_key(key)
                seq += 1
                for (h, epoch, kind), value in batch.items():
                    tomb = self._tombstone.get(h)
                    if tomb is not None and epoch <= tomb[0]:
                        continue  # a released incarnation's straggler; a fresh
                        # re-gate of the same rid carries a higher epoch and
                        # must pass, or the retry wedges forever.
                    table = (
                        self._span_reports
                        if kind == self._KIND_SPAN
                        else self._load_reports
                    )
                    table.setdefault((h, epoch), ({}, self._round))[0][stage] = value
            self._next_seq[stage] = seq

    def _form_verdicts(self):
        threshold = max(self.load_back_threshold, 1)
        for key, (sr, _) in list(self._span_reports.items()):
            if len(sr) < self.pp_size:
                continue
            del self._span_reports[key]
            span = min(d + hh for d, hh in sr.values())
            # A loader must clear the load threshold; every stage evaluates the
            # same condition from the same numbers at span-apply time.
            loaders = frozenset(s for s, (d, _) in sr.items() if span - d >= threshold)
            if loaders:
                dmap = {s: d for s, (d, _) in sr.items()}
                self._await_load[key] = (span, loaders, dmap, self._round)
                self._verdict_outbox.append((*key, self._KIND_SPAN, span))
            else:
                self._verdict_outbox.append(
                    (*key, self._KIND_FINAL, min(d for d, _ in sr.values()))
                )
        for key, (lr, _) in list(self._load_reports.items()):
            aw = self._await_load.get(key)
            if aw is None or not aw[1] <= lr.keys():
                continue
            span, loaders, dmap, _ = aw
            del self._load_reports[key]
            del self._await_load[key]
            # Admissible = what EVERY stage actually holds: loaders their loaded
            # length, non-loaders their device match, all capped by the span.
            final = min(
                [span]
                + [lr[s] for s in loaders]
                + [d for s, d in dmap.items() if s not in loaders]
            )
            self._verdict_outbox.append((*key, self._KIND_FINAL, final))

    def _apply_verdicts(self, buf):
        n = int(buf[0].item())
        flat = buf[1 : 1 + n * 4].tolist()
        for i in range(0, n * 4, 4):
            h, epoch, kind, value = flat[i : i + 4]
            rid = self._h_rid.get(h)
            st = self.ongoing_load_admit.get(rid) if rid is not None else None
            if st is None or st["epoch"] != epoch:
                continue  # released rid or a dead incarnation's verdict
            if kind == self._KIND_SPAN:
                self._apply_span(rid, st, value)
            else:
                self._admit_verdict[h] = value

    def _apply_span(self, rid, st, span):
        # SPAN is a cross-stage MIN, so quota = span - d never exceeds this
        # stage's own host chain: the allocation is a pure function of the
        # (uniform) verdict and allocator deltas stay PP-uniform. allow_evict is
        # off under PP -- an alloc-failure eviction would mutate one stage's
        # tree only; degrading to loaded=0 re-converges at FINAL instead.
        d, hh = st["d"], st["hh"]
        node = st["best_match_node"]
        quota = span - d
        swa_ok = self._swa_headroom_ok(quota) and self._full_headroom_ok(quota)
        if quota >= max(self.load_back_threshold, 1) and node is not None and swa_ok:
            node = self._clip_host_chain(node, d + hh - span, span - d)
            if node is not None:
                indices = self.load_back(node, allow_evict=self.pp_size <= 1)
                if indices is not None:
                    st["load_node"] = node
                    st["alloc"] = len(indices)
                    st["new_indices"] = indices
                    self._loadback_rid[node.id] = rid
                    return  # the LOADED report follows the local EIC ack
        # Nothing kicked on this stage: its admissible length is device-only.
        if not self._pp_active:
            self._admit_verdict[st["h"]] = d
        else:
            self._queue_report(st, self._KIND_FINAL, d)

    def _full_headroom_ok(self, quota):
        # The async admission gate allocates OUTSIDE add_one_req's rem_total_tokens
        # budget, so every queued req can pin a load-back until the pool is full
        # with #running-req: 0 and evictable_size 0 -- an unrecoverable prefill OOM
        # (the baseline's init_load_back sits inside that budget and self-bounds).
        # A refused load degrades to a device-only admit below, not a deadlock.
        reserve = self.load_back_reserve
        alloc = self.cache_controller.mem_pool_device_allocator
        return alloc.available_size() + self.evictable_size_ >= quota + reserve

    def _swa_headroom_ok(self, quota):
        swa = getattr(
            self.cache_controller.mem_pool_device_allocator, "swa_attn_allocator", None
        )
        if swa is None:
            return True
        swa_window = self.sliding_window_size or 0
        swa_needed = min(quota, max(swa_window, self.page_size))
        avail = swa.available_size()
        return avail >= swa_needed + swa.size // 2

    def _clip_host_chain(self, node, excess, quota):
        # Cut the evicted chain `excess` tokens above its bottom so the load
        # covers exactly [d, span). The cut lands on a verdict boundary
        # (PP-uniform) and touches only evicted (host-side) nodes, so device
        # node shapes stay PP-uniform. Returns None unless the clipped chain
        # still runs EXACTLY quota (= span - d) tokens down to the first
        # resident ancestor: during the span round trip the frozen chain can
        # break (host eviction / failed write ack) or its resident frontier can
        # move (a concurrent same-prefix load kick makes [d, x) resident; an
        # insert recompute re-evicts below d) -- loading anything but the frozen
        # [d, span) would silently break the cross-stage uniformity the verdict
        # promised, so any mismatch degrades to a device-only report instead.
        while excess > 0 and node.evicted:
            if not node.backuped:
                return None
            if len(node.key) <= excess:
                excess -= len(node.key)
                node = node.parent
            else:
                node = self._split_node(node.key, node, len(node.key) - excess)
                excess = 0
        if excess > 0 or not node.evicted:
            return None
        chain_len = 0
        walk = node
        while walk.evicted:
            if not walk.backuped:
                return None
            chain_len += len(walk.key)
            walk = walk.parent
        if chain_len != quota:
            return None
        return node

    def _sweep(self):
        self._tombstone = {
            h: t for h, t in self._tombstone.items() if t[1] > self._round
        }
        if self.rank != 0 or self.pp_rank != 0:
            return
        horizon = self._round - self._GC_AGE
        for table in (self._span_reports, self._load_reports, self._await_load):
            for key in list(table):
                if table[key][-1] > horizon:
                    continue
                if self._h_rid.get(key[0]) is None:
                    # dead-rid straggler (e.g. a report drained after a release
                    # already purged its pairing state) -- plain garbage.
                    del table[key]
                else:
                    logger.warning(
                        "EIC PP load-back wedged: rid_hash=%d has waited %d+ "
                        "rounds for peer reports (a stage's EIC ack may never "
                        "have arrived); the req stays deferred until aborted.",
                        key[0],
                        self._GC_AGE,
                    )

    def _drain_local_acks(self):
        queue_size = torch.tensor(
            self.cache_controller.ack_load_queue.qsize(), dtype=torch.int
        )
        if self.tp_size > 1:
            # TP/CP share a namespace -> acks match; MIN guards qsize skew only.
            torch.distributed.all_reduce(
                queue_size, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
        for _ in range(int(queue_size.item())):
            node_id, complete_token = self.cache_controller.ack_load_queue.get_nowait()
            rid = self._loadback_rid.pop(node_id, None)
            st = self.ongoing_load_admit.get(rid) if rid is not None else None
            if st is None or st["load_node"] is None or st["load_node"].id != node_id:
                # Req released before its load landed (the node_id check keeps a
                # dead incarnation's ack from being attributed to a re-gated
                # same-rid retry). Free the WHOLE span (a verdict-uniform
                # boundary); a partial free at the per-stage completed count
                # would fork device node shapes across stages.
                if node_id in self.ongoing_load_back:
                    self._free_failed_loadback(node_id, 0)
                continue
            st["complete"] = complete_token
            if not self._pp_active:
                self._admit_verdict[st["h"]] = st["d"] + complete_token
            else:
                self._queue_report(st, self._KIND_FINAL, st["d"] + complete_token)

    def prefix_loading(self, node: TreeNode) -> bool:
        # load_back publishes node.value before its DMA acks. A req adopting those
        # slots reads KV that is still landing, and if the load fails the tail is
        # freed under it; its insert then re-links the freed slots into the tree
        # (a page both free and cached). Such a req must wait for the load to settle.
        if not self.ongoing_load_back or self.pp_size > 1:
            # ponytail: PP stages hold per-stage chains, so deferring here would fork
            # admission across stages; PP>1 still adopts in-flight slots.
            return False
        loading = set()
        for start, end, _ in self.ongoing_load_back.values():
            while end is not start:
                loading.add(end.id)
                end = end.parent
        while node is not None and node is not self.root_node:
            if node.id in loading:
                return True
            node = node.parent
        return False

    def _free_failed_loadback(self, node_id, complete_token):
        # Local cleanup: release the load lock and free the failed-load tail so the
        # tree never holds garbage KV. Per-stage; the tree may diverge across PP.
        start_node, end_node, total_token_num = self.ongoing_load_back.pop(node_id)
        self.dec_lock_ref(end_node)
        failed_token_num = total_token_num - complete_token
        while end_node != start_node:
            if failed_token_num >= len(end_node.value):
                # node load back full fail
                # no need to delete failed node because the kvcache will be set after compute
                self.cache_controller.mem_pool_device_allocator.free(end_node.value)
                self.evictable_size_ -= len(end_node.value)
                failed_token_num -= len(end_node.value)
                end_node.value = None
                end_node.host_value = None
                self._update_leaf_status(end_node)
                self._update_leaf_status(end_node.parent)
            elif failed_token_num > 0:
                # node load back partial fail, split node
                split_len = len(end_node.value) - failed_token_num
                self._split_node(end_node.key, end_node, split_len)
                self.evictable_size_ -= failed_token_num
                self.cache_controller.mem_pool_device_allocator.free(end_node.value)
                failed_token_num -= len(end_node.value)
                end_node.value = None
                end_node.host_value = None
                self._update_leaf_status(end_node)
                self._update_leaf_status(end_node.parent)
                assert failed_token_num == 0, "failed_token_num should be zero"
            end_node = end_node.parent

    # TODO: is not correct for eic, but neednt to be fixed rightnow
    def evictable_size(self):
        return self.evictable_size_

    def full_evictable_size(self):
        return self.evictable_size_

    def swa_evictable_size(self):
        # EIC radix holds only FULL indices; SWA is allocator-managed. Report 0
        # to avoid double-counting vs swa_available_size in the idle leak check.
        return 0

    def full_protected_size(self):
        return self.protected_size_

    def swa_protected_size(self):
        # No SWA state in the EIC radix tree.
        return 0

    @property
    def swa_evict_release_prefix(self) -> bool:
        # EIC lacks SWARadixCache.dec_swa_lock_only, so let maybe_evict_swa drop
        # this request's out-of-window prefix SWA directly. Safe: SWA attention
        # never reads past the window, full KV is in the separate full pool, and
        # reuse reloads SWA from the EIC host backup.
        return True

    def supports_swa(self) -> bool:
        # EIC replaces the SWA-capable UnifiedRadixCache in kv_cache_builder, so
        # it must redeclare SWA support. Without this, supports_swa() falls back
        # to the RadixCache default (False), ScheduleBatch.maybe_evict_swa() is
        # gated off, and running sequences never release out-of-window SWA KV ->
        # the SWA pool fills up and prefill OOMs. The full-attention radix tree is
        # unaffected: only per-request SWA tails are freed in maybe_evict_swa.
        return self.sliding_window_size is not None

    def eic_swa_extend_eviction(self) -> bool:
        # EIC radix owns only FULL indices, so free out-of-window SWA in extend.
        return self.supports_swa()

    def sanity_check(self):
        # invariant_checker._check_tree_cache() calls this when
        # is_hybrid_swa and supports_swa(). EIC manages SWA freeing outside the
        # radix tree (per-request in maybe_evict_swa), so there is no tree
        # invariant to assert here.
        pass

    def evict(self, params: EvictParams, retry_times: int = 3) -> EvictResult:
        start_time = time.perf_counter()
        num_tokens = max(params.num_tokens, params.swa_num_tokens)
        # Throttle on the write-through backlog only. loading_check() holds a PP
        # collective; this loop fires on per-stage memory pressure (divergent
        # cadence), so calling it here would deadlock the PP group. The load
        # backlog drains at the single lockstep site (check_hicache_events) and
        # in the schedule_policy busy-wait, both PP-uniform.
        while len(self.ongoing_write_through) > 50:
            if time.perf_counter() - start_time > 30.0:
                logger.error(
                    "evict write-through throttle timed out after 30s: "
                    f"ongoing={len(self.ongoing_write_through)}; "
                    "EIC write thread likely stalled, proceeding with eviction"
                )
                break
            self.writing_check()
            time.sleep(0.1)

        num_evicted = 0
        while retry_times > 0:
            retry_times -= 1
            leaves = list(self.evictable_leaves)
            eviction_heap = [
                (self.eviction_strategy.get_priority(node), node) for node in leaves
            ]
            heapq.heapify(eviction_heap)

            write_back_nodes = []
            idx = 0

            logger.debug(
                f"evict {num_tokens} tokens, requested full {params.num_tokens}, "
                f"requested swa {params.swa_num_tokens}, current evictable size "
                f"{self.evictable_size_}, protect_size {self.protected_size_}, "
                f"leaves {len(leaves)}"
            )
            while num_evicted < num_tokens and len(eviction_heap):
                _priority, x = heapq.heappop(eviction_heap)
                logger.debug(
                    f"evicting {idx} node {x.id}, access {x.last_access_time}, value {x.value} {x.host_value}"
                )
                idx += 1

                if x.lock_ref > 0:
                    logger.debug(f"node {x.id} is locked, skip eviction")
                    continue

                if not x.backuped:
                    if self.cache_controller.write_policy == "write_back":
                        # write to host if the node is not backuped
                        num_evicted += self.write_backup(x, write_back=True)
                        write_back_nodes.append(x)
                    else:
                        num_evicted += self._evict_regular(x)
                else:
                    num_evicted += self._evict_backuped(x)

                for child in x.parent.children.values():
                    if child in write_back_nodes:
                        continue
                    if not child.evicted:
                        break
                else:
                    # all children are evicted or no children
                    new_priority = self.eviction_strategy.get_priority(x.parent)
                    heapq.heappush(eviction_heap, (new_priority, x.parent))

            if self.cache_controller.write_policy == "write_back":
                self.writing_check(write_back=True, blocking=True)
                for node in write_back_nodes:
                    if node.backuped:
                        self._evict_backuped(node)
                    else:
                        self._evict_regular(node)

            if num_evicted < num_tokens:
                logger.info(
                    f"only evicted {num_evicted} tokens, less than requested {num_tokens}"
                )
            return EvictResult(
                num_tokens_evicted=num_evicted,
                swa_num_tokens_evicted=num_evicted,
            )

    def _evict_backuped(self, node: TreeNode):
        if node.host_value is None:
            logger.error(f"host value is None for node {node.id}")
            return self._evict_regular(node)
        num_evicted = self.cache_controller.evict_device(node.value, node.host_value)
        assert num_evicted > 0
        self.evictable_size_ -= num_evicted
        node.value = None
        self._update_leaf_status(node)
        self._update_leaf_status(node.parent)
        return num_evicted

    def _evict_regular(self, node: TreeNode):
        # evict a node not initiated write to host
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        return num_evicted

    def evict_host(self, num_tokens: int):
        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)
            if x == self.root_node:
                break
            # only evict the host value of evicted nodes
            if not x.evicted:
                continue
            assert x.lock_ref == 0 and x.host_value is not None

            assert self.cache_controller.evict_host(x.host_value) > 0
            # Null the freed backup: frozen references (a deferring req's load
            # chain) probe `backuped` and must degrade, not DMA from freed slots.
            x.host_value = None
            for k, v in x.parent.children.items():
                if v == x:
                    break
            del x.parent.children[k]

            if len(x.parent.children) == 0 and x.parent.evicted:
                heapq.heappush(leaves, x.parent)

    def load_back(
        self,
        node: TreeNode,
        mem_quota: Optional[int] = None,
        allow_evict: bool = True,
    ) -> Optional[torch.Tensor]:
        # todo: more loading policies
        start_time = time.perf_counter()
        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # protect the ancestor nodes from eviction
        delta = self.inc_lock_ref(ancester_node)

        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancester_node)
            return None

        device_indices = self.cache_controller.load(
            host_indices=host_indices, node_id=last_hit_node.id
        )
        if device_indices is None and allow_evict:
            self.evict(EvictParams(num_tokens=len(host_indices)))
            device_indices = self.cache_controller.load(
                host_indices=host_indices, node_id=last_hit_node.id
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            return None

        self.ongoing_load_back[last_hit_node.id] = (
            ancester_node,
            last_hit_node,
            len(device_indices),
        )
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)]
            offset += len(node.host_value)
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        return device_indices

    def check_load_back_progress(self, req) -> bool:
        # Per-candidate admission gate. Under PP EVERY candidate passes here
        # (host metadata is per-stage, so needs_host_load_back can't select
        # candidates uniformly; gating all keeps report keysets symmetric,
        # keyed by rid). First encounter freezes (d, hh), locks the device
        # prefix and emits the SPAN report; the req then defers until the FINAL
        # verdict lands -- in the same DAG slot on every stage, so all stages
        # admit it in the same batch-forming loop. pp<=1 short-circuits both
        # verdicts locally (cold reqs admit with zero deferral, as before).
        st = self.ongoing_load_admit.get(req.rid)
        h = st["h"] if st is not None else self._rid_hash(req.rid)
        if st is None:
            d = len(req.prefix_indices)
            hh = req.host_hit_length if req.needs_host_load_back() else 0
            # Lock the frozen device prefix across the verdict round trips so
            # eviction can't reclaim slots the reports promised (lock the
            # deepest RESIDENT ancestor -- last_node itself may be an evicted
            # host node). Released in finalize/release.
            lock_node = req.last_node
            while lock_node.evicted:
                lock_node = lock_node.parent
            self.inc_lock_ref(lock_node)
            st = {
                "h": h,
                "epoch": self._bump_epoch(req.rid),
                "d": d,
                "hh": hh,
                "best_match_node": req.best_match_node if hh > 0 else None,
                "deferred_lock": lock_node,
                "alloc": 0,
                "load_node": None,
                "new_indices": None,
                "complete": None,
            }
            self.ongoing_load_admit[req.rid] = st
            self._h_rid[h] = req.rid
            if not self._pp_active:
                self._apply_span(req.rid, st, d + hh)
            else:
                self._queue_report(st, self._KIND_SPAN, (d, hh))
        if h not in self._admit_verdict:
            return False  # loading / awaiting verdicts; loading_check advances it
        return self._finalize_load_admit(req)

    def _finalize_load_admit(self, req) -> bool:
        st = self.ongoing_load_admit.pop(req.rid)
        h, d = st["h"], st["d"]
        self._h_rid.pop(h, None)
        admit = self._admit_verdict.pop(h)
        self.dec_lock_ref(st["deferred_lock"])
        if st["load_node"] is not None:
            # A FINAL verdict needs every loader's post-ack report, so it can
            # never outrun the local DMA or the local KV. Loud beats a silent
            # free of in-flight memory / a silent cross-stage length fork.
            assert st["complete"] is not None, "EIC PP verdict before local load ack"
            assert admit <= d + st["complete"], "EIC PP verdict exceeds local KV"
            # Free [admit, d + alloc): the failed tail AND loaded-beyond-verdict
            # in one cut at a verdict boundary, keeping trees and allocator
            # accounting PP-identical.
            self._free_failed_loadback(st["load_node"].id, max(admit - d, 0))
        else:
            assert admit <= d, "EIC PP verdict beyond device match without a load"
        if st["new_indices"] is not None and admit > d:
            req.prefix_indices = torch.cat(
                [req.prefix_indices, st["new_indices"][: admit - d]]
            )
        if admit < len(req.prefix_indices):
            req.prefix_indices = req.prefix_indices[:admit]
        prefix_len = len(req.prefix_indices)
        # Repoint last_node at the deepest RESIDENT node still spanning the
        # clamped prefix, seeding at the load chain's true depth (d + alloc);
        # path key lengths are split-invariant so the walk stays exact.
        node = st["load_node"] if st["load_node"] is not None else st["deferred_lock"]
        depth = d + st["alloc"]
        while node is not self.root_node and (
            node.value is None or depth - len(node.key) >= prefix_len
        ):
            depth -= len(node.key)
            node = node.parent
        req.last_node = node
        # Report the load-back through ep_main's cache breakdown: prepare_for_extend
        # turns these into cached_tokens_storage (sglang:cached_tokens_total
        # {cache_source="storage_unknown"}). The load is already in prefix_indices, so
        # PrefillAdder skips init_load_back and the host-hit subtraction under EIC.
        # PrefillAdder.add_one_req also sets extend_range from prefix_indices;
        # extend_range is still None here.
        req.host_hit_length = req.storage_hit_length = max(0, prefix_len - d)
        req.kv.cache_protected_len = prefix_len
        return True

    def release_load_admit(self, rid):
        # Cleanup for a deferring candidate removed from the waiting queue
        # WITHOUT admitting (abort / queued-limit): release every lock, drop all
        # per-req state, and tombstone the wire key so straggler reports and
        # verdicts of this incarnation die on arrival. Releases ride the same
        # relayed request stream on every stage, so the whole-span free below
        # stays PP-uniform. No-op if the rid never entered the gate.
        st = self.ongoing_load_admit.pop(rid, None)
        if st is None:
            return
        h = st["h"]
        self._h_rid.pop(h, None)
        self._admit_verdict.pop(h, None)
        self.dec_lock_ref(st["deferred_lock"])
        node_id = st["load_node"].id if st["load_node"] is not None else None
        if node_id is not None and node_id in self.ongoing_load_back:
            if st["complete"] is not None:
                self._loadback_rid.pop(node_id, None)
                self._free_failed_loadback(node_id, 0)  # whole span: uniform boundary
            # else: load still in flight -> _drain_local_acks frees it at ack.
        if self._pp_active:
            self._tombstone[h] = (st["epoch"], self._round + self._TOMBSTONE_TTL)
            key = (h, st["epoch"])
            for kind in (self._KIND_SPAN, self._KIND_FINAL):
                self._report_outbox.pop((h, st["epoch"], kind), None)
            self._span_reports.pop(key, None)
            self._load_reports.pop(key, None)
            self._await_load.pop(key, None)
            if self._verdict_outbox:
                self._verdict_outbox = [v for v in self._verdict_outbox if v[0] != h]

    def ready_to_load_host_cache(self):
        producer_index = self.cache_controller.layer_done_counter.update_producer()
        self.load_cache_event.set()
        return producer_index

    def check_hicache_events(self):
        self._drain_async_work()
        self.writing_check()
        self.loading_check()

    def match_prefix(self, params: MatchPrefixParams):
        key = params.key
        empty_value = torch.empty((0,), dtype=torch.int64, device=self.device)
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=empty_value,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                best_match_node=self.root_node,
                host_hit_length=0,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = empty_value

        host_hit_length = 0
        last_host_node = last_node
        while last_node.evicted:
            while not last_node.backuped and last_node.parent is not None:
                last_node = last_node.parent
                last_host_node = last_node
                host_hit_length = 0
            if not last_node.evicted:
                break
            host_hit_length += len(last_node.host_value)
            last_node = last_node.parent

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            best_match_node=last_host_node,
            host_hit_length=host_hit_length,
        )

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        node.last_access_time = time.monotonic()
        child_key = key.child_key(self.page_size)
        value = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            if node.evicted and not child.evicted:
                # A resident node under an evicted one (left by a failed
                # load-back) is unusable until the gap reloads: appending it
                # would splice KV across the gap.
                break
            child.last_access_time = time.monotonic()
            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                if not new_node.evicted:
                    value.append(new_node.value)
                node = new_node
                break
            else:
                if not child.evicted:
                    value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = key.child_key(self.page_size)

        return value, node

    def _split_node(self, key, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        new_node = TreeNode(priority=child.priority)
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len]
            child.value = child.value[split_len:]
        if child.backuped:
            new_node.host_value = child.host_value[:split_len]
            child.host_value = child.host_value[split_len:]
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[key.child_key(self.page_size)] = new_node
        return new_node

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        priority: int = 0,
        chunked: bool = False,
    ):
        if priority is None:
            priority = 0
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0, node

        child_key = key.child_key(self.page_size)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = node.key.match(key, page_size=self.page_size)

            if prefix_len == len(node.key):
                if node.evicted:
                    # change the reference if the node is evicted
                    # this often happens in the case of KV cache recomputation
                    node.value = value[:prefix_len].clone()
                    if not isinstance(self, EICPagedHiRadixCache):
                        self.token_to_kv_pool_host.free(node.host_value)
                    self.evictable_size_ += len(node.value)
                    self._update_leaf_status(node)
                    self._update_leaf_status(node.parent)
                    self.inc_hit_count(node, chunked)
                else:
                    self.inc_hit_count(node, chunked)
                    total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                if new_node.evicted:
                    new_node.value = value[:prefix_len].clone()
                    if not isinstance(self, EICPagedHiRadixCache):
                        self.token_to_kv_pool_host.free(new_node.host_value)
                    self.evictable_size_ += len(new_node.value)
                    self._update_leaf_status(new_node)
                    self._update_leaf_status(new_node.parent)
                    self.inc_hit_count(new_node, chunked)
                else:
                    self.inc_hit_count(new_node, chunked)
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)

            if self.cache_controller.write_policy != "write_back":
                self.inc_hit_count(new_node, chunked)
            node = new_node
        # RadixCache.insert unpacks (prefix_len, last_node).
        return total_prefix_length, node

    def _collect_leaves_device(self):
        def is_leaf(node):
            if node.evicted:
                return False
            if node == self.root_node:
                return False
            if len(node.children) == 0:
                return True
            for child in node.children.values():
                if not child.evicted:
                    return False
            return True

        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            if is_leaf(cur_node):
                ret_list.append(cur_node)
            else:
                for cur_child in cur_node.children.values():
                    if not cur_child.evicted:
                        stack.append(cur_child)
        return ret_list


DEFAULT_EIC_CHECK_MAX_NUM = 2048


def _need_calculate_hash(node: TreeNode, page_size: int):
    if node is None or node.key is None or len(node.key) == 0:
        return False
    return node.content_hash is None or len(node.key) // page_size != len(
        node.content_hash
    )


class EICPagedHiRadixCache(EICHiRadixCache):
    def __init__(
        self,
        params: CacheInitParams,
        server_args: ServerArgs,
    ):
        self.calculate_hash_fn = get_content_hash
        self.match_req_set = {}  # rid -> None, insertion-ordered for FIFO trim
        self.eic_check_max_num = DEFAULT_EIC_CHECK_MAX_NUM
        super().__init__(params, server_args)

    def init_hyper_params(self, config):
        super().init_hyper_params(config)
        self.eic_check_max_num = config.get(
            "eic_check_max_num", DEFAULT_EIC_CHECK_MAX_NUM
        )
        logger.info(
            f"EICPagedHiRadixCache eic_check_max_num set to {self.eic_check_max_num}"
        )
        self.load_back_check = config.get("load_back_check", False)

    def _calculate_content_hash(self, node: TreeNode):
        if _need_calculate_hash(node.parent, self.page_size):
            self._calculate_content_hash(node.parent)
        if node.parent is not None and node.parent.content_hash is not None:
            prev_node_hash = node.parent.content_hash[-1]
        else:
            prev_node_hash = None
        node.content_hash = self.calculate_hash_fn(
            node.key, self.page_size, prev_node_hash
        )

    def _split_node(self, key, child: TreeNode, split_len: int):
        assert (
            split_len % self.page_size == 0
        ), f"split_len {split_len} is not page aligned"
        # child node split into new_node -> child
        if _need_calculate_hash(child, self.page_size):
            self._calculate_content_hash(child)
        new_node = TreeNode(priority=child.priority)
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count
        split_hash_nums = split_len // self.page_size
        new_node.content_hash = child.content_hash[:split_hash_nums]
        child.content_hash = child.content_hash[split_hash_nums:]

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len]
            child.value = child.value[split_len:]
        if child.backuped:
            new_node.host_value = child.host_value[:split_len]
            child.host_value = child.host_value[split_len:]
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[key.child_key(self.page_size)] = new_node
        return new_node

    def _match_for_remote_fetch(self, node: TreeNode, key: RadixKey):
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        node.last_access_time = time.monotonic()
        child_key = key.child_key(self.page_size)
        local_prefix_len = 0

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = child.key.match(key, page_size=self.page_size)
            local_prefix_len += prefix_len
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                node = new_node
                break
            else:
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = key.child_key(self.page_size)
        return local_prefix_len, node

    def _insert_remote_node(self, node: TreeNode, key: RadixKey):
        node.last_access_time = time.monotonic()
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        if len(key) == 0:
            return 0

        child_key = key.child_key(self.page_size)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = node.key.match(key, page_size=self.page_size)

            if prefix_len == len(node.key):
                if node.evicted and node.host_value is None:
                    node.host_value = torch.arange(
                        len(node.key), dtype=torch.int32, device="cpu"
                    )
                if not node.evicted:
                    total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                if new_node.evicted and new_node.host_value is None:
                    new_node.host_value = torch.arange(
                        len(new_node.key), dtype=torch.int32, device="cpu"
                    )
                if not new_node.evicted:
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]

            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.host_value = torch.arange(
                len(key), dtype=torch.int32, device="cpu"
            )
            node.children[child_key] = new_node
            self._calculate_content_hash(new_node)
        return total_prefix_length

    def match_from_remote(self, waiting_queue: List[Req]):
        # waiting_queue is the same global stream on every rank, but a PP stage
        # can lag by a few requests (async forward). Gate on the common prefix so
        # every reduce below has a PP-invariant length and can't deadlock, even
        # if the local trees diverged.
        num_ready = torch.tensor(len(waiting_queue), dtype=torch.int64, device="cpu")
        self._reduce_min(num_ready)
        # PP0-authoritative length -> the reduce vector below is the same size on
        # every stage (so its p2p bcast can't hang). A lagging stage reads only
        # the reqs it actually has (local_n); the rest of the vector stays 0.
        num_ready = int(num_ready.item())
        local_n = min(num_ready, len(waiting_queue))
        if num_ready == 0:
            return
        if len(self.match_req_set) > 1000:
            self.match_req_set = dict(list(self.match_req_set.items())[500:])

        fetches = []  # (slot, last_node, compute_key, prev_hash)
        eic_keys = 0
        for slot in range(local_n):
            req = waiting_queue[slot]
            if req.rid in self.match_req_set:
                continue
            fill_ids = req.origin_input_ids + req.output_ids
            req_tokens = fill_ids[: len(fill_ids) - 1]
            if len(req_tokens) == 0:
                continue
            req_key = RadixKey(req_tokens, req.extra_key)
            prefix_len, last_node = self._match_for_remote_fetch(
                self.root_node, req_key
            )
            if len(req_key) - prefix_len < self.page_size:
                continue
            if _need_calculate_hash(last_node, self.page_size):
                self._calculate_content_hash(last_node)
            prev_hash = last_node.content_hash[-1] if last_node.content_hash else None
            fetches.append((slot, last_node, req_key[prefix_len:], prev_hash))
            eic_keys += (len(req_key) - prefix_len) // self.page_size
            if 0 < self.eic_check_max_num <= eic_keys:
                break

        # Query EIC per fetched slot, scatter into a queue-length vector (0 =
        # miss/fail), then MIN over CP/TP+PP so the admitted prefix -- and thus
        # the radix tree -- is identical on every stage.
        len_tensor = torch.zeros(num_ready, dtype=torch.int64, device="cpu")
        if fetches:
            lens = self.cache_controller.batch_find_longest_prefix_in_eic(
                [f[2] for f in fetches], [f[3] for f in fetches]
            )
            if len(lens) == len(fetches):
                for (slot, *_), n in zip(fetches, lens):
                    len_tensor[slot] = n
        # Unconditional: a lagging PP stage builds an empty fetches, and skipping
        # the reduce would orphan PP0's isend onto the next round's num_ready recv.
        self._reduce_min(len_tensor)

        for slot, last_node, compute_key, _ in fetches:
            eic_len = int(len_tensor[slot])
            if eic_len > 0:
                self._insert_remote_node(last_node, compute_key[:eic_len])
                # Only mark as matched when we actually admitted a remote prefix.
                # A miss (eic_len == 0) must be retried: the async write may not
                # have landed yet, or a sibling rank's write may still be in
                # flight. Permanently skipping it would strand the request.
                req = waiting_queue[slot]
                self.match_req_set[req.rid] = None

    def match_prefix(self, params: MatchPrefixParams):
        key = params.key
        empty_value = torch.empty((0,), dtype=torch.int64, device=self.device)
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=empty_value,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                best_match_node=self.root_node,
                host_hit_length=0,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = empty_value

        host_hit_length = 0
        last_host_node = last_node
        while last_node.evicted:
            while not last_node.backuped and last_node.parent is not None:
                last_node = last_node.parent
                last_host_node = last_node
                host_hit_length = 0
            if not last_node.evicted:
                break
            host_hit_length += len(last_node.host_value)
            last_node = last_node.parent

        # Load-back keeps only the trailing window of SWA KV; report that.
        swa_host_hit_length = 0
        if host_hit_length > 0 and self.sliding_window_size is not None:
            swa_host_hit_length = min(
                host_hit_length, max(self.sliding_window_size, self.page_size)
            )

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            best_match_node=last_host_node,
            host_hit_length=host_hit_length,
            swa_host_hit_length=swa_host_hit_length,
        )

    def write_backup(self, node: TreeNode, write_back=False):
        if node.evicted:
            return 0
        if not write_back and (
            node.parent != self.root_node and not node.parent.backuped
        ):
            return 0
        if _need_calculate_hash(node, self.page_size):
            self._calculate_content_hash(node)
        host_indices = self.cache_controller.write_page(
            device_indices=node.value,
            priority=-self.get_height(node),
            node_id=node.id,
            content_hash=node.content_hash,
        )
        if host_indices is not None:
            node.host_value = host_indices
            self.ongoing_write_through[node.id] = node
            if not write_back:
                self.inc_lock_ref(node)
        else:
            return 0

        return len(host_indices)

    def load_back(
        self,
        node: TreeNode,
        mem_quota: Optional[int] = None,
        allow_evict: bool = True,
    ) -> Optional[torch.Tensor]:
        # todo: more loading policies
        start_time = time.perf_counter()
        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # protect the ancestor nodes from eviction
        delta = self.inc_lock_ref(ancester_node)

        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancester_node)
            return None
        host_content_hash = []
        for n in nodes_to_load:
            host_content_hash.extend(n.content_hash)

        # check key existed
        if self.load_back_check:
            check_keys = host_content_hash[
                : self.load_back_threshold // self.page_size + 1
            ]
            mask = self.cache_controller.mem_pool_host.batch_exist_page(check_keys)
            check_ret = all(mask)
            if self.tp_size > 1:
                # gloo (cpu_group) has no ReduceOp.SUM for bool; use int32.
                check_tensor = torch.tensor(
                    0 if check_ret else 1, dtype=torch.int32, device="cpu"
                )
                torch.distributed.all_reduce(
                    check_tensor,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.tp_group,
                )
                check_ret = check_tensor.item() == 0
            if not check_ret:
                logger.warning(f"key has been evicted, skip load back")
                self.dec_lock_ref(ancester_node)
                return None

        device_indices = self.cache_controller.load_page(
            host_indices=host_indices,
            node_id=last_hit_node.id,
            content_hash=host_content_hash,
        )
        if device_indices is None and allow_evict:
            self.evict(EvictParams(num_tokens=len(host_indices)))
            device_indices = self.cache_controller.load_page(
                host_indices=host_indices,
                node_id=last_hit_node.id,
                content_hash=host_content_hash,
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            return None

        self.ongoing_load_back[last_hit_node.id] = (
            ancester_node,
            last_hit_node,
            len(device_indices),
        )
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)]
            offset += len(node.host_value)
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        return device_indices
