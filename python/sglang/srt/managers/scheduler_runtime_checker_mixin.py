from __future__ import annotations

import logging
import signal
import sys
import time
from typing import TYPE_CHECKING

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
from sglang.srt.utils.common import disable_request_logging, pyspy_dump_schedulers

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerRuntimeCheckerMixin:

    def _check_hybrid_memory(self: Scheduler):
        (
            full_num_used,
            swa_num_used,
            _,
            _,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        ) = self._get_swa_token_info()
        memory_leak = full_num_used != 0 or swa_num_used != 0
        token_msg = (
            f"{self.full_tokens_per_layer=}, {full_available_size=}, {full_evictable_size=}, {self.tree_cache.full_protected_size()=}\n"
            f"{self.swa_tokens_per_layer=}, {swa_available_size=}, {swa_evictable_size=}, {self.tree_cache.swa_protected_size()=}\n"
        )
        return memory_leak, token_msg

    def _check_mamba_memory(self: Scheduler):
        (
            full_num_used,
            mamba_num_used,
            _,
            _,
            full_available_size,
            full_evictable_size,
            mamba_available_size,
            mamba_evictable_size,
        ) = self._get_mamba_token_info()
        memory_leak = (
            full_num_used != self.tree_cache.full_protected_size()
            or mamba_num_used != self.tree_cache.mamba_protected_size()
        )
        token_msg = (
            f"{full_available_size=}, {full_evictable_size=}, {self.token_to_kv_pool_allocator.size=}, {self.tree_cache.full_protected_size()=}\n"
            f"{mamba_available_size=}, {mamba_evictable_size=}, {self.req_to_token_pool.mamba_pool.size=}, {self.tree_cache.mamba_protected_size()=}\n"
        )
        return memory_leak, token_msg

    def _check_radix_cache_memory(self: Scheduler):
        _, _, available_size, evictable_size = self._get_token_info()
        protected_size = self.tree_cache.protected_size()
        total_accounted = available_size + evictable_size + protected_size
        diff = self.max_total_num_tokens - total_accounted
        # Allow a small slack for in-flight/unaccounted tokens that are about to be
        # reflected by the tree/allocator (e.g., post-insert duplicates or tails).
        # This avoids transient false-positives during idle checks.
        try:
            page_size = int(getattr(self, "page_size", 1))
        except Exception:
            page_size = 1
        tolerance = max(32, page_size)
        memory_leak = not (0 <= diff <= tolerance)
        token_msg = (
            f"{self.max_total_num_tokens=}, {available_size=}, {evictable_size=}, "
            f"{protected_size=}, diff={diff}, tolerance={tolerance}\n"
        )
        return memory_leak, token_msg

    def _check_runtime_mem_leak(self: Scheduler):
        current_batch: ScheduleBatch = self.last_batch

        if current_batch is None:
            return

        _, _, available_size, evictable_size = self._get_token_info()
        protected_size = self.tree_cache.protected_size()

        extend_size = 0
        for i, req in enumerate(current_batch.reqs):
            seq_len = len(req.origin_input_ids) + len(req.output_ids)
            fill_len = len(req.fill_ids) if req.fill_ids is not None else 0
            prefix_len = (
                len(req.prefix_indices) if req.prefix_indices is not None else 0
            )

            if current_batch.forward_mode.is_decode():
                if req.finished():
                    unreleased_len = 1
                else:
                    unreleased_len = seq_len - prefix_len
            else:
                unreleased_len = fill_len - prefix_len

            extend_size += unreleased_len

        if (
            current_batch.forward_mode.is_extend()
            and self.running_batch is not None
            and not self.running_batch.is_empty()
            and self.running_batch.forward_mode.is_decode()
        ):
            for i, req in enumerate(self.running_batch.reqs):
                seq_len = len(req.origin_input_ids) + len(req.output_ids)
                prefix_len = (
                    len(req.prefix_indices) if req.prefix_indices is not None else 0
                )

                if req.finished():
                    unreleased_len = 0
                else:
                    unreleased_len = seq_len - prefix_len - 1

                extend_size += unreleased_len

        total_tokens = available_size + evictable_size + protected_size + extend_size

        assert (
            total_tokens == self.max_total_num_tokens
        ), f"Mem Leak Detected! {total_tokens=} vs {self.max_total_num_tokens=}"

    def _check_req_pool(self: Scheduler):
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            req_total_size = (
                self.req_to_token_pool.size + self.req_to_token_pool.pre_alloc_size
            )
        else:
            req_total_size = self.req_to_token_pool.size

        if len(self.req_to_token_pool.free_slots) != req_total_size:
            msg = (
                "req_to_token_pool memory leak detected!"
                f"available_size={len(self.req_to_token_pool.free_slots)}, "
                f"total_size={self.req_to_token_pool.size}\n"
            )
            raise ValueError(msg)

    def check_memory(self: Scheduler):
        if self.is_hybrid:
            memory_leak, token_msg = self._check_hybrid_memory()
        elif self.is_hybrid_gdn and isinstance(self.tree_cache, MambaRadixCache):
            memory_leak, token_msg = self._check_mamba_memory()
        else:
            memory_leak, token_msg = self._check_radix_cache_memory()

        if memory_leak:
            # Extra diagnostics to help pinpoint mismatched accounting
            try:
                alloc = self.token_to_kv_pool_allocator
                free_len = (
                    int(len(getattr(alloc, "free_pages", [])))
                    if getattr(alloc, "free_pages", None) is not None
                    else -1
                )
                release_len = (
                    int(len(getattr(alloc, "release_pages", [])))
                    if getattr(alloc, "release_pages", None) is not None
                    else -1
                )
                # Some trees expose total_size() for quick sanity
                tree_total = None
                try:
                    if hasattr(self.tree_cache, "total_size"):
                        total = self.tree_cache.total_size()
                        tree_total = total if isinstance(total, int) else str(total)
                except Exception:
                    tree_total = "n/a"
                print(
                    f"DEBUG {self.max_total_num_tokens=} "
                    f"free={free_len} "
                    f"release={release_len} "
                    f"evictable={self.tree_cache.evictable_size()} "
                    f"protected={self.tree_cache.protected_size()} "
                    f"avail={self.token_to_kv_pool_allocator.available_size()} "
                    f"tree_total={tree_total}"
                )
            except Exception:
                pass
            # Extra detailed breakdown including staged frees inside allocator free_group
            try:
                alloc = self.token_to_kv_pool_allocator
                page_size_dbg = int(getattr(alloc, "page_size", 1))
                is_open_free_group = bool(
                    hasattr(alloc, "is_not_in_free_group")
                    and (not alloc.is_not_in_free_group)
                )
                fg_list = getattr(alloc, "free_group", None)
                staged_groups = int(len(fg_list)) if fg_list is not None else 0
                staged_pages = 0
                if isinstance(fg_list, list) and staged_groups > 0:
                    # Sum lengths of page-id tensors staged for grouped frees
                    staged_pages = int(
                        sum(int(len(t)) for t in fg_list if t is not None)
                    )
                staged_tokens = staged_pages * page_size_dbg

                avail_now = int(self.token_to_kv_pool_allocator.available_size())
                evictable_now = int(self.tree_cache.evictable_size())
                protected_now = int(self.tree_cache.protected_size())
                total_accounted = avail_now + evictable_now + protected_now
                diff_now = int(self.max_total_num_tokens - total_accounted)
                diff_with_staged = int(
                    self.max_total_num_tokens - (total_accounted + staged_tokens)
                )
                reserved_decode = int(
                    getattr(self.server_args, "num_reserved_decode_tokens", 0)
                )
                running_nonempty = bool(
                    self.running_batch is not None and not self.running_batch.is_empty()
                )
                print(
                    "DEBUG+ breakdown: "
                    f"page_size={page_size_dbg} "
                    f"staged_groups={staged_groups} "
                    f"staged_pages={staged_pages} "
                    f"staged_tokens={staged_tokens} "
                    f"avail_now={avail_now} "
                    f"evictable_now={evictable_now} "
                    f"protected_now={protected_now} "
                    f"total_accounted={total_accounted} "
                    f"diff_now={diff_now} "
                    f"diff_with_staged={diff_with_staged} "
                    f"reserved_decode={reserved_decode} "
                    f"running_nonempty={running_nonempty}"
                )
            except Exception:
                pass
            # Decode-boundary slack estimate (pages that will be allocated next decode step)
            try:
                last_slack_pages = -1
                run_slack_pages = -1
                if getattr(self, "last_batch", None) is not None:
                    last_slack_pages = int(self.last_batch.new_page_count_next_decode())
                if (
                    getattr(self, "running_batch", None) is not None
                    and not self.running_batch.is_empty()
                ):
                    run_slack_pages = int(
                        self.running_batch.new_page_count_next_decode()
                    )
                slack_pages_total = max(
                    0, (0 if last_slack_pages < 0 else last_slack_pages)
                ) + max(0, (0 if run_slack_pages < 0 else run_slack_pages))
                slack_tokens = slack_pages_total * page_size_dbg
                diff_minus_slack = diff_now - slack_tokens
                last_bs = (
                    self.last_batch.batch_size() if self.last_batch is not None else 0
                )
                run_bs = (
                    self.running_batch.batch_size()
                    if self.running_batch is not None
                    and not self.running_batch.is_empty()
                    else 0
                )
                print(
                    "DEBUG+ decode_slack: "
                    f"last_slack_pages={last_slack_pages} "
                    f"run_slack_pages={run_slack_pages} "
                    f"slack_pages_total={slack_pages_total} "
                    f"slack_tokens={slack_tokens} "
                    f"diff_minus_slack={diff_minus_slack} "
                    f"last_batch_size={last_bs} "
                    f"running_batch_size={run_bs}"
                )
            except Exception:
                pass
            msg = "token_to_kv_pool_allocator memory leak detected! " f"{token_msg}"
            raise ValueError(msg)

        self._check_req_pool()

        if (
            self.enable_metrics
            and self.current_scheduler_metrics_enabled()
            and time.perf_counter() > self.metrics_collector.last_log_time + 30
        ):
            # During idle time, also collect metrics every 30 seconds.
            if self.is_hybrid:
                (
                    full_num_used,
                    swa_num_used,
                    full_token_usage,
                    swa_token_usage,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_swa_token_info()
                num_used = max(full_num_used, swa_num_used)
                token_usage = max(full_token_usage, swa_token_usage)
            elif self.is_hybrid_gdn:
                (
                    num_used,
                    _,
                    token_usage,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_mamba_token_info()
            else:
                num_used, token_usage, _, _ = self._get_token_info()
            num_running_reqs = len(self.running_batch.reqs)
            self.stats.num_running_reqs = num_running_reqs
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = round(token_usage, 2)
            self.stats.gen_throughput = 0
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.stats.num_grammar_queue_reqs = len(self.grammar_queue)
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = len(
                    self.disagg_prefill_bootstrap_queue.queue
                )
                self.stats.num_prefill_inflight_queue_reqs = len(
                    self.disagg_prefill_inflight_queue
                )
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = len(
                    self.disagg_decode_prealloc_queue.queue
                )
                self.stats.num_decode_transfer_queue_reqs = len(
                    self.disagg_decode_transfer_queue.queue
                )
            self.metrics_collector.log_stats(self.stats)
        self._publish_kv_events()

    def check_tree_cache(self: Scheduler):
        if (self.is_hybrid and isinstance(self.tree_cache, SWARadixCache)) or (
            self.is_hybrid_gdn and isinstance(self.tree_cache, MambaRadixCache)
        ):
            self.tree_cache.sanity_check()

    def self_check_during_idle(self: Scheduler):
        # Skip idle checks if there is an in-flight running batch to avoid counting
        # tokens that are not yet reflected in the radix tree or allocator lists.
        if self.running_batch is not None and not self.running_batch.is_empty():
            return

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            if len(self.disagg_prefill_inflight_queue) > 0:
                return
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            queue_size = (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
            )
            if self.server_args.disaggregation_decode_enable_offload_kvcache:
                queue_size += len(self.decode_offload_manager.ongoing_offload)
            if queue_size:
                return

        self.check_memory()
        self.check_tree_cache()
        self.new_token_ratio = self.init_new_token_ratio
        self.maybe_sleep_on_idle()

    def watchdog_thread(self: Scheduler):
        """A watch dog thread that will try to kill the server itself if one forward batch takes too long."""
        self.watchdog_last_forward_ct = 0
        self.watchdog_last_time = time.perf_counter()

        while True:
            current = time.perf_counter()
            if self.cur_batch is not None:
                if self.watchdog_last_forward_ct == self.forward_ct:
                    if current > self.watchdog_last_time + self.watchdog_timeout:
                        break
                else:
                    self.watchdog_last_forward_ct = self.forward_ct
                    self.watchdog_last_time = current
            time.sleep(self.watchdog_timeout // 2)

        if not disable_request_logging():
            # Print batch size and memory pool info to check whether there are de-sync issues.
            if self.is_hybrid:
                _, info_msg = self._check_hybrid_memory()
            elif self.is_hybrid_gdn and isinstance(self.tree_cache, MambaRadixCache):
                _, info_msg = self._check_mamba_memory()
            else:
                _, info_msg = self._check_radix_cache_memory()
            logger.error(
                f"{self.cur_batch.batch_size()=}\n"
                f"{self.cur_batch.reqs=}\n"
                f"{info_msg}"
            )

        pyspy_dump_schedulers()
        logger.error(f"Watchdog timeout ({self.watchdog_timeout=})")
        print(file=sys.stderr, flush=True)
        print(file=sys.stdout, flush=True)

        # Wait for some time so that the parent process can print the error.
        time.sleep(5)
        self.parent_process.send_signal(signal.SIGQUIT)
