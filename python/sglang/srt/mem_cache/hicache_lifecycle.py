"""HiCache's asynchronous transfer lifecycle, mixed into the existing tree owner.

Tree mutations run only on the scheduler thread. Controller ACKs progress I/O;
ready restore tickets are published only when their request retries admission.
"""

import json
import logging
import time

from sglang.srt.kv_compression.types import (
    BufferDrainError,
    CompressionCapacityError,
    new_page_refs,
)
from sglang.srt.mem_cache.l2_completion import (
    RestoreTicket,
    RestoreTransferResult,
    exceeds_load_quota,
    record_load_back_metrics,
    should_skip_full_load,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

logger = logging.getLogger(__name__)
REFS = "compression_page_refs"


class HiCacheLifecycleMixin:
    @property
    def async_l2(self):
        return getattr(getattr(self, "cache_controller", None), "async_l2", None)

    def get_kv_compression_context(self):
        state = self.async_l2
        return (state.runtime, state.provider) if state is not None else (None, None)

    def admission_failure_reason(self):
        state = self.async_l2
        if state is not None and (
            state.quarantined
            or getattr(state.runtime, "is_quarantined", lambda: False)()
        ):
            return "Compressed HiCache is quarantined; restart worker"
        return None

    def get_restore_reserved_tokens(self, req, match=None):
        state = self.async_l2
        ticket = state.restore_ticket if state is not None else None
        if ticket is None or ticket.req is not req:
            return 0
        anchor = match.best_match_node if match is not None else req.best_match_node
        host_hits = match.host_hit_length if match is not None else req.host_hit_length
        if (
            ticket is None
            or ticket.req is not req
            or not ticket.ready
            or ticket.consumed
            or ticket.abandoned
            or self.admission_failure_reason()
            or req.finished_reason is not None
            or req.to_finish is not None
            or getattr(req, "retracted_stain", False)
            or anchor != ticket.anchor
        ):
            return 0
        # Scheduler-thread snapshot only. Do not publish, allocate, or give
        # other candidates credit for memory owned by this request.
        kv, extra = self.tree_core.build_load_back_spec(ticket.anchor, req=req)
        if extra or len(kv.host_indices) != host_hits:
            return 0
        positions = {h: i for i, h in enumerate(ticket.host_indices.tolist())}
        current = kv.host_indices.tolist()  # Handles include the generation.
        if len(set(current)) != len(current) or any(
            h not in positions for h in current
        ):
            return 0
        refs = tuple(
            ref
            for nid in kv.nodes_to_load or []
            for ref in self.tree_core.node_by_id(nid)
            .component_data[ComponentType.FULL]
            .metadata.get(REFS, ())
        )
        selected = [positions[h] for h in current]
        if refs != tuple(ticket.page_refs[i] for i in selected):
            return 0
        if any(i >= len(ticket.device_indices) for i in selected):
            return 0
        return len(selected)

    def kv_page_refs(self, node):
        cd = node.component_data[ComponentType.FULL]
        refs = cd.metadata.get(REFS)
        if refs is None:
            refs = cd.metadata[REFS] = new_page_refs(len(node.key))
        if len(refs) != len(node.key):
            raise RuntimeError("KV identities do not match the split node")
        return refs

    def get_kv_transfer_refs(self, req, start, end, indices):
        if self.async_l2 is None:
            return None  # P/D-only execution assigns transient identities itself.
        from sglang.srt.mem_cache.radix_cache import RadixKey

        refs = list(new_page_refs(end - start))
        key = RadixKey(req.get_fill_ids(), req.extra_key, cache_salt=req.cache_salt)
        for node, pos, length in self.tree_core._walk_span(key, end):
            lo, hi = max(start, pos), min(end, pos + length)
            if lo >= hi:
                continue
            cd = node.component_data[ComponentType.FULL]
            if cd.value is not None and cd.value[
                lo - pos : hi - pos
            ].cpu().tolist() == list(map(int, indices[lo - start : hi - start])):
                refs[lo - start : hi - start] = self.kv_page_refs(node)[
                    lo - pos : hi - pos
                ]
        return tuple(refs)

    def on_kv_recomputed(self, node):
        if node.write_through_pending_id is not None:
            raise RuntimeError(
                "A pending backup must pin its source against recomputation"
            )
        cd = node.component_data[ComponentType.FULL]
        if cd.host_value is not None:
            self.async_l2.pool.free(cd.host_value)
            cd.host_value = None
        cd.metadata.pop(REFS, None)

    def _prepare_async_backup(self, node_id):
        state = self.async_l2
        node = self.tree_core.node_by_id(node_id)
        cd = node.component_data[ComponentType.FULL]
        if state.quarantined:
            return None
        if cd.host_value is not None or node.write_through_pending_id is not None:
            state.stats["backup_duplicate_skips"] += 1
            return None
        parent = node.parent
        if parent is not self.tree_core.root_node and not parent.backuped:
            state.stats["backup_parent_missing"] += 1
            return None
        count = len(node.key)
        if count * state.pool.reservation_bytes > state.pool.arena.numel():
            state.stats["admission_skips"] += 1
            return None
        refs = self.kv_page_refs(node)
        parent_pin, source_pin = None, None
        try:
            if parent is not self.tree_core.root_node:
                parent_pin = (
                    parent.id,
                    self.inc_host_lock_ref(parent.id).to_dec_params(),
                )
            source_pin = self.inc_lock_ref(node_id).to_dec_params()
            state.backup_pins[node_id] = (source_pin, parent_pin)
            parent_ack = getattr(parent, "write_through_pending_id", None)
            if parent_ack in state.backup_completions:
                state.backup_dependencies[node_id] = state.backup_completions[
                    parent_ack
                ]
            return refs
        except Exception:
            try:
                if source_pin is not None:
                    self.dec_lock_ref(node_id, source_pin)
                if parent_pin is not None:
                    self.dec_host_lock_ref(*parent_pin)
                state.backup_pins.pop(node_id, None)
                state.backup_dependencies.pop(node_id, None)
            except Exception as exc:
                state.quarantined.append((node_id, source_pin, parent_pin))
                raise BufferDrainError("L2 backup preparation cleanup failed") from exc
            raise

    def _release_backup_parent(self, node_id):
        pins = self.async_l2.backup_pins.pop(node_id, None)
        self.async_l2.backup_completions.pop(node_id, None)
        self.async_l2.backup_dependencies.pop(node_id, None)
        if pins is not None and pins[1] is not None:
            self.dec_host_lock_ref(*pins[1])

    def _rollback_async_backup(self, node_id):
        state = self.async_l2
        pending = self.ongoing_write_through.pop(node_id, None)
        if pending is not None:
            for nid in pending.publish_node_ids:
                node = self.tree_core.node_by_id(nid)
                cd = node.component_data[ComponentType.FULL]
                if cd.host_value is not None:
                    state.pool.free(cd.host_value)
                    cd.host_value = None
                node.write_through_pending_id = None
                self.tree_core._update_duplicate_tracking(node)
                self.tree_core._update_evictable_leaf_sets(node)
            self.dec_lock_ref(pending.node_id, pending.lock_params)
        else:
            pins = state.backup_pins.get(node_id)
            if pins is not None:
                self.dec_lock_ref(node_id, pins[0])
        self._release_backup_parent(node_id)

    def _complete_async_write_ack(self, ack):
        """Return success; failed-but-drained backups preserve the GPU copy."""
        state = self.async_l2
        try:
            count = ack.result()
        except BufferDrainError:
            state.quarantined.append(ack)
            logger.exception("L2 backup did not drain; restart worker")
            return False
        except Exception as exc:
            state.stats[
                "admission_skips"
                if isinstance(exc, CompressionCapacityError)
                else "backup_failures"
            ] += 1
            for nid in ack.node_ids:
                self._rollback_async_backup(nid)
            logger.exception("L2 backup rolled back")
            return False
        try:
            for nid in ack.node_ids:
                self._finish_write_through_ack(nid)
                self._release_backup_parent(nid)
            state.stats["d2h_bytes"] += count
            state.stats["backed_up_pages"] += ack.num_tokens
            state.stats["backup_completed"] += len(ack.node_ids)
            logger.info(
                "KV_COMPRESSION_BACKUP_DONE %s",
                json.dumps({"node_ids": ack.node_ids, "pages": ack.num_tokens}),
            )
        except Exception:
            state.quarantined.append(ack)
            logger.exception("L2 ACK publication failed; restart worker")
            return False
        return True

    def _drop_restore_ticket(self, ticket, *, free_targets=True):
        if ticket.consumed:
            return
        ticket.consumed = True
        try:
            if free_targets:
                self.token_to_kv_pool_allocator.free(ticket.device_indices)
            for lease in ticket.leases:
                lease.close()
            self.dec_lock_ref(ticket.anchor, ticket.device_lock)
            self.dec_host_lock_ref(ticket.anchor, ticket.host_lock)
        except Exception as exc:
            self.async_l2.quarantined.append(ticket)
            raise BufferDrainError(
                "L2 ticket release was only partially completed"
            ) from exc
        finally:
            if self.async_l2.restore_ticket is ticket:
                self.async_l2.restore_ticket = None

    def _invalidate_async_host_nodes(self, node_ids):
        for nid in node_ids:
            node = self.tree_core.node_by_id(nid)
            cd = node.component_data[ComponentType.FULL]
            if cd.host_value is not None:
                for handle, ref in zip(cd.host_value.tolist(), self.kv_page_refs(node)):
                    self.async_l2.pool.free_matching(handle, ref)
                cd.host_value = None
            self.tree_core._update_duplicate_tracking(node)
            self.tree_core._update_evictable_leaf_sets(node)

    def _complete_async_load_ack(self, ack):
        state = self.async_l2
        ticket = state.restore_ticket
        if ticket is None or ticket.completion is not ack.completion:
            raise RuntimeError("L2 restore ACK does not own the current ticket")
        if ticket.ready:
            return  # The same ready ticket cannot account for I/O twice.
        try:
            transfer = ack.result()
        except BufferDrainError:
            ticket.req.set_finish_with_abort(
                "L2 restore did not drain; restart worker", 500
            )
            state.quarantined.append(ticket)
            state.restore_ticket = None
            return
        except Exception as exc:
            if isinstance(exc, CompressionCapacityError):
                state.stats["restore_admission_retries"] += 1
            else:
                state.stats["restore_failures"] += 1
                ticket.req.set_finish_with_abort(
                    "L2 restore or verification failed", 500
                )
                kv, _ = self.tree_core.build_load_back_spec(
                    ticket.anchor, req=ticket.req
                )
                self._invalidate_async_host_nodes(kv.nodes_to_load or [])
            self._drop_restore_ticket(ticket)
            return
        # The ACK counts completed I/O, including work later discarded on cancel.
        # Request Host provenance is updated only when the ticket is consumed.
        if not isinstance(transfer, RestoreTransferResult):
            state.quarantined.append(ticket)
            state.restore_ticket = None
            ticket.req.set_finish_with_abort("Missing L2 restore completion data", 500)
            return
        state.stats["completed_restore_pages"] += transfer.pages
        state.stats["completed_restore_logical_bytes"] += transfer.logical_bytes
        state.stats["h2d_bytes"] += transfer.actual_bytes
        state.stats["restore_queue_seconds"] += transfer.queue_seconds
        state.stats["restore_execution_seconds"] += transfer.execution_seconds
        record_load_back_metrics(
            getattr(self, "metrics_collector", None),
            {"kv": transfer.pages},
            transfer.actual_bytes,
            transfer.gpu_seconds,
        )
        ticket.ready = True
        self._reap_cancelled_restore()

    def _reap_cancelled_restore(self):
        ticket = self.async_l2.restore_ticket
        if (
            ticket is not None
            and ticket.ready
            and (
                ticket.abandoned
                or ticket.req.finished_reason is not None
                or ticket.req.to_finish is not None
            )
        ):
            self.async_l2.stats["cancelled_restores"] += 1
            self._drop_restore_ticket(ticket)

    def init_async_load_back(self, params):
        from sglang.srt.managers.cache_controller import HiCacheAck
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        state, req = self.async_l2, params.req
        if state.quarantined:
            req.set_finish_with_abort("L2 is quarantined; restart worker", 500)
            return None
        ticket = state.restore_ticket
        if ticket is not None:
            if ticket.req is not req or not ticket.ready:
                return None
            return self._consume_restore_ticket(ticket, params)
        kv, extra = self.tree_core.build_load_back_spec(params.best_match_node, req=req)
        if extra:
            raise ValueError("Async compressed L2 supports FULL KV only")
        count = len(kv.host_indices)
        if should_skip_full_load(count, self.load_back_threshold):
            return self.tree_core.empty_match_result.device_indices, req.last_node
        anchor = params.best_match_node
        host_lock, device_lock = None, None
        indices, leases = None, []
        completion = None
        retain = False
        checking_objects = False
        try:
            host_lock = self.inc_host_lock_ref(anchor).to_dec_params()
            lock_result = self.inc_lock_ref(anchor)
            device_lock = lock_result.to_dec_params()
            if exceeds_load_quota(count, lock_result.delta, params.mem_quota):
                return self.tree_core.empty_match_result.device_indices, req.last_node
            allocator = self.token_to_kv_pool_allocator
            if allocator.available_size() < count:
                self.evict_for_alloc(
                    EvictParams(num_tokens=count - allocator.available_size())
                )
            indices = allocator.alloc(count)
            if indices is None:
                return self.tree_core.empty_match_result.device_indices, req.last_node
            checking_objects = True
            refs = tuple(
                ref
                for nid in kv.nodes_to_load or []
                for ref in self.kv_page_refs(self.tree_core.node_by_id(nid))
            )
            if len(refs) != count:
                raise RuntimeError("L2 restore identities do not match source range")
            for handle, ref in zip(kv.host_indices.tolist(), refs):
                leases.append(state.pool.acquire(handle, expected_ref=ref))
            checking_objects = False
            completion = self.cache_controller.l2_transfer_engine.submit_async_restore(
                state, leases, indices, self.cache_controller.load_fence_stream
            )
            ticket = RestoreTicket(
                anchor,
                kv.host_indices.clone(),
                indices,
                device_lock,
                host_lock,
                leases,
                completion,
                req,
                refs,
            )
            state.restore_ticket = ticket
            self.cache_controller.ack_load_queue.append(
                HiCacheAck(None, None, [anchor], count, completion=completion)
            )
            return None
        except BufferDrainError:
            retain = True
            state.quarantined.append(
                (completion, leases, indices, host_lock, device_lock)
            )
            state.restore_ticket = None
            req.set_finish_with_abort("L2 restore submission did not drain", 500)
            return None
        except CompressionCapacityError:
            if completion is not None:
                retain = True
                state.quarantined.append(
                    (completion, leases, indices, host_lock, device_lock)
                )
                state.restore_ticket = None
                req.set_finish_with_abort("L2 restore ownership was not committed", 500)
                return None
            state.stats["restore_admission_retries"] += 1
            return None
        except Exception:
            state.stats["restore_failures"] += 1
            req.set_finish_with_abort("L2 restore admission failed", 500)
            if completion is not None:
                # A task was submitted: don't free even if ticket/ACK creation fails.
                retain = True
                state.quarantined.append(
                    (completion, leases, indices, host_lock, device_lock)
                )
                state.restore_ticket = None
            elif checking_objects:
                self._invalidate_async_host_nodes(kv.nodes_to_load or [])
            logger.exception("L2 restore admission failed")
            return None
        finally:
            if state.restore_ticket is None and not retain:
                try:
                    for lease in leases:
                        lease.close()
                    if indices is not None:
                        self.token_to_kv_pool_allocator.free(indices)
                    if device_lock is not None:
                        self.dec_lock_ref(anchor, device_lock)
                    if host_lock is not None:
                        self.dec_host_lock_ref(anchor, host_lock)
                except Exception as exc:
                    state.quarantined.append(
                        (indices, leases, anchor, device_lock, host_lock)
                    )
                    raise BufferDrainError(
                        "L2 restore preparation cleanup failed"
                    ) from exc

    def _consume_restore_ticket(self, ticket, params):
        state = self.async_l2
        if ticket.consumed:
            ticket.req.set_finish_with_abort("Restore ticket consumed twice", 500)
            return None
        published = False
        try:
            if (
                ticket.req.finished_reason is not None
                or ticket.req.to_finish is not None
            ):
                self._drop_restore_ticket(ticket)
                return None
            if params.best_match_node != ticket.anchor:
                self._drop_restore_ticket(ticket)
                return None
            kv, extra = self.tree_core.build_load_back_spec(
                ticket.anchor, req=ticket.req
            )
            if extra:
                raise ValueError("Unexpected auxiliary restore state")
            positions = {h: i for i, h in enumerate(ticket.host_indices.tolist())}
            selected = [positions[h] for h in kv.host_indices.tolist()]
            refs = tuple(
                ref
                for nid in kv.nodes_to_load or []
                for ref in self.kv_page_refs(self.tree_core.node_by_id(nid))
            )
            if refs != tuple(ticket.page_refs[i] for i in selected):
                raise ValueError("Stale L2 materialization during restoration")
            adopted = ticket.device_indices[selected]
            if selected:
                published = True
                self._apply_cache_actions(
                    self.tree_core.commit_load_back(ticket.anchor, adopted, kv, {})
                )
                self.tree_core.finish_load_back(ticket.anchor)
            unused = sorted(set(range(len(ticket.device_indices))) - set(selected))
            if unused:
                self.token_to_kv_pool_allocator.free(ticket.device_indices[unused])
            state.stats["restored_pages"] += len(selected)
            lz4 = sum(
                ticket.leases[i].future.result().encoding == "lz4" for i in selected
            )
            verified = len(selected) if state.runtime.verify else 0
            state.stats["restored_lz4_pages"] += lz4
            state.stats["verified_restored_pages"] += verified
            state.stats["restore_seconds"] += (
                time.perf_counter() - ticket.completion.started
            )
            logger.info(
                "KV_COMPRESSION_L2_RESTORE %s",
                json.dumps(
                    {
                        "rid": ticket.req.rid,
                        "native_missing_pages": len(ticket.host_indices),
                        "adopted_pages": len(selected),
                        "lz4_pages": lz4,
                        "verified_pages": verified,
                        "seconds": time.perf_counter() - ticket.completion.started,
                    }
                ),
            )
            self._drop_restore_ticket(ticket, free_targets=False)
            return adopted, ticket.anchor
        except Exception:
            ticket.req.set_finish_with_abort("L2 restore publication failed", 500)
            if published:
                state.quarantined.append(ticket)
                state.restore_ticket = None
            else:
                self._drop_restore_ticket(ticket)
            logger.exception("L2 restore ticket could not be committed")
            return None

    def reconcile_restore_ticket(self, params, match):
        state = self.async_l2
        if state is None or state.restore_ticket is None:
            return
        ticket = state.restore_ticket
        if ticket.req is params.req and (
            match.host_hit_length == 0
            or getattr(ticket.req, "retracted_stain", False)
            or match.best_match_node != ticket.anchor
            or (
                ticket.ready
                and self.get_restore_reserved_tokens(ticket.req, match)
                != match.host_hit_length
            )
        ):
            ticket.abandoned = True
            self._reap_cancelled_restore()

    def has_pending_background_work(self):
        state = self.async_l2
        return state is not None and bool(
            self.cache_controller.ack_write_queue
            or self.cache_controller.ack_load_queue
            or state.restore_ticket
            or state.runtime.has_pending_work()
            or getattr(state.pool, "has_readers", lambda: False)()
        )

    def background_work_is_idle(self):
        state = self.async_l2
        return state is None or (
            state.idle()
            and not self.cache_controller.ack_write_queue
            and not self.cache_controller.ack_load_queue
        )

    def _poll_async_l2_housekeeping(self):
        state = self.async_l2
        if state is None:
            return
        self._reap_cancelled_restore()
        if time.monotonic() - state.last_metrics < 5:
            return
        state.last_metrics = time.monotonic()
        from collections import defaultdict

        stats = defaultdict(float, state.stats)
        stats.update(state.pool.io.stats)
        runtime, store = state.runtime.snapshot(), state.pool.snapshot()
        store.update(state.pool.allocator_snapshot())
        logger.info(
            "KV_COMPRESSION_STATS %s",
            json.dumps(
                {
                    "runtime": runtime,
                    "l2": store,
                    "operations": stats,
                    "pending_backups": len(self.cache_controller.write_queue)
                    + max(0, len(self.cache_controller.ack_write_queue) - 1),
                    "active_backup": (
                        {
                            "node_ids": self.cache_controller.ack_write_queue[
                                0
                            ].node_ids,
                            "pages": self.cache_controller.ack_write_queue[
                                0
                            ].num_tokens,
                        }
                        if self.cache_controller.ack_write_queue
                        else None
                    ),
                    "active_restore": state.restore_ticket.req.rid
                    if state.restore_ticket
                    else None,
                    "quarantined": len(state.quarantined),
                }
            ),
        )
        if getattr(self, "_enable_metrics_flag", False):
            from sglang.srt.kv_compression.metrics import get_compression_metrics

            get_compression_metrics().update(runtime, stats, store)

    def reset_async_l2(self):
        if self.async_l2 is not None:
            if not self.background_work_is_idle():
                raise RuntimeError(
                    "Cannot reset HiCache while async L2 is active or quarantined"
                )
            self.async_l2.pool.clear()

    def close_async_l2(self):
        if self.async_l2 is not None:
            self.async_l2.close()
