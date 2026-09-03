"""Runtime attach / detach of the HiCache (L3) storage backend.

``UnifiedRadixCache`` owns the tree; this component owns the lifecycle of the
storage backend behind it: validating the requested policies, applying the storage
runtime config, starting and stopping the controller's storage threads, and
cleaning up the bookkeeping a half-finished prefetch / backup leaves behind.

Keeping it beside the tree rather than inside it gives the three entry points --
startup, the admin API, and the atexit hook -- one implementation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.managers.cache_controller import StorageLifecycleConsensusError
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_STORAGE,
    StorageMetricsCollector,
    resolve_collector_class,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)

# Kept in sync with the server-args choices; validated here so an admin request
# carrying a bad policy is rejected before it can touch the controller.
_PREFETCH_POLICIES = ("best_effort", "wait_complete", "timeout")
_WRITE_POLICIES = ("write_back", "write_through", "write_through_selective")


class StorageAttachment:
    """Attach / detach the storage backend of one ``UnifiedRadixCache``."""

    def __init__(self, cache: UnifiedRadixCache):
        self._cache = cache
        # Custom group creation is a default-world collective.  Do it while all
        # scheduler ranks are still constructing the tree, before DP replicas
        # acquire independent request/idle state.
        controller = getattr(cache, "cache_controller", None)
        if controller is not None and hasattr(
            controller, "initialize_storage_data_sync_groups"
        ):
            controller.initialize_storage_data_sync_groups()

    # ---- Lifecycle entry points ----

    def attach(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
        local_ready: bool = True,
    ) -> tuple[bool, str]:
        """Enable the storage backend at runtime.

        Starts the storage threads inside the cache controller and turns on the
        prefetch / backup paths. The caller must ensure there are no running or
        queued requests, so this cannot race the scheduler thread.
        """
        cache = self._cache

        # Validate first: a rejected request must have no side effects.
        invalid = self._validate_policies(
            hicache_storage_prefetch_policy, hicache_write_policy
        )
        if invalid is not None:
            return False, invalid

        controller = cache.cache_controller
        if controller is None:
            return (
                False,
                "HiCache is not initialized (no cache controller); "
                "launch with --enable-hierarchical-cache to attach a backend.",
            )

        try:
            lifecycle_ready = controller.prepare_storage_lifecycle(local_ready)
        except StorageLifecycleConsensusError:
            raise
        except Exception as e:
            logger.exception("Failed to coordinate storage attach readiness.")
            return False, f"Failed to coordinate storage attach readiness: {e}"
        if not lifecycle_ready:
            return (
                False,
                "Reject attach: scheduler is not idle on this or a peer "
                "model-parallel rank.",
            )

        if cache.enable_storage:
            current_backend = controller.storage_backend_type
            if current_backend != storage_backend:
                return (
                    False,
                    f"HiCache storage backend is already enabled with backend "
                    f"'{current_backend}'. Cannot attach different backend "
                    f"'{storage_backend}'. Detach first.",
                )
            # Same backend: the request degenerates to a policy update.
            self._apply_policies(hicache_storage_prefetch_policy, hicache_write_policy)
            return (
                True,
                "HiCache storage backend already enabled with same backend; "
                "policies updated.",
            )

        logger.info(f"Attaching HiCache storage backend: {storage_backend}")
        parsed_config = None
        parse_error = None
        try:
            parsed_config = (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = HybridCacheController.parse_storage_backend_extra_config(
                storage_backend_extra_config_json
            )
        except Exception as error:
            logger.exception(
                "Failed to parse storage_backend_extra_config_json: %s", error
            )
            parse_error = error

        try:
            parse_agreed = controller.prepare_storage_lifecycle(parse_error is None)
        except StorageLifecycleConsensusError:
            raise
        except Exception as error:
            logger.exception("Failed to coordinate storage config parsing.")
            return (
                False,
                f"Failed to coordinate storage config parsing: {error}",
            )
        if not parse_agreed:
            where = "this rank" if parse_error is not None else "a peer rank"
            detail = f": {parse_error}" if parse_error is not None else ""
            return (
                False,
                f"Failed to parse storage_backend_extra_config_json on "
                f"{where}{detail}; backend was not attached.",
            )

        assert parsed_config is not None
        (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_base,
            prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys,
        ) = parsed_config

        # Apply policies before the controller attach, so the storage threads
        # observe the new values as soon as they start.
        self._apply_policies(hicache_storage_prefetch_policy, hicache_write_policy)

        try:
            controller.attach_storage_backend(
                storage_backend=storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=served_model_name,
                storage_backend_extra_config=extra_config,
                host_pools=controller.mem_pool_host.entries,
                coordinated_lifecycle=True,
            )
        except StorageLifecycleConsensusError:
            raise
        except Exception as e:
            logger.exception(
                f"Failed to attach storage backend '{storage_backend}': {e}"
            )
            return False, f"Failed to attach storage backend '{storage_backend}': {e}"

        publication_error = None
        try:
            self.apply_runtime_config(
                storage_backend=storage_backend,
                prefetch_threshold=prefetch_threshold,
                prefetch_timeout_base=prefetch_timeout_base,
                prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
                enable_storage=True,
                enable_storage_metrics=cache._enable_metrics_flag,
                extra_metric_labels=cache.extra_metric_labels,
            )
        except Exception as error:
            logger.exception("Failed to publish attached storage state to the tree.")
            publication_error = error

        try:
            publication_agreed = controller.finalize_storage_attach(
                publication_error is None
            )
        except StorageLifecycleConsensusError:
            cache.enable_storage = False
            cache.enable_storage_metrics = False
            raise
        except Exception as error:
            cache.enable_storage = False
            cache.enable_storage_metrics = False
            logger.exception("Failed to coordinate storage tree publication.")
            return False, f"Failed to coordinate storage tree publication: {error}"

        if not publication_agreed:
            cache.enable_storage = False
            cache.enable_storage_metrics = False
            where = "this rank" if publication_error is not None else "a peer rank"
            return (
                False,
                f"Storage tree publication failed on {where}; backend memory is "
                "retained and worker restart is required.",
            )
        return True, "Attached HiCache storage backend successfully."

    def detach(
        self,
        local_ready: bool = True,
        coordinated_lifecycle: bool = True,
    ) -> tuple[bool, str]:
        """Disable the storage backend at runtime.

        The caller must ensure there are no running or queued requests. Ordering
        matters and is the reason this is not just ``controller.detach()``:

        1. drain the control queues while the bookkeeping is still intact, so
           in-flight acks / releases can still be matched to their nodes --
           otherwise host pages and host locks leak;
        2. stop the storage threads;
        3. only then release whatever prefetch / backup is still tracked, since
           nothing can race the bookkeeping once the threads are gone;
        4. drain once more, because step 3 only *queues* the host pages -- this
           is what hands them back to the pool and its sidecars.
        """
        cache = self._cache
        controller = cache.cache_controller
        if controller is None:
            return False, "HiCache storage backend is not initialized."

        if coordinated_lifecycle:
            try:
                lifecycle_ready = controller.prepare_storage_lifecycle(local_ready)
            except StorageLifecycleConsensusError:
                raise
            except Exception as e:
                logger.exception("Failed to coordinate storage detach readiness.")
                return False, f"Failed to coordinate storage detach readiness: {e}"
            if not lifecycle_ready:
                return (
                    False,
                    "Reject detach: scheduler is not idle on this or a peer "
                    "model-parallel rank.",
                )

        drain_error = None
        try:
            cache.drain_storage_control_queues_local()
        except Exception as error:
            logger.exception("Failed to drain storage queues before detach.")
            drain_error = error

        try:
            drain_agreed = (
                controller.prepare_storage_lifecycle(drain_error is None)
                if coordinated_lifecycle
                else drain_error is None
            )
        except StorageLifecycleConsensusError:
            raise
        except Exception as e:
            logger.exception("Failed to coordinate storage detach preparation.")
            return False, f"Failed to coordinate storage detach preparation: {e}"
        if not drain_agreed:
            where = "this rank" if drain_error is not None else "a peer rank"
            detail = f": {drain_error}" if drain_error is not None else ""
            return (
                False,
                f"Storage detach preparation failed on {where}{detail}; "
                "backend remains attached.",
            )

        try:
            # Idempotent: ask the controller to clean up even when `enable_storage`
            # is already False, since that may be leftover state from an earlier
            # partial detach.
            controller.detach_storage_backend(
                coordinated_lifecycle=coordinated_lifecycle
            )
        except StorageLifecycleConsensusError:
            raise
        except Exception as e:
            logger.exception("Failed to detach storage backend.")
            # A coordinated teardown refusal disables the controller data path
            # while retaining native resources. Publish that disabled state to
            # the tree too, so requests cannot enqueue onto stopped threads.
            if not controller.enable_storage:
                cache.enable_storage = False
                cache.enable_storage_metrics = False
            return False, f"Failed to detach HiCache storage backend: {e}"

        try:
            self._release_pending_storage_ops()
            cache.drain_storage_control_queues_local()
        except Exception:
            logger.exception("Failed post-detach cleanup of storage bookkeeping.")

        cache.enable_storage = False
        cache.enable_storage_metrics = False
        return True, "Detached HiCache storage backend successfully."

    def shutdown(self) -> None:
        """Best-effort auto-detach on process shutdown.

        Keeps startup and runtime behavior consistent: a backend attached either
        via CLI args or via the admin API is detached on exit.
        """
        try:
            controller = getattr(self._cache, "cache_controller", None)
            backend = getattr(controller, "storage_backend", None)
            lifecycle_failed = bool(
                getattr(controller, "storage_lifecycle_failed", False)
            )
            if self._cache.enable_storage or backend is not None or lifecycle_failed:
                # Peers may already be gone on atexit.  Only local quiescence is
                # required because this process will not serve another request.
                self.detach(coordinated_lifecycle=False)
        except StorageLifecycleConsensusError:
            raise
        except Exception:
            logger.exception("Failed to detach storage backend on process shutdown.")

    def release_host_resources(self) -> None:
        """Detach storage before releasing host memory registered into it.

        Unlike :meth:`shutdown`, this graceful-release boundary is strict. A
        backend that refuses to close may still own native operations against
        the linker's host pools, so those pools must remain mapped until process
        reclaim rather than be destroyed underneath it.
        """
        cache = self._cache
        controller = getattr(cache, "cache_controller", None)
        backend = (
            getattr(controller, "storage_backend", None)
            if controller is not None
            else None
        )
        lifecycle_failed = bool(
            getattr(controller, "storage_lifecycle_failed", False)
            if controller is not None
            else False
        )
        if cache.enable_storage or backend is not None or lifecycle_failed:
            # Scheduler ``finally`` paths cannot assume peer ranks are alive.
            # A local close is sufficient to prove this rank's host mappings
            # are no longer reachable by native operations.
            detached, message = self.detach(coordinated_lifecycle=False)
            if not detached:
                raise RuntimeError(
                    "Cannot release HiCache host pools before storage detaches: "
                    f"{message}"
                )

        if cache.linker is not None:
            cache.linker.close()
        if cache.host_pool_group is not None:
            cache.host_pool_group.destroy()

    def clear(self) -> bool:
        """Drop everything the backend has stored, keeping it attached."""
        try:
            ok = self._cache.cache_controller.clear_storage_backend()
        except Exception as e:
            logger.error("Failed to clear hierarchical cache storage backend: %s", e)
            return False
        if ok:
            logger.info("Hierarchical cache storage backend cleared successfully!")
        return ok

    # ---- Runtime config ----

    def apply_runtime_config(
        self,
        *,
        storage_backend: Optional[str],
        prefetch_threshold: int,
        prefetch_timeout_base: float,
        prefetch_timeout_per_ki_token: float,
        hicache_storage_pass_prefix_keys: bool,
        enable_storage: bool,
        enable_storage_metrics: bool,
        extra_metric_labels: Optional[dict[str, str]],
    ) -> None:
        """Publish the storage knobs onto the tree; the single storage-enable point.

        Both startup and runtime attach funnel through here, so anything that must
        happen exactly when storage turns on belongs in this method.
        """
        cache = self._cache

        # Nodes already in the tree were built with hashing off. Fill them in as
        # storage turns on: a node hashed against an unhashed parent restarts the
        # page hash chain mid-sequence, so its L3 keys would cover only a suffix of
        # the prefix they claim to represent.
        if enable_storage and not cache.enable_storage:
            filled = cache.tree_core.backfill_missing_hash_values()
            if filled:
                logger.info(
                    "Hashed %d radix nodes that predate the storage backend.", filled
                )

        cache.enable_storage = enable_storage
        cache.prefetch_threshold = prefetch_threshold
        cache.prefetch_timeout_base = prefetch_timeout_base
        cache.prefetch_timeout_per_page = (
            cache.page_size / 1024 * prefetch_timeout_per_ki_token
        )
        cache.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        cache.enable_storage_metrics = enable_storage_metrics

        if enable_storage_metrics:
            cache.storage_metrics_collector = self._resolve_metrics_collector(
                storage_backend, extra_metric_labels
            )
        else:
            cache.storage_metrics_collector = None

    def _resolve_metrics_collector(
        self,
        storage_backend: Optional[str],
        extra_metric_labels: Optional[dict[str, str]],
    ) -> Optional[StorageMetricsCollector]:
        """Build the storage metrics collector, or relabel the existing one.

        A collector is created once and kept across detach / re-attach: building a
        second one with the same labels would register duplicate metrics.
        """
        cache = self._cache
        controller = cache.cache_controller
        attn_cp_rank, attn_cp_size = controller.get_attn_cp_rank_and_size()
        labels = {
            "storage_backend": storage_backend,
            "tp_rank": controller.tp_rank,
            "dp_rank": controller.dp_rank,
            "pp_rank": controller.pp_rank,
            "pp_size": controller.pp_size,
            "attn_cp_rank": attn_cp_rank,
            "attn_cp_size": attn_cp_size,
        }
        if extra_metric_labels:
            labels.update(extra_metric_labels)

        existing_collector = cache.storage_metrics_collector
        if existing_collector is None:
            storage_cls = resolve_collector_class(
                STAT_LOGGER_ROLE_STORAGE,
                StorageMetricsCollector,
            )
            return storage_cls(labels=labels)

        if set(existing_collector.labels.keys()) == set(labels.keys()):
            existing_collector.labels = labels
        else:
            logger.warning(
                "Storage metrics labels changed (%s -> %s). Keep existing labels to avoid duplicate metric registration.",
                sorted(existing_collector.labels.keys()),
                sorted(labels.keys()),
            )
        return existing_collector

    # ---- Internals ----

    @staticmethod
    def _validate_policies(
        hicache_storage_prefetch_policy: Optional[str],
        hicache_write_policy: Optional[str],
    ) -> Optional[str]:
        """The rejection reason for an invalid policy, or None when both are ok."""
        if (
            hicache_storage_prefetch_policy is not None
            and hicache_storage_prefetch_policy not in _PREFETCH_POLICIES
        ):
            return (
                f"Invalid hicache_storage_prefetch_policy: "
                f"{hicache_storage_prefetch_policy!r}. "
                f"Expected one of {list(_PREFETCH_POLICIES)}."
            )
        if (
            hicache_write_policy is not None
            and hicache_write_policy not in _WRITE_POLICIES
        ):
            return (
                f"Invalid hicache_write_policy: {hicache_write_policy!r}. "
                f"Expected one of {list(_WRITE_POLICIES)}."
            )
        return None

    def _apply_policies(
        self,
        hicache_storage_prefetch_policy: Optional[str],
        hicache_write_policy: Optional[str],
    ) -> None:
        cache = self._cache
        if hicache_storage_prefetch_policy is not None:
            cache.prefetch_stop_policy = hicache_storage_prefetch_policy
            logger.info(
                f"Set hicache_storage_prefetch_policy to "
                f"{hicache_storage_prefetch_policy}"
            )
        if hicache_write_policy is not None:
            cache.cache_controller.write_policy = hicache_write_policy
            cache.write_through_threshold = (
                1 if hicache_write_policy == "write_through" else 2
            )
            cache.is_write_back = hicache_write_policy == "write_back"
            logger.info(f"Set hicache_write_policy to {hicache_write_policy}")

    def _release_pending_storage_ops(self) -> None:
        """Release the host pages and locks that still-tracked ops hold.

        A safety net only: the scheduler refuses a detach unless `ongoing_prefetch`
        and `ongoing_backup` are already empty, so this normally finds nothing. It
        earns its keep on the atexit path, which has no such guard.

        Two deliberate departures from `release_aborted_request`, which does the
        same bookkeeping while the storage threads are still running:

        * no cross-rank barrier -- at process exit the other ranks may already be
          gone, and a barrier there would hang;
        * only the completed prefix of a prefetch is freed. The prefetch IO thread
          owns ``host_indices[completed_tokens:]`` and frees it as it drains, so
          freeing the whole range here would double-free whatever it already
          returned. Leaking the tail of an interrupted prefetch is the safer
          failure of the two.
        """
        cache = self._cache
        controller = cache.cache_controller

        for req_id in list(cache.ongoing_prefetch):
            info = cache.ongoing_prefetch[req_id]
            try:
                if info.host_indices is None:
                    # Host pages were never allocated for this operation.
                    cache.revoke_pending_prefetch(req_id)
                    continue
                completed_tokens, _ = controller.terminate_prefetch(info.operation)
                del cache.ongoing_prefetch[req_id]
                cache.dec_host_lock_ref(info.anchor_node_id, info.anchor_lock_params)
                controller.append_host_mem_release(
                    host_indices=info.host_indices[:completed_tokens],
                    extra_pools=[
                        x for xfers in info.comp_xfers.values() for x in xfers
                    ],
                )
                controller.prefetch_tokens_occupied = max(
                    0, controller.prefetch_tokens_occupied - len(info.prefetch_key)
                )
            except Exception:
                logger.exception("Failed to release pending prefetch %s", req_id)
                cache.ongoing_prefetch.pop(req_id, None)

        for ack_id in list(cache.ongoing_backup):
            node_id, lock_params = cache.ongoing_backup.pop(ack_id)
            try:
                cache.dec_host_lock_ref(node_id, lock_params)
            except Exception:
                logger.exception("Failed to release host lock for backup op %s", ack_id)

        cache.prefetch_loaded_tokens_by_reqid.clear()
