# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Runtime attach / detach of the HiCache (L3) storage backend.

``UnifiedRadixCache`` owns the tree; this component owns the admin-API lifecycle
of the storage backend behind it: validating the requested policies, starting and
stopping the controller's storage threads, and cleaning up the bookkeeping that a
half-finished prefetch / backup leaves behind.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)

# Kept in sync with the server-args choices; validated here so an admin request
# with a bad policy is rejected before it can touch the controller.
_PREFETCH_POLICIES = ("best_effort", "wait_complete", "timeout")
_WRITE_POLICIES = ("write_back", "write_through", "write_through_selective")


class StorageAttachment:
    """Attach / detach the storage backend of one ``UnifiedRadixCache``."""

    def __init__(self, cache: UnifiedRadixCache):
        self._cache = cache

    def attach(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Enable the storage backend at runtime.

        Starts the storage threads inside the cache controller and turns on the
        prefetch / backup paths. The caller must ensure there are no running or
        queued requests, so this does not race the scheduler thread.
        """
        cache = self._cache

        # Validate first: no side effects on a bad request.
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

        # Apply policies before the controller attach, so the storage threads
        # observe the new values as soon as they start.
        self._apply_policies(hicache_storage_prefetch_policy, hicache_write_policy)

        logger.info(f"Attaching HiCache storage backend: {storage_backend}")
        try:
            (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = HybridCacheController.parse_storage_backend_extra_config(
                storage_backend_extra_config_json
            )
        except Exception as e:
            logger.exception(f"Failed to parse storage_backend_extra_config_json: {e}")
            return (
                False,
                f"Failed to parse storage_backend_extra_config_json "
                f"'{storage_backend_extra_config_json}': {e}",
            )

        try:
            controller.attach_storage_backend(
                storage_backend=storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=served_model_name,
                storage_backend_extra_config=extra_config,
                host_pools=controller.mem_pool_host.entries,
            )
        except Exception as e:
            logger.exception(
                f"Failed to attach storage backend '{storage_backend}': {e}"
            )
            return False, f"Failed to attach storage backend '{storage_backend}': {e}"

        cache._apply_storage_runtime_config(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=controller.enable_storage,
            enable_storage_metrics=cache._enable_metrics_flag,
            extra_metric_labels=cache.extra_metric_labels,
        )
        return True, "Attached HiCache storage backend successfully."

    def detach(self) -> tuple[bool, str]:
        """Disable the storage backend at runtime.

        The caller must ensure there are no running or queued requests. Ordering
        matters and is the reason this is not just ``controller.detach()``:

        1. drain the control queues while the bookkeeping is still intact, so
           in-flight acks / releases can still be matched to their nodes --
           otherwise host pages and host locks leak;
        2. stop the storage threads;
        3. only then force-release whatever prefetch / backup is still tracked,
           since nothing can race the bookkeeping once the threads are gone;
        4. drain once more, because the release in step 3 only *queues* the host
           pages -- this is what hands them back to the pool and its sidecars.
        """
        cache = self._cache
        controller = cache.cache_controller
        if controller is None:
            return False, "HiCache storage backend is not initialized."

        try:
            cache.drain_storage_control_queues_local()
            # Idempotent: ask the controller to clean up even when
            # `enable_storage` is already False, since that may be leftover
            # state from an earlier partial detach.
            controller.detach_storage_backend()
        except Exception as e:
            logger.exception("Failed to detach storage backend.")
            # Never crash the server for an admin operation. The controller
            # raises while its threads are still alive, so leave `ongoing_*`
            # untouched -- a retry must still be able to match their acks.
            return False, f"Failed to detach HiCache storage backend: {e}"

        try:
            cache.drain_storage_control_queues_local()
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
            if self._cache.enable_storage:
                self.detach()
        except Exception:
            logger.exception("Failed to detach storage backend on process shutdown.")

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
        """Release the host pages and host locks still held by tracked ops.

        Prefetches go through ``release_aborted_request``, the same path the
        scheduler uses for an aborted request, so the host-page free, the
        ``dec_host_lock_ref`` and the ``prefetch_tokens_occupied`` accounting all
        stay in one place. Backups only hold a host lock on their anchor node.
        """
        cache = self._cache

        for req_id in list(cache.ongoing_prefetch):
            try:
                cache.release_aborted_request(req_id)
            except Exception:
                logger.exception(
                    "Failed to release pending prefetch %s during detach", req_id
                )
                cache.ongoing_prefetch.pop(req_id, None)

        for op_id, entry in list(cache.ongoing_backup.items()):
            try:
                node_id, lock_params = entry
                cache.dec_host_lock_ref(node_id, lock_params)
            except Exception:
                logger.exception(
                    "Failed to release host lock for backup op %s during detach", op_id
                )
            cache.ongoing_backup.pop(op_id, None)
