"""Scheduler-owned reference to a safely published continuous cache prefix."""

from dataclasses import dataclass


@dataclass
class CachePrefixLease:
    cache: object
    req: object
    cap: int
    device_lock: object
    host_node: object
    host_params: object
    released: bool = False

    @classmethod
    def acquire(cls, cache, req, limit):
        # Re-match AFTER policy completion has published the L3 result into the
        # tree. The native matcher joins continuous L1/L2 coverage, handles
        # overlap and namespaces, and finalizes required model components.
        req.init_next_round_input(cache, cow_mamba=False, max_cache_prefix_tokens=limit)
        cap = req.prefix_len + req.host_hit_length
        if cap < 0 or cap > limit or cap % cache.page_size:
            raise RuntimeError("invalid page-aligned cache full-prefix match")
        # FULL/KV-derived DSA needs a host lease only for a host-hit suffix.
        # On a device-only node native non-write-back host acquire is a no-op.
        # Releasing that no-op after selective backup creates a host copy would
        # steal the subsequent PUT's reference. Do not acquire/release it at all.
        host_node = req.best_match_node if req.host_hit_length > 0 else None
        device_lock = cache.lock(req.last_node)
        host_params = None
        try:
            if host_node is not None:
                host_params = cache.inc_host_lock_ref(host_node).to_dec_params()
        except Exception:
            cache.unlock(device_lock)
            raise
        return cls(cache, req, cap, device_lock, host_node, host_params)

    def restrict(self, cap):
        if cap == self.cap:
            return self
        # The lookup can split tree nodes. Acquire the shorter path before
        # dropping the old references; never expose an unprotected interval.
        shorter = type(self).acquire(self.cache, self.req, cap)
        self.release()
        return shorter

    def release(self):
        if self.released:
            return
        if self.host_node is not None:
            self.cache.dec_host_lock_ref(self.host_node, self.host_params)
        self.cache.unlock(self.device_lock)
        self.released = True
