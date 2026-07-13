# PD Flip Target Prefix Ownership Plan

1. Add a failing unit test around `_pd_flip_target_prealloc_and_send_metadata`
   showing that a matched prefix is not recorded as protected.
2. Add the one ownership assignment used by the normal decode prealloc path.
3. Run focused migration/HiCache/controller regressions and review.
4. Sync all workers, clean restart, and rerun the full trace.
5. Require 40/40, role flip success, target health, full req slots, and no pool
   invariant error after the previous delayed-crash boundary.
