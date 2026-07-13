# PD Flip Prefill Manager Reuse Plan

1. Add a failing scheduler unit test that provides an existing prefill manager
   and makes any new manager construction fail.
2. Update `_pd_flip_get_source_kv_manager` to prefer and cache the existing
   prefill manager, preserving the legacy construction fallback.
3. Run focused tests plus PD migration/controller regressions and review.
4. Sync all four nodes, restart node2/node3 for clean registrations, and rerun
   the complete 40-request trace.
5. Require 40/40, controller success, final prefill role, and all workers healthy
   beyond the prior 300-second bootstrap timeout/invariant-check boundary.
