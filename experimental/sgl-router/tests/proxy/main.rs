// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Full HTTP proxy integration tests.
//!
//! Each submodule spins up the router via `build_router(AppContext)` and
//! drives real requests through a `common::mock_worker::MockWorker`
//! backend. For component-scope tests that don't need the router, see
//! `tests/component/`.

mod common;

mod cache_aware_input_ids;
mod chat_routing;
mod failover;
mod graceful_shutdown;
mod h2c_forward;
mod header_forwarding;
mod inbound_h2c;
mod pd_bootstrap_injection;
mod pd_pool_isolation;
mod roundrobin_input_ids;
mod sticky_input_ids;
mod sticky_routing;
mod timeout;
