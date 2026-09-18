// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared mutable state that selection reads: engine load reports,
//! router-local in-flight accounting, the KV-event cache index, and
//! affinity assignments.

pub mod active_load;
pub mod affinity_store;
pub mod engine_load;
pub mod kv_events;
pub mod load_view;

pub use affinity_store::AffinityStore;
pub use load_view::LoadView;
