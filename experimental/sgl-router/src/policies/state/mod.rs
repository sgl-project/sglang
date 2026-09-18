// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared, mutable state that selection policies read: the KV-event cache
//! index, engine load accounting, and affinity assignments.

pub mod affinity_store;
pub mod engine_load;
pub mod kv_events;

pub use affinity_store::AffinityStore;
