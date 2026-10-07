// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared mutable state that selection reads: the KV-event cache index, load
//! monitoring, and affinity assignments. Both the legacy policies and
//! `policies_reorg` read it here.

pub mod affinity_store;
pub mod kv_events;
pub mod load_monitor;

pub use affinity_store::AffinityStore;
