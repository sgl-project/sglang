// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared mutable state that selection reads: engine load reports,
//! router-local in-flight accounting, and the KV-event cache index.

pub mod active_load;
pub mod engine_load;
pub mod kv_events;
