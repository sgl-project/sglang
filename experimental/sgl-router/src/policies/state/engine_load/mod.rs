// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Engine load as seen by the router: engine-reported telemetry
//! ([`reports`]) and router-local in-flight tracking ([`inflight`]).

pub mod inflight;
pub mod reports;

pub use inflight::*;
pub use reports::*;
