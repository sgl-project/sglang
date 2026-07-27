// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::load_monitor::LoadMonitorSnapshot;
use crate::server::app_context::AppContext;
use axum::extract::State;
use axum::Json;
use std::sync::Arc;

/// Returns one immutable diagnostic snapshot of the Router load monitor.
pub async fn snapshot(State(ctx): State<Arc<AppContext>>) -> Json<LoadMonitorSnapshot> {
    Json(ctx.load_monitor.snapshot())
}
