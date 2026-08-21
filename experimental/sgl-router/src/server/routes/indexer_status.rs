// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use axum::extract::{Json, State};
use axum::http::StatusCode;
use sgl_kv_indexer::IndexerStatusReport;

use crate::server::app_context::AppContext;

pub async fn report(
    State(ctx): State<Arc<AppContext>>,
    Json(report): Json<IndexerStatusReport>,
) -> Result<StatusCode, (StatusCode, String)> {
    let Some(index) = ctx.prefix_index.as_ref() else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "external KV Indexer is not configured".into(),
        ));
    };
    index
        .status_registry()
        .record(report)
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_payload_round_trips_json() {
        let report = IndexerStatusReport {
            indexer_id: "i-1".into(),
            endpoint: "http://127.0.0.1:50051".into(),
            ready: true,
            normalized_load: 0.25,
            ready_workers: 2,
            total_workers: 2,
        };
        let json = serde_json::to_string(&report).unwrap();
        assert_eq!(
            serde_json::from_str::<IndexerStatusReport>(&json).unwrap(),
            report
        );
    }
}
