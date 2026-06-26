// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::app_context::AppContext;
use axum::extract::State;
use axum::Json;
use serde::Serialize;
use std::sync::Arc;

#[derive(Serialize)]
pub struct ModelsList {
    pub object: &'static str,
    pub data: Vec<ModelEntry>,
}

#[derive(Serialize)]
pub struct ModelEntry {
    pub id: String,
    pub object: &'static str,
    pub owned_by: &'static str,
}

pub async fn list_models(State(ctx): State<Arc<AppContext>>) -> Json<ModelsList> {
    // The router serves a single configured model; OpenAI clients still
    // expect a list shape, so return a one-element `data` array.
    let m = &ctx.config.model;
    let data = vec![ModelEntry {
        id: m.id.clone(),
        object: "model",
        owned_by: "sglang",
    }];
    Json(ModelsList {
        object: "list",
        data,
    })
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use crate::config::PolicyKind;

    #[tokio::test]
    async fn lists_configured_model() {
        let mut ctx = crate::server::app_context::AppContext::stub();
        ctx.config.model = crate::config::ModelConfig {
            id: "qwen3".into(),
            tokenizer_path: "x".into(),
            policy: PolicyKind::RoundRobin,
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
        };
        let app = crate::server::app::build_router(std::sync::Arc::new(ctx));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "list");
        let ids: Vec<&str> = v["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["qwen3"]);
        assert_eq!(v["data"][0]["object"], "model");
        // Pin `owned_by` so a refactor that flips the hardcoded value to
        // "openai" / "" / a typo would fail loudly here. OpenAI clients
        // expect this field and some (e.g. langchain-openai) treat
        // `owned_by != "system"` as a meaningful signal.
        assert_eq!(v["data"][0]["owned_by"], "sglang");
    }
}
