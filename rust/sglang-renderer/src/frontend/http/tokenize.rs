//! HTTP tokenization and detokenization adapter.

use super::error::{json_rejection_response, response_error};
use crate::{
    RendererService,
    openai::tokenize::{
        DetokenizeRequest, TokenizeRequest, detokenize as detokenize_request,
        tokenize as tokenize_request,
    },
};
use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    response::Response,
    routing::post,
};
use serde_json::Value;
use std::sync::Arc;

pub(super) fn routes(renderer: Arc<RendererService>) -> Router<()> {
    Router::new()
        .route("/tokenize", post(tokenize))
        .route("/v1/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/v1/detokenize", post(detokenize))
        .with_state(renderer)
}

async fn tokenize(
    State(renderer): State<Arc<RendererService>>,
    body: Result<Json<TokenizeRequest>, JsonRejection>,
) -> Result<Json<Value>, Response> {
    let Json(request) = body.map_err(json_rejection_response)?;
    tokenize_request(&renderer, request)
        .await
        .map(Json)
        .map_err(response_error)
}

async fn detokenize(
    State(renderer): State<Arc<RendererService>>,
    body: Result<Json<DetokenizeRequest>, JsonRejection>,
) -> Result<Json<Value>, Response> {
    let Json(request) = body.map_err(json_rejection_response)?;
    detokenize_request(&renderer, request)
        .await
        .map(Json)
        .map_err(response_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode},
    };
    use serde_json::json;
    use tower::ServiceExt;

    use crate::{RendererConfig, RendererLimits, SamplingDefaults, TextTokenizer};
    use sglang_processor::ProcessorError;

    struct PrefixTokenizer;

    impl TextTokenizer for PrefixTokenizer {
        fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>, ProcessorError> {
            Ok(add_special_tokens
                .then_some(1)
                .into_iter()
                .chain(text.split_whitespace().map(|_| 7))
                .chain(add_special_tokens.then_some(2))
                .collect())
        }

        fn decode(
            &self,
            token_ids: &[u32],
            skip_special_tokens: bool,
        ) -> Result<String, RendererError> {
            Ok(token_ids
                .iter()
                .filter(|&&id| !(skip_special_tokens && matches!(id, 1 | 2)))
                .map(|id| format!("t{id}"))
                .collect::<Vec<_>>()
                .join(" "))
        }
    }

    fn app() -> Router<()> {
        let config = RendererConfig {
            served_model_name: "model".into(),
            tokenizer_path: ".".into(),
            revision: None,
            model_path: String::new(),
            chat_template: Some("chatml".into()),
            tool_call_parser: None,
            reasoning_parser: None,
            default_chat_template_kwargs: Default::default(),
            stream_response_default_include_usage: false,
            default_sampling_params: SamplingDefaults::default(),
            limits: RendererLimits {
                vocab_size: 100,
                context_len: 64,
                num_reserved_tokens: 0,
                allow_auto_truncate: false,
                enable_return_hidden_states: false,
            },
        };
        routes(Arc::new(RendererService::with_tokenizer(
            config,
            Arc::new(PrefixTokenizer),
            2,
            2,
        )))
    }

    async fn post(body: Value) -> (StatusCode, Value) {
        post_to("/v1/tokenize", body).await
    }

    async fn post_to(uri: &str, body: Value) -> (StatusCode, Value) {
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let body =
            serde_json::from_slice(&to_bytes(response.into_body(), 64 * 1024).await.unwrap())
                .unwrap();
        (status, body)
    }

    #[tokio::test]
    async fn prompt_tokenization_preserves_batch_shape_and_special_token_choice() {
        let (status, body) = post(json!({
            "prompt": ["one two", ""],
            "add_special_tokens": false
        }))
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["tokens"], json!([[7, 7], []]));
        assert_eq!(body["count"], json!([2, 0]));

        let (_, body) = post(json!({"prompt": "one"})).await;
        assert_eq!(body["tokens"], json!([1, 7, 2]));
    }

    #[tokio::test]
    async fn detokenization_preserves_batch_shape_and_special_token_choice() {
        let (status, body) = post_to("/v1/detokenize", json!({"tokens": [1, 7, 2]})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, json!({"text": "t7"}));

        let (status, body) = post_to(
            "/detokenize",
            json!({"tokens": [[1, 7], [], [9]], "skip_special_tokens": false}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["text"], json!(["t1 t7", "", "t9"]));

        let (status, body) = post_to("/v1/detokenize", json!({"tokens": []})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["text"], json!(""));
    }

    #[tokio::test]
    async fn detokenization_rejects_token_ids_outside_the_vocabulary() {
        for tokens in [json!([7, 100]), json!([[7], [-1]])] {
            let (status, body) = post_to("/v1/detokenize", json!({"tokens": tokens})).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{tokens}");
            assert!(
                body["error"]["message"]
                    .as_str()
                    .is_some_and(|message| message.contains("out-of-vocabulary")),
                "{body}"
            );
        }

        let (status, _) = post_to("/v1/detokenize", json!({"tokens": ["7"]})).await;
        assert!(status.is_client_error());
    }

    #[tokio::test]
    async fn chat_tokenization_applies_the_template_without_generation_limits() {
        let (status, body) = post(json!({
            "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": 10_000
        }))
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            body["tokens"]
                .as_array()
                .is_some_and(|tokens| !tokens.is_empty())
        );
        assert_ne!(body["tokens"][0], json!(1));
        assert_ne!(
            body["tokens"][body["tokens"].as_array().unwrap().len() - 1],
            json!(2)
        );
        assert_eq!(
            body["count"],
            json!(body["tokens"].as_array().unwrap().len())
        );
    }

    #[tokio::test]
    async fn chat_tokenization_continues_the_final_assistant_message() {
        let (_, regular) = post(json!({
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "partial answer"}
            ]
        }))
        .await;
        let (status, continued) = post(json!({
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "partial answer"}
            ],
            "continue_final_message": true,
            "chat_template_kwargs": {
                "continue_final_message": false,
                "add_generation_prompt": true
            }
        }))
        .await;

        assert_eq!(status, StatusCode::OK);
        assert!(continued["count"].as_u64().unwrap() < regular["count"].as_u64().unwrap());
    }
}
