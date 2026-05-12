// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! OpenAI-compatible HTTP API request / response types.
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// ──────────────────────────────── shared ────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum StopCondition {
    Single(String),
    Multiple(Vec<String>),
}

impl StopCondition {
    pub fn as_vec(&self) -> Vec<String> {
        match self {
            StopCondition::Single(s) => vec![s.clone()],
            StopCondition::Multiple(v) => v.clone(),
        }
    }
}

#[derive(Debug, Serialize, Clone, Default)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ─────────────────────────── /v1/completions ────────────────────────────────

#[derive(Debug, Deserialize, Clone)]
pub struct CompletionRequest {
    pub model: String,
    #[serde(default)]
    pub prompt: PromptInput,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub min_p: Option<f64>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<StopCondition>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub repetition_penalty: Option<f64>,
    #[serde(default)]
    pub logprobs: Option<u32>,
    #[serde(default)]
    pub echo: Option<bool>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub ignore_eos: Option<bool>,
    #[serde(default)]
    pub skip_special_tokens: Option<bool>,
    // Structured output
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
}

#[derive(Debug, Deserialize, Clone, Default)]
#[serde(untagged)]
pub enum PromptInput {
    #[default]
    Empty,
    Text(String),
    Tokens(Vec<u32>),
    Batch(Vec<String>),
}

impl PromptInput {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            PromptInput::Text(s) => Some(s.as_str()),
            _ => None,
        }
    }
    pub fn as_token_ids(&self) -> Option<&[u32]> {
        match self {
            PromptInput::Tokens(ids) => Some(ids.as_slice()),
            _ => None,
        }
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: Option<String>,
    pub logprobs: Option<Value>,
}

#[derive(Debug, Serialize, Clone)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: UsageInfo,
}

// Streaming chunk
#[derive(Debug, Serialize, Clone)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
}

#[derive(Debug, Serialize, Clone)]
pub struct CompletionChunkChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: Option<String>,
    pub logprobs: Option<Value>,
}

// ──────────────────────── /v1/chat/completions ──────────────────────────────

#[derive(Debug, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<ChatContent>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Value>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum ChatContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl ChatContent {
    pub fn as_text(&self) -> String {
        match self {
            ChatContent::Text(s) => s.clone(),
            ChatContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: Value },
}

#[derive(Debug, Deserialize, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub min_p: Option<f64>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<StopCondition>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub repetition_penalty: Option<f64>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub ignore_eos: Option<bool>,
    #[serde(default)]
    pub skip_special_tokens: Option<bool>,
    #[serde(default)]
    pub tools: Option<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: UsageInfo,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatChoice {
    pub index: usize,
    pub message: AssistantMessage,
    pub finish_reason: Option<String>,
    pub logprobs: Option<Value>,
}

#[derive(Debug, Serialize, Clone)]
pub struct AssistantMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
}

// Streaming chunk
#[derive(Debug, Serialize, Clone)]
pub struct ChatChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageInfo>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: DeltaMessage,
    pub finish_reason: Option<String>,
    pub logprobs: Option<Value>,
}

#[derive(Debug, Serialize, Clone, Default)]
pub struct DeltaMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
}

// ───────────────────────────── /v1/models ───────────────────────────────────

#[derive(Debug, Serialize, Clone)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelCard>,
}

// ─────────────────────────── structured output ──────────────────────────────

#[derive(Debug, Deserialize, Clone)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub fmt_type: String,
    #[serde(default)]
    pub json_schema: Option<HashMap<String, Value>>,
}

impl ResponseFormat {
    pub fn json_schema_str(&self) -> Option<String> {
        if self.fmt_type == "json_schema" {
            self.json_schema.as_ref().and_then(|m| {
                m.get("schema").map(|v| v.to_string())
            })
        } else if self.fmt_type == "json_object" {
            Some("{}".to_string())
        } else {
            None
        }
    }
}

// ─────────────────────────── error response ─────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub code: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

impl ErrorResponse {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        ErrorResponse {
            error: ErrorDetail {
                message: msg.into(),
                r#type: "invalid_request_error".into(),
                code: None,
            },
        }
    }
    pub fn internal(msg: impl Into<String>) -> Self {
        ErrorResponse {
            error: ErrorDetail {
                message: msg.into(),
                r#type: "internal_server_error".into(),
                code: None,
            },
        }
    }
}
