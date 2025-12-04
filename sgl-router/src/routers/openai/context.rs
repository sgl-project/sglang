//! Request context types for OpenAI router pipeline.

use std::sync::Arc;

use axum::http::HeaderMap;
use serde_json::Value;

use super::provider::Provider;
use crate::{
    core::Worker,
    data_connector::{ConversationItemStorage, ConversationStorage, ResponseStorage},
    mcp::McpManager,
    protocols::{chat::ChatCompletionRequest, responses::ResponsesRequest},
};

pub struct RequestContext {
    pub input: RequestInput,
    pub components: ComponentRefs,
    pub state: ProcessingState,
}

pub struct RequestInput {
    pub request_type: RequestType,
    pub headers: Option<HeaderMap>,
    #[allow(dead_code)]
    pub model_id: Option<String>,
}

pub enum RequestType {
    Chat(Arc<ChatCompletionRequest>),
    Responses(Arc<ResponsesRequest>),
}

#[derive(Clone)]
pub struct SharedComponents {
    pub client: reqwest::Client,
}

pub struct ResponsesComponents {
    pub shared: SharedComponents,
    pub mcp_manager: Arc<McpManager>,
    pub response_storage: Arc<dyn ResponseStorage>,
    pub conversation_storage: Arc<dyn ConversationStorage>,
    pub conversation_item_storage: Arc<dyn ConversationItemStorage>,
}

pub enum ComponentRefs {
    Shared(Arc<SharedComponents>),
    Responses(Arc<ResponsesComponents>),
}

impl ComponentRefs {
    pub fn client(&self) -> &reqwest::Client {
        match self {
            ComponentRefs::Shared(s) => &s.client,
            ComponentRefs::Responses(r) => &r.shared.client,
        }
    }

    pub fn mcp_manager(&self) -> Option<&Arc<McpManager>> {
        match self {
            ComponentRefs::Shared(_) => None,
            ComponentRefs::Responses(r) => Some(&r.mcp_manager),
        }
    }

    pub fn response_storage(&self) -> Option<&Arc<dyn ResponseStorage>> {
        match self {
            ComponentRefs::Shared(_) => None,
            ComponentRefs::Responses(r) => Some(&r.response_storage),
        }
    }

    pub fn conversation_storage(&self) -> Option<&Arc<dyn ConversationStorage>> {
        match self {
            ComponentRefs::Shared(_) => None,
            ComponentRefs::Responses(r) => Some(&r.conversation_storage),
        }
    }

    pub fn conversation_item_storage(&self) -> Option<&Arc<dyn ConversationItemStorage>> {
        match self {
            ComponentRefs::Shared(_) => None,
            ComponentRefs::Responses(r) => Some(&r.conversation_item_storage),
        }
    }
}

#[derive(Default)]
pub struct ProcessingState {
    pub worker: Option<WorkerSelection>,
    pub payload: Option<PayloadState>,
}

pub struct WorkerSelection {
    pub worker: Arc<dyn Worker>,
    #[allow(dead_code)]
    pub provider: Arc<dyn Provider>,
}

pub struct PayloadState {
    pub json: Value,
    pub url: String,
    pub previous_response_id: Option<String>,
}

impl RequestContext {
    pub fn for_responses(
        request: Arc<ResponsesRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: ComponentRefs,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Responses(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    pub fn for_chat(
        request: Arc<ChatCompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: ComponentRefs,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Chat(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
        }
    }
}

impl RequestContext {
    pub fn responses_request(&self) -> &ResponsesRequest {
        match &self.input.request_type {
            RequestType::Responses(req) => req.as_ref(),
            _ => panic!("Expected responses request"),
        }
    }

    #[allow(dead_code)]
    pub fn responses_request_arc(&self) -> Arc<ResponsesRequest> {
        match &self.input.request_type {
            RequestType::Responses(req) => Arc::clone(req),
            _ => panic!("Expected responses request"),
        }
    }

    pub fn is_streaming(&self) -> bool {
        match &self.input.request_type {
            RequestType::Chat(req) => req.stream,
            RequestType::Responses(req) => req.stream.unwrap_or(false),
        }
    }

    pub fn headers(&self) -> Option<&HeaderMap> {
        self.input.headers.as_ref()
    }

    #[allow(dead_code)]
    pub fn model_id(&self) -> Option<&str> {
        self.input.model_id.as_deref()
    }

    pub fn worker(&self) -> Option<&Arc<dyn Worker>> {
        self.state.worker.as_ref().map(|w| &w.worker)
    }

    #[allow(dead_code)]
    pub fn provider(&self) -> Option<&dyn Provider> {
        self.state.worker.as_ref().map(|w| w.provider.as_ref())
    }

    pub fn payload(&self) -> Option<&PayloadState> {
        self.state.payload.as_ref()
    }

    pub fn take_payload(&mut self) -> Option<PayloadState> {
        self.state.payload.take()
    }
}

pub struct StorageHandles {
    pub response: Arc<dyn ResponseStorage>,
    pub conversation: Arc<dyn ConversationStorage>,
    pub conversation_item: Arc<dyn ConversationItemStorage>,
}

pub struct OwnedStreamingContext {
    pub url: String,
    pub payload: Value,
    pub original_body: ResponsesRequest,
    pub previous_response_id: Option<String>,
    pub storage: StorageHandles,
}

impl RequestContext {
    pub fn into_streaming_context(mut self) -> OwnedStreamingContext {
        let payload_state = self.take_payload().expect("Payload not prepared");

        OwnedStreamingContext {
            url: payload_state.url,
            payload: payload_state.json,
            original_body: self.responses_request().clone(),
            previous_response_id: payload_state.previous_response_id,
            storage: StorageHandles {
                response: self
                    .components
                    .response_storage()
                    .expect("Response storage required")
                    .clone(),
                conversation: self
                    .components
                    .conversation_storage()
                    .expect("Conversation storage required")
                    .clone(),
                conversation_item: self
                    .components
                    .conversation_item_storage()
                    .expect("Conversation item storage required")
                    .clone(),
            },
        }
    }
}

pub struct StreamingEventContext<'a> {
    pub server_label: &'a str,
    pub original_request: &'a ResponsesRequest,
    pub previous_response_id: Option<&'a str>,
}

pub type StreamingRequest = OwnedStreamingContext;
