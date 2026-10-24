//! The kwargs SGLang's `_process_messages` hands a template without an SGLang
//! request path (Jinja, Dynamo's encoders, legacy conversations).

use std::collections::HashMap;

use dynamo_protocols::types::{ChatCompletionRequestMessage, CreateChatCompletionRequest};
use dynamo_renderer::{OAIChatLikeRequest, RenderedPrompt, TextInput};
use serde_json::Value;

use super::{ChatFormatter, TemplateError};

/// What SGLang's chat layer reads off a raw chat body (not re-validated) for its template.
#[derive(Debug, Clone, Default)]
pub struct TemplateArgs {
    kwargs: HashMap<String, Value>,
}

impl TemplateArgs {
    /// The server `defaults` under the request's own `chat_template_kwargs`.
    pub fn new(body: &Value, defaults: &HashMap<String, Value>) -> Self {
        let mut kwargs = defaults.clone();
        if let Some(own) = body["chat_template_kwargs"].as_object() {
            kwargs.extend(own.clone());
        }
        Self { kwargs }
    }
}

impl ChatFormatter {
    /// Render `request` through the formatter's own template (Jinja, Dynamo's
    /// encoder, or the legacy conversation) as SGLang's chat layer does.
    pub fn render_template(
        &self,
        request: &CreateChatCompletionRequest,
        args: &TemplateArgs,
    ) -> Result<RenderedPrompt, TemplateError> {
        let request = TemplateRequest {
            request,
            kwargs: (!args.kwargs.is_empty()).then_some(&args.kwargs),
        };
        self.render_prompt(&request)
    }
}

/// The wire request plus its `chat_template_kwargs`, which the protocol type
/// does not carry.
struct TemplateRequest<'a> {
    request: &'a CreateChatCompletionRequest,
    kwargs: Option<&'a HashMap<String, Value>>,
}

impl OAIChatLikeRequest for TemplateRequest<'_> {
    fn model(&self) -> String {
        self.request.model()
    }
    fn messages(&self) -> minijinja::Value {
        self.request.messages()
    }
    fn typed_messages(&self) -> Option<&[ChatCompletionRequestMessage]> {
        self.request.typed_messages()
    }
    fn tools(&self) -> Option<minijinja::Value> {
        self.request.tools()
    }
    fn tool_choice(&self) -> Option<minijinja::Value> {
        self.request.tool_choice()
    }
    fn response_format(&self) -> Option<minijinja::Value> {
        self.request.response_format()
    }
    fn reasoning_effort(&self) -> Option<minijinja::Value> {
        self.request.reasoning_effort()
    }
    fn should_add_generation_prompt(&self) -> bool {
        self.request.should_add_generation_prompt()
    }
    fn chat_template_args(&self) -> Option<&HashMap<String, Value>> {
        self.kwargs
    }
    fn extract_text(&self) -> Option<TextInput> {
        self.request.extract_text()
    }
    fn mm_processor_kwargs(&self) -> Option<&Value> {
        self.request.mm_processor_kwargs()
    }
}
