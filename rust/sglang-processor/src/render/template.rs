//! The kwargs SGLang's `_process_messages` hands a template without an SGLang
//! request path (Jinja, Dynamo's encoders, legacy conversations).

use std::collections::HashMap;

use dynamo_protocols::types::CreateChatCompletionRequest;
use dynamo_renderer::RenderedPrompt;
use serde_json::Value;

use super::{ChatFormatter, TemplateArgsRequest, TemplateError};

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
        let args = args.kwargs.clone();
        self.render_prompt(&TemplateArgsRequest { request, args })
    }
}
