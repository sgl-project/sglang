//! The kwargs SGLang's `_process_messages` hands a template without an SGLang
//! request path (Jinja, Dynamo's encoders, legacy conversations).

use std::collections::HashMap;

use dynamo_protocols::types::CreateChatCompletionRequest;
use dynamo_renderer::RenderedPrompt;
use serde_json::Value;

use super::{ChatFormatter, TemplateArgsRequest, TemplateError, requested_thinking};

/// What SGLang's chat layer reads off a raw chat body (not re-validated) for its template.
#[derive(Debug, Clone, Default)]
pub struct TemplateArgs {
    kwargs: HashMap<String, Value>,
}

impl TemplateArgs {
    /// The request's own `chat_template_kwargs`, then the thinking toggle its
    /// `reasoning` or `reasoning_effort` implies (`normalize_reasoning_inputs`),
    /// then the server `defaults`.
    pub fn new(body: &Value, defaults: &HashMap<String, Value>) -> Self {
        let mut kwargs: HashMap<String, Value> = body["chat_template_kwargs"]
            .as_object()
            .map(|own| own.clone().into_iter().collect())
            .unwrap_or_default();
        if let Some(thinking) = requested_thinking(body) {
            for key in ["thinking", "enable_thinking"] {
                kwargs
                    .entry(key.to_owned())
                    .or_insert(Value::Bool(thinking));
            }
        }
        for (key, value) in defaults {
            kwargs.entry(key.clone()).or_insert_with(|| value.clone());
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// `reasoning_effort` sets both thinking toggles beneath the request's own
    /// kwargs, and the server defaults fill in only what is left.
    #[test]
    fn effort_sets_thinking_beneath_the_requests_kwargs() {
        let defaults = HashMap::from([
            ("enable_thinking".to_owned(), json!(true)),
            ("other".to_owned(), json!(1)),
        ]);
        let body = json!({
            "reasoning_effort": "none",
            "chat_template_kwargs": {"thinking": true},
        });
        let args = TemplateArgs::new(&body, &defaults);
        assert_eq!(
            args.kwargs,
            HashMap::from([
                ("thinking".to_owned(), json!(true)),
                ("enable_thinking".to_owned(), json!(false)),
                ("other".to_owned(), json!(1)),
            ])
        );
    }
}
