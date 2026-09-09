//! SGLang-compatible prompt and chat tokenization.

use dynamo_protocols::types::{
    ChatCompletionRequestMessage, ChatCompletionTool, ChatCompletionToolChoiceOption,
};
use futures::future::try_join_all;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{ChatRequest, OneOrMany, ReasoningEffort, RendererService, ResponseError};

use super::{protocol::normalize_reasoning_inputs, renderer_error};

pub(crate) async fn tokenize(
    renderer: &RendererService,
    mut request: TokenizeRequest,
) -> Result<Value, ResponseError> {
    let has_prompt = request.prompt.is_some();
    let has_messages = request.messages.is_some();
    if has_prompt == has_messages {
        return Err(ResponseError {
            status_code: 400,
            message: "Exactly one of 'prompt' or 'messages' must be provided.".into(),
        });
    }
    let (tokens, count) = match request.prompt.take() {
        Some(prompt) => {
            let add_special_tokens = request.add_special_tokens;
            match prompt {
                OneOrMany::One(text) => {
                    let tokens = renderer
                        .tokenize_prompt(text, add_special_tokens)
                        .await
                        .map_err(renderer_error)?;
                    (json!(tokens), json!(tokens.len()))
                }
                OneOrMany::Many(texts) => {
                    let tokens = try_join_all(
                        texts
                            .into_iter()
                            .map(|text| renderer.tokenize_prompt(text, add_special_tokens)),
                    )
                    .await
                    .map_err(renderer_error)?;
                    let count = tokens.iter().map(Vec::len).collect::<Vec<_>>();
                    (json!(tokens), json!(count))
                }
            }
        }
        None => {
            let request = request
                .into_chat(&renderer.config().served_model_name)
                .map_err(renderer_error)?;
            let tokens = renderer
                .tokenize_chat(request)
                .await
                .map_err(renderer_error)?;
            (json!(tokens), json!(tokens.len()))
        }
    };
    Ok(json!({
        "tokens": tokens,
        "count": count,
        "max_model_len": renderer.config().limits.context_len,
    }))
}

#[derive(Deserialize)]
pub(crate) struct TokenizeRequest {
    #[serde(default)]
    prompt: Option<OneOrMany<String>>,
    #[serde(default)]
    messages: Option<Vec<ChatCompletionRequestMessage>>,
    #[serde(default = "default_true")]
    add_special_tokens: bool,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    tools: Option<Vec<ChatCompletionTool>>,
    #[serde(default)]
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    #[serde(default)]
    reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    reasoning: Option<Value>,
    #[serde(default)]
    continue_final_message: bool,
    #[serde(default)]
    chat_template_kwargs: Option<std::collections::HashMap<String, Value>>,
}

impl TokenizeRequest {
    fn into_chat(mut self, served_model: &str) -> Result<ChatRequest, crate::RendererError> {
        normalize_reasoning_inputs(
            &mut self.reasoning_effort,
            self.reasoning.take(),
            &mut self.chat_template_kwargs,
        )?;
        let model = self.model.unwrap_or_else(|| served_model.to_owned());
        if model != served_model {
            return Err(format!("The model `{model}` does not exist").into());
        }
        Ok(ChatRequest {
            rid: "tokenize".into(),
            model,
            messages: self
                .messages
                .take()
                .expect("chat tokenization request has messages"),
            tools: self.tools,
            tool_choice: self.tool_choice,
            response_format: None,
            reasoning_effort: self.reasoning_effort,
            continue_final_message: self.continue_final_message,
            chat_template_args: self.chat_template_kwargs,
            sampling_params: Default::default(),
            choice_count: 1,
            stream: false,
            return_logprob: false,
            top_logprobs_num: 0,
            parallel_tool_calls: true,
            metadata: crate::GenerateRequestMetadata::default(),
        })
    }
}

const fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chat_tokenization_lowers_tokenize_specific_options() {
        let request: TokenizeRequest = serde_json::from_value(json!({
            "messages": [{"role": "assistant", "content": "partial"}],
            "reasoning_effort": "high",
            "continue_final_message": true,
            "chat_template_kwargs": {"marker": true}
        }))
        .unwrap();

        let chat = request.into_chat("model").unwrap();

        assert!(chat.continue_final_message);
        assert_eq!(
            chat.chat_template_args
                .as_ref()
                .and_then(|args| args.get("marker")),
            Some(&json!(true))
        );
        assert_eq!(
            serde_json::to_value(chat.reasoning_effort).unwrap(),
            json!("high")
        );
    }
}
