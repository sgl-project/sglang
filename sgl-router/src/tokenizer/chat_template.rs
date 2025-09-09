//! Chat template support for tokenizers using Jinja2 templates
//!
//! This module provides functionality to apply chat templates to messages,
//! similar to HuggingFace transformers' apply_chat_template method.

use anyhow::{anyhow, Result};
use minijinja::{context, Environment, Value};
use serde::{Deserialize, Serialize};
use serde_json;

/// Represents a chat message with role and content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        ChatMessage {
            role: role.into(),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

/// Chat template processor using Jinja2
pub struct ChatTemplateProcessor {
    template: String,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl ChatTemplateProcessor {
    /// Create a new chat template processor
    pub fn new(template: String, bos_token: Option<String>, eos_token: Option<String>) -> Self {
        ChatTemplateProcessor {
            template,
            bos_token,
            eos_token,
        }
    }

    /// Apply the chat template to a list of messages
    ///
    /// This mimics the behavior of HuggingFace's apply_chat_template method
    /// but returns the formatted string instead of token IDs.
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let mut env = Environment::new();

        // Register the template
        env.add_template("chat", &self.template)
            .map_err(|e| anyhow!("Failed to add template: {}", e))?;

        // Get the template
        let tmpl = env
            .get_template("chat")
            .map_err(|e| anyhow!("Failed to get template: {}", e))?;

        // Convert messages to a format Jinja can work with
        let messages_value: Vec<Value> = messages
            .iter()
            .map(|msg| {
                context! {
                    role => msg.role.clone(),
                    content => msg.content.clone()
                }
            })
            .collect();

        // Render the template
        let rendered = tmpl
            .render(context! {
                messages => messages_value,
                add_generation_prompt => add_generation_prompt,
                bos_token => self.bos_token.clone().unwrap_or_default(),
                eos_token => self.eos_token.clone().unwrap_or_default()
            })
            .map_err(|e| anyhow!("Failed to render template: {}", e))?;

        Ok(rendered)
    }
}

/// Load chat template from tokenizer config JSON
pub fn load_chat_template_from_config(config_path: &str) -> Result<Option<String>> {
    use std::fs;

    let content = fs::read_to_string(config_path)?;
    let config: serde_json::Value = serde_json::from_str(&content)?;

    // Look for chat_template in the config
    if let Some(template) = config.get("chat_template") {
        if let Some(template_str) = template.as_str() {
            return Ok(Some(template_str.to_string()));
        }
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_creation() {
        let msg = ChatMessage::system("You are a helpful assistant");
        assert_eq!(msg.role, "system");
        assert_eq!(msg.content, "You are a helpful assistant");

        let user_msg = ChatMessage::user("Hello!");
        assert_eq!(user_msg.role, "user");

        let assistant_msg = ChatMessage::assistant("Hi there!");
        assert_eq!(assistant_msg.role, "assistant");
    }

    #[test]
    fn test_simple_chat_template() {
        // Simple template that formats messages
        let template = r#"
{%- for message in messages -%}
{{ message.role }}: {{ message.content }}
{% endfor -%}
{%- if add_generation_prompt -%}
assistant:
{%- endif -%}
"#;

        let processor = ChatTemplateProcessor::new(template.to_string(), None, None);

        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
        ];

        let result = processor.apply_chat_template(&messages, true).unwrap();
        assert!(result.contains("system: You are helpful"));
        assert!(result.contains("user: Hello"));
        assert!(result.contains("assistant:"));
    }

    #[test]
    fn test_chat_template_with_tokens() {
        // Template that uses special tokens
        let template = r#"
{{ bos_token }}
{%- for message in messages -%}
{{ message.role }}: {{ message.content }}{{ eos_token }}
{% endfor -%}
"#;

        let processor = ChatTemplateProcessor::new(
            template.to_string(),
            Some("<s>".to_string()),
            Some("</s>".to_string()),
        );

        let messages = vec![ChatMessage::user("Test")];

        let result = processor.apply_chat_template(&messages, false).unwrap();
        assert!(result.contains("<s>"));
        assert!(result.contains("</s>"));
    }
}
