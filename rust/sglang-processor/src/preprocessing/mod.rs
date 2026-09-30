//! Chat rendering and tokenization over Dynamo's renderer and tokenizers.

mod template;
mod tokenizer;

pub use template::{
    ChatFormatter, ChatTemplateConfig, DeepSeekV4Profile, OneOrMany, TemplateError,
    ThinkingTemplates, load_chat_formatter, load_chat_support,
};
#[cfg(feature = "test-support")]
pub use template::{test_hugging_face_formatter, test_hugging_face_formatter_from_config};
pub use tokenizer::{DynamoTokenizer, TextTokenizer, load_tokenizer};
pub use tokenizer::{resolve_chat_template_file, resolve_model_file, resolve_tokenizer_file};
