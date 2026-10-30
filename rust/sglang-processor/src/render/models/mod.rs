//! Model-specific SGLang behaviour around Dynamo's formatters.

mod deepseek_v4;
mod glm53;
mod kimi_k25;
mod kimi_k3;
mod pydantic_dump;
mod qwen38;
mod schema_types;

pub use self::deepseek_v4::DeepSeekTokenizer;
pub(super) use self::deepseek_v4::render as render_deepseek_v4;
pub(crate) use self::deepseek_v4::thinking as deepseek_v4_thinking;
#[cfg(feature = "openai")]
pub(crate) use self::deepseek_v4::{
    continued_message as deepseek_v4_continued_message, parse_start as deepseek_v4_parse_start,
    special_tokens as deepseek_v4_special_tokens, task as deepseek_v4_task,
};
pub(super) use self::glm53::{is_glm53, render as render_glm53};
pub(super) use self::kimi_k3::render as render_kimi_k3;
pub(super) use self::kimi_k25::{deep_sort, encode_tools_to_typescript};
pub use self::pydantic_dump::continues_final_message;
pub(crate) use self::pydantic_dump::pydantic_bool;
pub(super) use self::pydantic_dump::{function_dump, split_continuation};
pub(super) use self::qwen38::{is_qwen38, render as render_qwen38, takes_content_parts};
