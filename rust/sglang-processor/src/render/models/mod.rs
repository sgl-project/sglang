//! Model-specific SGLang behaviour around Dynamo's formatters.

mod deepseek_v4;
mod deepseek_v41;
mod glm53;
mod kimi_k25;
mod kimi_k3;
mod qwen38;
mod schema_types;

pub use self::deepseek_v4::DeepSeekV4Profile;
pub(super) use self::deepseek_v4::{
    render as render_deepseek_v4, resolve_dsv4_profile, thinking as deepseek_v4_thinking,
};
pub(super) use self::deepseek_v41::render as render_deepseek_v41;
pub(super) use self::glm53::{is_glm53, render as render_glm53};
pub(super) use self::kimi_k3::render as render_kimi_k3;
pub(super) use self::kimi_k25::{deep_sort, encode_tools_to_typescript};
pub(super) use self::qwen38::{is_qwen38, render as render_qwen38, takes_content_parts};
