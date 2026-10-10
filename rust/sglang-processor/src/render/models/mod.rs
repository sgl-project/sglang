//! Model-specific SGLang behaviour around Dynamo's formatters.

mod deepseek_v4;
mod deepseek_v41;
mod kimi_k25;

pub use self::deepseek_v4::DeepSeekV4Profile;
pub(super) use self::deepseek_v4::{
    render as render_deepseek_v4, resolve_dsv4_profile, thinking as deepseek_v4_thinking,
};
pub(super) use self::deepseek_v41::render as render_deepseek_v41;
pub(super) use self::kimi_k25::{deep_sort, encode_tools_to_typescript};
