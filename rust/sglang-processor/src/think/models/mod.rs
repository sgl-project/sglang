//! Parser names SGLang ports from Python, where Dynamo's parsers split output
//! differently. Every other name goes to Dynamo.

mod deepseek_v4;
mod glm45;
mod kimi_k3;

pub(crate) use self::kimi_k3::KimiK3Detector;
#[cfg(feature = "openai")]
pub(crate) use self::kimi_k3::{
    THINK_CLOSE as KIMI_K3_THINK_CLOSE, THINK_OPEN as KIMI_K3_THINK_OPEN,
};
use super::ThinkConfig;

/// The detector a `--reasoning-parser` name selects.
pub(crate) fn think_config(reasoning_parser: &str) -> Option<ThinkConfig> {
    match reasoning_parser {
        "deepseek-v4" => Some(ThinkConfig::Tags(deepseek_v4::THINK)),
        "deepseek-v41" => Some(ThinkConfig::Tags(deepseek_v4::THINK_V41)),
        "glm45" => Some(ThinkConfig::Tags(glm45::THINK)),
        "kimi_k3" => Some(ThinkConfig::KimiK3),
        _ => None,
    }
}
