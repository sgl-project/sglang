//! Parser names SGLang ports from Python, where Dynamo's parsers split output
//! differently. Every other name goes to Dynamo.

mod deepseek_v4;
mod glm45;
mod kimi_k3;
mod qwen3;

pub(crate) use self::kimi_k3::KimiK3Detector;
use super::ThinkConfig;

/// The detector a `--reasoning-parser` name selects.
pub(crate) fn think_config(reasoning_parser: &str) -> Option<ThinkConfig> {
    match reasoning_parser {
        "deepseek-v4" => Some(ThinkConfig::Tags(deepseek_v4::THINK)),
        "deepseek-v41" => Some(ThinkConfig::Tags(deepseek_v4::THINK_V41)),
        "glm45" => Some(ThinkConfig::Tags(glm45::THINK)),
        "kimi_k3" => Some(ThinkConfig::KimiK3),
        "qwen3" => Some(ThinkConfig::Tags(qwen3::THINK)),
        _ => None,
    }
}
