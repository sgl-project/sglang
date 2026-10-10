//! Parser names SGLang ports from Python, where Dynamo's parsers split output
//! differently. Every other name goes to Dynamo.

mod deepseek_v4;

use super::ThinkConfig;

/// The base-detector tokens a `--reasoning-parser` name selects.
pub(crate) fn think_config(reasoning_parser: &str) -> Option<ThinkConfig> {
    match reasoning_parser {
        "deepseek-v4" => Some(deepseek_v4::THINK),
        "deepseek-v41" => Some(deepseek_v4::THINK_V41),
        _ => None,
    }
}
