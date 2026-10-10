//! DeepSeek-V4 and V4.1 (`DeepSeekV4Detector` and `DeepSeekV41ReasoningDetector`
//! in `parser/reasoning_parser.py`).

use crate::think::ThinkConfig;

pub(super) const THINK: ThinkConfig = ThinkConfig {
    start: "<think>",
    end: "</think>",
    tool_start: Some("<｜DSML｜tool_calls>"),
    tool_start_at_line_start: true,
};

/// V4.1's tool-call anchor is the spaced `<｜DSML｜ calls>` tag.
pub(super) const THINK_V41: ThinkConfig = ThinkConfig {
    tool_start: Some("<｜DSML｜ calls>"),
    ..THINK
};
