//! DeepSeek-V4 (`DeepSeekV4Detector` in `parser/reasoning_parser.py`).

use crate::parser::think::ThinkConfig;

pub(super) const THINK: ThinkConfig = ThinkConfig {
    start: "<think>",
    end: "</think>",
    tool_start: Some("<｜DSML｜tool_calls>"),
    tool_start_at_line_start: true,
};
