//! Qwen3 through Qwen3.8 (`Qwen3Detector` in `parser/reasoning_parser.py`):
//! a `<tool_call>` anywhere also ends the reasoning.

use crate::think::TagConfig;

pub(super) const THINK: TagConfig = TagConfig {
    start: "<think>",
    end: "</think>",
    tool_start: Some("<tool_call>"),
    tool_start_at_line_start: false,
};
