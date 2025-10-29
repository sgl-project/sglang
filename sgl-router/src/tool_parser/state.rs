/// Placeholder for Harmony streaming metadata captured during token-aware parsing.
#[derive(Debug, Clone, Default)]
pub struct HarmonyStreamState {
    /// All tokens observed so far for the current assistant response.
    pub tokens: Vec<u32>,
    /// Number of tokens that have already been processed by the Harmony parser.
    pub processed_tokens: usize,
    /// Number of tool calls emitted downstream.
    pub emitted_calls: usize,
    /// Pending analysis-channel content awaiting flush into normal text output.
    pub analysis_buffer: String,
    /// Whether the tool name has been surfaced for the current call.
    pub emitted_name: bool,
    /// Whether arguments have been surfaced for the current call.
    pub emitted_args: bool,
}
