//! Built-in tool support for Responses API
//!
//! Built-in tools provide a high-level abstraction over MCP servers,
//! making common operations like web search, file search, and code interpretation
//! easier to use and more user-friendly.
//!
//! ## Architecture
//!
//! Built-in tools follow a 4-layer pipeline:
//! 1. **Detector**: Identifies built-in tools in requests and validates them
//! 2. **Converter**: Maps built-in tool types to static MCP servers
//! 3. **MCP Handler**: Executes tools via MCP (existing infrastructure)
//! 4. **Formatter**: Transforms MCP results into built-in tool output

pub mod converter;
pub mod detector;
pub mod formatter;
pub mod types;

pub use converter::BuiltinToolConverter;
pub use detector::BuiltinToolDetector;
pub use formatter::BuiltinToolFormatter;
pub use types::{BuiltinToolCall, BuiltinToolResult, BuiltinToolType};
