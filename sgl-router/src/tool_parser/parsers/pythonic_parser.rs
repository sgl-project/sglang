/// Pythonic format parser for tool calls
///
/// Handles Python function call syntax within square brackets:
/// ```text
/// [tool1(arg1=val1, arg2=val2), tool2(arg1=val3)]
/// ```
///
/// This format is used by Llama models and uses Python literals
/// rather than JSON for arguments.
use async_trait::async_trait;
use num_traits::ToPrimitive;
use regex::Regex;
use rustpython_parser::ast::{Constant, Expr, Mod, UnaryOp};
use rustpython_parser::{parse, Mode};
use serde_json::{Map, Number, Value};
use std::sync::OnceLock;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    state::ParseState,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

static PYTHONIC_BLOCK_REGEX: OnceLock<Regex> = OnceLock::new();

/// Lazily compiled regex that locates pythonic tool call blocks.
fn pythonic_block_regex() -> &'static Regex {
    PYTHONIC_BLOCK_REGEX.get_or_init(|| {
        // Matches one or more function calls inside a list. The `(?s)` flag allows
        // newlines inside argument lists while keeping the pattern anchored to
        // identifiers followed by parentheses, preventing plain lists like
        // `[1, 2, 3]` from matching.
        Regex::new(r"(?s)\[\s*[A-Za-z_]\w*\s*\(.*?\)\s*(?:,\s*[A-Za-z_]\w*\s*\(.*?\)\s*)*\]")
            .expect("pythonic tool call regex must compile")
    })
}

/// Parser for Pythonic tool call format
#[derive(Default)]
pub struct PythonicParser;

impl PythonicParser {
    /// Create a new Pythonic parser
    pub fn new() -> Self {
        Self
    }

    /// Extract the first pythonic tool call block and return it along with the
    /// surrounding "normal" content.
    fn extract_tool_calls(&self, text: &str) -> Option<(String, String)> {
        pythonic_block_regex().find(text).map(|mat| {
            let block = mat.as_str().to_string();
            let normal = format!("{}{}", &text[..mat.start()], &text[mat.end()..]);
            (block, normal)
        })
    }

    /// Strip special tokens that Llama models might output
    fn strip_special_tokens(text: &str) -> String {
        text.replace("<|python_start|>", "")
            .replace("<|python_end|>", "")
    }

    fn parse_tool_call_block(&self, block: &str) -> ToolParserResult<Vec<ToolCall>> {
        let expr = parse_python_expression(block)?;
        match expr {
            Expr::List(list_expr) => list_expr
                .elts
                .into_iter()
                .enumerate()
                .map(|(idx, call_expr)| build_tool_call(call_expr, idx))
                .collect(),
            _ => Err(ToolParserError::ParsingFailed(
                "Expected a list of function calls in pythonic tool call".to_string(),
            )),
        }
    }
}

#[async_trait]
impl ToolParser for PythonicParser {
    async fn parse_complete(&self, text: &str) -> ToolParserResult<(String, Vec<ToolCall>)> {
        let cleaned = Self::strip_special_tokens(text);

        if let Some((tool_calls_text, normal_text)) = self.extract_tool_calls(&cleaned) {
            match self.parse_tool_call_block(&tool_calls_text) {
                Ok(calls) => {
                    if calls.is_empty() {
                        // No tools successfully parsed despite having markers
                        Ok((text.to_string(), vec![]))
                    } else {
                        Ok((normal_text, calls))
                    }
                }
                Err(e) => {
                    // Log warning and return entire text as fallback
                    tracing::warn!("Failed to parse pythonic tool calls: {}", e);
                    Ok((text.to_string(), vec![]))
                }
            }
        } else {
            Ok((text.to_string(), vec![]))
        }
    }

    async fn parse_incremental(
        &self,
        chunk: &str,
        state: &mut ParseState,
    ) -> ToolParserResult<StreamingParseResult> {
        state.buffer.push_str(chunk);
        let mut result = StreamingParseResult::new();

        // Phase 1: Check for normal text before tool markers
        if !state.in_tool_section {
            // Look for start of pythonic tool call block
            if let Some(block_start) = self.find_pythonic_block_start(&state.buffer) {
                if block_start > 0 {
                    result.normal_text = state.buffer.drain(..block_start).collect();
                    state.in_tool_section = true;
                    return Ok(result);
                }
                state.in_tool_section = true;
            } else {
                // Check if we might have a partial start
                if self.has_partial_pythonic_start(&state.buffer) {
                    return Ok(result);
                }

                // No tool calls, return as normal text
                result.normal_text = std::mem::take(&mut state.buffer);
                return Ok(result);
            }
        }

        // Phase 2: Process pythonic tool calls
        if state.in_tool_section {
            self.process_pythonic_calls(state, &mut result)?;
        }

        Ok(result)
    }

    fn detect_format(&self, text: &str) -> bool {
        let cleaned = Self::strip_special_tokens(text);
        if pythonic_block_regex().is_match(&cleaned) {
            return true;
        }

        false
    }
}

impl PythonicParser {
    fn find_pythonic_block_start(&self, buffer: &str) -> Option<usize> {
        // Look for pattern like "[function_name("
        let mut in_string = false;
        let mut escape = false;

        for (i, ch) in buffer.char_indices() {
            if escape {
                escape = false;
                continue;
            }

            match ch {
                '\\' if in_string => escape = true,
                '"' | '\'' => in_string = !in_string,
                '[' if !in_string => {
                    // Check if followed by a function name pattern
                    let rest = &buffer[i + 1..];
                    let trimmed = rest.trim_start();
                    if trimmed.len() > 0 {
                        let first_char = trimmed.chars().next().unwrap();
                        if first_char.is_ascii_alphabetic() || first_char == '_' {
                            // Likely a function name
                            return Some(i);
                        }
                    }
                }
                _ => {}
            }
        }
        None
    }

    fn has_partial_pythonic_start(&self, buffer: &str) -> bool {
        // Check if buffer ends with '[' potentially starting a tool call
        buffer.trim_end().ends_with('[') ||
        // Or ends with '[' followed by partial function name
        buffer.chars().rev().take(20).any(|c| c == '[')
    }

    fn process_pythonic_calls(
        &self,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        // Try to find complete pythonic block
        if let Some(mat) = pythonic_block_regex().find(&state.buffer) {
            let block = mat.as_str().to_string();
            let mat_end = mat.end();

            // Parse the block
            match self.parse_pythonic_block(&block) {
                Ok(tools) => {
                    for (idx, (name, args)) in tools.iter().enumerate() {
                        self.emit_tool_call(name, args, idx, state, result)?;
                    }

                    // Remove processed block
                    state.buffer.drain(..mat_end);

                    // Check if there's more content
                    if state.buffer.trim().is_empty() {
                        state.mode = crate::tool_parser::state::ParseMode::Complete;
                    }
                }
                Err(_) => {
                    // Failed to parse, try partial extraction
                    self.try_extract_partial_pythonic(state, result)?;
                }
            }
        } else {
            // No complete block, try partial extraction
            self.try_extract_partial_pythonic(state, result)?;
        }

        Ok(())
    }

    fn parse_pythonic_block(&self, block: &str) -> ToolParserResult<Vec<(String, String)>> {
        // Parse as Python expression
        let ast = parse(block, Mode::Expression, "<tool>")
            .map_err(|e| ToolParserError::ParsingFailed(format!("Python parse error: {}", e)))?;

        let mut tools = Vec::new();

        if let Mod::Expression(ref expr) = ast {
            if let Expr::List(list) = expr.body.as_ref() {
                for element in &list.elts {
                    if let Expr::Call(call) = element {
                        // Extract function name
                        let name = if let Expr::Name(name) = call.func.as_ref() {
                            name.id.to_string()
                        } else {
                            continue;
                        };

                        // Convert arguments to JSON
                        let args_json = self.convert_call_args_to_json(call)?;
                        let args_str = serde_json::to_string(&args_json)?;

                        tools.push((name, args_str));
                    }
                }
            }
        }

        Ok(tools)
    }

    fn try_extract_partial_pythonic(
        &self,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        // Try to extract partial function calls
        // Pattern: function_name(partial_args
        let pattern = r"([A-Za-z_]\w*)\s*\([^)]*";
        let re = regex::Regex::new(pattern).unwrap();

        let mut tool_index = 0;
        let buffer = state.buffer.clone();
        for mat in re.find_iter(&buffer) {
            let text = mat.as_str();

            // Extract function name
            if let Some(paren_pos) = text.find('(') {
                let func_name = text[..paren_pos].trim();

                if !func_name.is_empty() {
                    // Ensure we have partial tool entry
                    let partial = state.ensure_partial_tool(tool_index);

                    if !partial.name_sent {
                        partial.name = Some(func_name.to_string());
                        partial.id = Some(format!("pythonic_call_{}", uuid::Uuid::new_v4()));
                        partial.name_sent = true;

                        result.tool_calls.push(ToolCallItem {
                            tool_index,
                            id: partial.id.clone(),
                            name: partial.name.clone(),
                            arguments_delta: String::new(),
                        });
                    }

                    // Try to extract partial arguments
                    let args_start = paren_pos + 1;
                    if args_start < text.len() {
                        let partial_args = &text[args_start..];

                        // Try to parse partial Python args and convert to JSON
                        if !partial_args.trim().is_empty() {
                            // For now, emit raw text as arguments
                            // In production, would parse Python literals properly
                            if partial_args.len() > partial.streamed_arguments.len() {
                                let delta = &partial_args[partial.streamed_arguments.len()..];

                                result.tool_calls.push(ToolCallItem {
                                    tool_index,
                                    id: None,
                                    name: None,
                                    arguments_delta: delta.to_string(),
                                });

                                partial.streamed_arguments.push_str(delta);
                            }
                        }
                    }

                    tool_index += 1;
                }
            }
        }

        Ok(())
    }

    fn convert_call_args_to_json(&self, call: &rustpython_parser::ast::ExprCall) -> ToolParserResult<Value> {
        if !call.args.is_empty() {
            return Err(ToolParserError::ParsingFailed(
                "Positional arguments are not supported in pythonic tool calls".to_string(),
            ));
        }

        let mut arguments_map = Map::with_capacity(call.keywords.len());
        for keyword in &call.keywords {
            let arg_name = keyword.arg.as_ref().ok_or_else(|| {
                ToolParserError::ParsingFailed(
                    "pythonic tool calls do not support **kwargs".to_string(),
                )
            })?;
            let value_json = expression_to_json(&keyword.value)?;
            arguments_map.insert(arg_name.to_string(), value_json);
        }

        Ok(Value::Object(arguments_map))
    }

    fn emit_tool_call(
        &self,
        name: &str,
        args: &str,
        index: usize,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        // Ensure we have partial tool entry
        let partial = state.ensure_partial_tool(index);
        partial.name = Some(name.to_string());
        partial.id = Some(format!("pythonic_call_{}", uuid::Uuid::new_v4()));

        result.tool_calls.push(ToolCallItem {
            tool_index: index,
            id: partial.id.clone(),
            name: partial.name.clone(),
            arguments_delta: args.to_string(),
        });

        Ok(())
    }


}

fn parse_python_expression(source: &str) -> ToolParserResult<Expr> {
    let module = parse(source, Mode::Expression, "<pythonic_tool_call>")
        .map_err(|err| ToolParserError::ParsingFailed(err.to_string()))?;

    match module {
        Mod::Expression(expr_mod) => Ok(*expr_mod.body),
        _ => Err(ToolParserError::ParsingFailed(
            "Expected a Python expression".to_string(),
        )),
    }
}

fn build_tool_call(expr: Expr, index: usize) -> ToolParserResult<ToolCall> {
    match expr {
        Expr::Call(call_expr) => {
            if !call_expr.args.is_empty() {
                return Err(ToolParserError::ParsingFailed(
                    "Positional arguments are not supported in pythonic tool calls".to_string(),
                ));
            }

            let function_name = match *call_expr.func {
                Expr::Name(name_expr) => name_expr.id.to_string(),
                _ => {
                    return Err(ToolParserError::ParsingFailed(
                        "Unsupported function reference in pythonic tool call".to_string(),
                    ))
                }
            };

            let mut arguments_map = Map::with_capacity(call_expr.keywords.len());
            for keyword in call_expr.keywords {
                let arg_name = keyword.arg.ok_or_else(|| {
                    ToolParserError::ParsingFailed(
                        "pythonic tool calls do not support **kwargs".to_string(),
                    )
                })?;
                let value_json = expression_to_json(&keyword.value)?;
                arguments_map.insert(arg_name.to_string(), value_json);
            }

            let arguments_json = Value::Object(arguments_map);
            let arguments_string = serde_json::to_string(&arguments_json)?;

            Ok(ToolCall {
                id: format!("call-{}", index + 1),
                r#type: "function".to_string(),
                function: FunctionCall {
                    name: function_name,
                    arguments: arguments_string,
                },
            })
        }
        _ => Err(ToolParserError::ParsingFailed(
            "Expected function calls inside pythonic tool call list".to_string(),
        )),
    }
}

fn expression_to_json(expr: &Expr) -> ToolParserResult<Value> {
    match expr {
        Expr::Constant(expr_constant) => constant_to_json(&expr_constant.value),
        Expr::List(list_expr) => collect_sequence(&list_expr.elts).map(Value::Array),
        Expr::Tuple(tuple_expr) => collect_sequence(&tuple_expr.elts).map(Value::Array),
        Expr::Dict(dict_expr) => {
            collect_dict(&dict_expr.keys, &dict_expr.values).map(Value::Object)
        }
        Expr::UnaryOp(unary_expr) => match unary_expr.op {
            UnaryOp::USub => match unary_expr.operand.as_ref() {
                Expr::Constant(const_expr) => negate_constant(&const_expr.value),
                _ => Err(ToolParserError::ParsingFailed(
                    "Unsupported unary operand in pythonic tool call".to_string(),
                )),
            },
            UnaryOp::UAdd => expression_to_json(unary_expr.operand.as_ref()),
            _ => Err(ToolParserError::ParsingFailed(format!(
                "Unsupported unary operator in pythonic tool call: {:?}",
                unary_expr.op
            ))),
        },
        Expr::Name(name_expr) => Ok(Value::String(name_expr.id.to_string())),
        _ => Err(ToolParserError::ParsingFailed(format!(
            "Unsupported expression in pythonic tool call: {:?}",
            expr
        ))),
    }
}

fn constant_to_json(constant: &Constant) -> ToolParserResult<Value> {
    match constant {
        Constant::None => Ok(Value::Null),
        Constant::Bool(b) => Ok(Value::Bool(*b)),
        Constant::Int(value) => Ok(integer_constant_to_value(value, false)),
        Constant::Float(f) => Number::from_f64(*f).map(Value::Number).ok_or_else(|| {
            ToolParserError::ParsingFailed(
                "Invalid float literal in pythonic tool call".to_string(),
            )
        }),
        Constant::Str(s) => Ok(Value::String(s.clone())),
        Constant::Bytes(bytes) => Ok(Value::String(String::from_utf8_lossy(bytes).into_owned())),
        Constant::Tuple(values) => constant_tuple_to_array(values).map(Value::Array),
        Constant::Ellipsis | Constant::Complex { .. } => Err(ToolParserError::ParsingFailed(
            "Unsupported literal in pythonic tool call".to_string(),
        )),
    }
}

fn negate_constant(constant: &Constant) -> ToolParserResult<Value> {
    match constant {
        Constant::Int(value) => Ok(integer_constant_to_value(value, true)),
        Constant::Float(f) => Number::from_f64(-f).map(Value::Number).ok_or_else(|| {
            ToolParserError::ParsingFailed(
                "Invalid float literal in pythonic tool call".to_string(),
            )
        }),
        _ => Err(ToolParserError::ParsingFailed(
            "Unsupported unary operand in pythonic tool call".to_string(),
        )),
    }
}

fn value_to_key_string(value: Value) -> ToolParserResult<String> {
    match value {
        Value::String(s) => Ok(s),
        Value::Number(num) => Ok(num.to_string()),
        Value::Bool(b) => Ok(b.to_string()),
        Value::Null => Ok("null".to_string()),
        other => Err(ToolParserError::ParsingFailed(format!(
            "Unsupported key type in pythonic tool call: {:?}",
            other
        ))),
    }
}

fn collect_sequence(elements: &[Expr]) -> ToolParserResult<Vec<Value>> {
    elements.iter().map(expression_to_json).collect()
}

fn collect_dict(keys: &[Option<Expr>], values: &[Expr]) -> ToolParserResult<Map<String, Value>> {
    let mut map = Map::with_capacity(keys.len());
    for (key_expr, value_expr) in keys.iter().zip(values.iter()) {
        let key_expr = key_expr.as_ref().ok_or_else(|| {
            ToolParserError::ParsingFailed(
                "pythonic tool calls do not support **kwargs".to_string(),
            )
        })?;
        let key_value = expression_to_json(key_expr)?;
        let key = value_to_key_string(key_value)?;
        let value_json = expression_to_json(value_expr)?;
        map.insert(key, value_json);
    }
    Ok(map)
}

fn constant_tuple_to_array(values: &[Constant]) -> ToolParserResult<Vec<Value>> {
    values.iter().map(constant_to_json).collect()
}

fn integer_constant_to_value<T>(value: &T, negate: bool) -> Value
where
    T: ToPrimitive + std::fmt::Display,
{
    if let Some(mut i) = value.to_i64() {
        if negate {
            i = -i;
        }
        return Value::Number(Number::from(i));
    }

    if negate {
        if let Some(u) = value.to_u64() {
            if u <= i64::MAX as u64 {
                return Value::Number(Number::from(-(u as i64)));
            }
            return Value::String(format!("-{}", value));
        }
        Value::String(format!("-{}", value))
    } else if let Some(u) = value.to_u64() {
        Value::Number(Number::from(u))
    } else {
        Value::String(value.to_string())
    }
}
