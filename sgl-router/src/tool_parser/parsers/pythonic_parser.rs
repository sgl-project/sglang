use std::sync::OnceLock;

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
use rustpython_parser::{
    ast::{Constant, Expr, Mod, UnaryOp},
    parse, Mode,
};
use serde_json::{Map, Number, Value};

use crate::{
    protocols::common::Tool,
    tool_parser::{
        errors::{ParserError, ParserResult},
        parsers::helpers,
        traits::ToolParser,
        types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
    },
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
pub struct PythonicParser {
    /// Buffer for accumulating chunks
    buffer: String,
}

impl Default for PythonicParser {
    fn default() -> Self {
        Self::new()
    }
}

impl PythonicParser {
    /// Create a new Pythonic parser
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
        }
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

    fn parse_tool_call_block(&self, block: &str) -> ParserResult<Vec<ToolCall>> {
        let expr = parse_python_expression(block)?;
        match expr {
            Expr::List(list_expr) => list_expr
                .elts
                .into_iter()
                .enumerate()
                .map(|(idx, call_expr)| build_tool_call(call_expr, idx))
                .collect(),
            _ => Err(ParserError::ParsingFailed(
                "Expected a list of function calls in pythonic tool call".to_string(),
            )),
        }
    }
}

#[async_trait]
impl ToolParser for PythonicParser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
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
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);

        let cleaned = Self::strip_special_tokens(&self.buffer);

        // Look for opening bracket
        if let Some(start) = cleaned.find('[') {
            let normal_text = if start > 0 {
                cleaned[..start].to_string()
            } else {
                String::new()
            };

            // Look for matching closing bracket
            if let Some(end) = find_matching_bracket(&cleaned, start) {
                // Found complete tool call - extract it and parse using parse_complete
                let call_text = &cleaned[start..=end];

                match self.parse_complete(call_text).await {
                    Ok((_, calls)) => {
                        // Update buffer with remaining text after tool call
                        let remaining_text = &cleaned[end + 1..];
                        self.buffer = remaining_text.to_string();

                        // Validate tool names and convert ToolCall to ToolCallItem
                        let tool_indices = helpers::get_tool_indices(tools);
                        let items: Vec<ToolCallItem> = calls
                            .into_iter()
                            .enumerate()
                            .filter_map(|(idx, tool)| {
                                if !tool_indices.contains_key(&tool.function.name) {
                                    tracing::warn!(
                                        "Invalid tool name '{}' - skipping",
                                        tool.function.name
                                    );
                                    return None;
                                }

                                Some(ToolCallItem {
                                    tool_index: idx,
                                    name: Some(tool.function.name),
                                    parameters: tool.function.arguments,
                                })
                            })
                            .collect();

                        return Ok(StreamingParseResult {
                            normal_text,
                            calls: items,
                        });
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse pythonic tool call: {}", e);
                        // Clear buffer on error
                        self.buffer.clear();
                        return Ok(StreamingParseResult::default());
                    }
                }
            } else {
                // We have an opening bracket but no closing bracket yet
                // Put back everything from the bracket onwards
                self.buffer = cleaned[start..].to_string();

                if !normal_text.is_empty() {
                    return Ok(StreamingParseResult {
                        normal_text,
                        calls: vec![],
                    });
                }

                // Still accumulating a potential tool call
                return Ok(StreamingParseResult::default());
            }
        }

        // No tool call bracket found
        self.buffer.clear();
        Ok(StreamingParseResult {
            normal_text: cleaned,
            calls: vec![],
        })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        let cleaned = Self::strip_special_tokens(text);
        if pythonic_block_regex().is_match(&cleaned) {
            return true;
        }

        false
    }
}

/// Find the matching closing bracket for the opening bracket at start position.
/// Properly handles nested brackets.
fn find_matching_bracket(buffer: &str, start: usize) -> Option<usize> {
    let mut bracket_count = 0;
    let chars: Vec<char> = buffer.chars().collect();

    for (i, &ch) in chars.iter().enumerate().skip(start) {
        if ch == '[' {
            bracket_count += 1;
        } else if ch == ']' {
            bracket_count -= 1;
            if bracket_count == 0 {
                return Some(i);
            }
        }
    }
    None // No matching bracket found
}

fn parse_python_expression(source: &str) -> ParserResult<Expr> {
    let module = parse(source, Mode::Expression, "<pythonic_tool_call>")
        .map_err(|err| ParserError::ParsingFailed(err.to_string()))?;

    match module {
        Mod::Expression(expr_mod) => Ok(*expr_mod.body),
        _ => Err(ParserError::ParsingFailed(
            "Expected a Python expression".to_string(),
        )),
    }
}

fn build_tool_call(expr: Expr, _index: usize) -> ParserResult<ToolCall> {
    match expr {
        Expr::Call(call_expr) => {
            if !call_expr.args.is_empty() {
                return Err(ParserError::ParsingFailed(
                    "Positional arguments are not supported in pythonic tool calls".to_string(),
                ));
            }

            let function_name = match *call_expr.func {
                Expr::Name(name_expr) => name_expr.id.to_string(),
                _ => {
                    return Err(ParserError::ParsingFailed(
                        "Unsupported function reference in pythonic tool call".to_string(),
                    ))
                }
            };

            let mut arguments_map = Map::with_capacity(call_expr.keywords.len());
            for keyword in call_expr.keywords {
                let arg_name = keyword.arg.ok_or_else(|| {
                    ParserError::ParsingFailed(
                        "pythonic tool calls do not support **kwargs".to_string(),
                    )
                })?;
                let value_json = expression_to_json(&keyword.value)?;
                arguments_map.insert(arg_name.to_string(), value_json);
            }

            let arguments_json = Value::Object(arguments_map);
            let arguments_string = serde_json::to_string(&arguments_json)?;

            Ok(ToolCall {
                function: FunctionCall {
                    name: function_name,
                    arguments: arguments_string,
                },
            })
        }
        _ => Err(ParserError::ParsingFailed(
            "Expected function calls inside pythonic tool call list".to_string(),
        )),
    }
}

fn expression_to_json(expr: &Expr) -> ParserResult<Value> {
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
                _ => Err(ParserError::ParsingFailed(
                    "Unsupported unary operand in pythonic tool call".to_string(),
                )),
            },
            UnaryOp::UAdd => expression_to_json(unary_expr.operand.as_ref()),
            _ => Err(ParserError::ParsingFailed(format!(
                "Unsupported unary operator in pythonic tool call: {:?}",
                unary_expr.op
            ))),
        },
        Expr::Name(name_expr) => Ok(Value::String(name_expr.id.to_string())),
        _ => Err(ParserError::ParsingFailed(format!(
            "Unsupported expression in pythonic tool call: {:?}",
            expr
        ))),
    }
}

fn constant_to_json(constant: &Constant) -> ParserResult<Value> {
    match constant {
        Constant::None => Ok(Value::Null),
        Constant::Bool(b) => Ok(Value::Bool(*b)),
        Constant::Int(value) => Ok(integer_constant_to_value(value, false)),
        Constant::Float(f) => Number::from_f64(*f).map(Value::Number).ok_or_else(|| {
            ParserError::ParsingFailed("Invalid float literal in pythonic tool call".to_string())
        }),
        Constant::Str(s) => Ok(Value::String(s.clone())),
        Constant::Bytes(bytes) => Ok(Value::String(String::from_utf8_lossy(bytes).into_owned())),
        Constant::Tuple(values) => constant_tuple_to_array(values).map(Value::Array),
        Constant::Ellipsis | Constant::Complex { .. } => Err(ParserError::ParsingFailed(
            "Unsupported literal in pythonic tool call".to_string(),
        )),
    }
}

fn negate_constant(constant: &Constant) -> ParserResult<Value> {
    match constant {
        Constant::Int(value) => Ok(integer_constant_to_value(value, true)),
        Constant::Float(f) => Number::from_f64(-f).map(Value::Number).ok_or_else(|| {
            ParserError::ParsingFailed("Invalid float literal in pythonic tool call".to_string())
        }),
        _ => Err(ParserError::ParsingFailed(
            "Unsupported unary operand in pythonic tool call".to_string(),
        )),
    }
}

fn value_to_key_string(value: Value) -> ParserResult<String> {
    match value {
        Value::String(s) => Ok(s),
        Value::Number(num) => Ok(num.to_string()),
        Value::Bool(b) => Ok(b.to_string()),
        Value::Null => Ok("null".to_string()),
        other => Err(ParserError::ParsingFailed(format!(
            "Unsupported key type in pythonic tool call: {:?}",
            other
        ))),
    }
}

fn collect_sequence(elements: &[Expr]) -> ParserResult<Vec<Value>> {
    elements.iter().map(expression_to_json).collect()
}

fn collect_dict(keys: &[Option<Expr>], values: &[Expr]) -> ParserResult<Map<String, Value>> {
    let mut map = Map::with_capacity(keys.len());
    for (key_expr, value_expr) in keys.iter().zip(values.iter()) {
        let key_expr = key_expr.as_ref().ok_or_else(|| {
            ParserError::ParsingFailed("pythonic tool calls do not support **kwargs".to_string())
        })?;
        let key_value = expression_to_json(key_expr)?;
        let key = value_to_key_string(key_value)?;
        let value_json = expression_to_json(value_expr)?;
        map.insert(key, value_json);
    }
    Ok(map)
}

fn constant_tuple_to_array(values: &[Constant]) -> ParserResult<Vec<Value>> {
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
