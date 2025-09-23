/// Minimal Python literal parser for Pythonic tool call format
///
/// This module provides a recursive descent parser for Python literals
/// (strings, numbers, booleans, None, lists, dicts) without requiring
/// a full Python AST parser.
use serde_json::{json, Value};
use std::collections::HashMap;

use crate::tool_parser::errors::{ToolParserError, ToolParserResult};

/// Token types for Python literals
#[derive(Debug, Clone, PartialEq)]
enum Token {
    // Literals
    String(String),
    Number(String),
    True,
    False,
    None,

    // Delimiters
    LeftBracket,  // [
    RightBracket, // ]
    LeftBrace,    // {
    RightBrace,   // }
    LeftParen,    // (
    RightParen,   // )
    Comma,        // ,
    Colon,        // :
    Equals,       // =

    // Identifier for function names
    Identifier(String),

    // End of input
    Eof,
}

/// Lexer for Python literals
struct Lexer {
    input: Vec<char>,
    position: usize,
}

impl Lexer {
    fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            position: 0,
        }
    }

    fn current_char(&self) -> Option<char> {
        self.input.get(self.position).copied()
    }

    fn advance(&mut self) {
        if self.position < self.input.len() {
            self.position += 1;
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current_char() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn read_string(&mut self, quote_char: char) -> Result<String, ToolParserError> {
        let mut result = String::new();
        self.advance(); // Skip opening quote

        while let Some(ch) = self.current_char() {
            if ch == '\\' {
                self.advance();
                if let Some(escaped) = self.current_char() {
                    match escaped {
                        'n' => result.push('\n'),
                        't' => result.push('\t'),
                        'r' => result.push('\r'),
                        '\\' => result.push('\\'),
                        '\'' => result.push('\''),
                        '"' => result.push('"'),
                        _ => {
                            result.push('\\');
                            result.push(escaped);
                        }
                    }
                    self.advance();
                }
            } else if ch == quote_char {
                self.advance(); // Skip closing quote
                return Ok(result);
            } else {
                result.push(ch);
                self.advance();
            }
        }

        Err(ToolParserError::ParsingFailed("Unterminated string".into()))
    }

    fn read_number(&mut self) -> String {
        let mut result = String::new();

        // Handle negative numbers
        if self.current_char() == Some('-') {
            result.push('-');
            self.advance();
        }

        // Read digits and decimal point
        while let Some(ch) = self.current_char() {
            if ch.is_ascii_digit() || ch == '.' || ch == 'e' || ch == 'E' || ch == '+' || ch == '-'
            {
                result.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        result
    }

    fn read_identifier(&mut self) -> String {
        let mut result = String::new();

        while let Some(ch) = self.current_char() {
            if ch.is_alphanumeric() || ch == '_' {
                result.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        result
    }

    fn next_token(&mut self) -> Result<Token, ToolParserError> {
        self.skip_whitespace();

        match self.current_char() {
            None => Ok(Token::Eof),
            Some('[') => {
                self.advance();
                Ok(Token::LeftBracket)
            }
            Some(']') => {
                self.advance();
                Ok(Token::RightBracket)
            }
            Some('{') => {
                self.advance();
                Ok(Token::LeftBrace)
            }
            Some('}') => {
                self.advance();
                Ok(Token::RightBrace)
            }
            Some('(') => {
                self.advance();
                Ok(Token::LeftParen)
            }
            Some(')') => {
                self.advance();
                Ok(Token::RightParen)
            }
            Some(',') => {
                self.advance();
                Ok(Token::Comma)
            }
            Some(':') => {
                self.advance();
                Ok(Token::Colon)
            }
            Some('=') => {
                self.advance();
                Ok(Token::Equals)
            }
            Some('"') => Ok(Token::String(self.read_string('"')?)),
            Some('\'') => Ok(Token::String(self.read_string('\'')?)),
            Some(ch) if ch == '-' || ch.is_ascii_digit() => Ok(Token::Number(self.read_number())),
            Some(ch) if ch.is_alphabetic() || ch == '_' => {
                let ident = self.read_identifier();
                match ident.as_str() {
                    "True" => Ok(Token::True),
                    "False" => Ok(Token::False),
                    "None" => Ok(Token::None),
                    _ => Ok(Token::Identifier(ident)),
                }
            }
            Some(ch) => Err(ToolParserError::ParsingFailed(format!(
                "Unexpected character: {}",
                ch
            ))),
        }
    }
}

/// Parser for Python literals
pub struct PythonLiteralParser {
    lexer: Lexer,
    current_token: Token,
}

impl PythonLiteralParser {
    pub fn new(input: &str) -> Result<Self, ToolParserError> {
        let mut lexer = Lexer::new(input);
        let current_token = lexer.next_token()?;
        Ok(Self {
            lexer,
            current_token,
        })
    }

    fn advance(&mut self) -> Result<(), ToolParserError> {
        self.current_token = self.lexer.next_token()?;
        Ok(())
    }

    fn expect(&mut self, expected: Token) -> Result<(), ToolParserError> {
        if self.current_token == expected {
            self.advance()?;
            Ok(())
        } else {
            Err(ToolParserError::ParsingFailed(format!(
                "Expected {:?}, got {:?}",
                expected, self.current_token
            )))
        }
    }

    /// Parse a Python literal value
    pub fn parse_value(&mut self) -> Result<Value, ToolParserError> {
        match &self.current_token.clone() {
            Token::String(s) => {
                let value = s.clone();
                self.advance()?;
                Ok(json!(value))
            }
            Token::Number(n) => {
                let value = if let Ok(int_val) = n.parse::<i64>() {
                    json!(int_val)
                } else if let Ok(float_val) = n.parse::<f64>() {
                    json!(float_val)
                } else {
                    return Err(ToolParserError::ParsingFailed(format!(
                        "Invalid number: {}",
                        n
                    )));
                };
                self.advance()?;
                Ok(value)
            }
            Token::True => {
                self.advance()?;
                Ok(json!(true))
            }
            Token::False => {
                self.advance()?;
                Ok(json!(false))
            }
            Token::None => {
                self.advance()?;
                Ok(Value::Null)
            }
            Token::LeftBracket => self.parse_list(),
            Token::LeftBrace => self.parse_dict(),
            _ => Err(ToolParserError::ParsingFailed(format!(
                "Unexpected token: {:?}",
                self.current_token
            ))),
        }
    }

    /// Parse a Python list: [item1, item2, ...]
    fn parse_list(&mut self) -> Result<Value, ToolParserError> {
        self.expect(Token::LeftBracket)?;
        let mut items = Vec::new();

        // Handle empty list
        if self.current_token == Token::RightBracket {
            self.advance()?;
            return Ok(json!(items));
        }

        loop {
            items.push(self.parse_value()?);

            if self.current_token == Token::Comma {
                self.advance()?;
                // Handle trailing comma
                if self.current_token == Token::RightBracket {
                    break;
                }
            } else if self.current_token == Token::RightBracket {
                break;
            } else {
                return Err(ToolParserError::ParsingFailed(format!(
                    "Expected ',' or ']', got {:?}",
                    self.current_token
                )));
            }
        }

        self.expect(Token::RightBracket)?;
        Ok(json!(items))
    }

    /// Parse a Python dict: {key1: value1, key2: value2, ...}
    fn parse_dict(&mut self) -> Result<Value, ToolParserError> {
        self.expect(Token::LeftBrace)?;
        let mut map = HashMap::new();

        // Handle empty dict
        if self.current_token == Token::RightBrace {
            self.advance()?;
            return Ok(json!(map));
        }

        loop {
            // Parse key (must be a string)
            let key = match &self.current_token {
                Token::String(s) => {
                    let k = s.clone();
                    self.advance()?;
                    k
                }
                _ => {
                    return Err(ToolParserError::ParsingFailed(format!(
                        "Expected string key, got {:?}",
                        self.current_token
                    )))
                }
            };

            self.expect(Token::Colon)?;

            // Parse value
            let value = self.parse_value()?;
            map.insert(key, value);

            if self.current_token == Token::Comma {
                self.advance()?;
                // Handle trailing comma
                if self.current_token == Token::RightBrace {
                    break;
                }
            } else if self.current_token == Token::RightBrace {
                break;
            } else {
                return Err(ToolParserError::ParsingFailed(format!(
                    "Expected ',' or '}}', got {:?}",
                    self.current_token
                )));
            }
        }

        self.expect(Token::RightBrace)?;
        Ok(json!(map))
    }
}

/// Parse a Python literal string into a JSON value
pub fn parse_python_literal(input: &str) -> ToolParserResult<Value> {
    let mut parser = PythonLiteralParser::new(input)?;
    parser.parse_value()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_primitives() {
        assert_eq!(parse_python_literal("True").unwrap(), json!(true));
        assert_eq!(parse_python_literal("False").unwrap(), json!(false));
        assert_eq!(parse_python_literal("None").unwrap(), Value::Null);
        assert_eq!(parse_python_literal("42").unwrap(), json!(42));
        assert_eq!(parse_python_literal("12.345").unwrap(), json!(12.345));
        assert_eq!(parse_python_literal("-42").unwrap(), json!(-42));
        assert_eq!(parse_python_literal("\"hello\"").unwrap(), json!("hello"));
        assert_eq!(parse_python_literal("'world'").unwrap(), json!("world"));
    }

    #[test]
    fn test_parse_list() {
        assert_eq!(parse_python_literal("[]").unwrap(), json!([]));
        assert_eq!(parse_python_literal("[1, 2, 3]").unwrap(), json!([1, 2, 3]));
        assert_eq!(
            parse_python_literal("[\"a\", \"b\", \"c\"]").unwrap(),
            json!(["a", "b", "c"])
        );
        assert_eq!(
            parse_python_literal("[True, False, None]").unwrap(),
            json!([true, false, null])
        );
        // Nested list
        assert_eq!(
            parse_python_literal("[[1, 2], [3, 4]]").unwrap(),
            json!([[1, 2], [3, 4]])
        );
    }

    #[test]
    fn test_parse_dict() {
        assert_eq!(parse_python_literal("{}").unwrap(), json!({}));
        assert_eq!(
            parse_python_literal("{\"a\": 1, \"b\": 2}").unwrap(),
            json!({"a": 1, "b": 2})
        );
        assert_eq!(
            parse_python_literal("{'x': True, 'y': False}").unwrap(),
            json!({"x": true, "y": false})
        );
        // Nested dict
        assert_eq!(
            parse_python_literal("{\"nested\": {\"value\": [1, 2, 3]}}").unwrap(),
            json!({"nested": {"value": [1, 2, 3]}})
        );
    }

    #[test]
    fn test_complex_nested() {
        let input = r#"{"config": {"nested": {"value": [1, 2, 3]}, "enabled": True}}"#;
        let expected = json!({
            "config": {
                "nested": {
                    "value": [1, 2, 3]
                },
                "enabled": true
            }
        });
        assert_eq!(parse_python_literal(input).unwrap(), expected);
    }
}
