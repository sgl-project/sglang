use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    traits::PartialJsonParser,
};
use serde_json::{Map, Value};

/// Parser for incomplete JSON
pub struct PartialJson {
    /// Maximum depth for nested structures
    max_depth: usize,
    /// Whether to allow incomplete values
    allow_incomplete: bool,
}

impl PartialJson {
    /// Create a new partial JSON parser
    pub fn new(max_depth: usize, allow_incomplete: bool) -> Self {
        Self {
            max_depth,
            allow_incomplete,
        }
    }

    /// Parse potentially incomplete JSON, returning parsed value and consumed bytes
    pub fn parse_value(&self, input: &str) -> ToolParserResult<(Value, usize)> {
        let mut parser = Parser::new(input, self.max_depth, self.allow_incomplete);
        let value = parser.parse_value(0)?;
        Ok((value, parser.position))
    }
}

impl Default for PartialJson {
    fn default() -> Self {
        Self::new(32, true)
    }
}

impl PartialJsonParser for PartialJson {
    fn parse(&self, input: &str) -> ToolParserResult<(Value, usize)> {
        self.parse_value(input)
    }

    fn is_complete(&self, input: &str) -> bool {
        // Try to parse as complete JSON
        serde_json::from_str::<Value>(input).is_ok()
    }

    fn max_depth(&self) -> usize {
        self.max_depth
    }
}

/// Internal parser state
struct Parser<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
    position: usize,
    max_depth: usize,
    allow_incomplete: bool,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str, max_depth: usize, allow_incomplete: bool) -> Self {
        Self {
            chars: input.chars().peekable(),
            position: 0,
            max_depth,
            allow_incomplete,
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.chars.peek().copied()
    }

    fn advance(&mut self) {
        if self.chars.next().is_some() {
            self.position += 1;
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self, depth: usize) -> ToolParserResult<Value> {
        if depth > self.max_depth {
            return Err(ToolParserError::DepthExceeded(self.max_depth));
        }

        self.skip_whitespace();

        match self.peek() {
            Some('{') => self.parse_object(depth + 1),
            Some('[') => self.parse_array(depth + 1),
            Some('"') => self.parse_string(),
            Some('t') | Some('f') => self.parse_bool(),
            Some('n') => self.parse_null(),
            Some(c) if c == '-' || c.is_ascii_digit() => self.parse_number(),
            _ => {
                if self.allow_incomplete {
                    Ok(Value::Null)
                } else {
                    Err(ToolParserError::ParsingFailed(
                        "Unexpected character".into(),
                    ))
                }
            }
        }
    }

    fn parse_object(&mut self, depth: usize) -> ToolParserResult<Value> {
        if depth > self.max_depth {
            return Err(ToolParserError::DepthExceeded(self.max_depth));
        }

        let mut object = Map::new();

        // Consume '{'
        self.advance();
        self.skip_whitespace();

        // Check for empty object
        if self.peek() == Some('}') {
            self.advance();
            return Ok(Value::Object(object));
        }

        loop {
            // Parse key
            let key = match self.parse_string() {
                Ok(Value::String(s)) => s,
                Err(_) if self.allow_incomplete => {
                    // Incomplete object
                    return Ok(Value::Object(object));
                }
                Err(e) => return Err(e),
                _ => return Err(ToolParserError::ParsingFailed("Expected string key".into())),
            };

            self.skip_whitespace();

            // Expect ':'
            if self.peek() != Some(':') {
                if self.allow_incomplete {
                    // Add null value for incomplete pair
                    object.insert(key, Value::Null);
                    return Ok(Value::Object(object));
                }
                return Err(ToolParserError::ParsingFailed("Expected ':'".into()));
            }
            self.advance();
            self.skip_whitespace();

            // Parse value (keep same depth - we already incremented in parse_object)
            let value = match self.parse_value(depth) {
                Ok(v) => v,
                Err(_) if self.allow_incomplete => {
                    // Add null for incomplete value
                    object.insert(key, Value::Null);
                    return Ok(Value::Object(object));
                }
                Err(e) => return Err(e),
            };

            object.insert(key, value);
            self.skip_whitespace();

            match self.peek() {
                Some(',') => {
                    self.advance();
                    self.skip_whitespace();
                    // Check for trailing comma
                    if self.peek() == Some('}') {
                        self.advance();
                        return Ok(Value::Object(object));
                    }
                }
                Some('}') => {
                    self.advance();
                    return Ok(Value::Object(object));
                }
                None if self.allow_incomplete => {
                    return Ok(Value::Object(object));
                }
                _ => {
                    if self.allow_incomplete {
                        return Ok(Value::Object(object));
                    }
                    return Err(ToolParserError::ParsingFailed("Expected ',' or '}'".into()));
                }
            }
        }
    }

    fn parse_array(&mut self, depth: usize) -> ToolParserResult<Value> {
        if depth > self.max_depth {
            return Err(ToolParserError::DepthExceeded(self.max_depth));
        }

        let mut array = Vec::new();

        // Consume '['
        self.advance();
        self.skip_whitespace();

        // Check for empty array
        if self.peek() == Some(']') {
            self.advance();
            return Ok(Value::Array(array));
        }

        loop {
            // Parse value (keep same depth - we already incremented in parse_object)
            let value = match self.parse_value(depth) {
                Ok(v) => v,
                Err(_) if self.allow_incomplete => {
                    return Ok(Value::Array(array));
                }
                Err(e) => return Err(e),
            };

            array.push(value);
            self.skip_whitespace();

            match self.peek() {
                Some(',') => {
                    self.advance();
                    self.skip_whitespace();
                    // Check for trailing comma
                    if self.peek() == Some(']') {
                        self.advance();
                        return Ok(Value::Array(array));
                    }
                }
                Some(']') => {
                    self.advance();
                    return Ok(Value::Array(array));
                }
                None if self.allow_incomplete => {
                    return Ok(Value::Array(array));
                }
                _ => {
                    if self.allow_incomplete {
                        return Ok(Value::Array(array));
                    }
                    return Err(ToolParserError::ParsingFailed("Expected ',' or ']'".into()));
                }
            }
        }
    }

    fn parse_string(&mut self) -> ToolParserResult<Value> {
        if self.peek() != Some('"') {
            return Err(ToolParserError::ParsingFailed("Expected '\"'".into()));
        }

        // Consume opening quote
        self.advance();

        let mut string = String::new();
        let mut escaped = false;

        while let Some(ch) = self.peek() {
            if escaped {
                // Handle escape sequences
                let escaped_char = match ch {
                    '"' | '\\' | '/' => ch,
                    'b' => '\u{0008}',
                    'f' => '\u{000C}',
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    'u' => {
                        // Unicode escape
                        self.advance();
                        let hex = self.parse_unicode_escape()?;
                        string.push(hex);
                        escaped = false;
                        continue;
                    }
                    _ => ch, // Invalid escape, but be lenient
                };
                string.push(escaped_char);
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                // End of string
                self.advance();
                return Ok(Value::String(string));
            } else {
                string.push(ch);
            }
            self.advance();
        }

        // Incomplete string
        if self.allow_incomplete {
            Ok(Value::String(string))
        } else {
            Err(ToolParserError::ParsingFailed("Unterminated string".into()))
        }
    }

    fn parse_unicode_escape(&mut self) -> ToolParserResult<char> {
        let mut hex = String::new();
        for _ in 0..4 {
            if let Some(ch) = self.peek() {
                if ch.is_ascii_hexdigit() {
                    hex.push(ch);
                    self.advance();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if hex.len() == 4 {
            u32::from_str_radix(&hex, 16)
                .ok()
                .and_then(char::from_u32)
                .ok_or_else(|| ToolParserError::ParsingFailed("Invalid unicode escape".into()))
        } else if self.allow_incomplete {
            Ok('\u{FFFD}') // Replacement character
        } else {
            Err(ToolParserError::ParsingFailed(
                "Incomplete unicode escape".into(),
            ))
        }
    }

    fn parse_number(&mut self) -> ToolParserResult<Value> {
        let mut number = String::new();

        // Handle negative sign
        if self.peek() == Some('-') {
            number.push('-');
            self.advance();
        }

        // Parse integer part
        if self.peek() == Some('0') {
            number.push('0');
            self.advance();
        } else {
            while let Some(ch) = self.peek() {
                if ch.is_ascii_digit() {
                    number.push(ch);
                    self.advance();
                } else {
                    break;
                }
            }
        }

        // Parse decimal part
        if self.peek() == Some('.') {
            number.push('.');
            self.advance();

            while let Some(ch) = self.peek() {
                if ch.is_ascii_digit() {
                    number.push(ch);
                    self.advance();
                } else {
                    break;
                }
            }
        }

        // Parse exponent
        if let Some(ch) = self.peek() {
            if ch == 'e' || ch == 'E' {
                number.push(ch);
                self.advance();

                if let Some(sign) = self.peek() {
                    if sign == '+' || sign == '-' {
                        number.push(sign);
                        self.advance();
                    }
                }

                while let Some(ch) = self.peek() {
                    if ch.is_ascii_digit() {
                        number.push(ch);
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
        }

        // Try to parse as integer first, then as float
        if let Ok(n) = number.parse::<i64>() {
            Ok(Value::Number(serde_json::Number::from(n)))
        } else if let Ok(n) = number.parse::<f64>() {
            Ok(Value::Number(
                serde_json::Number::from_f64(n).unwrap_or_else(|| serde_json::Number::from(0)),
            ))
        } else if self.allow_incomplete {
            Ok(Value::Number(serde_json::Number::from(0)))
        } else {
            Err(ToolParserError::ParsingFailed("Invalid number".into()))
        }
    }

    fn parse_bool(&mut self) -> ToolParserResult<Value> {
        let mut word = String::new();

        // Peek at upcoming characters to validate it looks like a boolean
        let mut temp_chars = self.chars.clone();
        while let Some(&ch) = temp_chars.peek() {
            if ch.is_alphabetic() && word.len() < 5 {
                // "false" is 5 chars
                word.push(ch);
                temp_chars.next();
            } else {
                break;
            }
        }

        // Check if it's a valid boolean prefix
        let is_valid = word == "true"
            || word == "false"
            || (self.allow_incomplete && ("true".starts_with(&word) || "false".starts_with(&word)));

        if !is_valid {
            return Err(ToolParserError::ParsingFailed("Invalid boolean".into()));
        }

        // Now actually consume the characters
        word.clear();
        while let Some(ch) = self.peek() {
            if ch.is_alphabetic() {
                word.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        match word.as_str() {
            "true" => Ok(Value::Bool(true)),
            "false" => Ok(Value::Bool(false)),
            partial if self.allow_incomplete => {
                if "true".starts_with(partial) {
                    Ok(Value::Bool(true))
                } else if "false".starts_with(partial) {
                    Ok(Value::Bool(false))
                } else {
                    Err(ToolParserError::ParsingFailed("Invalid boolean".into()))
                }
            }
            _ => Err(ToolParserError::ParsingFailed("Invalid boolean".into())),
        }
    }

    fn parse_null(&mut self) -> ToolParserResult<Value> {
        let mut word = String::new();

        // Peek at upcoming characters to validate it looks like "null"
        let mut temp_chars = self.chars.clone();
        while let Some(&ch) = temp_chars.peek() {
            if ch.is_alphabetic() && word.len() < 4 {
                // "null" is 4 chars
                word.push(ch);
                temp_chars.next();
            } else {
                break;
            }
        }

        // Check if it's a valid null prefix
        let is_valid = word == "null" || (self.allow_incomplete && "null".starts_with(&word));

        if !is_valid {
            return Err(ToolParserError::ParsingFailed("Invalid null".into()));
        }

        // Now actually consume the characters
        word.clear();
        while let Some(ch) = self.peek() {
            if ch.is_alphabetic() {
                word.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        if word == "null" || (self.allow_incomplete && "null".starts_with(&word)) {
            Ok(Value::Null)
        } else {
            Err(ToolParserError::ParsingFailed("Invalid null".into()))
        }
    }
}

/// Utility function to check if a string contains complete JSON
pub fn is_complete_json(input: &str) -> bool {
    serde_json::from_str::<Value>(input).is_ok()
}

/// Utility function to find common prefix between two strings
pub fn find_common_prefix(s1: &str, s2: &str) -> usize {
    s1.chars()
        .zip(s2.chars())
        .take_while(|(a, b)| a == b)
        .count()
}

/// Utility function to compute diff between old and new strings
pub fn compute_diff(old: &str, new: &str) -> String {
    let common_len = find_common_prefix(old, new);
    // Convert character count to byte offset
    new.chars().skip(common_len).collect()
}
