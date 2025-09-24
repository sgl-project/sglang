//! Chat template support for tokenizers using Jinja2 templates
//!
//! This module provides functionality to apply chat templates to messages,
//! similar to HuggingFace transformers' apply_chat_template method.

use anyhow::{anyhow, Result};
use minijinja::{context, machinery, Environment, Value};
use serde_json;
use std::collections::HashMap;

/// Chat template content format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplateContentFormat {
    /// Content is a simple string
    String,
    /// Content is a list of structured parts (OpenAI format)
    OpenAI,
}

impl Default for ChatTemplateContentFormat {
    fn default() -> Self {
        Self::String
    }
}

impl std::fmt::Display for ChatTemplateContentFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String => write!(f, "string"),
            Self::OpenAI => write!(f, "openai"),
        }
    }
}

/// Detect the content format expected by a Jinja2 chat template
///
/// This implements the same detection logic as SGLang's detect_jinja_template_content_format
/// which uses AST parsing to look for content iteration patterns.
///
/// Returns:
/// - ChatTemplateContentFormat::OpenAI if template expects structured content (list of parts)
/// - ChatTemplateContentFormat::String if template expects simple string content
pub fn detect_chat_template_content_format(template: &str) -> ChatTemplateContentFormat {
    // Use AST-based detection (enabled by default)
    if let Some(format) = detect_format_with_ast(template) {
        return format;
    }

    // Default to string format if AST parsing fails
    ChatTemplateContentFormat::String
}

/// AST-based detection using minijinja's unstable machinery
/// This implements the exact same logic as SGLang's _is_var_or_elems_access functions
fn detect_format_with_ast(template: &str) -> Option<ChatTemplateContentFormat> {
    use minijinja::machinery::{parse, WhitespaceConfig};
    use minijinja::syntax::SyntaxConfig;

    // Parse the template into AST
    let ast = match parse(
        template,
        "template",
        SyntaxConfig {},
        WhitespaceConfig::default(),
    ) {
        Ok(ast) => ast,
        Err(_) => return Some(ChatTemplateContentFormat::String),
    };

    // Traverse AST looking for patterns that indicate OpenAI format
    let has_iteration = find_content_iteration_in_ast(&ast);
    let has_structure_checks = find_content_structure_checks_in_ast(&ast);
    let has_assignment_patterns = find_variable_assignment_patterns_in_ast(&ast);

    if has_iteration || has_structure_checks || has_assignment_patterns {
        Some(ChatTemplateContentFormat::OpenAI)
    } else {
        Some(ChatTemplateContentFormat::String)
    }
}

/// Find content iteration patterns in AST
/// Implements the same logic as SGLang's AST traversal
fn find_content_iteration_in_ast(ast: &machinery::ast::Stmt) -> bool {
    use machinery::ast::Stmt;

    match ast {
        Stmt::Template(template) => {
            // Recursively check all children
            template
                .children
                .iter()
                .any(|child| find_content_iteration_in_ast(child))
        }
        Stmt::ForLoop(for_loop) => {
            // Check if this for-loop iterates over message content
            is_var_or_elems_access(&for_loop.iter, "message", "content") ||
            is_var_or_elems_access(&for_loop.iter, "msg", "content") ||
            is_var_or_elems_access(&for_loop.iter, "m", "content") ||
            // Also check the body for nested loops
            for_loop.body.iter().any(|stmt| find_content_iteration_in_ast(stmt))
        }
        Stmt::IfCond(if_cond) => {
            // Check true and false branches
            if_cond
                .true_body
                .iter()
                .any(|stmt| find_content_iteration_in_ast(stmt))
                || if_cond
                    .false_body
                    .iter()
                    .any(|stmt| find_content_iteration_in_ast(stmt))
        }
        _ => false, // Other statement types don't contain loops
    }
}

/// Check if expression accesses varname['key'] or varname.key
/// Implements SGLang's _is_var_or_elems_access logic using actual AST nodes
fn is_var_or_elems_access(expr: &machinery::ast::Expr, varname: &str, key: &str) -> bool {
    use machinery::ast::Expr;

    match expr {
        // Check for attribute access: varname.key
        Expr::GetAttr(getattr) => is_var_access(&getattr.expr, varname) && getattr.name == key,
        // Check for item access: varname['key'] or varname["key"]
        Expr::GetItem(getitem) => {
            is_var_access(&getitem.expr, varname) && is_const_string(&getitem.subscript_expr, key)
        }
        // Handle filters and tests that might wrap the access
        Expr::Filter(filter) => {
            if let Some(ref expr) = filter.expr {
                is_var_or_elems_access(expr, varname, key)
            } else {
                false
            }
        }
        Expr::Test(test) => is_var_or_elems_access(&test.expr, varname, key),
        _ => false,
    }
}

/// Check if expression is a variable access (like {{ varname }})
/// Implements SGLang's _is_var_access logic
fn is_var_access(expr: &machinery::ast::Expr, varname: &str) -> bool {
    matches!(expr, machinery::ast::Expr::Var(var) if var.id == varname)
}

/// Check if expression is a constant string with the given value
fn is_const_string(expr: &machinery::ast::Expr, value: &str) -> bool {
    matches!(expr, machinery::ast::Expr::Const(const_expr)
        if const_expr.value.as_str() == Some(value))
}

/// Find content structure checks in AST (like content[0], content|length)
fn find_content_structure_checks_in_ast(ast: &machinery::ast::Stmt) -> bool {
    use machinery::ast::Stmt;

    match ast {
        Stmt::Template(template) => template
            .children
            .iter()
            .any(|child| find_content_structure_checks_in_ast(child)),
        Stmt::ForLoop(for_loop) => for_loop
            .body
            .iter()
            .any(|stmt| find_content_structure_checks_in_ast(stmt)),
        Stmt::IfCond(if_cond) => {
            // Check if condition has content structure checks
            has_content_structure_check_expr(&if_cond.expr)
                || if_cond
                    .true_body
                    .iter()
                    .any(|stmt| find_content_structure_checks_in_ast(stmt))
                || if_cond
                    .false_body
                    .iter()
                    .any(|stmt| find_content_structure_checks_in_ast(stmt))
        }
        Stmt::EmitExpr(expr) => has_content_structure_check_expr(&expr.expr),
        _ => false,
    }
}

/// Find variable assignment patterns like set content = message['content']
fn find_variable_assignment_patterns_in_ast(ast: &machinery::ast::Stmt) -> bool {
    use machinery::ast::Stmt;

    match ast {
        Stmt::Template(template) => template
            .children
            .iter()
            .any(|child| find_variable_assignment_patterns_in_ast(child)),
        Stmt::ForLoop(for_loop) => {
            // Check if this for-loop body contains both assignment and iteration
            let has_assignment = for_loop
                .body
                .iter()
                .any(|stmt| is_content_assignment_stmt(stmt));
            let has_iteration = for_loop.body.iter().any(|stmt| {
                is_content_variable_iteration(stmt)
                    || matches!(stmt, Stmt::IfCond(if_cond) if
                        if_cond.true_body.iter().any(|s| is_content_variable_iteration(s)) ||
                        if_cond.false_body.iter().any(|s| is_content_variable_iteration(s))
                    )
            });

            (has_assignment && has_iteration)
                || for_loop
                    .body
                    .iter()
                    .any(|stmt| find_variable_assignment_patterns_in_ast(stmt))
        }
        Stmt::IfCond(if_cond) => {
            if_cond
                .true_body
                .iter()
                .any(|stmt| find_variable_assignment_patterns_in_ast(stmt))
                || if_cond
                    .false_body
                    .iter()
                    .any(|stmt| find_variable_assignment_patterns_in_ast(stmt))
        }
        _ => false,
    }
}

/// Check if expression has content structure checks (index access, length, etc.)
fn has_content_structure_check_expr(expr: &machinery::ast::Expr) -> bool {
    use machinery::ast::Expr;

    match expr {
        // Check for content[0] - index access
        Expr::GetItem(getitem) => {
            is_content_access(&getitem.expr) && is_numeric_constant(&getitem.subscript_expr)
        }
        // Check for content|length - filter with length
        Expr::Filter(filter) => {
            if let Some(ref filter_expr) = filter.expr {
                is_content_access(filter_expr) && filter.name == "length"
            } else {
                false
            }
        }
        // Check for content is sequence/iterable
        Expr::Test(test) => {
            is_content_access(&test.expr) && (test.name == "sequence" || test.name == "iterable")
        }
        _ => false,
    }
}

/// Check if statement assigns message content to a variable
fn is_content_assignment_stmt(stmt: &machinery::ast::Stmt) -> bool {
    use machinery::ast::Stmt;

    match stmt {
        Stmt::Set(set_stmt) => {
            // Check if this is setting content = message['content']
            is_var_access(&set_stmt.target, "content")
                && is_var_or_elems_access(&set_stmt.expr, "message", "content")
        }
        _ => false,
    }
}

/// Check if statement iterates over content variable
fn is_content_variable_iteration(stmt: &machinery::ast::Stmt) -> bool {
    use machinery::ast::{Expr, Stmt};

    match stmt {
        Stmt::ForLoop(for_loop) => {
            // Check if iterating over a variable named "content"
            matches!(for_loop.iter, Expr::Var(ref var) if var.id == "content")
        }
        _ => false,
    }
}

/// Check if expression accesses content (message.content, message['content'], etc.)
fn is_content_access(expr: &machinery::ast::Expr) -> bool {
    is_var_or_elems_access(expr, "message", "content")
        || is_var_or_elems_access(expr, "msg", "content")
        || is_var_or_elems_access(expr, "m", "content")
}

/// Check if expression is a numeric constant (for index access)
fn is_numeric_constant(expr: &machinery::ast::Expr) -> bool {
    matches!(expr, machinery::ast::Expr::Const(const_expr) if const_expr.value.is_number())
}

/// Parameters for chat template application
#[derive(Default)]
pub struct ChatTemplateParams<'a> {
    pub add_generation_prompt: bool,
    pub continue_final_message: bool,
    pub tools: Option<&'a [serde_json::Value]>,
    pub documents: Option<&'a [serde_json::Value]>,
    pub template_kwargs: Option<&'a HashMap<String, serde_json::Value>>,
}

/// Chat template processor using Jinja2 - simple wrapper like HuggingFace
pub struct ChatTemplateProcessor {
    template: String,
}

impl ChatTemplateProcessor {
    /// Create a new chat template processor
    pub fn new(template: String) -> Self {
        ChatTemplateProcessor { template }
    }

    /// Apply the chat template to a list of messages
    ///
    /// This mimics the behavior of HuggingFace's apply_chat_template method
    /// but returns the formatted string instead of token IDs.
    /// Messages should be pre-processed into the format expected by the template.
    pub fn apply_chat_template(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> Result<String> {
        // Validate incompatible options
        if params.continue_final_message && params.add_generation_prompt {
            return Err(anyhow!("continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead."));
        }
        let mut env = Environment::new();

        // Register the template
        env.add_template("chat", &self.template)
            .map_err(|e| anyhow!("Failed to add template: {}", e))?;

        // Get the template
        let tmpl = env
            .get_template("chat")
            .map_err(|e| anyhow!("Failed to get template: {}", e))?;

        // Convert messages to minijinja::Value (messages already processed by router)
        let minijinja_messages: Vec<Value> = messages.iter().map(Value::from_serialize).collect();

        let base_context = context! {
            messages => &minijinja_messages,
            add_generation_prompt => params.add_generation_prompt,
            tools => params.tools,
            documents => params.documents,
        };

        // Merge with template_kwargs if provided
        let ctx = if let Some(kwargs) = params.template_kwargs {
            context! {
                ..base_context,
                ..Value::from_serialize(kwargs)
            }
        } else {
            base_context
        };

        // Render the template
        let rendered = tmpl
            .render(&ctx)
            .map_err(|e| anyhow!("Failed to render template: {}", e))?;

        Ok(rendered)
    }
}

/// Load chat template from tokenizer config JSON
pub fn load_chat_template_from_config(config_path: &str) -> Result<Option<String>> {
    use std::fs;

    let content = fs::read_to_string(config_path)?;
    let config: serde_json::Value = serde_json::from_str(&content)?;

    // Look for chat_template in the config
    if let Some(template) = config.get("chat_template") {
        if let Some(template_str) = template.as_str() {
            return Ok(Some(template_str.to_string()));
        }
    }

    Ok(None)
}
