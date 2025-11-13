//! Chat template support for tokenizers using Jinja2 templates
//!
//! This module provides functionality to apply chat templates to messages,
//! similar to HuggingFace transformers' apply_chat_template method.

use std::{collections::HashMap, fs};

use anyhow::{anyhow, Result};
use minijinja::{
    context,
    machinery::{
        ast::{Expr, Stmt},
        parse, WhitespaceConfig,
    },
    syntax::SyntaxConfig,
    Environment, Value,
};
use serde_json;

/// Chat template content format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatTemplateContentFormat {
    /// Content is a simple string
    #[default]
    String,
    /// Content is a list of structured parts (OpenAI format)
    OpenAI,
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

/// Flags tracking which OpenAI-style patterns we've seen
#[derive(Default, Debug, Clone, Copy)]
struct Flags {
    saw_iteration: bool,
    saw_structure: bool,
    saw_assignment: bool,
    saw_macro: bool,
}

impl Flags {
    fn any(self) -> bool {
        self.saw_iteration || self.saw_structure || self.saw_assignment || self.saw_macro
    }
}

/// Single-pass AST detector with scope tracking
struct Detector<'a> {
    ast: &'a Stmt<'a>,
    /// Message loop vars currently in scope (e.g., `message`, `m`, `msg`)
    scope: std::collections::VecDeque<String>,
    scope_set: std::collections::HashSet<String>,
    flags: Flags,
}

impl<'a> Detector<'a> {
    fn new(ast: &'a Stmt<'a>) -> Self {
        Self {
            ast,
            scope: std::collections::VecDeque::new(),
            scope_set: std::collections::HashSet::new(),
            flags: Flags::default(),
        }
    }

    fn run(mut self) -> Flags {
        self.walk_stmt(self.ast);
        self.flags
    }

    fn push_scope(&mut self, var: String) {
        self.scope.push_back(var.clone());
        self.scope_set.insert(var);
    }

    fn pop_scope(&mut self) {
        if let Some(v) = self.scope.pop_back() {
            self.scope_set.remove(&v);
        }
    }

    fn is_var_access(expr: &Expr, varname: &str) -> bool {
        matches!(expr, Expr::Var(v) if v.id == varname)
    }

    fn is_const_str(expr: &Expr, value: &str) -> bool {
        matches!(expr, Expr::Const(c) if c.value.as_str() == Some(value))
    }

    fn is_numeric_const(expr: &Expr) -> bool {
        matches!(expr, Expr::Const(c) if c.value.is_number())
    }

    /// Check if expr is varname.content or varname["content"]
    fn is_var_dot_content(expr: &Expr, varname: &str) -> bool {
        match expr {
            Expr::GetAttr(g) => Self::is_var_access(&g.expr, varname) && g.name == "content",
            Expr::GetItem(g) => {
                Self::is_var_access(&g.expr, varname)
                    && Self::is_const_str(&g.subscript_expr, "content")
            }
            // Unwrap filters/tests that just wrap the same expr
            Expr::Filter(f) => f
                .expr
                .as_ref()
                .is_some_and(|e| Self::is_var_dot_content(e, varname)),
            Expr::Test(t) => Self::is_var_dot_content(&t.expr, varname),
            _ => false,
        }
    }

    /// Check if expr accesses .content on any variable in our scope, or any descendant of it.
    fn is_any_scope_var_content(&self, expr: &Expr) -> bool {
        let mut current_expr = expr;
        loop {
            // Check if current level matches <scopeVar>.content
            if self
                .scope_set
                .iter()
                .any(|v| Self::is_var_dot_content(current_expr, v))
            {
                return true;
            }
            // Walk up the expression tree
            match current_expr {
                Expr::GetAttr(g) => current_expr = &g.expr,
                Expr::GetItem(g) => current_expr = &g.expr,
                _ => return false,
            }
        }
    }

    fn walk_stmt(&mut self, stmt: &Stmt) {
        // Early exit if we've already detected an OpenAI pattern
        if self.flags.any() {
            return;
        }

        match stmt {
            Stmt::Template(t) => {
                for ch in &t.children {
                    self.walk_stmt(ch);
                }
            }
            // {% for message in messages %}
            Stmt::ForLoop(fl) => {
                // Detect "for X in messages" → push X into scope
                if let Expr::Var(iter) = &fl.iter {
                    if iter.id == "messages" {
                        if let Expr::Var(target) = &fl.target {
                            self.push_scope(target.id.to_string());
                        }
                    }
                }

                // Also detect "for ... in message.content" or "for ... in content"
                // - Iterating directly over <scopeVar>.content => OpenAI style
                if self.is_any_scope_var_content(&fl.iter) {
                    self.flags.saw_iteration = true;
                }
                // - Iterating over a local var named "content"
                if matches!(&fl.iter, Expr::Var(v) if v.id == "content") {
                    self.flags.saw_iteration = true;
                }

                for b in &fl.body {
                    self.walk_stmt(b);
                }

                // Pop scope if we pushed it
                if let Expr::Var(iter) = &fl.iter {
                    if iter.id == "messages" && matches!(&fl.target, Expr::Var(_)) {
                        self.pop_scope();
                    }
                }
            }
            Stmt::IfCond(ic) => {
                self.inspect_expr_for_structure(&ic.expr);
                for b in &ic.true_body {
                    self.walk_stmt(b);
                }
                for b in &ic.false_body {
                    self.walk_stmt(b);
                }
            }
            Stmt::EmitExpr(e) => {
                self.inspect_expr_for_structure(&e.expr);
            }
            // {% set content = message.content %}
            Stmt::Set(s) => {
                if Self::is_var_access(&s.target, "content")
                    && self.is_any_scope_var_content(&s.expr)
                {
                    self.flags.saw_assignment = true;
                }
            }
            Stmt::Macro(m) => {
                // Heuristic: macro that checks type (via `is` test) and also has any loop
                let mut has_type_check = false;
                let mut has_loop = false;
                Self::scan_macro_body(&m.body, &mut has_type_check, &mut has_loop);
                if has_type_check && has_loop {
                    self.flags.saw_macro = true;
                }
            }
            _ => {}
        }
    }

    fn inspect_expr_for_structure(&mut self, expr: &Expr) {
        if self.flags.saw_structure {
            return;
        }

        match expr {
            // content[0] or message.content[0]
            Expr::GetItem(gi) => {
                if (matches!(&gi.expr, Expr::Var(v) if v.id == "content")
                    || self.is_any_scope_var_content(&gi.expr))
                    && Self::is_numeric_const(&gi.subscript_expr)
                {
                    self.flags.saw_structure = true;
                }
            }
            // content|length or message.content|length
            Expr::Filter(f) => {
                if f.name == "length" {
                    if let Some(inner) = &f.expr {
                        // Box derefs automatically, so `&**inner` is `&Expr`
                        let inner_ref: &Expr = inner;
                        let is_content_var = matches!(inner_ref, Expr::Var(v) if v.id == "content");
                        if is_content_var || self.is_any_scope_var_content(inner_ref) {
                            self.flags.saw_structure = true;
                        }
                    }
                } else if let Some(inner) = &f.expr {
                    let inner_ref: &Expr = inner;
                    self.inspect_expr_for_structure(inner_ref);
                }
            }
            // content is sequence/iterable OR message.content is sequence/iterable
            Expr::Test(t) => {
                if t.name == "sequence" || t.name == "iterable" || t.name == "string" {
                    if matches!(&t.expr, Expr::Var(v) if v.id == "content")
                        || self.is_any_scope_var_content(&t.expr)
                    {
                        self.flags.saw_structure = true;
                    }
                } else {
                    self.inspect_expr_for_structure(&t.expr);
                }
            }
            Expr::GetAttr(g) => {
                // Keep walking; nested expressions can hide structure checks
                self.inspect_expr_for_structure(&g.expr);
            }
            // Handle binary operations like: if (message.content is string) and other_cond
            Expr::BinOp(op) => {
                self.inspect_expr_for_structure(&op.left);
                self.inspect_expr_for_structure(&op.right);
            }
            // Handle unary operations like: if not (message.content is string)
            Expr::UnaryOp(op) => {
                self.inspect_expr_for_structure(&op.expr);
            }
            _ => {}
        }
    }

    fn scan_macro_body(body: &[Stmt], has_type_check: &mut bool, has_loop: &mut bool) {
        for s in body {
            if *has_type_check && *has_loop {
                return;
            }

            match s {
                Stmt::IfCond(ic) => {
                    if matches!(&ic.expr, Expr::Test(_)) {
                        *has_type_check = true;
                    }
                    Self::scan_macro_body(&ic.true_body, has_type_check, has_loop);
                    Self::scan_macro_body(&ic.false_body, has_type_check, has_loop);
                }
                Stmt::ForLoop(fl) => {
                    *has_loop = true;
                    Self::scan_macro_body(&fl.body, has_type_check, has_loop);
                }
                Stmt::Template(t) => {
                    Self::scan_macro_body(&t.children, has_type_check, has_loop);
                }
                _ => {}
            }
        }
    }
}

/// AST-based detection using minijinja's unstable machinery
/// Single-pass detector with scope tracking
fn detect_format_with_ast(template: &str) -> Option<ChatTemplateContentFormat> {
    let ast = match parse(
        template,
        "template",
        SyntaxConfig {},
        WhitespaceConfig::default(),
    ) {
        Ok(ast) => ast,
        Err(_) => return Some(ChatTemplateContentFormat::String),
    };

    let flags = Detector::new(&ast).run();
    Some(if flags.any() {
        ChatTemplateContentFormat::OpenAI
    } else {
        ChatTemplateContentFormat::String
    })
}

/// Parameters for chat template application
#[derive(Default)]
pub struct ChatTemplateParams<'a> {
    pub add_generation_prompt: bool,
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
        let mut env = Environment::new();

        // Register the template
        env.add_template("chat", &self.template)
            .map_err(|e| anyhow!("Failed to add template: {}", e))?;

        // Enable Python method compatibility (e.g., str.startswith, str.endswith)
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

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
