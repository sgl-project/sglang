//! Python's `jinja_template_utils.detect_jinja_template_content_format`.

use minijinja::machinery::{ast, parse};

/// `Some(true)` is OpenAI content parts, `Some(false)` is strings. A syntax
/// extension minijinja cannot inspect is left to the host's fallback path.
pub(super) fn takes_content_parts(template: &str) -> Option<bool> {
    if ["image", "audio", "video", "vision"]
        .iter()
        .any(|keyword| template.contains(keyword))
    {
        return Some(true);
    }
    let tree = parse(
        template,
        "chat_template",
        Default::default(),
        Default::default(),
    )
    .ok()?;
    Some(has_content_loop(&tree))
}

fn has_content_loop(stmt: &ast::Stmt<'_>) -> bool {
    use ast::Stmt;
    let any = |body: &[Stmt<'_>]| body.iter().any(has_content_loop);
    match stmt {
        Stmt::Template(node) => any(&node.children),
        Stmt::ForLoop(node) => {
            is_content_access(&node.iter) || any(&node.body) || any(&node.else_body)
        }
        Stmt::IfCond(node) => any(&node.true_body) || any(&node.false_body),
        Stmt::WithBlock(node) => any(&node.body),
        Stmt::SetBlock(node) => any(&node.body),
        Stmt::AutoEscape(node) => any(&node.body),
        Stmt::FilterBlock(node) => any(&node.body),
        Stmt::Block(node) => any(&node.body),
        Stmt::Macro(node) => any(&node.body),
        Stmt::CallBlock(node) => any(&node.macro_decl.body),
        _ => false,
    }
}

/// Python recognizes `message.content`, `msg.content`, and `m.content`, with
/// item access, filters, tests, and slices around the loop iterable.
fn is_content_access(expr: &ast::Expr<'_>) -> bool {
    use ast::Expr;
    let is_message = |expr: &Expr<'_>| matches!(expr, Expr::Var(var) if ["message", "msg", "m"].contains(&var.id));
    match expr {
        Expr::Filter(node) => node.expr.as_ref().is_some_and(is_content_access),
        Expr::Test(node) => is_content_access(&node.expr),
        Expr::Slice(node) => is_content_access(&node.expr),
        Expr::GetAttr(node) => node.name == "content" && is_message(&node.expr),
        Expr::GetItem(node) => {
            is_message(&node.expr)
                && matches!(&node.subscript_expr, Expr::Const(key) if key.value.as_str() == Some("content"))
        }
        _ => false,
    }
}
