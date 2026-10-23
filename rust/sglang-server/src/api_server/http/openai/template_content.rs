//! Match Python's Jinja content-format selection before message normalization.

use minijinja::machinery::{
    Token,
    ast::{Expr, Stmt},
    parse, tokenize,
};

// The existing minijinja dependency exposes its AST only through the unstable
// machinery feature. Keep that dependency here; classification follows Python's
// jinja_template_utils.detect_jinja_template_content_format, not render probes.
pub(crate) fn requires_content_arrays(source: &str) -> bool {
    if ["image", "audio", "video", "vision"]
        .iter()
        .any(|keyword| source.contains(keyword))
    {
        return true;
    }
    let Some(source) = normalize_generation_tags(source) else {
        return false;
    };
    parse(
        &source,
        "chat_template",
        Default::default(),
        Default::default(),
    )
    .is_ok_and(|template| contains_content_loop(&template))
}

// Renderer 7.1.1 maps executable generation/endgeneration tags to with/endwith,
// including whitespace-control forms. Token spans preserve strings, comments,
// raw text, and the original +/- markers without another Jinja lexer.
fn normalize_generation_tags(source: &str) -> Option<String> {
    let mut out = String::with_capacity(source.len());
    let mut copied = 0;
    let mut block_start = false;
    let mut replacement = None;
    for token in tokenize(source, false, Default::default(), Default::default()) {
        let (token, span) = token.ok()?;
        if matches!(token, Token::BlockEnd) {
            if let Some((start, end, name)) = replacement.take() {
                out.push_str(&source[copied..start]);
                out.push_str(name);
                copied = end;
            }
        } else {
            replacement = match (&token, block_start) {
                (Token::Ident("generation"), true) => {
                    Some((span.start_offset as usize, span.end_offset as usize, "with"))
                }
                (Token::Ident("endgeneration"), true) => Some((
                    span.start_offset as usize,
                    span.end_offset as usize,
                    "endwith",
                )),
                _ => None,
            };
        }
        block_start = matches!(token, Token::BlockStart);
    }
    out.push_str(&source[copied..]);
    Some(out)
}

fn content_iterator(expr: &Expr<'_>) -> bool {
    match expr {
        Expr::Filter(filter) => filter.expr.as_ref().is_some_and(content_iterator),
        Expr::Test(test) => content_iterator(&test.expr),
        Expr::Slice(slice) => content_iterator(&slice.expr),
        Expr::GetAttr(attr) => attr.name == "content" && message_variable(&attr.expr),
        Expr::GetItem(item) => {
            matches!(&item.subscript_expr, Expr::Const(key) if key.value.as_str() == Some("content"))
                && message_variable(&item.expr)
        }
        _ => false,
    }
}

fn message_variable(expr: &Expr<'_>) -> bool {
    matches!(expr, Expr::Var(var) if matches!(var.id, "message" | "msg" | "m"))
}

fn contains_content_loop(stmt: &Stmt<'_>) -> bool {
    match stmt {
        Stmt::Template(template) => template.children.iter().any(contains_content_loop),
        Stmt::ForLoop(node) => {
            content_iterator(&node.iter)
                || node
                    .body
                    .iter()
                    .chain(&node.else_body)
                    .any(contains_content_loop)
        }
        Stmt::IfCond(node) => node
            .true_body
            .iter()
            .chain(&node.false_body)
            .any(contains_content_loop),
        Stmt::WithBlock(node) => node.body.iter().any(contains_content_loop),
        Stmt::SetBlock(node) => node.body.iter().any(contains_content_loop),
        Stmt::AutoEscape(node) => node.body.iter().any(contains_content_loop),
        Stmt::FilterBlock(node) => node.body.iter().any(contains_content_loop),
        Stmt::Block(node) => node.body.iter().any(contains_content_loop),
        Stmt::Macro(node) => node.body.iter().any(contains_content_loop),
        Stmt::CallBlock(node) => node.macro_decl.body.iter().any(contains_content_loop),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::requires_content_arrays;

    #[test]
    fn content_loop_uses_openai_parts() {
        assert!(requires_content_arrays(
            "{% for part in message.content %}{{ part.text }}{% endfor %}"
        ));
    }
    #[test]
    fn filtered_and_sliced_content_preserves_parts() {
        for source in [
            "{% for part in msg['content'] | list %}{{ part.text }}{% endfor %}",
            "{% for part in m.content[1:] %}{{ part.text }}{% endfor %}",
            "{% for part in message.content | select('mapping') | list %}{{ part.text }}{% endfor %}",
        ] {
            assert!(requires_content_arrays(source), "{source}");
        }
    }

    #[test]
    fn content_loops_are_found_in_nested_blocks() {
        for source in [
            "{% if false %}{% for part in message.content %}{{ part.text }}{% endfor %}{% endif %}",
            "{% if true %}string{% else %}{% for part in message.content %}{{ part.text }}{% endfor %}{% endif %}",
            "{% macro render(message) %}{% for part in message.content %}{{ part.text }}{% endfor %}{% endmacro %}",
            "{% call wrapper() %}{% for part in message.content %}{{ part.text }}{% endfor %}{% endcall %}",
            "{% with m = message %}{% for part in m.content %}{{ part.text }}{% endfor %}{% endwith %}",
            "{% set captured %}{% for part in message.content %}{{ part.text }}{% endfor %}{% endset %}",
            "{% filter trim %}{% for part in message.content %}{{ part.text }}{% endfor %}{% endfilter %}",
            "{% autoescape false %}{% for part in message.content %}{{ part.text }}{% endfor %}{% endautoescape %}",
            "{% block body %}{% for part in message.content %}{{ part.text }}{% endfor %}{% endblock %}",
            "{% for x in [] %}{% else %}{% for part in message.content %}{{ part.text }}{% endfor %}{% endfor %}",
            "{% if message.content is string %}{{ message.content }}{% else %}{% for part in message.content %}{{ part.text }}{% endfor %}{% endif %}",
        ] {
            assert!(requires_content_arrays(source), "{source}");
        }
    }

    #[test]
    fn media_keywords_select_parts_before_template_parsing() {
        for source in ["image", "audio", "video", "vision", "{% broken image"] {
            assert!(requires_content_arrays(source), "{source}");
        }
    }

    #[test]
    fn string_templates_do_not_confuse_serialized_parts_with_text() {
        for source in [
            "{{ message.content }}",
            "{{ [{'type': 'text', 'text': 'template_test'}] }}",
            "{% for message in messages %}{{ message.content }}{% endfor %}",
            "{% for part in other.content %}{{ part.text }}{% endfor %}",
            "{% for part in message.other %}{{ part.text }}{% endfor %}",
            "{% for part in message.content[0] %}{{ part }}{% endfor %}",
            "{# {% for part in message.content %} #}",
            "{% raw %}{% for part in message.content %}{% endraw %}",
            "{% broken",
            "VISION",
        ] {
            assert!(!requires_content_arrays(source), "{source}");
        }
    }
    #[test]
    fn generation_tags_admitted_by_renderer_preserve_classification() {
        for source in [
            "{% generation %}{% for part in message.content %}{{ part.text }}{% endfor %}{% endgeneration %}",
            "{%- generation -%}{% for part in message.content %}{{ part.text }}{% endfor %}{%- endgeneration -%}",
            "{%+ generation +%}{% for part in message.content %}{{ part.text }}{% endfor %}{%+ endgeneration +%}",
        ] {
            assert!(requires_content_arrays(source), "{source}");
        }
    }
}
