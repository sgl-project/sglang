//! The messages SGLang hands a Jinja template, passed through Dynamo's formatter
//! unchanged.

use std::collections::HashMap;

use dynamo_renderer::OAIChatLikeRequest;
use serde_json::Value;

/// Dynamo's HF formatter rewrites message contents (joining text parts or
/// wrapping strings in lists). SGLang has already normalized them, so restore
/// that context immediately before evaluating the template. Generic callers
/// without this private context keep Dynamo's normal rendering path. Returns the
/// context key: `name`, lengthened until the template does not mention it.
pub(crate) fn preserve_messages(config: &mut Value, name: &str) -> String {
    let source = config["chat_template"].to_string();
    let mut context = name.to_owned();
    while source.contains(&context) {
        context.push('_');
    }
    let wrap = |template: &mut Value| {
        if let Some(source) = template.as_str() {
            *template = format!(
                "{{% if {context} is defined %}}{{% set messages = {context} %}}{{% endif +%}}{source}"
            )
            .into();
        }
    };
    match &mut config["chat_template"] {
        template @ Value::String(_) => wrap(template),
        Value::Array(templates) => {
            for templates in templates.iter_mut().filter_map(Value::as_object_mut) {
                templates.values_mut().for_each(wrap);
            }
        }
        _ => {}
    }
    context
}

/// The template context SGLang builds, as the request Dynamo renders: the
/// messages travel in `args` under the [`preserve_messages`] key, so Dynamo's
/// own `messages`, which the wrapped template replaces, stay empty.
pub(crate) struct PreservedRequest {
    pub tools: Option<Vec<Value>>,
    pub args: HashMap<String, Value>,
}

impl OAIChatLikeRequest for PreservedRequest {
    fn model(&self) -> String {
        String::new()
    }

    fn messages(&self) -> minijinja::Value {
        minijinja::Value::from(Vec::<minijinja::Value>::new())
    }

    fn tools(&self) -> Option<minijinja::Value> {
        self.tools.as_ref().map(minijinja::Value::from_serialize)
    }

    fn should_add_generation_prompt(&self) -> bool {
        true
    }

    fn chat_template_args(&self) -> Option<&HashMap<String, Value>> {
        Some(&self.args)
    }
}
