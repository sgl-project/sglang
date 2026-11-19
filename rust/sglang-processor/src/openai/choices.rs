//! `n` choices of a lowered request as a batch, for an engine that samples
//! one sequence per request.

use serde_json::Value;

/// A prompt's fields whose list form is repeated with it across `n` choices.
pub const PROMPT_LIST_FIELDS: [&str; 6] = [
    "bootstrap_host",
    "bootstrap_port",
    "bootstrap_room",
    "extra_key",
    "cache_salt",
    "lora_path",
];

/// The `n` of a lowered `/generate` body.
pub fn choices_per_prompt(generate: &Value) -> usize {
    let n = generate["sampling_params"]["n"].as_u64().unwrap_or(1);
    usize::try_from(n).unwrap_or(usize::MAX)
}

/// `value` as its prompts: a batch's items, or the one prompt.
fn prompts(key: &str, value: Value) -> Vec<Value> {
    match value {
        Value::Array(items) if key == "text" || items.first().is_some_and(Value::is_array) => items,
        prompt => vec![prompt],
    }
}

/// Make `n` choices `n` copies of each prompt, in prompt order: the indexed
/// frames Python's `n` streams. The host bounds `n` before the copy.
pub fn copy_prompt_per_choice(generate: &mut Value) {
    let n = choices_per_prompt(generate);
    if n <= 1 {
        return;
    }
    let key = match generate["input_ids"] {
        Value::Null => "text",
        _ => "input_ids",
    };
    generate["sampling_params"]["n"] = 1.into();
    let copy = |items: Vec<Value>| -> Vec<Value> {
        let copies = items.into_iter().flat_map(|item| vec![item; n]);
        copies.collect()
    };
    generate[key] = Value::Array(copy(prompts(key, generate[key].take())));
    // Python's parallel sampling repeats the prompt's list-form fields with it.
    for key in PROMPT_LIST_FIELDS {
        if let Some(Value::Array(items)) = generate.get_mut(key) {
            *items = copy(std::mem::take(items));
        }
    }
    // A rid names its prompt, whose choices are that rid numbered.
    generate["rid"] = match generate["rid"].take() {
        Value::Array(mut rids) if rids.len() == 1 => rids.remove(0),
        Value::Array(rids) => {
            let numbered = rids.iter().flat_map(|rid| {
                let rid = rid.as_str().unwrap_or_default();
                (0..n).map(move |i| Value::from(format!("{rid}_{i}")))
            });
            Value::Array(numbered.collect())
        }
        rid => rid,
    };
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn choices_copy_each_prompt_in_prompt_order() {
        let mut one = json!({"input_ids": [1, 2], "rid": ["r"], "bootstrap_room": [5],
            "sampling_params": {"n": 2}});
        copy_prompt_per_choice(&mut one);
        assert_eq!(one["input_ids"], json!([[1, 2], [1, 2]]));
        assert_eq!(one["bootstrap_room"], json!([5, 5]));
        assert_eq!(one["rid"], "r");
        assert_eq!(one["sampling_params"]["n"], 1);

        let mut batch = json!({"text": ["a", "b"], "rid": ["x", "y"],
            "sampling_params": {"n": 2}});
        copy_prompt_per_choice(&mut batch);
        assert_eq!(batch["text"], json!(["a", "a", "b", "b"]));
        assert_eq!(batch["rid"], json!(["x_0", "x_1", "y_0", "y_1"]));

        let mut single = json!({"text": "a", "sampling_params": {"n": 1}});
        copy_prompt_per_choice(&mut single);
        assert_eq!(single["text"], "a");
    }
}
