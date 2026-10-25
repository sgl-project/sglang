//! The images a chat request carries, in the order SGLang's Python path collects them.

use serde_json::Value;

/// The source (URL, `data:` URL, file path or bare base64) of every image part
/// of `messages`, in message order then part order, any role. Mirrors
/// `parser/jinja_template_utils.py::process_content_for_template_format` (the
/// "openai" content format), which builds `image_data` from the same walk.
pub fn extract_images(messages: &Value) -> Vec<String> {
    let parts = (messages.as_array().into_iter().flatten())
        .filter_map(|message| message.get("content")?.as_array())
        .flatten();
    parts.filter_map(image_url).map(str::to_owned).collect()
}

/// `image_url` is an object with `url`, or (the flat form) the URL itself.
fn image_url(part: &Value) -> Option<&str> {
    if !matches!(part.get("type")?.as_str()?, "image_url" | "input_image") {
        return None;
    }
    match &part["image_url"] {
        Value::String(url) => Some(url),
        object => object["url"].as_str(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn collects_images_in_message_then_part_order_for_any_role() {
        let messages = json!([
            {"role": "system", "content": [{"type": "image_url", "image_url": {"url": "a"}}]},
            {"role": "user", "content": "text only"},
            {"role": "user", "content": [
                {"type": "text", "text": "hi"},
                {"type": "input_image", "image_url": "b"},
                {"type": "video_url", "video_url": {"url": "c"}},
                {"type": "image_url", "image_url": {"url": "d", "detail": "high"}},
            ]},
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "e"}}]},
        ]);
        assert_eq!(extract_images(&messages), ["a", "b", "d", "e"]);
        assert!(extract_images(&json!([{"role": "user", "content": "x"}])).is_empty());
    }
}
