//! The images a chat request carries, in the order SGLang's Python path collects them.

use serde_json::Value;

/// The source (URL, `data:` URL, file path or bare base64) of every image part
/// of `messages`, in message order then part order, any role. Mirrors
/// `parser/jinja_template_utils.py::process_content_for_template_format` (the
/// "openai" content format), which builds `image_data` from the same walk and
/// fails on an image part without a URL.
pub fn extract_images(messages: &Value) -> Result<Vec<String>, String> {
    let parts = (messages.as_array().into_iter().flatten())
        .filter_map(|message| message.get("content")?.as_array())
        .flatten()
        .filter(|part| matches!(part["type"].as_str(), Some("image_url" | "input_image")));
    parts
        .map(|part| {
            image_url(part)
                .map(str::to_owned)
                .ok_or_else(|| "image content part has no image_url url".to_owned())
        })
        .collect()
}

/// `image_url` is an object with `url`, or (the flat form) the URL itself.
fn image_url(part: &Value) -> Option<&str> {
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
        assert_eq!(extract_images(&messages).unwrap(), ["a", "b", "d", "e"]);
        assert!(
            extract_images(&json!([{"role": "user", "content": "x"}]))
                .unwrap()
                .is_empty()
        );
    }

    /// An image part without a URL (say, only a `uuid`) fails the request, as
    /// in Python, rather than leaving its placeholder without an image.
    #[test]
    fn refuses_image_part_without_url() {
        for part in [
            json!({"type": "image_url", "image_url": null, "uuid": "cached"}),
            json!({"type": "image_url", "image_url": {"detail": "high"}}),
            json!({"type": "input_image"}),
        ] {
            let messages = json!([{"role": "user", "content": [part]}]);
            assert!(extract_images(&messages).is_err(), "{messages}");
        }
    }
}
