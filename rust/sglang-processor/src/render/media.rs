//! The media a chat request carries, in the order SGLang's Python path collects it.

use serde_json::Value;

/// A media modality a chat content part can carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaModality {
    Image,
    Video,
    Audio,
}

/// One media part of a chat request: its modality and its source string
/// (URL, `data:` URL, file path or bare base64), as the client sent it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaRef {
    pub modality: MediaModality,
    pub url: String,
}

/// Every media part of `messages`, in message order then part order, any role.
/// Mirrors `parser/jinja_template_utils.py::process_content_for_template_format`
/// (the "openai" content format), which builds `image_data`, `video_data` and
/// `audio_data` from the same walk.
pub fn extract_media(messages: &Value) -> Vec<MediaRef> {
    let parts = (messages.as_array().into_iter().flatten())
        .filter_map(|message| message.get("content")?.as_array())
        .flatten();
    parts.filter_map(media_ref).collect()
}

fn media_ref(part: &Value) -> Option<MediaRef> {
    let (modality, url) = match part.get("type")?.as_str()? {
        "image_url" | "input_image" => (MediaModality::Image, image_url(part)?),
        "video_url" => (MediaModality::Video, part["video_url"]["url"].as_str()?),
        "audio_url" => (MediaModality::Audio, part["audio_url"]["url"].as_str()?),
        _ => return None,
    };
    Some(MediaRef {
        modality,
        url: url.to_owned(),
    })
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
    fn collects_media_in_message_then_part_order_for_any_role() {
        let messages = json!([
            {"role": "system", "content": [{"type": "image_url", "image_url": {"url": "a"}}]},
            {"role": "user", "content": "text only"},
            {"role": "user", "content": [
                {"type": "text", "text": "hi"},
                {"type": "input_image", "image_url": "b"},
                {"type": "video_url", "video_url": {"url": "c"}},
                {"type": "image_url", "image_url": {"url": "d", "detail": "high"}},
            ]},
            {"role": "tool", "content": [{"type": "audio_url", "audio_url": {"url": "e"}}]},
        ]);
        let refs = extract_media(&messages);
        let refs: Vec<(MediaModality, &str)> =
            refs.iter().map(|r| (r.modality, r.url.as_str())).collect();
        assert_eq!(
            refs,
            [
                (MediaModality::Image, "a"),
                (MediaModality::Image, "b"),
                (MediaModality::Video, "c"),
                (MediaModality::Image, "d"),
                (MediaModality::Audio, "e"),
            ]
        );
        assert!(extract_media(&json!([{"role": "user", "content": "x"}])).is_empty());
    }
}
