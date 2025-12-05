//! Utilities for extracting multimodal content from chat messages.
//!
//! This module provides functions to extract image URLs and data from
//! OpenAI-style chat completion requests.

use crate::protocols::{
    chat::{ChatMessage, MessageContent},
    common::ContentPart,
};

/// Extracted multimodal content from chat messages.
#[derive(Debug, Default)]
pub struct ExtractedMultiModal {
    /// Image URLs found in messages
    pub image_urls: Vec<String>,
    /// Image detail levels (corresponding to image_urls)
    pub image_details: Vec<Option<String>>,
    /// Base64 encoded image data (data URLs decoded)
    pub image_data: Vec<Vec<u8>>,
}

impl ExtractedMultiModal {
    /// Check if any multimodal content was found.
    pub fn is_empty(&self) -> bool {
        self.image_urls.is_empty() && self.image_data.is_empty()
    }

    /// Get total number of images (URLs + inline data).
    pub fn image_count(&self) -> usize {
        self.image_urls.len() + self.image_data.len()
    }
}

/// Extract multimodal content from chat messages.
///
/// Scans all messages for image_url content parts and extracts:
/// - HTTP/HTTPS URLs
/// - Data URLs (decoded from base64)
///
/// # Arguments
/// * `messages` - Chat messages to scan
///
/// # Returns
/// Extracted multimodal content.
pub fn extract_multimodal_from_messages(messages: &[ChatMessage]) -> ExtractedMultiModal {
    let mut result = ExtractedMultiModal::default();

    for message in messages {
        let content = match message {
            ChatMessage::User { content, .. } => Some(content),
            ChatMessage::System { content, .. } => Some(content),
            ChatMessage::Assistant { content, .. } => content.as_ref(),
            ChatMessage::Tool { content, .. } => Some(content),
            // Skip function messages - they don't contain images
            ChatMessage::Function { .. } => None,
        };

        if let Some(MessageContent::Parts(parts)) = content {
            for part in parts {
                if let ContentPart::ImageUrl { image_url } = part {
                    process_image_url(&image_url.url, &image_url.detail, &mut result);
                }
            }
        }
    }

    result
}

/// Process a single image URL, handling both HTTP URLs and data URLs.
fn process_image_url(url: &str, detail: &Option<String>, result: &mut ExtractedMultiModal) {
    if url.starts_with("data:") {
        // Data URL - extract base64 content
        if let Some(data) = decode_data_url(url) {
            result.image_data.push(data);
        }
    } else {
        // Regular URL
        result.image_urls.push(url.to_string());
        result.image_details.push(detail.clone());
    }
}

/// Decode a data URL to raw bytes.
///
/// Supports format: `data:[<mediatype>][;base64],<data>`
fn decode_data_url(url: &str) -> Option<Vec<u8>> {
    // Find the comma that separates metadata from data
    let comma_pos = url.find(',')?;
    let metadata = &url[5..comma_pos]; // Skip "data:"
    let data = &url[comma_pos + 1..];

    // Check if it's base64 encoded
    if metadata.ends_with(";base64") {
        base64::Engine::decode(&base64::engine::general_purpose::STANDARD, data).ok()
    } else {
        // URL-encoded data (less common)
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::ImageUrl;

    #[test]
    fn test_extract_image_urls() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "What's in this image?".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/image.jpg".to_string(),
                        detail: Some("high".to_string()),
                    },
                },
            ]),
            name: None,
        }];

        let extracted = extract_multimodal_from_messages(&messages);

        assert_eq!(extracted.image_urls.len(), 1);
        assert_eq!(extracted.image_urls[0], "https://example.com/image.jpg");
        assert_eq!(extracted.image_details[0], Some("high".to_string()));
    }

    #[test]
    fn test_extract_data_url() {
        // Small red pixel PNG as base64
        let png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
        let data_url = format!("data:image/png;base64,{}", png_base64);

        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: data_url,
                    detail: None,
                },
            }]),
            name: None,
        }];

        let extracted = extract_multimodal_from_messages(&messages);

        assert!(extracted.image_urls.is_empty());
        assert_eq!(extracted.image_data.len(), 1);
        // Verify it's valid PNG data (starts with PNG magic bytes)
        assert_eq!(&extracted.image_data[0][..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    #[test]
    fn test_extract_multiple_images() {
        let messages = vec![
            ChatMessage::User {
                content: MessageContent::Parts(vec![
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/image1.jpg".to_string(),
                            detail: None,
                        },
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/image2.jpg".to_string(),
                            detail: Some("low".to_string()),
                        },
                    },
                ]),
                name: None,
            },
            ChatMessage::Assistant {
                content: Some(MessageContent::Text("I see two images.".to_string())),
                name: None,
                tool_calls: None,
                reasoning_content: None,
            },
            ChatMessage::User {
                content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/image3.jpg".to_string(),
                        detail: None,
                    },
                }]),
                name: None,
            },
        ];

        let extracted = extract_multimodal_from_messages(&messages);

        assert_eq!(extracted.image_urls.len(), 3);
        assert_eq!(extracted.image_count(), 3);
    }

    #[test]
    fn test_extract_empty() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Text("Hello, how are you?".to_string()),
            name: None,
        }];

        let extracted = extract_multimodal_from_messages(&messages);

        assert!(extracted.is_empty());
    }
}
