use serde::{Deserialize, Serialize};

use super::common::GenerationRequest;

// ============================================================================
// Images Generation API (/v1/images/generations)
// ============================================================================

/// Request for generating images from a text prompt
/// OpenAI-compatible: https://platform.openai.com/docs/api-reference/images/create
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageGenerationRequest {
    /// A text description of the desired image(s)
    pub prompt: String,

    /// The model to use for image generation
    pub model: String,

    /// The number of images to generate (1-10)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// The quality of the image ("standard" or "hd")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,

    /// The format in which the generated images are returned ("url" or "b64_json")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,

    /// The size of the generated images (e.g., "1024x1024", "1792x1024", "1024x1792")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,

    /// The style of the generated images ("vivid" or "natural")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,

    /// A unique identifier representing your end-user
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl GenerationRequest for ImageGenerationRequest {
    fn is_stream(&self) -> bool {
        // Image generation is non-streaming
        false
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn extract_text_for_routing(&self) -> String {
        self.prompt.clone()
    }
}

// ============================================================================
// Images Edit API (/v1/images/edits)
// ============================================================================

/// Request for editing images with a text prompt
/// OpenAI-compatible: https://platform.openai.com/docs/api-reference/images/createEdit
///
/// Note: This endpoint uses multipart/form-data encoding. The image and mask
/// fields contain base64-encoded image data when received via the gateway.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageEditRequest {
    /// The image to edit. Must be a valid PNG file, less than 4MB, and square.
    /// Base64-encoded image data.
    pub image: String,

    /// A text description of the desired image(s)
    pub prompt: String,

    /// The model to use for image editing
    #[serde(default = "default_edit_model")]
    pub model: String,

    /// An additional image whose fully transparent areas indicate where the
    /// original image should be edited. Must be a valid PNG file, less than 4MB,
    /// and have the same dimensions as the original image.
    /// Base64-encoded image data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask: Option<String>,

    /// The number of images to generate (1-10)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// The size of the generated images (e.g., "256x256", "512x512", "1024x1024")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,

    /// The format in which the generated images are returned ("url" or "b64_json")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,

    /// A unique identifier representing your end-user
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

fn default_edit_model() -> String {
    "dall-e-2".to_string()
}

impl GenerationRequest for ImageEditRequest {
    fn is_stream(&self) -> bool {
        // Image editing is non-streaming
        false
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn extract_text_for_routing(&self) -> String {
        self.prompt.clone()
    }
}

// ============================================================================
// Response Types
// ============================================================================

/// Response from image generation or editing endpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageResponse {
    /// Unix timestamp of when the response was created
    pub created: i64,

    /// Array of generated image objects
    pub data: Vec<ImageObject>,
}

/// Individual image object in the response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageObject {
    /// The base64-encoded JSON of the generated image (if response_format is "b64_json")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,

    /// The URL of the generated image (if response_format is "url")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// The prompt that was used to generate the image (for dall-e-3)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}
