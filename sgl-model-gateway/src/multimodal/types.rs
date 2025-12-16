use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    path::PathBuf,
    sync::{Arc, LazyLock},
};

use image::DynamicImage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Supported multimodal modalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Image,
    ImageEmbeds,
    Audio,
    Video,
}

impl fmt::Display for Modality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Modality::Image => write!(f, "image"),
            Modality::ImageEmbeds => write!(f, "image_embeds"),
            Modality::Audio => write!(f, "audio"),
            Modality::Video => write!(f, "video"),
        }
    }
}

/// Simple helper used for default placeholder tokens per modality.
pub static DEFAULT_PLACEHOLDERS: LazyLock<HashMap<Modality, &'static str>> = LazyLock::new(|| {
    HashMap::from([
        (Modality::Image, "<image>"),
        (Modality::ImageEmbeds, "<image_embeds>"),
        (Modality::Audio, "<audio>"),
        (Modality::Video, "<video>"),
    ])
});

/// Detail level passed by OpenAI style APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ImageDetail {
    #[default]
    Auto,
    Low,
    High,
}

/// A normalized content part understood by the tracker.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatContentPart {
    Text {
        text: String,
    },
    ImageUrl {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
    },
    ImageData {
        data: Vec<u8>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },
    ImageEmbeds {
        payload: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
    },
}

/// Represents one segment in the conversation after placeholders are inserted.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversationSegment {
    Text(String),
    Placeholder { token: String },
}

impl ConversationSegment {
    pub fn text<S: Into<String>>(text: S) -> Self {
        ConversationSegment::Text(text.into())
    }

    pub fn placeholder<S: Into<String>>(token: S) -> Self {
        ConversationSegment::Placeholder {
            token: token.into(),
        }
    }
}

/// Metadata describing where placeholder tokens were inserted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaceholderHandle {
    pub modality: Modality,
    pub item_index: usize,
    pub text_position: usize,
}

pub type PlaceholderMap = HashMap<String, Vec<PlaceholderHandle>>;

/// Image source metadata (useful for hashing & tracing).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageSource {
    Url { url: String },
    DataUrl,
    InlineBytes,
    File { path: PathBuf },
}

/// Concrete image payload captured by the media connector.
#[derive(Debug)]
pub struct ImageFrame {
    image: DynamicImage,
    raw_bytes: Arc<Vec<u8>>,
    pub detail: ImageDetail,
    pub source: ImageSource,
}

impl ImageFrame {
    pub fn new(
        image: DynamicImage,
        raw_bytes: Arc<Vec<u8>>,
        detail: ImageDetail,
        source: ImageSource,
    ) -> Self {
        Self {
            image,
            raw_bytes,
            detail,
            source,
        }
    }

    pub fn data(&self) -> &DynamicImage {
        &self.image
    }

    pub fn raw_bytes(&self) -> &[u8] {
        self.raw_bytes.as_slice()
    }

    pub fn source(&self) -> &ImageSource {
        &self.source
    }

    pub fn size(&self) -> ImageSize {
        ImageSize::new(self.image.width(), self.image.height())
    }
}

/// Container for all supported multimodal media objects.
#[derive(Debug, Clone)]
pub enum TrackedMedia {
    Image(Arc<ImageFrame>),
    /// Placeholder variants for future modalities.
    Audio,
    Video,
    Embeddings,
}

pub type MultiModalData = HashMap<Modality, Vec<TrackedMedia>>;
pub type MultiModalUUIDs = HashMap<Modality, Vec<Option<String>>>;

pub type TokenId = i32;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

impl ImageSize {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlaceholderRange {
    pub offset: usize,
    pub length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalTensor {
    pub shape: Vec<usize>,
    pub dtype: String,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MultiModalValue {
    Tensor(MultiModalTensor),
    Json(Value),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MultiModalInputs {
    pub prompt_token_ids: Vec<u32>,
    #[serde(default)]
    pub mm_kwargs: BTreeMap<String, Vec<MultiModalValue>>,
    #[serde(default)]
    pub mm_hashes: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub mm_placeholders: BTreeMap<String, Vec<PlaceholderRange>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_salt: Option<String>,
}

impl MultiModalInputs {
    pub fn new(prompt_token_ids: Vec<u32>) -> Self {
        Self {
            prompt_token_ids,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct PromptReplacement {
    pub modality: Modality,
    pub placeholder_token: String,
    pub tokens: Vec<TokenId>,
}

impl PromptReplacement {
    pub fn repeated(
        modality: Modality,
        placeholder_token: &str,
        token_id: TokenId,
        count: usize,
    ) -> Self {
        Self {
            modality,
            placeholder_token: placeholder_token.to_string(),
            tokens: vec![token_id; count],
        }
    }

    pub fn sequence(modality: Modality, placeholder_token: &str, sequence: Vec<TokenId>) -> Self {
        Self {
            modality,
            placeholder_token: placeholder_token.to_string(),
            tokens: sequence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multimodal_inputs_defaults() {
        let inputs = MultiModalInputs::new(vec![1, 2, 3]);
        assert_eq!(inputs.prompt_token_ids, vec![1, 2, 3]);
        assert!(inputs.mm_kwargs.is_empty());
    }

    #[test]
    fn placeholder_range_serializes() {
        let range = PlaceholderRange {
            offset: 10,
            length: 4,
        };
        let json = serde_json::to_string(&range).unwrap();
        assert!(json.contains("offset"));
    }

    #[test]
    fn prompt_replacement_builders() {
        let rep = PromptReplacement::repeated(Modality::Image, "<image>", 100, 3);
        assert_eq!(rep.tokens, vec![100, 100, 100]);
    }
}
