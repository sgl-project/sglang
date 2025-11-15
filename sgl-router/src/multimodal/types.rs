use std::{
    collections::HashMap,
    fmt,
    path::PathBuf,
    sync::{Arc, LazyLock},
};

use image::DynamicImage;
use serde::{Deserialize, Serialize};

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
        payload: serde_json::Value,
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
