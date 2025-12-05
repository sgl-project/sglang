pub mod error;
pub mod extract;
pub mod media;
pub mod pipeline;
pub mod registry;
pub mod tracker;
pub mod types;
pub mod vision;

pub use error::{MediaConnectorError, MultiModalError, MultiModalResult};
pub use extract::{extract_multimodal_from_messages, ExtractedMultiModal};
pub use media::{ImageFetchConfig, MediaConnector, MediaConnectorConfig, MediaSource};
pub use pipeline::{
    MultiModalPipeline, PipelineConfig, PipelineError, ProcessedImage, ProcessingResult,
};
pub use registry::{ModelProcessorSpec, ModelRegistry};
pub use tracker::{AsyncMultiModalTracker, TrackerConfig, TrackerOutput};
pub use types::*;
// Re-export vision processing components
pub use vision::{
    ImagePreProcessor, ImageProcessorRegistry, Llama4VisionProcessor, LlavaNextProcessor,
    LlavaProcessor, ModelSpecificValue, Phi3VisionProcessor, Phi4VisionProcessor, PixtralProcessor,
    PreProcessorConfig, PreprocessedImages, Qwen2VLProcessor, Qwen3VLProcessor, TransformError,
};
