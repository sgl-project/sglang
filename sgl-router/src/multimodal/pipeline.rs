//! Multimodal processing pipeline for the model gateway.
//!
//! This module provides the orchestration layer that:
//! 1. Extracts multimodal content from chat messages
//! 2. Fetches images via MediaConnector
//! 3. Preprocesses images using the appropriate Rust processor
//!
//! # Architecture
//!
//! ```text
//! ChatMessage with images
//!         │
//!         ▼
//! ┌───────────────────┐
//! │ MultiModalPipeline │
//! └───────────────────┘
//!         │
//!         ├──► MediaConnector (fetch)
//!         │         │
//!         │         ▼
//!         │    DynamicImage[]
//!         │
//!         ├──► ImageProcessorRegistry (lookup)
//!         │         │
//!         │         ▼
//!         │    ImagePreProcessor
//!         │
//!         └──► processor.preprocess()
//!                   │
//!                   ▼
//!              ProcessingResult
//!              (original images + preprocessed + metadata)
//! ```

use std::sync::Arc;

use image::DynamicImage;
use thiserror::Error;
use tracing::{debug, warn};

use super::{
    error::MultiModalError,
    media::{ImageFetchConfig, MediaConnector, MediaSource},
    types::{ImageDetail, ImageFrame, Modality},
    vision::{ImagePreProcessor, ImageProcessorRegistry, PreProcessorConfig, PreprocessedImages},
};

/// Errors that can occur during multimodal processing.
#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("No processor found for model: {0}")]
    NoProcessor(String),

    #[error("Image fetch failed: {0}")]
    FetchError(#[from] MultiModalError),

    #[error("Image preprocessing failed: {0}")]
    PreprocessError(#[from] super::vision::TransformError),

    #[error("No images provided for processing")]
    NoImages,

    #[error("Task join error: {0}")]
    JoinError(#[from] tokio::task::JoinError),
}

/// Configuration for the multimodal processing pipeline.
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// Model ID used to select the appropriate processor
    pub model_id: String,
    /// Optional preprocessor config from HuggingFace
    pub preprocessor_config: Option<PreProcessorConfig>,
    /// Default image detail level
    pub default_detail: ImageDetail,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            preprocessor_config: None,
            default_detail: ImageDetail::Auto,
        }
    }
}

impl PipelineConfig {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            ..Default::default()
        }
    }

    pub fn with_preprocessor_config(mut self, config: PreProcessorConfig) -> Self {
        self.preprocessor_config = Some(config);
        self
    }

    pub fn with_default_detail(mut self, detail: ImageDetail) -> Self {
        self.default_detail = detail;
        self
    }
}

/// A single processed image with all associated data.
#[derive(Debug, Clone)]
pub struct ProcessedImage {
    /// Original image before preprocessing
    pub original: DynamicImage,
    /// Original image size (width, height)
    pub original_size: (u32, u32),
    /// Detail level requested
    pub detail: ImageDetail,
}

/// Result of multimodal processing.
#[derive(Debug)]
pub struct ProcessingResult {
    /// Original images that were processed
    pub original_images: Vec<ProcessedImage>,
    /// Preprocessed images ready for the model
    pub preprocessed: PreprocessedImages,
    /// List of modalities processed
    pub modalities: Vec<Modality>,
}

impl ProcessingResult {
    /// Check if preprocessing was successful.
    pub fn is_empty(&self) -> bool {
        self.preprocessed.batch_size() == 0
    }

    /// Get the number of images processed.
    pub fn image_count(&self) -> usize {
        self.original_images.len()
    }

    /// Get number of tokens for each image.
    pub fn num_tokens(&self) -> &[usize] {
        &self.preprocessed.num_img_tokens
    }

    /// Get total number of image tokens.
    pub fn total_tokens(&self) -> usize {
        self.preprocessed.total_tokens()
    }

    /// Get original sizes of all images.
    pub fn original_sizes(&self) -> Vec<(u32, u32)> {
        self.original_images
            .iter()
            .map(|img| img.original_size)
            .collect()
    }

    /// Get the preprocessed image sizes (from the processor output).
    pub fn processed_sizes(&self) -> &[(u32, u32)] {
        &self.preprocessed.image_sizes
    }
}

/// Multimodal processing pipeline.
///
/// This struct orchestrates the complete multimodal processing flow:
/// fetching and preprocessing images.
pub struct MultiModalPipeline {
    media_connector: Arc<MediaConnector>,
    processor_registry: ImageProcessorRegistry,
}

impl MultiModalPipeline {
    /// Create a new pipeline with the given media connector.
    pub fn new(media_connector: Arc<MediaConnector>) -> Self {
        Self {
            media_connector,
            processor_registry: ImageProcessorRegistry::with_defaults(),
        }
    }

    /// Create a new pipeline with custom processor registry.
    pub fn with_registry(
        media_connector: Arc<MediaConnector>,
        processor_registry: ImageProcessorRegistry,
    ) -> Self {
        Self {
            media_connector,
            processor_registry,
        }
    }

    /// Check if a processor is available for the given model.
    pub fn has_processor(&self, model_id: &str) -> bool {
        self.processor_registry.has_processor(model_id)
    }

    /// Get the processor for a model (for inspection/testing).
    pub fn get_processor(&self, model_id: &str) -> Option<&dyn ImagePreProcessor> {
        self.processor_registry.find(model_id)
    }

    /// Process images from URLs.
    ///
    /// # Arguments
    /// * `urls` - Image URLs to fetch and process
    /// * `config` - Pipeline configuration
    ///
    /// # Returns
    /// Processing result with original and preprocessed images, or error.
    pub async fn process_image_urls(
        &self,
        urls: &[String],
        config: &PipelineConfig,
    ) -> Result<ProcessingResult, PipelineError> {
        if urls.is_empty() {
            return Err(PipelineError::NoImages);
        }

        // Find processor for model
        let processor = self
            .processor_registry
            .find(&config.model_id)
            .ok_or_else(|| PipelineError::NoProcessor(config.model_id.clone()))?;

        debug!(
            model_id = %config.model_id,
            processor = processor.model_name(),
            num_images = urls.len(),
            "Processing images with Rust processor"
        );

        // Fetch all images concurrently
        let fetch_config = ImageFetchConfig {
            detail: config.default_detail,
        };

        let fetch_futures: Vec<_> = urls
            .iter()
            .map(|url| {
                let connector = Arc::clone(&self.media_connector);
                let source = MediaSource::Url(url.clone());
                let cfg = fetch_config;
                async move { connector.fetch_image(source, cfg).await }
            })
            .collect();

        let fetch_results = futures::future::join_all(fetch_futures).await;

        // Collect successful fetches
        let mut original_images: Vec<ProcessedImage> = Vec::with_capacity(fetch_results.len());
        let mut images_for_processing: Vec<DynamicImage> = Vec::with_capacity(fetch_results.len());

        for (i, result) in fetch_results.into_iter().enumerate() {
            match result {
                Ok(frame) => {
                    let size = frame.size();
                    original_images.push(ProcessedImage {
                        original: frame.data().clone(),
                        original_size: (size.width, size.height),
                        detail: frame.detail,
                    });
                    images_for_processing.push(frame.data().clone());
                }
                Err(e) => {
                    warn!(url = %urls[i], error = %e, "Failed to fetch image, skipping");
                }
            }
        }

        if images_for_processing.is_empty() {
            return Err(PipelineError::NoImages);
        }

        // Preprocess images
        let preproc_config = config.preprocessor_config.clone().unwrap_or_default();
        let preprocessed = processor.preprocess(&images_for_processing, &preproc_config)?;

        Ok(ProcessingResult {
            original_images,
            preprocessed,
            modalities: vec![Modality::Image; images_for_processing.len()],
        })
    }

    /// Process images from raw bytes.
    ///
    /// # Arguments
    /// * `image_data` - Raw image bytes to process
    /// * `config` - Pipeline configuration
    ///
    /// # Returns
    /// Processing result with original and preprocessed images, or error.
    pub async fn process_image_bytes(
        &self,
        image_data: &[Vec<u8>],
        config: &PipelineConfig,
    ) -> Result<ProcessingResult, PipelineError> {
        if image_data.is_empty() {
            return Err(PipelineError::NoImages);
        }

        // Find processor for model
        let processor = self
            .processor_registry
            .find(&config.model_id)
            .ok_or_else(|| PipelineError::NoProcessor(config.model_id.clone()))?;

        debug!(
            model_id = %config.model_id,
            processor = processor.model_name(),
            num_images = image_data.len(),
            "Processing image bytes with Rust processor"
        );

        // Decode all images concurrently
        let fetch_config = ImageFetchConfig {
            detail: config.default_detail,
        };

        let fetch_futures: Vec<_> = image_data
            .iter()
            .map(|bytes| {
                let connector = Arc::clone(&self.media_connector);
                let source = MediaSource::InlineBytes(bytes.clone());
                let cfg = fetch_config;
                async move { connector.fetch_image(source, cfg).await }
            })
            .collect();

        let fetch_results = futures::future::join_all(fetch_futures).await;

        // Collect successful decodes
        let mut original_images: Vec<ProcessedImage> = Vec::with_capacity(fetch_results.len());
        let mut images_for_processing: Vec<DynamicImage> = Vec::with_capacity(fetch_results.len());

        for (i, result) in fetch_results.into_iter().enumerate() {
            match result {
                Ok(frame) => {
                    let size = frame.size();
                    original_images.push(ProcessedImage {
                        original: frame.data().clone(),
                        original_size: (size.width, size.height),
                        detail: frame.detail,
                    });
                    images_for_processing.push(frame.data().clone());
                }
                Err(e) => {
                    warn!(index = i, error = %e, "Failed to decode image, skipping");
                }
            }
        }

        if images_for_processing.is_empty() {
            return Err(PipelineError::NoImages);
        }

        // Preprocess images
        let preproc_config = config.preprocessor_config.clone().unwrap_or_default();
        let preprocessed = processor.preprocess(&images_for_processing, &preproc_config)?;

        Ok(ProcessingResult {
            original_images,
            preprocessed,
            modalities: vec![Modality::Image; images_for_processing.len()],
        })
    }

    /// Process already-fetched image frames.
    ///
    /// This is useful when images have already been fetched by the tracker.
    ///
    /// # Arguments
    /// * `frames` - Already-fetched image frames
    /// * `config` - Pipeline configuration
    ///
    /// # Returns
    /// Processing result with original and preprocessed images, or error.
    pub fn process_image_frames(
        &self,
        frames: &[Arc<ImageFrame>],
        config: &PipelineConfig,
    ) -> Result<ProcessingResult, PipelineError> {
        if frames.is_empty() {
            return Err(PipelineError::NoImages);
        }

        // Find processor for model
        let processor = self
            .processor_registry
            .find(&config.model_id)
            .ok_or_else(|| PipelineError::NoProcessor(config.model_id.clone()))?;

        debug!(
            model_id = %config.model_id,
            processor = processor.model_name(),
            num_images = frames.len(),
            "Processing image frames with Rust processor"
        );

        // Extract images
        let original_images: Vec<ProcessedImage> = frames
            .iter()
            .map(|f| {
                let size = f.size();
                ProcessedImage {
                    original: f.data().clone(),
                    original_size: (size.width, size.height),
                    detail: f.detail,
                }
            })
            .collect();

        let images_for_processing: Vec<DynamicImage> =
            frames.iter().map(|f| f.data().clone()).collect();

        // Preprocess images
        let preproc_config = config.preprocessor_config.clone().unwrap_or_default();
        let preprocessed = processor.preprocess(&images_for_processing, &preproc_config)?;

        Ok(ProcessingResult {
            original_images,
            preprocessed,
            modalities: vec![Modality::Image; frames.len()],
        })
    }

    /// Process images directly without fetching.
    ///
    /// Use this when you already have DynamicImage instances.
    ///
    /// # Arguments
    /// * `images` - Images to process
    /// * `config` - Pipeline configuration
    ///
    /// # Returns
    /// Processing result with preprocessed images, or error.
    pub fn process_images(
        &self,
        images: &[DynamicImage],
        config: &PipelineConfig,
    ) -> Result<ProcessingResult, PipelineError> {
        if images.is_empty() {
            return Err(PipelineError::NoImages);
        }

        // Find processor for model
        let processor = self
            .processor_registry
            .find(&config.model_id)
            .ok_or_else(|| PipelineError::NoProcessor(config.model_id.clone()))?;

        debug!(
            model_id = %config.model_id,
            processor = processor.model_name(),
            num_images = images.len(),
            "Processing images directly with Rust processor"
        );

        // Build original image info
        let original_images: Vec<ProcessedImage> = images
            .iter()
            .map(|img| ProcessedImage {
                original: img.clone(),
                original_size: (img.width(), img.height()),
                detail: config.default_detail,
            })
            .collect();

        // Preprocess images
        let preproc_config = config.preprocessor_config.clone().unwrap_or_default();
        let preprocessed = processor.preprocess(images, &preproc_config)?;

        Ok(ProcessingResult {
            original_images,
            preprocessed,
            modalities: vec![Modality::Image; images.len()],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_builder() {
        let config = PipelineConfig::new("llava-v1.5-7b")
            .with_preprocessor_config(PreProcessorConfig::default())
            .with_default_detail(ImageDetail::High);

        assert_eq!(config.model_id, "llava-v1.5-7b");
        assert!(config.preprocessor_config.is_some());
        assert_eq!(config.default_detail, ImageDetail::High);
    }

    #[test]
    fn test_processing_result_accessors() {
        use ndarray::Array4;

        let pixel_values = Array4::<f32>::zeros((2, 3, 224, 224));
        let preprocessed =
            PreprocessedImages::new(pixel_values, vec![196, 196], vec![(224, 224), (224, 224)]);

        // Create dummy original images
        let original_images = vec![
            ProcessedImage {
                original: DynamicImage::new_rgb8(640, 480),
                original_size: (640, 480),
                detail: ImageDetail::Auto,
            },
            ProcessedImage {
                original: DynamicImage::new_rgb8(800, 600),
                original_size: (800, 600),
                detail: ImageDetail::High,
            },
        ];

        let result = ProcessingResult {
            original_images,
            preprocessed,
            modalities: vec![Modality::Image, Modality::Image],
        };

        assert_eq!(result.image_count(), 2);
        assert_eq!(result.num_tokens(), &[196, 196]);
        assert_eq!(result.total_tokens(), 392);
        assert_eq!(result.original_sizes(), vec![(640, 480), (800, 600)]);
        assert_eq!(result.processed_sizes(), &[(224, 224), (224, 224)]);
        assert!(!result.is_empty());
    }
}
