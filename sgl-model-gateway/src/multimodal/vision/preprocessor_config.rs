//! HuggingFace preprocessor_config.json parsing.
//!
//! This module parses the `preprocessor_config.json` files from HuggingFace model
//! repositories, providing the configuration needed for image preprocessing.

use std::collections::HashMap;

use image::imageops::FilterType;
use serde::{Deserialize, Deserializer};

use super::transforms;

/// Struct to represent patch_size as dict {"height": x, "width": y}
#[derive(Debug, Clone, Deserialize, Default)]
pub struct PatchSize {
    pub height: Option<u32>,
    pub width: Option<u32>,
}

/// Custom deserializer for patch_size that handles both integer and dict formats.
/// - Integer format: `"patch_size": 16` -> PatchSize { height: 16, width: 16 }
/// - Dict format: `"patch_size": {"height": 16, "width": 16}` -> PatchSize { height: 16, width: 16 }
fn deserialize_patch_size<'de, D>(deserializer: D) -> Result<Option<PatchSize>, D::Error>
where
    D: Deserializer<'de>,
{
    use std::fmt;

    use serde::de::{self, MapAccess, Visitor};

    struct PatchSizeVisitor;

    impl<'de> Visitor<'de> for PatchSizeVisitor {
        type Value = Option<PatchSize>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("an integer, a dict with height/width, or null")
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            let v = value as u32;
            Ok(Some(PatchSize {
                height: Some(v),
                width: Some(v),
            }))
        }

        fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            let v = value as u32;
            Ok(Some(PatchSize {
                height: Some(v),
                width: Some(v),
            }))
        }

        fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            let mut height = None;
            let mut width = None;

            while let Some(key) = map.next_key::<String>()? {
                match key.as_str() {
                    "height" => height = Some(map.next_value::<u32>()?),
                    "width" => width = Some(map.next_value::<u32>()?),
                    _ => {
                        let _ = map.next_value::<de::IgnoredAny>()?;
                    }
                }
            }

            Ok(Some(PatchSize { height, width }))
        }
    }

    deserializer.deserialize_any(PatchSizeVisitor)
}

/// HuggingFace preprocessor_config.json structure.
///
/// This struct captures the common fields across different vision model processors.
/// Model-specific fields are accessed via the flexible `extra` field.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct PreProcessorConfig {
    /// Processor class name (e.g., "CLIPImageProcessor", "Qwen2VLImageProcessor")
    #[serde(default)]
    pub image_processor_type: Option<String>,

    /// Whether to convert to RGB
    #[serde(default)]
    pub do_convert_rgb: Option<bool>,

    /// Whether to normalize with mean/std
    #[serde(default)]
    pub do_normalize: Option<bool>,

    /// Whether to pad images
    #[serde(default)]
    pub do_pad: Option<bool>,

    /// Whether to rescale pixel values (typically by 1/255)
    #[serde(default)]
    pub do_rescale: Option<bool>,

    /// Whether to resize images
    #[serde(default)]
    pub do_resize: Option<bool>,

    /// Whether to center crop after resizing
    #[serde(default)]
    pub do_center_crop: Option<bool>,

    /// Per-channel normalization mean
    #[serde(default, alias = "norm_mean")]
    pub image_mean: Option<Vec<f64>>,

    /// Per-channel normalization std
    #[serde(default, alias = "norm_std")]
    pub image_std: Option<Vec<f64>>,

    /// Rescale factor (typically 1/255 = 0.00392156862745098)
    #[serde(default)]
    pub rescale_factor: Option<f64>,

    /// PIL resampling filter enum (0=Nearest, 1=Lanczos, 2=Bilinear, 3=Bicubic)
    #[serde(default, alias = "resample")]
    pub resampling: Option<usize>,

    /// Target size for resizing
    /// Can be {"height": H, "width": W} or {"shortest_edge": S}
    #[serde(default)]
    pub size: Option<HashMap<String, u32>>,

    /// Target size for center cropping
    #[serde(default)]
    pub crop_size: Option<HashMap<String, u32>>,

    // =====================
    // Model-specific fields
    // =====================
    /// Vision encoder patch size (typically 14 or 16)
    /// Can be an integer or a dict {"height": x, "width": y}
    #[serde(default, deserialize_with = "deserialize_patch_size")]
    pub patch_size: Option<PatchSize>,

    /// Qwen-VL: merge size for token reduction
    #[serde(default)]
    pub merge_size: Option<usize>,

    /// Qwen-VL: minimum total pixels
    #[serde(default)]
    pub min_pixels: Option<usize>,

    /// Qwen-VL: maximum total pixels
    #[serde(default)]
    pub max_pixels: Option<usize>,

    /// Qwen-VL: temporal patch size for video
    #[serde(default)]
    pub temporal_patch_size: Option<usize>,

    /// Phi3-Vision: number of image crops
    #[serde(default)]
    pub num_crops: Option<usize>,

    /// Phi4-Vision: dynamic HD max crops
    #[serde(default)]
    pub dynamic_hd: Option<usize>,

    /// LLaMA-Vision: maximum image tiles
    #[serde(default)]
    pub max_image_tiles: Option<usize>,

    /// Fixed number of image tokens (some models use this)
    #[serde(default)]
    pub num_img_tokens: Option<usize>,

    // =====================
    // Special tokens
    // =====================
    /// Image start token
    #[serde(default)]
    pub im_start_token: Option<String>,

    /// Image end token
    #[serde(default)]
    pub im_end_token: Option<String>,

    /// Slice start token (for multi-crop)
    #[serde(default)]
    pub slice_start_token: Option<String>,

    /// Slice end token
    #[serde(default)]
    pub slice_end_token: Option<String>,

    /// Vision start token (alternative naming)
    #[serde(default)]
    pub vision_start_token: Option<String>,

    /// Vision end token
    #[serde(default)]
    pub vision_end_token: Option<String>,

    /// Catch-all for model-specific fields not explicitly defined
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl PreProcessorConfig {
    /// Parse from JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Parse from JSON value.
    pub fn from_value(value: serde_json::Value) -> Result<Self, serde_json::Error> {
        serde_json::from_value(value)
    }

    /// Get patch size as a simple usize.
    ///
    /// Returns the height value from PatchSize if available, falling back to provided default.
    pub fn get_patch_size(&self, default: usize) -> usize {
        self.patch_size
            .as_ref()
            .and_then(|p| p.height)
            .map(|h| h as usize)
            .unwrap_or(default)
    }

    /// Get image mean as fixed array, with fallback to CLIP defaults.
    pub fn get_image_mean(&self) -> [f64; 3] {
        self.image_mean
            .as_ref()
            .and_then(|v| {
                if v.len() >= 3 {
                    Some([v[0], v[1], v[2]])
                } else {
                    None
                }
            })
            .unwrap_or(Self::CLIP_MEAN)
    }

    /// Get image std as fixed array, with fallback to CLIP defaults.
    pub fn get_image_std(&self) -> [f64; 3] {
        self.image_std
            .as_ref()
            .and_then(|v| {
                if v.len() >= 3 {
                    Some([v[0], v[1], v[2]])
                } else {
                    None
                }
            })
            .unwrap_or(Self::CLIP_STD)
    }

    /// Get target size from various config formats.
    ///
    /// Handles both `{"height": H, "width": W}` and `{"shortest_edge": S}` formats.
    /// Returns (height, width).
    pub fn get_target_size(&self) -> Option<(u32, u32)> {
        self.size.as_ref().map(|s| {
            // Try explicit height/width first
            let h = s
                .get("height")
                .or_else(|| s.get("shortest_edge"))
                .copied()
                .unwrap_or(224);
            let w = s
                .get("width")
                .or_else(|| s.get("shortest_edge"))
                .copied()
                .unwrap_or(224);
            (h, w)
        })
    }

    /// Get crop size.
    ///
    /// Returns (height, width).
    pub fn get_crop_size(&self) -> Option<(u32, u32)> {
        self.crop_size.as_ref().map(|s| {
            let h = s.get("height").copied().unwrap_or(224);
            let w = s.get("width").copied().unwrap_or(224);
            (h, w)
        })
    }

    /// Get the interpolation filter for resizing.
    pub fn get_filter(&self) -> FilterType {
        transforms::pil_to_filter(self.resampling)
    }

    /// Check if normalization should be applied.
    pub fn should_normalize(&self) -> bool {
        self.do_normalize.unwrap_or(true)
    }

    /// Check if rescaling should be applied.
    pub fn should_rescale(&self) -> bool {
        self.do_rescale.unwrap_or(false)
    }

    /// Check if resizing should be applied.
    pub fn should_resize(&self) -> bool {
        self.do_resize.unwrap_or(true)
    }

    /// Check if center cropping should be applied.
    pub fn should_center_crop(&self) -> bool {
        self.do_center_crop.unwrap_or(false)
    }

    /// Get rescale factor with default.
    pub fn get_rescale_factor(&self) -> f64 {
        self.rescale_factor.unwrap_or(1.0 / 255.0)
    }

    /// Get a typed extra field.
    pub fn get_extra<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.extra
            .get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    // Common default values
    pub const CLIP_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
    pub const CLIP_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

    pub const IMAGENET_MEAN: [f64; 3] = [0.485, 0.456, 0.406];
    pub const IMAGENET_STD: [f64; 3] = [0.229, 0.224, 0.225];

    pub const SIGLIP_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
    pub const SIGLIP_STD: [f64; 3] = [0.5, 0.5, 0.5];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_clip_config() {
        let json = r#"{
            "do_center_crop": true,
            "do_normalize": true,
            "do_resize": true,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "resample": 3,
            "size": {"shortest_edge": 224}
        }"#;

        let config = PreProcessorConfig::from_json(json).unwrap();

        assert!(config.should_normalize());
        assert!(config.should_center_crop());
        assert!(config.should_resize());
        assert_eq!(config.resampling, Some(3));

        let (h, w) = config.get_target_size().unwrap();
        assert_eq!(h, 224);
        assert_eq!(w, 224);

        let mean = config.get_image_mean();
        assert!((mean[0] - 0.48145466).abs() < 1e-6);
    }

    #[test]
    fn test_parse_qwen_vl_config() {
        let json = r#"{
            "do_normalize": true,
            "do_rescale": true,
            "do_resize": true,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "min_pixels": 200704,
            "max_pixels": 1003520,
            "patch_size": 14,
            "merge_size": 2,
            "temporal_patch_size": 2,
            "rescale_factor": 0.00392156862745098
        }"#;

        let config = PreProcessorConfig::from_json(json).unwrap();

        assert_eq!(config.min_pixels, Some(200704));
        assert_eq!(config.max_pixels, Some(1003520));
        assert_eq!(config.get_patch_size(0), 14);
        assert_eq!(config.merge_size, Some(2));
        assert!((config.get_rescale_factor() - 1.0 / 255.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_size_formats() {
        // Height/width format
        let json1 = r#"{"size": {"height": 336, "width": 336}}"#;
        let config1 = PreProcessorConfig::from_json(json1).unwrap();
        assert_eq!(config1.get_target_size(), Some((336, 336)));

        // Shortest edge format
        let json2 = r#"{"size": {"shortest_edge": 224}}"#;
        let config2 = PreProcessorConfig::from_json(json2).unwrap();
        assert_eq!(config2.get_target_size(), Some((224, 224)));
    }

    #[test]
    fn test_defaults() {
        let config = PreProcessorConfig::default();

        // Should use CLIP defaults when not specified
        let mean = config.get_image_mean();
        assert!((mean[0] - PreProcessorConfig::CLIP_MEAN[0]).abs() < 1e-6);

        // Default behaviors
        assert!(config.should_normalize()); // true by default
        assert!(!config.should_rescale()); // false by default
        assert!(config.should_resize()); // true by default
        assert!(!config.should_center_crop()); // false by default
    }

    #[test]
    fn test_filter_conversion() {
        let json = r#"{"resampling": 3}"#;
        let config = PreProcessorConfig::from_json(json).unwrap();
        assert!(matches!(config.get_filter(), FilterType::CatmullRom));
    }

    #[test]
    fn test_extra_fields() {
        let json = r#"{
            "custom_field": 42,
            "nested": {"foo": "bar"}
        }"#;

        let config = PreProcessorConfig::from_json(json).unwrap();

        let custom: Option<i32> = config.get_extra("custom_field");
        assert_eq!(custom, Some(42));

        let nested: Option<HashMap<String, String>> = config.get_extra("nested");
        assert_eq!(
            nested.as_ref().unwrap().get("foo"),
            Some(&"bar".to_string())
        );
    }
}
