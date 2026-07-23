//! Model processor registries.
//!
//! Two registries live here:
//! * [`ImageProcessorSpec`] / [`ProcessorRegistry`] — the Python-facing batch
//!   preprocess interface (e.g. Inkling), looked up by name at init time.
//! * [`VisionProcessor`] / [`native_pipeline_from_spec`] — the pure-Rust
//!   pipeline `sglang-server`'s MM workers drive natively. Each model family
//!   implements the trait in `src/<model>/mod.rs`; the Python side selects one
//!   by serializing a spec (`{"family": ..., resolved processor params}`).

/// Trait that each model's (Python-facing) image processor must implement.
pub trait ImageProcessorSpec: Send + Sync {
    /// Short identifier, e.g. "inkling".
    fn name(&self) -> &'static str;

    /// Process a batch of raw image bytes: decode + preprocess + hash.
    ///
    /// Returns `(height, width, patches_as_u16_bits, content_hash)` per image.
    fn preprocess_batch(
        &self,
        datas: &[Vec<u8>],
        patch_size: usize,
        rescale_frac: Option<f64>,
        rescale_cap: Option<i64>,
    ) -> Result<Vec<(usize, usize, Vec<u16>, u64)>, String>;
}

/// Global registry of available processors.
pub struct ProcessorRegistry {
    specs: Vec<Box<dyn ImageProcessorSpec>>,
}

impl ProcessorRegistry {
    pub fn new() -> Self {
        Self { specs: Vec::new() }
    }

    pub fn register(&mut self, spec: Box<dyn ImageProcessorSpec>) {
        self.specs.push(spec);
    }

    pub fn lookup(&self, name: &str) -> Option<&dyn ImageProcessorSpec> {
        self.specs
            .iter()
            .find(|s| s.name() == name)
            .map(|s| s.as_ref())
    }

    pub fn list_names(&self) -> Vec<&'static str> {
        self.specs.iter().map(|s| s.name()).collect()
    }
}

impl Default for ProcessorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the default registry with all compiled-in processors.
pub fn default_registry() -> ProcessorRegistry {
    let mut reg = ProcessorRegistry::new();
    reg.register(Box::new(crate::inkling::InklingProcessor));
    reg
}

// --- Native (pure-Rust) vision pipeline ---

/// One preprocessed image: the model-ready feature tensor plus its patch grid.
pub struct ProcessedImage {
    /// `[grid_h * grid_w, feature_dim]` f32, flattened row-major in the
    /// model's HF processor order.
    pub pixel_values: Vec<f32>,
    /// `[t, h, w]` patch grid (`t` = 1 for still images).
    pub grid_thw: [u32; 3],
}

/// One media item's placement for M-RoPE: inclusive token range + patch grid.
pub struct MropeItem {
    pub start: u32,
    pub end: u32,
    pub grid: [u32; 3],
}

/// The per-model-family stages of the native pipeline (preprocess + token
/// geometry). Adding a family = implementing this in `src/<model>/mod.rs` and
/// adding its `family` arm to [`native_pipeline_from_spec`].
pub trait VisionProcessor: Send + Sync {
    /// Decode-agnostic preprocess of one RGB image (HWC u8): resize,
    /// normalize, patchify — the model's HF image-processor equivalent.
    fn process_image(&self, rgb: &[u8], h: usize, w: usize) -> Result<ProcessedImage, String>;

    /// LLM tokens one image occupies given its grid (placeholder expansion).
    fn tokens_per_image(&self, grid: &[u32; 3]) -> usize;

    /// Second dim of `pixel_values` (constant per model config).
    fn feature_dim(&self) -> usize;

    /// Image-only M-RoPE: flattened `[3, input_len]` positions + the position
    /// delta. Only called for requests with images as the sole modality.
    fn mrope_image_only(
        &self,
        input_len: usize,
        items: &[MropeItem],
    ) -> Result<(Vec<i64>, i64), String>;
}

/// A ready-to-run native pipeline: the family processor plus the token ids the
/// server needs for placeholder expansion.
pub struct NativePipeline {
    pub image_token_id: i32,
    pub processor: Box<dyn VisionProcessor>,
}

/// Build a native pipeline from the Python-side spec JSON. `Err` on an unknown
/// family or malformed spec — the caller treats that as "no native pipeline".
pub fn native_pipeline_from_spec(json: &str) -> Result<NativePipeline, String> {
    #[derive(serde::Deserialize)]
    struct Header {
        family: String,
        image_token_id: i32,
    }
    let header: Header = serde_json::from_str(json).map_err(|e| format!("native mm spec: {e}"))?;
    let processor: Box<dyn VisionProcessor> = match header.family.as_str() {
        "qwen_vl" => Box::new(crate::qwen_vl::QwenVlProcessor::from_spec_json(json)?),
        other => return Err(format!("unknown native mm family: {other}")),
    };
    Ok(NativePipeline {
        image_token_id: header.image_token_id,
        processor,
    })
}
