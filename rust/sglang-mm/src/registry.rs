//! Model processor registries.
//!
//! Two registries live here:
//! * [`ImageProcessorSpec`] / [`ProcessorRegistry`] — the Python-facing batch
//!   preprocess interface (e.g. Inkling), looked up by name at init time.
//! * [`pipeline_from_spec`] — the pure-Rust request pipeline `sglang-server`'s
//!   MM workers drive. Each model family implements
//!   [`crate::pipeline::MmFamilyProcessor`] in `src/<model>/mod.rs`; the Python
//!   side selects one by serializing a spec
//!   (`{"family": ..., resolved processor params}`).

/// `(height, width, patches_as_u16_bits, content_hash)` for one image.
pub type PreprocessedImage = (usize, usize, Vec<u16>, u64);

/// Trait that each model's image processor must implement.
pub trait ImageProcessorSpec: Send + Sync {
    /// Short identifier, e.g. "inkling".
    fn name(&self) -> &'static str;

    /// Process a batch of raw image bytes: decode + preprocess + hash.
    fn preprocess_batch(
        &self,
        datas: &[Vec<u8>],
        patch_size: usize,
        rescale_frac: Option<f64>,
        rescale_cap: Option<i64>,
    ) -> Result<Vec<PreprocessedImage>, String>;
}

/// Global registry of available processors.
pub struct ProcessorRegistry {
    specs: Vec<Box<dyn ImageProcessorSpec>>,
}

impl Default for ProcessorRegistry {
    fn default() -> Self {
        Self::new()
    }
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

/// Build the default registry with all compiled-in processors.
pub fn default_registry() -> ProcessorRegistry {
    let mut reg = ProcessorRegistry::new();
    reg.register(Box::new(crate::inkling::InklingProcessor));
    reg
}
// --- Server (pure-Rust) request pipeline ---

/// Build a family processor from the Python-side spec JSON. `Err` on an
/// unknown family or malformed spec — the caller treats that as "no Rust
/// pipeline".
pub fn pipeline_from_spec(
    json: &str,
) -> Result<Box<dyn crate::pipeline::MmFamilyProcessor>, String> {
    #[derive(serde::Deserialize)]
    struct Header {
        family: String,
    }
    let header: Header = serde_json::from_str(json).map_err(|e| format!("mm spec: {e}"))?;
    match header.family.as_str() {
        "qwen_vl" => Ok(Box::new(crate::qwen_vl::QwenVlProcessor::from_spec_json(
            json,
        )?)),
        other => Err(format!("unknown mm family: {other}")),
    }
}
