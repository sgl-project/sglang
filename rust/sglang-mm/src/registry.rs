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

/// The resolved parameters of one family pipeline — the typed form of the
/// Python-side spec, one variant per family arm. `sglang-server` builds it
/// directly from its `MmSpec` pyclass; the JSON parity API reaches it through
/// [`pipeline_from_spec`], where the `family` key selects the variant.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(tag = "family", rename_all = "snake_case")]
pub enum PipelineSpec {
    QwenVl(crate::qwen_vl::QwenVlSpec),
}

/// Build a family processor from a typed spec. `Err` when the family
/// rejects its parameters (e.g. a zero patch size).
pub fn build_pipeline(
    spec: PipelineSpec,
) -> Result<Box<dyn crate::pipeline::MmFamilyProcessor>, String> {
    match spec {
        PipelineSpec::QwenVl(spec) => Ok(Box::new(crate::qwen_vl::QwenVlProcessor::new(spec)?)),
    }
}

/// Build a family processor from the Python-side spec JSON
/// (`{"family": ..., resolved processor params}`). `Err` on an unknown family
/// or malformed spec — the caller treats that as "no Rust pipeline".
pub fn pipeline_from_spec(
    json: &str,
) -> Result<Box<dyn crate::pipeline::MmFamilyProcessor>, String> {
    let spec: PipelineSpec = serde_json::from_str(json).map_err(|e| format!("mm spec: {e}"))?;
    build_pipeline(spec)
}
