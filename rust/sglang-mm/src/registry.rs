//! Model processor registry.
//!
//! Each model implements `ImageProcessorSpec` and registers itself. The Python
//! layer looks up a processor by model name at init time.

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
