//! Validated configuration for the first Rust model-execution architecture.

use std::fs;
use std::path::{Path, PathBuf};

use safetensors::Dtype;
use serde::Deserialize;

use crate::checkpoint::{CheckpointError, SafeTensorCheckpoint};

/// Validated dense Llama causal-language-model configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct LlamaConfig {
    /// Transformer hidden width.
    pub hidden_size: usize,
    /// SwiGLU intermediate width.
    pub intermediate_size: usize,
    /// Number of transformer blocks.
    pub num_hidden_layers: usize,
    /// Query attention heads.
    pub num_attention_heads: usize,
    /// Key/value attention heads (GQA/MQA aware).
    pub num_key_value_heads: usize,
    /// Width of each attention head.
    pub head_dim: usize,
    /// Token vocabulary width.
    pub vocab_size: usize,
    /// Maximum configured context length.
    pub max_position_embeddings: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// Rotary position embedding base.
    pub rope_theta: f64,
    /// Whether the LM head shares token embedding weights.
    pub tie_word_embeddings: bool,
}

#[derive(Deserialize)]
struct RawLlamaConfig {
    model_type: String,
    architectures: Option<Vec<String>>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    vocab_size: usize,
    max_position_embeddings: usize,
    rms_norm_eps: Option<f64>,
    rope_theta: Option<f64>,
    tie_word_embeddings: Option<bool>,
    hidden_act: Option<String>,
    rope_scaling: Option<serde_json::Value>,
}

impl LlamaConfig {
    /// Read and validate a local model directory or its `config.json` file.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ModelConfigError> {
        let path = path.as_ref();
        let config_path = if path.is_dir() {
            path.join("config.json")
        } else {
            path.to_path_buf()
        };
        let contents = fs::read_to_string(&config_path).map_err(|source| ModelConfigError::Io {
            path: config_path.clone(),
            source,
        })?;
        let raw: RawLlamaConfig =
            serde_json::from_str(&contents).map_err(|source| ModelConfigError::InvalidJson {
                path: config_path,
                message: source.to_string(),
            })?;
        Self::try_from(raw)
    }

    fn validate(&self) -> Result<(), ModelConfigError> {
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.vocab_size == 0
            || self.max_position_embeddings == 0
        {
            return Err(ModelConfigError::ZeroDimension);
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads)
            || self.head_dim != self.hidden_size / self.num_attention_heads
        {
            return Err(ModelConfigError::InconsistentHeadDimension);
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(ModelConfigError::InconsistentKeyValueHeads);
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(ModelConfigError::InvalidScalar("rms_norm_eps"));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(ModelConfigError::InvalidScalar("rope_theta"));
        }
        Ok(())
    }

    /// Tensor names and shapes expected by the dense Hugging Face Llama layout.
    pub fn expected_tensor_shapes(&self) -> Vec<TensorSpec> {
        let mut specs = vec![TensorSpec {
            name: "model.embed_tokens.weight".into(),
            shape: vec![self.vocab_size, self.hidden_size],
        }];
        for layer in 0..self.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            let kv_width = self.num_key_value_heads * self.head_dim;
            let query_width = self.num_attention_heads * self.head_dim;
            specs.extend([
                TensorSpec {
                    name: format!("{prefix}.self_attn.q_proj.weight"),
                    shape: vec![query_width, self.hidden_size],
                },
                TensorSpec {
                    name: format!("{prefix}.self_attn.k_proj.weight"),
                    shape: vec![kv_width, self.hidden_size],
                },
                TensorSpec {
                    name: format!("{prefix}.self_attn.v_proj.weight"),
                    shape: vec![kv_width, self.hidden_size],
                },
                TensorSpec {
                    name: format!("{prefix}.self_attn.o_proj.weight"),
                    shape: vec![self.hidden_size, query_width],
                },
                TensorSpec {
                    name: format!("{prefix}.mlp.gate_proj.weight"),
                    shape: vec![self.intermediate_size, self.hidden_size],
                },
                TensorSpec {
                    name: format!("{prefix}.mlp.up_proj.weight"),
                    shape: vec![self.intermediate_size, self.hidden_size],
                },
                TensorSpec {
                    name: format!("{prefix}.mlp.down_proj.weight"),
                    shape: vec![self.hidden_size, self.intermediate_size],
                },
                TensorSpec {
                    name: format!("{prefix}.input_layernorm.weight"),
                    shape: vec![self.hidden_size],
                },
                TensorSpec {
                    name: format!("{prefix}.post_attention_layernorm.weight"),
                    shape: vec![self.hidden_size],
                },
            ]);
        }
        specs.push(TensorSpec {
            name: "model.norm.weight".into(),
            shape: vec![self.hidden_size],
        });
        if !self.tie_word_embeddings {
            specs.push(TensorSpec {
                name: "lm_head.weight".into(),
                shape: vec![self.vocab_size, self.hidden_size],
            });
        }
        specs
    }

    /// Validate required tensor presence, shape, and supported floating dtype.
    pub fn validate_checkpoint(
        &self,
        checkpoint: &SafeTensorCheckpoint,
    ) -> Result<(), ModelConfigError> {
        for spec in self.expected_tensor_shapes() {
            let tensor = checkpoint.tensor(&spec.name).map_err(|error| match error {
                CheckpointError::TensorNotFound(name) => ModelConfigError::MissingTensor(name),
                other => ModelConfigError::Checkpoint(other.to_string()),
            })?;
            if tensor.shape() != spec.shape {
                return Err(ModelConfigError::TensorShapeMismatch {
                    name: spec.name,
                    expected: spec.shape,
                    actual: tensor.shape().to_vec(),
                });
            }
            if !matches!(tensor.dtype(), Dtype::F32 | Dtype::F16 | Dtype::BF16) {
                return Err(ModelConfigError::UnsupportedTensorDtype {
                    name: spec.name,
                    dtype: tensor.dtype(),
                });
            }
        }
        Ok(())
    }
}

/// One required named tensor in a model checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    /// Hugging Face safetensors key.
    pub name: String,
    /// Expected dimensions in checkpoint order.
    pub shape: Vec<usize>,
}

impl TryFrom<RawLlamaConfig> for LlamaConfig {
    type Error = ModelConfigError;

    fn try_from(raw: RawLlamaConfig) -> Result<Self, Self::Error> {
        if raw.model_type != "llama" {
            return Err(ModelConfigError::UnsupportedModelType(raw.model_type));
        }
        if raw.architectures.as_ref().is_some_and(|architectures| {
            !architectures.iter().any(|name| name == "LlamaForCausalLM")
        }) {
            return Err(ModelConfigError::UnsupportedArchitecture(
                raw.architectures.unwrap_or_default(),
            ));
        }
        if raw
            .hidden_act
            .as_deref()
            .is_some_and(|activation| activation != "silu")
        {
            return Err(ModelConfigError::UnsupportedActivation(
                raw.hidden_act.unwrap_or_default(),
            ));
        }
        if raw
            .rope_scaling
            .as_ref()
            .is_some_and(|value| !value.is_null())
        {
            return Err(ModelConfigError::UnsupportedRopeScaling);
        }
        let head_dim = raw
            .head_dim
            .unwrap_or(raw.hidden_size / raw.num_attention_heads.max(1));
        let config = Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads.unwrap_or(raw.num_attention_heads),
            head_dim,
            vocab_size: raw.vocab_size,
            max_position_embeddings: raw.max_position_embeddings,
            rms_norm_eps: raw.rms_norm_eps.unwrap_or(1e-6),
            rope_theta: raw.rope_theta.unwrap_or(10_000.0),
            tie_word_embeddings: raw.tie_word_embeddings.unwrap_or(false),
        };
        config.validate()?;
        Ok(config)
    }
}

/// Unsupported or invalid local Llama model configuration.
#[derive(Debug, thiserror::Error)]
pub enum ModelConfigError {
    #[error("model config I/O failed at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid model config JSON at {path}: {message}")]
    InvalidJson { path: PathBuf, message: String },
    #[error("unsupported model_type for Rust executor: {0}")]
    UnsupportedModelType(String),
    #[error("unsupported model architectures: {0:?}")]
    UnsupportedArchitecture(Vec<String>),
    #[error("unsupported Llama activation: {0}")]
    UnsupportedActivation(String),
    #[error("Llama rope_scaling is not supported by the current executor")]
    UnsupportedRopeScaling,
    #[error("model dimensions must all be non-zero")]
    ZeroDimension,
    #[error("hidden_size, num_attention_heads, and head_dim are inconsistent")]
    InconsistentHeadDimension,
    #[error("num_key_value_heads must divide num_attention_heads")]
    InconsistentKeyValueHeads,
    #[error("invalid model scalar: {0}")]
    InvalidScalar(&'static str),
    #[error("required model tensor is missing: {0}")]
    MissingTensor(String),
    #[error("invalid checkpoint for model: {0}")]
    Checkpoint(String),
    #[error("tensor {name} has shape {actual:?}; expected {expected:?}")]
    TensorShapeMismatch {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("tensor {name} has unsupported dtype {dtype:?}")]
    UnsupportedTensorDtype { name: String, dtype: Dtype },
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use safetensors::tensor::{TensorView, serialize};

    fn write_config(contents: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "sglang-model-config-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("config.json"), contents).unwrap();
        dir
    }

    fn write_checkpoint(dir: &Path, specs: &[TensorSpec], omit: Option<&str>) {
        for (index, spec) in specs.iter().enumerate() {
            if omit == Some(spec.name.as_str()) {
                continue;
            }
            let count: usize = spec.shape.iter().product();
            let data = vec![0_u8; count * 4];
            let tensor = TensorView::new(Dtype::F32, spec.shape.clone(), &data).unwrap();
            let bytes = serialize([(spec.name.as_str(), tensor)], None).unwrap();
            fs::write(dir.join(format!("shard-{index:05}.safetensors")), bytes).unwrap();
        }
    }

    const VALID: &str = r#"{
        "model_type":"llama",
        "architectures":["LlamaForCausalLM"],
        "hidden_size":16,
        "intermediate_size":32,
        "num_hidden_layers":2,
        "num_attention_heads":4,
        "num_key_value_heads":2,
        "vocab_size":128,
        "max_position_embeddings":256
    }"#;

    #[test]
    fn loads_local_config_with_llama_defaults() {
        let dir = write_config(VALID);
        let config = LlamaConfig::from_path(&dir).unwrap();
        assert_eq!(config.head_dim, 4);
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert_eq!(config.rope_theta, 10_000.0);
        assert!(!config.tie_word_embeddings);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_unsupported_model_types_and_rope_scaling() {
        let dir = write_config(&VALID.replace("llama", "qwen2"));
        assert!(matches!(
            LlamaConfig::from_path(&dir),
            Err(ModelConfigError::UnsupportedModelType(_))
        ));
        fs::remove_dir_all(dir).unwrap();

        let scaled = VALID.replace(
            "\"max_position_embeddings\":256",
            "\"max_position_embeddings\":256,\"rope_scaling\":{\"type\":\"linear\",\"factor\":2.0}",
        );
        let dir = write_config(&scaled);
        assert!(matches!(
            LlamaConfig::from_path(&dir),
            Err(ModelConfigError::UnsupportedRopeScaling)
        ));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_inconsistent_attention_shapes() {
        let invalid = VALID.replace("\"num_key_value_heads\":2", "\"num_key_value_heads\":3");
        let dir = write_config(&invalid);
        assert!(matches!(
            LlamaConfig::from_path(&dir),
            Err(ModelConfigError::InconsistentKeyValueHeads)
        ));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn validates_llama_checkpoint_shapes_and_required_tensors() {
        let config_dir = write_config(VALID);
        let config = LlamaConfig::from_path(&config_dir).unwrap();
        let checkpoint_dir = config_dir.join("weights");
        fs::create_dir_all(&checkpoint_dir).unwrap();
        let specs = config.expected_tensor_shapes();
        write_checkpoint(&checkpoint_dir, &specs, None);
        let checkpoint = SafeTensorCheckpoint::open(&checkpoint_dir).unwrap();
        config.validate_checkpoint(&checkpoint).unwrap();
        drop(checkpoint);

        fs::remove_dir_all(&checkpoint_dir).unwrap();
        fs::create_dir_all(&checkpoint_dir).unwrap();
        write_checkpoint(&checkpoint_dir, &specs, Some("lm_head.weight"));
        let checkpoint = SafeTensorCheckpoint::open(&checkpoint_dir).unwrap();
        assert!(matches!(
            config.validate_checkpoint(&checkpoint),
            Err(ModelConfigError::MissingTensor(name)) if name == "lm_head.weight"
        ));
        fs::remove_dir_all(config_dir).unwrap();
    }

    #[test]
    fn rejects_wrong_llama_tensor_shape() {
        let dir = write_config(VALID);
        let config = LlamaConfig::from_path(&dir).unwrap();
        let checkpoint_dir = dir.join("weights");
        fs::create_dir_all(&checkpoint_dir).unwrap();
        let mut specs = config.expected_tensor_shapes();
        let embedding = specs
            .iter_mut()
            .find(|spec| spec.name == "model.embed_tokens.weight")
            .unwrap();
        embedding.shape = vec![config.vocab_size, config.hidden_size - 1];
        write_checkpoint(&checkpoint_dir, &specs, None);
        let checkpoint = SafeTensorCheckpoint::open(&checkpoint_dir).unwrap();
        assert!(matches!(
            config.validate_checkpoint(&checkpoint),
            Err(ModelConfigError::TensorShapeMismatch { name, .. }) if name == "model.embed_tokens.weight"
        ));
        fs::remove_dir_all(dir).unwrap();
    }
}
