//! Memory-mapped local safetensors checkpoint index for model executors.

use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

/// A local single-file or sharded safetensors checkpoint.
///
/// Files are memory mapped read-only. The checkpoint files must not be modified
/// or replaced while the index is alive. Tensor access returns a borrowed view
/// and does not copy the tensor payload.
pub struct SafeTensorCheckpoint {
    files: Vec<MappedShard>,
    tensors: HashMap<String, usize>,
}

struct MappedShard {
    path: PathBuf,
    mmap: Mmap,
}

impl SafeTensorCheckpoint {
    /// Open one `.safetensors` file or all top-level shards in a directory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, CheckpointError> {
        let path = path.as_ref();
        let mut paths = if path.is_file() {
            vec![path.to_path_buf()]
        } else if path.is_dir() {
            let mut paths = Vec::new();
            for entry in fs::read_dir(path).map_err(|source| CheckpointError::Io {
                path: path.to_path_buf(),
                source,
            })? {
                let entry = entry.map_err(|source| CheckpointError::Io {
                    path: path.to_path_buf(),
                    source,
                })?;
                let shard = entry.path();
                if shard.extension().is_some_and(|ext| ext == "safetensors") {
                    paths.push(shard);
                }
            }
            paths
        } else {
            return Err(CheckpointError::NotFound(path.to_path_buf()));
        };
        paths.sort();
        if paths.is_empty() {
            return Err(CheckpointError::NoShards(path.to_path_buf()));
        }

        let mut files = Vec::with_capacity(paths.len());
        let mut tensors = HashMap::new();
        for path in paths {
            let file = File::open(&path).map_err(|source| CheckpointError::Io {
                path: path.clone(),
                source,
            })?;
            // SAFETY: the mapping is read-only and owned by this checkpoint.
            // The documented API contract forbids mutating/replacing files
            // while the checkpoint is alive.
            let mmap = unsafe { Mmap::map(&file) }.map_err(|source| CheckpointError::Io {
                path: path.clone(),
                source,
            })?;
            let parsed = SafeTensors::deserialize(&mmap).map_err(|source| {
                CheckpointError::InvalidShard {
                    path: path.clone(),
                    message: source.to_string(),
                }
            })?;
            let file_index = files.len();
            for (name, _) in parsed.tensors() {
                if tensors.insert(name.clone(), file_index).is_some() {
                    return Err(CheckpointError::DuplicateTensor(name));
                }
            }
            files.push(MappedShard { path, mmap });
        }
        Ok(Self { files, tensors })
    }

    /// Number of tensor names indexed across all shards.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Sorted tensor names for deterministic inspection and loading.
    pub fn tensor_names(&self) -> Vec<&str> {
        let mut names: Vec<_> = self.tensors.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// Borrow a tensor payload without copying it out of the mapped shard.
    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>, CheckpointError> {
        let shard_index = *self
            .tensors
            .get(name)
            .ok_or_else(|| CheckpointError::TensorNotFound(name.into()))?;
        SafeTensors::deserialize(&self.files[shard_index].mmap)
            .and_then(|tensors| tensors.tensor(name))
            .map_err(|source| CheckpointError::InvalidShard {
                path: self.files[shard_index].path.clone(),
                message: source.to_string(),
            })
    }
}

/// Local checkpoint discovery, indexing, or tensor access error.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("checkpoint path does not exist: {0}")]
    NotFound(PathBuf),
    #[error("no safetensors shards found under {0}")]
    NoShards(PathBuf),
    #[error("checkpoint I/O failed at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid safetensors shard {path}: {message}")]
    InvalidShard { path: PathBuf, message: String },
    #[error("tensor appears in more than one shard: {0}")]
    DuplicateTensor(String),
    #[error("checkpoint has no tensor named {0}")]
    TensorNotFound(String),
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use safetensors::Dtype;
    use safetensors::tensor::{TensorView as OwnedView, serialize};

    use super::*;

    fn temp_dir() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("sglang-checkpoint-{}-{nonce}", std::process::id()))
    }

    fn write_shard(path: &Path, name: &str, values: &[f32]) {
        let data: Vec<_> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let tensor = OwnedView::new(Dtype::F32, vec![values.len()], &data).unwrap();
        let bytes = serialize([(name, tensor)], None).unwrap();
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn opens_shards_and_borrows_typed_tensor_views() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        write_shard(&dir.join("model-00002.safetensors"), "second", &[3.0]);
        write_shard(&dir.join("model-00001.safetensors"), "first", &[1.0, 2.0]);
        fs::write(dir.join("config.json"), b"{}").unwrap();

        let checkpoint = SafeTensorCheckpoint::open(&dir).unwrap();
        assert_eq!(checkpoint.tensor_count(), 2);
        assert_eq!(checkpoint.tensor_names(), ["first", "second"]);
        let tensor = checkpoint.tensor("first").unwrap();
        assert_eq!(tensor.dtype(), Dtype::F32);
        assert_eq!(tensor.shape(), [2]);
        assert_eq!(tensor.data(), [0, 0, 128, 63, 0, 0, 0, 64]);
        assert!(matches!(
            checkpoint.tensor("missing"),
            Err(CheckpointError::TensorNotFound(_))
        ));
        drop(checkpoint);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_duplicate_tensor_names_across_shards() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        write_shard(&dir.join("a.safetensors"), "duplicate", &[1.0]);
        write_shard(&dir.join("b.safetensors"), "duplicate", &[2.0]);
        assert!(matches!(
            SafeTensorCheckpoint::open(&dir),
            Err(CheckpointError::DuplicateTensor(name)) if name == "duplicate"
        ));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reports_missing_checkpoint_or_empty_directory() {
        let dir = temp_dir();
        assert!(matches!(
            SafeTensorCheckpoint::open(&dir),
            Err(CheckpointError::NotFound(_))
        ));
        fs::create_dir_all(&dir).unwrap();
        assert!(matches!(
            SafeTensorCheckpoint::open(&dir),
            Err(CheckpointError::NoShards(_))
        ));
        fs::remove_dir_all(dir).unwrap();
    }
}
