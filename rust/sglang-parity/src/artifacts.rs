//! Persist observations without changing their contents or deciding test outcomes.

use std::path::{Path, PathBuf};

use serde::Serialize;
use uuid::Uuid;

/// Artifact directory belonging to one run.
#[derive(Debug)]
pub struct Artifacts {
    root: PathBuf,
}

impl Artifacts {
    pub fn create(output: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(output)?;
        let root = output.join(Uuid::new_v4().to_string());
        std::fs::create_dir(&root)?;
        Ok(Self {
            root: std::fs::canonicalize(root)?,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn directory(&self, relative: impl AsRef<Path>) -> std::io::Result<PathBuf> {
        let path = self.root.join(relative);
        std::fs::create_dir_all(&path)?;
        Ok(path)
    }

    /// Replace a JSON artifact atomically so interrupted runs keep a readable report.
    pub fn write_json(&self, path: &Path, value: &impl Serialize) -> std::io::Result<()> {
        let mut bytes = serde_json::to_vec_pretty(value)?;
        bytes.push(b'\n');
        write_atomic(path, &bytes)
    }
}

/// Replace an artifact without exposing a partially written file to readers.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut pending = path.as_os_str().to_owned();
    pending.push(".pending");
    std::fs::write(&pending, bytes)?;
    std::fs::rename(pending, path)
}
