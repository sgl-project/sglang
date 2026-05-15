pub mod types;
pub use types::*;

use anyhow::Context as _;
use anyhow::{anyhow, Result};
use std::path::Path;

impl Config {
    pub fn from_path(p: &Path) -> Result<Self> {
        let raw =
            std::fs::read_to_string(p).with_context(|| format!("read config {}", p.display()))?;
        let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("");
        let cfg: Config = match ext {
            "yaml" | "yml" => serde_yaml::from_str(&raw)
                .map_err(|e| anyhow!("parse yaml {}: {e}", p.display()))?,
            "toml" => {
                toml::from_str(&raw).map_err(|e| anyhow!("parse toml {}: {e}", p.display()))?
            }
            other => {
                return Err(anyhow!(
                    "unsupported config extension {other:?}; want yaml/yml/toml"
                ))
            }
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.workers.is_empty() {
            return Err(anyhow!("at least one worker required"));
        }
        for w in &self.workers {
            if w.url.is_empty() {
                return Err(anyhow!("worker.url must be set"));
            }
        }
        for m in &self.models {
            if m.id.is_empty() {
                return Err(anyhow!("model.id must be non-empty"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_minimal_yaml() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.yaml");
        std::fs::write(
            &p,
            r#"
server:
  host: "0.0.0.0"
  port: 8090
models:
  - id: "qwen3-0.6b"
    tokenizer_path: "/tmp/qwen.json"
workers:
  - url: "http://127.0.0.1:30000"
"#,
        )
        .unwrap();
        let c = Config::from_path(&p).unwrap();
        assert_eq!(c.server.port, 8090);
        assert_eq!(c.models[0].id, "qwen3-0.6b");
        assert_eq!(c.workers[0].url, "http://127.0.0.1:30000");
    }

    #[test]
    fn loads_minimal_toml() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.toml");
        std::fs::write(
            &p,
            r#"
[server]
host = "0.0.0.0"
port = 8090
[[models]]
id = "qwen3-0.6b"
tokenizer_path = "/tmp/qwen.json"
[[workers]]
url = "http://127.0.0.1:30000"
"#,
        )
        .unwrap();
        let c = Config::from_path(&p).unwrap();
        assert_eq!(c.server.port, 8090);
    }

    #[test]
    fn loads_workers_array() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.toml");
        std::fs::write(
            &p,
            r#"
[server]
host = "127.0.0.1"
port = 8090
[[models]]
id = "qwen3-0.6b"
tokenizer_path = "/tmp/qwen.json"
[[workers]]
url = "http://127.0.0.1:30000"
"#,
        )
        .unwrap();
        let c = Config::from_path(&p).unwrap();
        assert_eq!(c.workers.len(), 1);
        assert_eq!(c.workers[0].url, "http://127.0.0.1:30000");
    }

    #[test]
    fn rejects_empty_workers() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.yaml");
        std::fs::write(
            &p,
            "server:\n  host: \"0.0.0.0\"\n  port: 8090\nmodels: []\nworkers: []\n",
        )
        .unwrap();
        let err = Config::from_path(&p).unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("worker"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_missing_workers() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.yaml");
        std::fs::write(
            &p,
            "server:\n  host: \"0.0.0.0\"\n  port: 8090\nmodels: []\n",
        )
        .unwrap();
        let err = Config::from_path(&p).unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("worker"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_unknown_extension() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.txt");
        std::fs::write(&p, "").unwrap();
        let err = Config::from_path(&p).unwrap_err();
        assert!(err.to_string().contains("yaml") && err.to_string().contains("toml"));
    }
}
