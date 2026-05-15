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
        const VALID_POLICIES: &[&str] = &["round_robin", "random", "power_of_two"];

        for m in &self.models {
            if m.id.is_empty() {
                return Err(anyhow!("model.id must be non-empty"));
            }
            if !VALID_POLICIES.contains(&m.policy.as_str()) {
                return Err(anyhow!(
                    "model.policy = {:?} not recognized; valid: {VALID_POLICIES:?}",
                    m.policy
                ));
            }
        }
        match &self.discovery.backend {
            DiscoveryBackend::StaticFile(s) => {
                if s.path.is_empty() {
                    return Err(anyhow!("discovery.static_file.path must be set"));
                }
            }
            DiscoveryBackend::K8s(k) => {
                if k.namespace.is_empty() {
                    return Err(anyhow!("discovery.k8s.namespace must be set"));
                }
                // label_selector may be empty (matches everything in namespace)
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
discovery:
  backend: static_file
  static_file:
    path: "/tmp/workers.toml"
"#,
        )
        .unwrap();
        let c = Config::from_path(&p).unwrap();
        assert_eq!(c.server.port, 8090);
        assert_eq!(c.models[0].id, "qwen3-0.6b");
        match &c.discovery.backend {
            DiscoveryBackend::StaticFile(s) => assert_eq!(s.path, "/tmp/workers.toml"),
            _ => panic!("expected static_file backend"),
        }
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
[discovery]
backend = "static_file"
[discovery.static_file]
path = "/tmp/workers.toml"
"#,
        )
        .unwrap();
        let c = Config::from_path(&p).unwrap();
        assert_eq!(c.server.port, 8090);
        match &c.discovery.backend {
            DiscoveryBackend::StaticFile(s) => assert_eq!(s.path, "/tmp/workers.toml"),
            _ => panic!("expected static_file backend"),
        }
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
            err.to_string().to_lowercase().contains("discovery")
                || err.to_string().to_lowercase().contains("missing"),
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

    #[test]
    fn loads_static_file_discovery() {
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
policy = "round_robin"
[discovery]
backend = "static_file"
[discovery.static_file]
path = "/etc/experimental/sgl-router/workers.toml"
"#,
        )
        .unwrap();
        let c = Config::from_path(&p).unwrap();
        match &c.discovery.backend {
            DiscoveryBackend::StaticFile(s) => {
                assert_eq!(s.path, "/etc/experimental/sgl-router/workers.toml");
            }
            _ => panic!("expected static_file backend"),
        }
        assert_eq!(c.models[0].policy, "round_robin");
    }

    #[test]
    fn loads_k8s_discovery() {
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
policy = "round_robin"
[discovery]
backend = "k8s"
[discovery.k8s]
namespace = "default"
label_selector = "app=sglang"
"#,
        )
        .unwrap();
        let c = Config::from_path(&p).unwrap();
        match &c.discovery.backend {
            DiscoveryBackend::K8s(k) => {
                assert_eq!(k.namespace, "default");
                assert_eq!(k.label_selector, "app=sglang");
            }
            _ => panic!("expected k8s backend"),
        }
    }

    #[test]
    fn rejects_unknown_policy_name() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.yaml");
        std::fs::write(
            &p,
            "
server:
  host: 0.0.0.0
  port: 8090
discovery:
  backend: static_file
  static_file:
    path: /tmp/w.toml
models:
  - id: qwen
    tokenizer_path: /tmp/qwen.json
    policy: bogus_policy
",
        )
        .unwrap();
        let err = Config::from_path(&p).unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("bogus_policy")
                || err.to_string().to_lowercase().contains("policy"),
            "got: {err}"
        );
    }

    #[test]
    fn defaults_policy_to_round_robin() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.toml");
        std::fs::write(
            &p,
            r#"
[server]
host = "127.0.0.1"
port = 8090
[[models]]
id = "qwen"
tokenizer_path = "/tmp/qwen.json"
[discovery]
backend = "static_file"
[discovery.static_file]
path = "/tmp/w.toml"
"#,
        )
        .unwrap();
        let c = Config::from_path(&p).unwrap();
        assert_eq!(c.models[0].policy, "round_robin");
    }
}
