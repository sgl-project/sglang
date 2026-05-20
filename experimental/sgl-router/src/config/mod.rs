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
        // Unknown policy names are rejected by serde via `PolicyKind`'s
        // `rename_all = "snake_case"`; threshold = 0 is rejected by
        // `NonZeroU32`.  Only fields without a type-system constraint are
        // checked here.
        for m in &self.models {
            if m.id.is_empty() {
                return Err(anyhow!("model.id must be non-empty"));
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
                // Reject invalid selector combinations at load time (no
                // selectors, mixed plain+PD, partial PD).
                k.mode().map_err(|e| anyhow!("{e}"))?;
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
        assert_eq!(c.models[0].policy, PolicyKind::RoundRobin);
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
                assert_eq!(k.label_selector.as_deref(), Some("app=sglang"));
                assert!(k.prefill_selector.is_none());
                assert!(k.decode_selector.is_none());
            }
            _ => panic!("expected k8s backend"),
        }
    }

    #[test]
    fn loads_k8s_pd_discovery_with_prefill_and_decode_selectors() {
        // K8s PD: selectors drive slice-classification; the actual
        // bootstrap_port for each prefill worker comes from
        // `/server_info` post-discovery (see
        // `crate::workers::introspect`). The config layer only validates
        // the selector combination here — successful load means PD on
        // K8s is wired through to the manager.
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
[discovery]
backend = "k8s"
[discovery.k8s]
namespace = "default"
prefill_selector = "app=sglang,role=prefill"
decode_selector = "app=sglang,role=decode"
"#,
        )
        .unwrap();
        let c = Config::from_path(&p).expect("k8s PD config must load");
        match &c.discovery.backend {
            DiscoveryBackend::K8s(k) => {
                assert_eq!(k.namespace, "default");
                assert_eq!(k.prefill_selector.as_deref(), Some("app=sglang,role=prefill"));
                assert_eq!(k.decode_selector.as_deref(), Some("app=sglang,role=decode"));
                assert!(k.label_selector.is_none());
            }
            _ => panic!("expected k8s backend"),
        }
    }

    #[test]
    fn rejects_k8s_config_with_no_selector() {
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
backend = "k8s"
[discovery.k8s]
namespace = "default"
"#,
        )
        .unwrap();
        let err = Config::from_path(&p).unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(msg.contains("selector"), "got: {err}");
    }

    fn k8s_cfg(
        label: Option<&str>,
        prefill: Option<&str>,
        decode: Option<&str>,
    ) -> K8sDiscoveryConfig {
        K8sDiscoveryConfig {
            namespace: "default".into(),
            label_selector: label.map(str::to_string),
            prefill_selector: prefill.map(str::to_string),
            decode_selector: decode.map(str::to_string),
        }
    }

    #[test]
    fn k8s_mode_rejects_when_no_selector_is_set() {
        let err = k8s_cfg(None, None, None).mode().unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("selector"),
            "expected selector error, got: {err}"
        );
    }

    #[test]
    fn k8s_mode_rejects_mixed_plain_and_pd_selectors() {
        let err = k8s_cfg(
            Some("app=sglang"),
            Some("role=prefill"),
            Some("role=decode"),
        )
        .mode()
        .unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("label_selector") && msg.contains("prefill"),
            "expected mixed-mode error, got: {err}"
        );
    }

    #[test]
    fn k8s_mode_rejects_partial_pd_selectors() {
        let err = k8s_cfg(None, Some("role=prefill"), None)
            .mode()
            .unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("prefill_selector") && msg.contains("decode_selector"),
            "expected partial-PD error, got: {err}"
        );

        let err = k8s_cfg(None, None, Some("role=decode")).mode().unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("prefill_selector") && msg.contains("decode_selector"),
            "expected partial-PD error, got: {err}"
        );
    }

    /// Plain `label_selector = Some("")` STAYS valid — empty selector
    /// matches every EndpointSlice in the namespace, which is a documented
    /// k8s behavior and the user explicitly opts in to "match all" by
    /// setting plain mode. The empty-rejection is a PD-mode-only safeguard.
    #[test]
    fn k8s_mode_accepts_empty_plain_label_selector() {
        let mode = k8s_cfg(Some(""), None, None)
            .mode()
            .expect("plain mode valid");
        match mode {
            K8sDiscoveryMode::Plain { label_selector } => {
                assert_eq!(label_selector, "");
            }
            other => panic!("expected Plain, got {other:?}"),
        }
    }

    #[test]
    fn k8s_mode_accepts_plain_with_label_selector() {
        let mode = k8s_cfg(Some("app=sglang"), None, None).mode().unwrap();
        match mode {
            K8sDiscoveryMode::Plain { label_selector } => {
                assert_eq!(label_selector, "app=sglang")
            }
            other => panic!("expected Plain, got {other:?}"),
        }
    }

    #[test]
    fn k8s_mode_constructs_pd_disaggregation() {
        // Selectors drive slice-classification only; the per-worker
        // bootstrap_port is filled from `/server_info` by the worker
        // manager. K8s PD is fully supported as of the
        // /server_info-derived-disaggregation-role commit.
        let mode = k8s_cfg(None, Some("role=prefill"), Some("role=decode"))
            .mode()
            .expect("PD mode is valid");
        assert_eq!(
            mode,
            K8sDiscoveryMode::PdDisaggregation {
                prefill_selector: "role=prefill".to_string(),
                decode_selector: "role=decode".to_string(),
            }
        );
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
        assert_eq!(c.models[0].policy, PolicyKind::RoundRobin);
    }
}
