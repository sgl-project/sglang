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
        // Per-worker URL syntax + scheme is enforced by the typed
        // `WorkerConfig.url: reqwest::Url` deserializer; no further check
        // needed here.
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
        assert_eq!(c.workers[0].url.as_str(), "http://127.0.0.1:30000/");
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
        assert_eq!(c.workers[0].url.as_str(), "http://127.0.0.1:30000/");
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

    // Worker URL parsing.
    //
    // The field is `reqwest::Url` (re-export of `url::Url`). The `url` crate
    // normalizes a bare authority to include a `/` path: parsing
    // `"http://x:30000"` yields a `Url` whose `as_str()` is
    // `"http://x:30000/"`. So both forms below converge on the same string.

    #[test]
    fn loads_config_with_url_having_trailing_slash() {
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
  - url: "http://127.0.0.1:30000/"
"#,
        )
        .unwrap();
        let c = Config::from_path(&p).unwrap();
        // url crate normalizes — both with and without trailing slash end up
        // with a path of "/". We pin the normalized form here.
        assert_eq!(c.workers[0].url.as_str(), "http://127.0.0.1:30000/");
    }

    #[test]
    fn loads_config_without_trailing_slash_normalizes_to_trailing_slash() {
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
        assert_eq!(c.workers[0].url.as_str(), "http://127.0.0.1:30000/");
    }

    #[test]
    fn loads_config_rejects_malformed_url() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.yaml");
        std::fs::write(
            &p,
            r#"
server:
  host: "0.0.0.0"
  port: 8090
models: []
workers:
  - url: "not-a-url"
"#,
        )
        .unwrap();
        let err = Config::from_path(&p).unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("url") || msg.contains("relative") || msg.contains("scheme"),
            "expected URL parse failure, got: {err}"
        );
    }

    #[test]
    fn loads_config_rejects_url_without_scheme() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.yaml");
        std::fs::write(
            &p,
            r#"
server:
  host: "0.0.0.0"
  port: 8090
models: []
workers:
  - url: "127.0.0.1:30000"
"#,
        )
        .unwrap();
        // "127.0.0.1:30000" parses as scheme "127.0.0.1" — the url crate
        // accepts this (it's a syntactically valid URI with an opaque path).
        // Our validator rejects schemes other than http/https.
        let err = Config::from_path(&p).unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("scheme") || msg.contains("http") || msg.contains("url"),
            "expected scheme validation failure, got: {err}"
        );
    }

    #[test]
    fn loads_config_rejects_empty_url() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.yaml");
        std::fs::write(
            &p,
            r#"
server:
  host: "0.0.0.0"
  port: 8090
models: []
workers:
  - url: ""
"#,
        )
        .unwrap();
        let err = Config::from_path(&p).unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("url") || msg.contains("empty") || msg.contains("relative"),
            "expected empty-URL failure, got: {err}"
        );
    }

    #[test]
    fn loads_config_rejects_url_with_path() {
        // A worker URL with a non-trivial path silently loses the path when
        // we later join `/v1/tokenize` etc., so we reject up-front.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.yaml");
        std::fs::write(
            &p,
            r#"
server:
  host: "0.0.0.0"
  port: 8090
models: []
workers:
  - url: "http://x:30000/api/"
"#,
        )
        .unwrap();
        let err = Config::from_path(&p).unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("path"),
            "expected path-rejection error, got: {err}"
        );
    }

    #[test]
    fn loads_config_rejects_url_with_query() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.yaml");
        std::fs::write(
            &p,
            r#"
server:
  host: "0.0.0.0"
  port: 8090
models: []
workers:
  - url: "http://x:30000/?key=foo"
"#,
        )
        .unwrap();
        let err = Config::from_path(&p).unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("query"),
            "expected query-rejection error, got: {err}"
        );
    }

    #[test]
    fn loads_config_rejects_url_with_fragment() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.yaml");
        std::fs::write(
            &p,
            r#"
server:
  host: "0.0.0.0"
  port: 8090
models: []
workers:
  - url: "http://x:30000/#frag"
"#,
        )
        .unwrap();
        let err = Config::from_path(&p).unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("fragment"),
            "expected fragment-rejection error, got: {err}"
        );
    }

    #[test]
    fn loads_config_accepts_bare_authority() {
        // Both with and without trailing slash must pass — they're the
        // canonical forms our deserializer allows.
        for url in &["http://x:30000", "http://x:30000/"] {
            let dir = tempfile::tempdir().unwrap();
            let p = dir.path().join("c.yaml");
            std::fs::write(
                &p,
                format!(
                    r#"
server:
  host: "0.0.0.0"
  port: 8090
models: []
workers:
  - url: "{url}"
"#
                ),
            )
            .unwrap();
            let c = Config::from_path(&p)
                .unwrap_or_else(|e| panic!("expected {url} to pass, got: {e}"));
            assert_eq!(c.workers[0].url.as_str(), "http://x:30000/");
        }
    }

    #[test]
    fn url_join_drops_existing_path_for_absolute_input() {
        // Pin the Url::join semantics we rely on in proxy::forward_*. An
        // absolute path ("/v1/...") replaces the existing path, so a
        // trailing-slash worker URL and a non-trailing one both produce the
        // same joined URL — no double-slash bug.
        let base_with = reqwest::Url::parse("http://x:30000/").unwrap();
        let base_without = reqwest::Url::parse("http://x:30000").unwrap();
        assert_eq!(
            base_with.join("/v1/tokenize").unwrap().as_str(),
            "http://x:30000/v1/tokenize"
        );
        assert_eq!(
            base_without.join("/v1/tokenize").unwrap().as_str(),
            "http://x:30000/v1/tokenize"
        );
    }
}
