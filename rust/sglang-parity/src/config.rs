//! Explicit run selection and the built-in platform definitions.

use std::collections::BTreeSet;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::environment::EnvironmentConfig;
use crate::{CheckTarget, RunConfig, ServerConfig};

/// The complete user-facing run configuration. Every field is required.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunSpec {
    pub environment: String,
    pub suites: Vec<String>,
    pub check: CheckTarget,
    pub output_dir: PathBuf,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EnvironmentSpec {
    environment: EnvironmentConfig,
    server: ServerConfig,
    startup_timeout_secs: u64,
    request_timeout_secs: u64,
    shutdown_timeout_secs: u64,
}

impl RunSpec {
    /// Parse the single supported input format and explain obsolete inputs.
    pub fn parse(text: &str) -> Result<Self, String> {
        let value: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        if value.get("server").is_some() || value.get("profiles").is_some() {
            return Err("obsolete run configuration: use exactly environment, suites, check and output_dir; server settings belong to the built-in environment and profiles belong to the suites. Old settings cannot be carried over implicitly".into());
        }
        let spec: Self = serde_json::from_str(text).map_err(|e| e.to_string())?;
        spec.validate()?;
        Ok(spec)
    }

    fn validate(&self) -> Result<(), String> {
        if !matches!(self.environment.as_str(), "mlx" | "cuda") {
            return Err(format!(
                "unknown environment {:?}; expected mlx or cuda",
                self.environment
            ));
        }
        let mut seen = BTreeSet::new();
        if self.suites.is_empty()
            || self
                .suites
                .iter()
                .any(|name| name.trim().is_empty() || !seen.insert(name))
        {
            return Err("suites must be nonempty and contain distinct nonempty names".into());
        }
        if self.output_dir.as_os_str().is_empty() {
            return Err("output_dir must not be empty".into());
        }
        Ok(())
    }

    /// Expand a platform definition without installing or observing Git state.
    pub fn resolve_environment(&self) -> Result<RunConfig, String> {
        self.validate()?;
        let text = match self.environment.as_str() {
            "mlx" => include_str!("../configs/environments/mlx.json"),
            "cuda" => include_str!("../configs/environments/cuda.json"),
            _ => unreachable!("validated environment"),
        };
        let spec: EnvironmentSpec = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let config = RunConfig {
            environment: spec.environment,
            server: spec.server,
            profiles: Default::default(),
            startup_timeout_secs: spec.startup_timeout_secs,
            request_timeout_secs: spec.request_timeout_secs,
            shutdown_timeout_secs: spec.shutdown_timeout_secs,
            output_dir: self.output_dir.clone(),
        };
        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn selection_is_explicit_and_platforms_preserve_pinned_settings() {
        let valid = json!({"environment":"mlx", "suites":["native_generate"],
            "check":"full-response", "output_dir":"target/parity"});
        for key in ["environment", "suites", "check", "output_dir"] {
            let mut missing = valid.clone();
            missing.as_object_mut().unwrap().remove(key);
            assert!(RunSpec::parse(&missing.to_string()).is_err(), "{key}");
        }
        for (key, value) in [
            ("environment", json!("auto")),
            ("suites", json!([])),
            ("suites", json!(["one", "one"])),
            ("output_dir", json!("")),
            ("check", json!("output")),
            ("typo", json!(true)),
        ] {
            let mut invalid = valid.clone();
            invalid[key] = value;
            assert!(RunSpec::parse(&invalid.to_string()).is_err(), "{key}");
        }
        assert!(
            RunSpec::parse(r#"{"server":{"model":"old"}}"#)
                .unwrap_err()
                .contains("obsolete")
        );
        for text in [
            include_str!("../configs/mlx.json"),
            include_str!("../configs/cuda.json"),
        ] {
            let spec = RunSpec::parse(text).unwrap();
            let config = spec.resolve_environment().unwrap();
            assert_eq!(config.output_dir, PathBuf::from("target/parity"));
            assert!(config.profiles.is_empty());
            assert!(config.server.python.is_none());
            assert!(
                config
                    .server
                    .args
                    .windows(2)
                    .any(|p| p[0] == "--revision" && p[1].len() == 40)
            );
        }
    }
}
