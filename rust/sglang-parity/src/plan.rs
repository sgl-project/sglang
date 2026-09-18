//! Resolve named SGLang configurations and explicit case bindings before execution.

use crate::compare::ComparisonRules;
use crate::environment::{Backend, EnvironmentConfig};
use crate::http::HttpCase;
use crate::process::ServerConfig;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

/// Environment and lifecycle limits; comparison rules belong to [`HttpSuite`].
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    pub server: ServerConfig,
    /// Historical declarations retained when reading reports, never executed.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub profiles: BTreeMap<String, Value>,
    #[serde(default)]
    pub environment: EnvironmentConfig,
    #[serde(default = "default_startup_timeout")]
    pub startup_timeout_secs: u64,
    #[serde(default = "default_request_timeout")]
    pub request_timeout_secs: u64,
    #[serde(default = "default_shutdown_timeout")]
    pub shutdown_timeout_secs: u64,
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,
}

fn default_startup_timeout() -> u64 {
    600
}
fn default_request_timeout() -> u64 {
    120
}
fn default_shutdown_timeout() -> u64 {
    60
}
fn default_output_dir() -> PathBuf {
    PathBuf::from("target/parity")
}

/// The response view an API policy validates and prepares for comparison.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CheckTarget {
    #[default]
    FullResponse,
    GeneratedContent,
}

/// The single resolved specification used by describe, execution, and reporting.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HttpSuite {
    pub name: String,
    #[serde(default)]
    pub check: CheckTarget,
    pub response_implementation: String,
    pub output_mode: String,
    /// Opaque API rules retained for review; only the suite interprets them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_policy: Option<Value>,
    pub comparison: ComparisonRules,
    pub cases: Vec<HttpCase>,
}

impl RunConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.profiles.is_empty() {
            return Err("profiles belong to the suite; RunConfig.profiles is only retained for reading old reports".into());
        }
        self.server.validate()?;
        if self.environment.setup_timeout_secs == 0
            || self.startup_timeout_secs == 0
            || self.request_timeout_secs == 0
            || !(1..=60).contains(&self.shutdown_timeout_secs)
        {
            return Err(
                "timeouts must be positive; shutdown timeout must be at most 60 seconds".into(),
            );
        }
        if self.output_dir.as_os_str().is_empty() {
            return Err("output_dir must not be empty".into());
        }
        Ok(())
    }
}

impl HttpSuite {
    pub fn validate(&self) -> Result<(), String> {
        self.comparison.validate()?;
        if self.name.trim().is_empty()
            || self.response_implementation.trim().is_empty()
            || self.output_mode.trim().is_empty()
            || self.cases.is_empty()
        {
            return Err(
                "suite identity, response implementation, output mode, and cases are required"
                    .into(),
            );
        }
        let mut names = BTreeSet::new();
        let mut groups: BTreeMap<&str, usize> = BTreeMap::new();
        for case in &self.cases {
            if case.name.is_empty()
                || !case
                    .name
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
                || !names.insert(&case.name)
            {
                return Err(format!(
                    "case name must be unique and contain only letters, digits, '_' or '-': {:?}",
                    case.name
                ));
            }
            case.request
                .validate()
                .map_err(|e| format!("{}: {e}", case.name))?;
            case.requires.validate()?;
            let mut assertions = BTreeSet::new();
            if case
                .assertions
                .iter()
                .any(|name| name.trim().is_empty() || !assertions.insert(name))
            {
                return Err(format!(
                    "{}: assertion names must be nonempty and unique",
                    case.name
                ));
            }
            for step in &case.before_each {
                step.validate()?;
            }
            if let Some(group) = &case.equivalence_group {
                if group.trim().is_empty() {
                    return Err("equivalence group must not be empty".into());
                }
                *groups.entry(group).or_default() += 1;
            }
        }
        if let Some((group, _)) = groups.iter().find(|(_, count)| **count < 2) {
            return Err(format!(
                "equivalence group {group:?} needs at least two cases"
            ));
        }
        Ok(())
    }
}

/// Host requirements can be tightened by an individual case.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Requirements {
    #[serde(default)]
    pub backends: Vec<Backend>,
    #[serde(default)]
    pub min_cuda_devices: u32,
}

impl Requirements {
    pub fn validate(&self) -> Result<(), String> {
        if self.backends.contains(&Backend::Auto) {
            return Err("requirements must name mlx or cuda, not auto".into());
        }
        let mut seen = BTreeSet::new();
        if self.backends.iter().any(|b| !seen.insert(format!("{b:?}"))) {
            return Err("duplicate required backend".into());
        }
        if self.min_cuda_devices > 0
            && !self.backends.is_empty()
            && !self.backends.contains(&Backend::Cuda)
        {
            return Err("CUDA device requirements conflict with selected backends".into());
        }
        Ok(())
    }

    pub fn combine(&self, other: &Self) -> Result<Self, String> {
        self.validate()?;
        other.validate()?;
        let backends = [Backend::Mlx, Backend::Cuda]
            .into_iter()
            .filter(|b| {
                (self.backends.is_empty() || self.backends.contains(b))
                    && (other.backends.is_empty() || other.backends.contains(b))
                    && (self.min_cuda_devices.max(other.min_cuda_devices) == 0
                        || *b == Backend::Cuda)
            })
            .collect::<Vec<_>>();
        if backends.is_empty() {
            return Err("profile and case requirements cannot be satisfied together".into());
        }
        Ok(Self {
            backends,
            min_cuda_devices: self.min_cuda_devices.max(other.min_cuda_devices),
        })
    }

    pub fn unavailable(&self, backend: Backend, devices: Option<u32>) -> Option<String> {
        if !self.backends.is_empty() && !self.backends.contains(&backend) {
            return Some(format!(
                "requires {:?}; current backend is {backend:?}",
                self.backends
            ));
        }
        if self.min_cuda_devices > 0
            && (backend != Backend::Cuda || devices.is_none_or(|n| n < self.min_cuda_devices))
        {
            return Some(format!(
                "requires {} CUDA devices; available {}",
                self.min_cuda_devices,
                devices.unwrap_or(0)
            ));
        }
        None
    }
}

/// A scenario's additions to the environment's base server configuration.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileSpec {
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    pub radix_cache: Option<bool>,
    #[serde(default)]
    pub requires: Requirements,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ResolvedProfile {
    pub id: String,
    pub server: ServerConfig,
    pub requires: Requirements,
}

/// A profile's compiled API policy and only the cases explicitly bound to it.
pub struct ProfilePlan<P> {
    pub profile: ResolvedProfile,
    pub suite: HttpSuite,
    pub policy: P,
}

pub struct ExecutionPlan<P> {
    pub profiles: Vec<ProfilePlan<P>>,
}

pub fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
}

impl ProfileSpec {
    /// Resolve one explicit profile without changing the base server.
    pub fn resolve(&self, id: &str, base: &ServerConfig) -> Result<ResolvedProfile, String> {
        if !valid_name(id) {
            return Err(format!("invalid profile name {id:?}"));
        }
        self.requires.validate()?;
        let mut server = base.clone();
        server.args.extend(self.args.clone());
        let mut options: Vec<&str> = Vec::new();
        for arg in &server.args {
            if arg.starts_with("--") {
                let option = arg.split('=').next().unwrap();
                if let Some(previous) = options
                    .iter()
                    .find(|previous| option.starts_with(**previous) || previous.starts_with(option))
                {
                    return Err(format!(
                        "profile {id}: duplicate or ambiguous options {previous} and {option}"
                    ));
                }
                options.push(option);
            }
        }
        for (key, value) in &self.env {
            if server.env.insert(key.clone(), value.clone()).is_some() {
                return Err(format!(
                    "profile {id}: environment variable {key} is already set by the environment"
                ));
            }
        }
        if let Some(enabled) = self.radix_cache {
            server.radix_cache = enabled;
        }
        server.validate()?;
        Ok(ResolvedProfile {
            id: id.into(),
            server,
            requires: self.requires.clone(),
        })
    }
}

impl<P> ExecutionPlan<P> {
    pub fn validate(&self) -> Result<(), String> {
        if self.profiles.is_empty() {
            return Err("no cases are bound to profiles".into());
        }
        let mut ids = BTreeSet::new();
        for entry in &self.profiles {
            if entry.suite.check != self.profiles[0].suite.check {
                return Err("all profiles must use the same check target".into());
            }
            if !valid_name(&entry.profile.id) || !ids.insert(&entry.profile.id) {
                return Err("profile names must be valid and unique".into());
            }
            entry.profile.server.validate()?;
            entry.profile.requires.validate()?;
            entry.suite.validate()?;
            for case in &entry.suite.cases {
                entry.profile.requires.combine(&case.requires)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn profiles_append_settings_and_reject_conflicts() {
        let base: ServerConfig = serde_json::from_value(json!({
            "model":"base", "seed":7, "args":["--device","mps"], "env":{"COMMON":"yes"}
        }))
        .unwrap();
        let profile: ProfileSpec = serde_json::from_value(json!({
            "args":["--incremental-streaming-output"], "env":{"SCENARIO":"enabled"}, "radix_cache":true
        })).unwrap();
        let resolved = profile.resolve("variant", &base).unwrap();
        assert_eq!(resolved.server.model, base.model);
        assert_eq!(resolved.server.seed, base.seed);
        assert_eq!(
            resolved.server.args,
            ["--device", "mps", "--incremental-streaming-output"]
        );
        assert_eq!(resolved.server.env["COMMON"], "yes");
        assert_eq!(resolved.server.env["SCENARIO"], "enabled");
        assert!(resolved.server.radix_cache);
        assert!(!base.radix_cache);
        assert_eq!(
            ProfileSpec::default()
                .resolve("default", &base)
                .unwrap()
                .server,
            base
        );
        for name in ["bad/name", ""] {
            assert!(profile.resolve(name, &base).is_err());
        }
        for value in [
            json!({"args":["--device=cuda"]}),
            json!({"args":["--dev","mps"]}),
            json!({"args":["--device-extra"]}),
            json!({"env":{"COMMON":"override"}}),
            json!({"args":["--port","9999"]}),
            json!({"args":["--foo=1","--foo","2"]}),
        ] {
            let invalid: ProfileSpec = serde_json::from_value(value.clone()).unwrap();
            assert!(invalid.resolve("invalid", &base).is_err(), "{value}");
        }
        for obsolete in [
            json!({"server":{}}),
            json!({"model":"other"}),
            json!({"seed":99}),
            json!({"extra_args":[]}),
        ] {
            assert!(serde_json::from_value::<ProfileSpec>(obsolete).is_err());
        }
        let mlx = Requirements {
            backends: vec![Backend::Mlx],
            ..Default::default()
        };
        let two_devices = Requirements {
            min_cuda_devices: 2,
            ..Default::default()
        };
        assert!(mlx.combine(&two_devices).is_err());
        let cuda = Requirements::default().combine(&two_devices).unwrap();
        for (backend, devices, available) in [
            (Backend::Mlx, Some(2), false),
            (Backend::Cuda, None, false),
            (Backend::Cuda, Some(1), false),
            (Backend::Cuda, Some(2), true),
        ] {
            assert_eq!(cuda.unavailable(backend, devices).is_none(), available);
        }
    }
}
