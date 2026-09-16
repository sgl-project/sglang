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
    #[serde(default)]
    pub profiles: BTreeMap<String, ProfileSpec>,
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

/// The single resolved specification used by describe, execution, and reporting.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HttpSuite {
    pub name: String,
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

/// Startup overrides. Argument vectors replace rather than append to the base.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServerOverrides {
    pub model: Option<String>,
    pub seed: Option<u64>,
    pub args: Option<Vec<String>>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    pub radix_cache: Option<bool>,
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

/// One named startup configuration; there is no profile-to-profile inheritance.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileSpec {
    #[serde(default)]
    pub server: ServerOverrides,
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

impl RunConfig {
    /// Resolve the implicit default first, followed by named profiles in key order.
    pub fn resolve_profiles(&self) -> Result<Vec<ResolvedProfile>, String> {
        self.validate()?;
        let mut result = vec![ResolvedProfile {
            id: "default".into(),
            server: self.server.clone(),
            requires: Requirements::default(),
        }];
        for (id, spec) in &self.profiles {
            if id == "default" || !valid_name(id) {
                return Err(format!("invalid or reserved profile name {id:?}"));
            }
            spec.requires.validate()?;
            let mut server = self.server.clone();
            if let Some(model) = &spec.server.model {
                server.model.clone_from(model);
            }
            if let Some(seed) = spec.server.seed {
                server.seed = seed;
            }
            if let Some(args) = &spec.server.args {
                server.args.clone_from(args);
            }
            server.env.extend(spec.server.env.clone());
            if let Some(enabled) = spec.server.radix_cache {
                server.radix_cache = enabled;
            }
            server.validate()?;
            result.push(ResolvedProfile {
                id: id.clone(),
                server,
                requires: spec.requires.clone(),
            });
        }
        Ok(result)
    }
}

impl<P> ExecutionPlan<P> {
    pub fn validate(&self) -> Result<(), String> {
        if self.profiles.is_empty() {
            return Err("no cases are bound to profiles".into());
        }
        let mut ids = BTreeSet::new();
        for entry in &self.profiles {
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
    fn overrides_replace_argv_overlay_environment_and_reject_conflicts() {
        let config: RunConfig = serde_json::from_value(json!({
            "server": {"model":"base", "args":["--device","mps"], "env":{"COMMON":"yes","MODE":"base"}},
            "profiles": {"variant": {"server": {"model":"other", "seed":7,
                "args":["--incremental-streaming-output"], "env":{"MODE":"variant"}, "radix_cache":true}}}
        })).unwrap();
        let profiles = config.resolve_profiles().unwrap();
        assert_eq!(
            profiles.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(),
            ["default", "variant"]
        );
        let server = &profiles[1].server;
        assert_eq!(server.model, "other");
        assert_eq!(server.args, ["--incremental-streaming-output"]);
        assert_eq!(server.env["COMMON"], "yes");
        assert_eq!(server.env["MODE"], "variant");
        assert!(server.radix_cache);
        assert!(!profiles[0].server.radix_cache);
        assert_eq!(server.seed, 7);
        for name in ["default", "bad/name", ""] {
            let mut invalid = config.clone();
            invalid.profiles.insert(name.into(), ProfileSpec::default());
            assert!(invalid.resolve_profiles().is_err());
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
