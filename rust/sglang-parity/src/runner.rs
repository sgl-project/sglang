//! Execute one resolved suite against both SGLang implementations.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::artifacts::Artifacts;
use crate::compare::{ComparisonRules, Difference, Violation, compare_json, prepare_comparison};
use crate::environment::{self, EnvironmentConfig, EnvironmentPlan, PreparedEnvironment};
use crate::http::{self, CaptureMode, HttpCase, HttpObservation};
use crate::process::{Implementation, ServerConfig, SglangProcess};
use crate::progress::track;

/// Environment and lifecycle limits; comparison rules belong to [`HttpSuite`].
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    pub server: ServerConfig,
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
    pub comparison: ComparisonRules,
    pub cases: Vec<HttpCase>,
}

/// Validate an API response and return its full final JSON without masking fields.
pub trait ResponsePolicy {
    /// Validate the API contract and reconstruct the complete response.
    ///
    /// Implementations must preserve unknown final fields and must not mask
    /// differences. Return violations for invalid responses, including in-band
    /// errors. This method performs no I/O and receives no implementation identity.
    fn prepare(
        &self,
        case: &HttpCase,
        observation: &HttpObservation,
    ) -> Result<Value, Vec<Violation>>;
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
            reqwest::Method::from_bytes(case.method.as_bytes()).map_err(|e| e.to_string())?;
            if !case.path.starts_with('/')
                || case.path.starts_with("//")
                || case.path.contains(['#', '\\'])
                || case
                    .path
                    .bytes()
                    .any(|c| c.is_ascii_whitespace() || c.is_ascii_control())
            {
                return Err(format!(
                    "case {} requires an absolute local HTTP path",
                    case.name
                ));
            }
            if !(200..=599).contains(&case.expect_status) {
                return Err(format!(
                    "case {} has an invalid final HTTP status",
                    case.name
                ));
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

/// The executable specification, also saved verbatim in the run artifacts.
#[derive(Clone, Debug, Serialize)]
pub struct EffectiveSuite {
    pub environment: EnvironmentPlan,
    pub suite: HttpSuite,
    pub repeats_per_implementation: usize,
    pub implementation_order: [Implementation; 2],
}

/// Resolve the review view without starting a process or issuing requests.
///
/// # Errors
/// Returns a configuration error for invalid run settings or suite declarations.
pub fn describe(config: &RunConfig, suite: &HttpSuite) -> Result<EffectiveSuite, RunError> {
    config.validate().map_err(RunError::Config)?;
    suite.validate().map_err(RunError::Config)?;
    Ok(EffectiveSuite {
        environment: environment::describe(config).map_err(RunError::Config)?,
        suite: suite.clone(),
        repeats_per_implementation: 2,
        implementation_order: Implementation::ALL,
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Status {
    #[default]
    NotRun,
    Pass,
    Fail,
    Unstable,
    Skipped,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Check {
    pub status: Status,
    pub differences: Vec<Difference>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Attempt {
    pub directory: PathBuf,
    pub observation: Option<HttpObservation>,
    pub final_json: Option<PathBuf>,
    pub violations: Vec<Violation>,
    #[serde(skip)]
    comparison: Option<Value>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SideResult {
    pub attempts: Vec<Attempt>,
    pub repeatability: Check,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CaseResult {
    pub name: String,
    pub implementations: BTreeMap<String, SideResult>,
    pub parity: Check,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EquivalenceResult {
    pub group: String,
    pub implementation: Implementation,
    pub left: String,
    pub right: String,
    pub check: Check,
}

/// All completed and incomplete work, with paths to unmodified observations.
#[derive(Debug, Serialize, Deserialize)]
pub struct Report {
    pub state: String,
    pub directory: PathBuf,
    pub effective_suite: PathBuf,
    pub config: RunConfig,
    pub environment: Option<Value>,
    pub runtime_errors: Vec<String>,
    pub cases: Vec<CaseResult>,
    pub equivalence: Vec<EquivalenceResult>,
}

impl Report {
    /// Exit 2 takes precedence over protocol/parity failures (1); only a fully
    /// completed, valid, stable, equivalent run returns 0.
    pub fn exit_code(&self) -> i32 {
        if self.state != "complete"
            || !self.runtime_errors.is_empty()
            || self.cases.iter().any(|case| {
                case.implementations.values().any(|side| {
                    side.repeatability.status == Status::Unstable
                        || side.attempts.len() != 2
                        || side.attempts.iter().any(|attempt| {
                            attempt
                                .observation
                                .as_ref()
                                .is_none_or(|o| o.transport_error.is_some())
                        })
                })
            })
        {
            return 2;
        }
        if self.cases.iter().any(|case| {
            case.parity.status != Status::Pass
                || case
                    .implementations
                    .values()
                    .any(|side| side.repeatability.status != Status::Pass)
        }) || self
            .equivalence
            .iter()
            .any(|result| result.check.status != Status::Pass)
        {
            return 1;
        }
        0
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("artifact I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("HTTP client setup failed: {0}")]
    Client(#[from] reqwest::Error),
}

struct RunArtifacts {
    artifacts: Artifacts,
    report: Option<Report>,
}

impl RunArtifacts {
    fn verify_source(&mut self, prepared: &PreparedEnvironment) -> bool {
        match prepared.verify_source() {
            Ok(()) => true,
            Err(error) => {
                self.report
                    .as_mut()
                    .unwrap()
                    .runtime_errors
                    .push(format!("source verification: {error}"));
                false
            }
        }
    }

    fn save(&self) -> std::io::Result<()> {
        let report = self.report.as_ref().expect("active report");
        self.artifacts
            .write_json(&self.artifacts.root().join("report.json"), report)?;
        if matches!(report.state.as_str(), "complete" | "interrupted") {
            crate::report::ReportView::new(report, self.artifacts.root()).write_html()?;
        }
        Ok(())
    }
}

impl Drop for RunArtifacts {
    fn drop(&mut self) {
        if let Some(report) = &mut self.report {
            report.state = "interrupted".into();
            report
                .runtime_errors
                .push("Run did not complete; incomplete attempts are not passing results.".into());
            // Best effort on cancellation; normal writes surface their I/O errors.
            let _ = self.save();
        }
    }
}

/// Run both managed implementations sequentially using the same suite and policy.
///
/// Each request runs twice per implementation. Invalid responses are failures;
/// unstable valid responses skip cross-implementation and equivalence conclusions.
/// The library persists raw data and a partial report, including when cancelled.
/// Dropping the returned future terminates and reaps the managed process group.
///
/// # Errors
/// Returns configuration, client setup, or artifact I/O errors. Service startup,
/// shutdown, environment preparation, and request failures are recorded in the
/// returned [`Report`].
pub async fn run(
    config: &RunConfig,
    suite: &HttpSuite,
    policy: &impl ResponsePolicy,
) -> Result<Report, RunError> {
    tracing::info!("Validating configuration and source revision");
    let effective = describe(config, suite)?;
    let client = http::client()?;
    let artifacts = Artifacts::create(&config.output_dir)?;
    tracing::info!(commit = %effective.environment.commit, backend = ?effective.environment.profile.backend, suite = %suite.name, cases = suite.cases.len(), "Starting parity run");
    tracing::info!(directory = %artifacts.root().display(), "Run artifacts and logs");
    let effective_path = artifacts.root().join("effective_suite.json");
    artifacts.write_json(&effective_path, &effective)?;
    let report = Report {
        state: "preparing".into(),
        directory: artifacts.root().to_owned(),
        effective_suite: effective_path,
        config: config.clone(),
        environment: None,
        runtime_errors: Vec::new(),
        cases: suite
            .cases
            .iter()
            .map(|case| CaseResult {
                name: case.name.clone(),
                implementations: Implementation::ALL
                    .into_iter()
                    .map(|implementation| (implementation.as_str().into(), SideResult::default()))
                    .collect(),
                parity: Check::default(),
            })
            .collect(),
        equivalence: Vec::new(),
    };
    let mut state = RunArtifacts {
        artifacts,
        report: Some(report),
    };
    state.save()?;
    let prepared =
        match environment::prepare(config, &effective.environment, state.artifacts.root()).await {
            Ok(prepared) => prepared,
            Err(error) => {
                tracing::error!(%error, "Environment preparation failed");
                let report = state.report.as_mut().unwrap();
                report
                    .runtime_errors
                    .push(format!("environment preparation: {error}"));
                for case in &mut report.cases {
                    case.parity.status = Status::Skipped;
                }
                report.state = "complete".into();
                state.save()?;
                return Ok(state.report.take().unwrap());
            }
        };
    let report = state.report.as_mut().unwrap();
    report.environment = Some(prepared.record.clone());
    report.state = "running".into();
    state.save()?;
    let mut source_valid = true;
    let requests = suite
        .cases
        .iter()
        .map(|case| serde_json::to_vec_pretty(&case.body))
        .collect::<Result<Vec<_>, _>>()
        .map_err(std::io::Error::from)?;
    for implementation in Implementation::ALL {
        if !state.verify_source(&prepared) {
            source_valid = false;
            break;
        }
        let side_name = implementation.as_str();
        let side_dir = state.artifacts.directory(side_name)?;
        let server_log = side_dir.join("server.log");
        let mut process = match track(
            &format!("Starting {side_name} server; waiting for readiness"),
            &server_log,
            SglangProcess::start(
                &prepared.server,
                implementation,
                &server_log,
                Duration::from_secs(config.startup_timeout_secs),
                Duration::from_secs(config.shutdown_timeout_secs),
            ),
        )
        .await
        {
            Ok(process) => process,
            Err(error) => {
                tracing::error!(implementation = side_name, %error, "Server startup failed");
                state
                    .report
                    .as_mut()
                    .unwrap()
                    .runtime_errors
                    .push(format!("{side_name} startup: {error}"));
                source_valid = state.verify_source(&prepared);
                state.save()?;
                if !source_valid {
                    break;
                }
                continue;
            }
        };
        tracing::info!(implementation = side_name, "Server ready");
        for (index, case) in suite.cases.iter().enumerate() {
            for repeat in 1..=2 {
                let directory = state
                    .artifacts
                    .directory(format!("{side_name}/{}/{repeat}", case.name))?;
                std::fs::write(directory.join("request.json"), &requests[index])?;
                let attempts = &mut state.report.as_mut().unwrap().cases[index]
                    .implementations
                    .get_mut(side_name)
                    .unwrap()
                    .attempts;
                attempts.push(Attempt {
                    directory: directory.clone(),
                    observation: None,
                    final_json: None,
                    violations: Vec::new(),
                    comparison: None,
                });
                state.save()?;
                // The only transport dispatch point; API interpretation follows capture.
                let observation = track(
                    &format!(
                        "{side_name}: case {}/{} {}, repeat {repeat}/2",
                        index + 1,
                        suite.cases.len(),
                        case.name
                    ),
                    &server_log,
                    http::capture(
                        &client,
                        &process.base_url(),
                        case,
                        &requests[index],
                        &directory.join("response.body"),
                        Duration::from_secs(config.request_timeout_secs),
                    ),
                )
                .await?;
                if case.capture == CaptureMode::Sse {
                    state
                        .artifacts
                        .write_json(&directory.join("events.json"), &observation.events)?;
                }
                let mut violations = observation.violations.clone();
                let mut comparison = None;
                let mut final_json = None;
                if observation.transport_error.is_none() && violations.is_empty() {
                    match policy.prepare(case, &observation) {
                        Ok(value) => {
                            let path = directory.join("final.json");
                            state.artifacts.write_json(&path, &value)?;
                            final_json = Some(path);
                            match prepare_comparison(
                                &value,
                                case.comparison_scope,
                                &suite.comparison,
                            ) {
                                Ok(value) => comparison = Some(value),
                                Err(errors) => violations.extend(errors),
                            }
                        }
                        Err(errors) => {
                            violations.extend(errors);
                            if violations.is_empty() {
                                violations.push(Violation::new(
                                    "",
                                    "response policy rejected the response without diagnostics",
                                ));
                            }
                        }
                    }
                }
                let side = state.report.as_mut().unwrap().cases[index]
                    .implementations
                    .get_mut(side_name)
                    .unwrap();
                if observation.transport_error.is_some() || !violations.is_empty() {
                    tracing::warn!(implementation = side_name, case = %case.name, repeat, status = ?observation.status, error = ?observation.transport_error, violations = violations.len(), "Request failed validation; details retained in report");
                }
                *side.attempts.last_mut().unwrap() = Attempt {
                    directory,
                    observation: Some(observation),
                    final_json,
                    violations,
                    comparison,
                };
                state.save()?;
            }
            let side = state.report.as_mut().unwrap().cases[index]
                .implementations
                .get_mut(side_name)
                .unwrap();
            side.repeatability = match (&side.attempts[0].comparison, &side.attempts[1].comparison)
            {
                (Some(left), Some(right)) => check(left, right, Status::Unstable),
                _ => Check {
                    status: Status::Skipped,
                    ..Check::default()
                },
            };
            tracing::info!(implementation = side_name, case = %case.name, repeatability = ?side.repeatability.status, "Case complete");
            state.save()?;
        }
        let shutdown = track(
            &format!("Stopping {side_name} server"),
            &server_log,
            process.shutdown(),
        )
        .await;
        source_valid = state.verify_source(&prepared);
        if let Err(error) = shutdown {
            tracing::error!(implementation = side_name, %error, "Server shutdown failed");
            state
                .report
                .as_mut()
                .unwrap()
                .runtime_errors
                .push(format!("{side_name} shutdown: {error}"));
            // A failed cleanup cannot justify starting another managed service.
            break;
        }
        state.save()?;
        if !source_valid {
            break;
        }
    }
    let report = state.report.as_mut().unwrap();
    for case in &mut report.cases {
        case.parity = match (
            source_valid,
            stable_value(&case.implementations["python"]),
            stable_value(&case.implementations["rust"]),
        ) {
            (true, Some(left), Some(right)) => check(left, right, Status::Fail),
            _ => Check {
                status: Status::Skipped,
                ..Check::default()
            },
        };
    }
    let mut groups: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (index, case) in suite.cases.iter().enumerate() {
        if let Some(group) = &case.equivalence_group {
            groups.entry(group).or_default().push(index);
        }
    }
    for (group, members) in groups {
        for implementation in Implementation::ALL {
            let left = &report.cases[members[0]];
            for &index in &members[1..] {
                let right = &report.cases[index];
                let comparison = match (
                    source_valid,
                    stable_value(&left.implementations[implementation.as_str()]),
                    stable_value(&right.implementations[implementation.as_str()]),
                ) {
                    (true, Some(left), Some(right)) => check(left, right, Status::Fail),
                    _ => Check {
                        status: Status::Skipped,
                        ..Check::default()
                    },
                };
                report.equivalence.push(EquivalenceResult {
                    group: group.into(),
                    implementation,
                    left: left.name.clone(),
                    right: right.name.clone(),
                    check: comparison,
                });
            }
        }
    }
    report.state = "complete".into();
    state.save()?;
    Ok(state.report.take().unwrap())
}

fn stable_value(side: &SideResult) -> Option<&Value> {
    (side.repeatability.status == Status::Pass)
        .then(|| side.attempts[0].comparison.as_ref())
        .flatten()
}

fn check(left: &Value, right: &Value, failure: Status) -> Check {
    let differences = compare_json(left, right);
    Check {
        status: if differences.is_empty() {
            Status::Pass
        } else {
            failure
        },
        differences,
    }
}
