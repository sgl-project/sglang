//! Execute one resolved suite against both SGLang implementations.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::artifacts::Artifacts;
use crate::compare::{Difference, Violation, compare_json, prepare_comparison};
use crate::environment::{self, EnvironmentPlan, PreparedEnvironment};
use crate::http::{self, CaptureMode, HttpCase, HttpObservation, Isolation};
use crate::plan::{ExecutionPlan, ProfilePlan, ResolvedProfile};
pub use crate::plan::{HttpSuite, RunConfig};
use crate::process::{Implementation, SglangProcess};
use crate::progress::track;

/// A complete reconstructed response, before applying comparison exceptions.
#[derive(Clone, Debug)]
pub struct PreparedResponse {
    /// Scenario checks never suppress comparison of an otherwise valid response.
    pub assertions: Vec<AssertionResult>,
    pub value: Value,
    /// Result JSON pointers mapped to zero-based observation event indices.
    /// Descendants inherit their closest ancestor's sources.
    pub origins: BTreeMap<String, Vec<usize>>,
}

impl From<Value> for PreparedResponse {
    fn from(value: Value) -> Self {
        Self {
            value,
            origins: BTreeMap::new(),
            assertions: Vec::new(),
        }
    }
}

/// Validate an API response and reconstruct its complete, unmasked result.
pub trait ResponsePolicy {
    /// Validate the API contract and reconstruct the complete response.
    ///
    /// Implementations must preserve fields or explicitly reject unsupported
    /// semantics, never silently discard differences. Return violations for
    /// invalid responses, including in-band errors. This method performs no I/O
    /// and receives no implementation identity.
    fn prepare(
        &self,
        case: &HttpCase,
        observation: &HttpObservation,
    ) -> Result<PreparedResponse, Vec<Violation>>;
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
    if !config.profiles.is_empty() {
        return Err(RunError::Config(
            "named profiles require describe_plan/run_plan".into(),
        ));
    }
    suite.validate().map_err(RunError::Config)?;
    Ok(EffectiveSuite {
        environment: environment::describe(config).map_err(RunError::Config)?,
        suite: suite.clone(),
        repeats_per_implementation: 2,
        implementation_order: Implementation::ALL,
    })
}

/// One run's explicit profile bindings and pinned environment descriptions.
#[derive(Clone, Debug, Serialize)]
pub struct EffectivePlan {
    pub profiles: Vec<EffectiveProfile>,
    pub repeats_per_implementation: usize,
    pub implementation_order: [Implementation; 2],
}

#[derive(Clone, Debug, Serialize)]
pub struct EffectiveProfile {
    pub profile: ResolvedProfile,
    pub suite: HttpSuite,
    pub environment: EnvironmentPlan,
}

/// Resolve every profile against one observed source revision without mutation.
pub fn describe_plan<P>(
    config: &RunConfig,
    plan: &ExecutionPlan<P>,
) -> Result<EffectivePlan, RunError> {
    let declared = config.resolve_profiles().map_err(RunError::Config)?;
    plan.validate().map_err(RunError::Config)?;
    for entry in &plan.profiles {
        if !declared.contains(&entry.profile) {
            return Err(RunError::Config(format!(
                "profile {} no longer matches RunConfig; recompile the plan",
                entry.profile.id
            )));
        }
    }
    let source = environment::describe(config).map_err(RunError::Config)?;
    let profiles = plan
        .profiles
        .iter()
        .map(|entry| {
            Ok(EffectiveProfile {
                profile: entry.profile.clone(),
                suite: entry.suite.clone(),
                environment: environment::for_server(&source, &entry.profile.server)
                    .map_err(RunError::Config)?,
            })
        })
        .collect::<Result<_, RunError>>()?;
    Ok(EffectivePlan {
        profiles,
        repeats_per_implementation: 2,
        implementation_order: Implementation::ALL,
    })
}

impl<T: ResponsePolicy + ?Sized> ResponsePolicy for &T {
    fn prepare(
        &self,
        case: &HttpCase,
        observation: &HttpObservation,
    ) -> Result<PreparedResponse, Vec<Violation>> {
        (**self).prepare(case, observation)
    }
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

/// API-owned scenario evidence, distinct from response protocol validation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssertionResult {
    pub name: String,
    pub violations: Vec<Violation>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Attempt {
    #[serde(default)]
    pub assertions: Vec<AssertionResult>,
    #[serde(default)]
    pub before_each: Vec<Attempt>,
    #[serde(default)]
    pub server_log: Option<PathBuf>,
    pub directory: PathBuf,
    pub observation: Option<HttpObservation>,
    pub final_json: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub origins: BTreeMap<String, Vec<usize>>,
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
    /// Unique execution identity: profile/case for plans, legacy case name otherwise.
    #[serde(default = "default_profile")]
    pub profile_id: String,
    pub name: String,
    pub implementations: BTreeMap<String, SideResult>,
    pub parity: Check,
    #[serde(default)]
    pub unavailable: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EquivalenceResult {
    pub group: String,
    pub implementation: Implementation,
    pub left: String,
    pub right: String,
    pub check: Check,
}

fn default_profile() -> String {
    "default".into()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProfileResult {
    pub state: String,
    pub environment: Option<PathBuf>,
    pub diagnostics: Vec<String>,
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
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub profiles: BTreeMap<String, ProfileResult>,
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
                    .flat_map(|s| &s.attempts)
                    .flat_map(|a| &a.assertions)
                    .any(|a| !a.violations.is_empty())
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
    let effective = describe(config, suite)?;
    let plan = ExecutionPlan {
        profiles: vec![ProfilePlan {
            profile: ResolvedProfile {
                id: "default".into(),
                server: config.server.clone(),
                requires: Default::default(),
            },
            suite: suite.clone(),
            policy,
        }],
    };
    let resolved = EffectivePlan {
        profiles: vec![EffectiveProfile {
            profile: plan.profiles[0].profile.clone(),
            suite: suite.clone(),
            environment: effective.environment.clone(),
        }],
        repeats_per_implementation: 2,
        implementation_order: Implementation::ALL,
    };
    execute(config, &plan, resolved, Some(effective)).await
}

/// Execute explicit profile/case bindings with one source snapshot and one report.
pub async fn run_plan<P: ResponsePolicy>(
    config: &RunConfig,
    plan: &ExecutionPlan<P>,
) -> Result<Report, RunError> {
    let effective = describe_plan(config, plan)?;
    execute(config, plan, effective, None).await
}

async fn execute<P: ResponsePolicy>(
    config: &RunConfig,
    plan: &ExecutionPlan<P>,
    effective: EffectivePlan,
    legacy: Option<EffectiveSuite>,
) -> Result<Report, RunError> {
    let client = http::client()?;
    let artifacts = Artifacts::create(&config.output_dir)?;
    let legacy_paths = legacy.is_some();
    let effective_path = artifacts.root().join(if legacy_paths {
        "effective_suite.json"
    } else {
        "effective_plan.json"
    });
    if let Some(legacy) = legacy {
        artifacts.write_json(&effective_path, &legacy)?;
    } else {
        artifacts.write_json(&effective_path, &effective)?;
    }
    tracing::info!(directory = %artifacts.root().display(), profiles = plan.profiles.len(), "Run artifacts and logs");
    let cases = plan
        .profiles
        .iter()
        .flat_map(|entry| {
            entry.suite.cases.iter().map(|case| CaseResult {
                profile_id: entry.profile.id.clone(),
                name: if legacy_paths {
                    case.name.clone()
                } else {
                    format!("{}/{}", entry.profile.id, case.name)
                },
                implementations: Implementation::ALL
                    .into_iter()
                    .map(|i| (i.as_str().into(), SideResult::default()))
                    .collect(),
                parity: Check::default(),
                unavailable: None,
            })
        })
        .collect();
    let profiles = plan
        .profiles
        .iter()
        .map(|p| {
            (
                p.profile.id.clone(),
                ProfileResult {
                    state: "pending".into(),
                    environment: None,
                    diagnostics: Vec::new(),
                },
            )
        })
        .collect();
    let mut state = RunArtifacts {
        artifacts,
        report: Some(Report {
            state: "preparing".into(),
            directory: PathBuf::new(),
            effective_suite: effective_path,
            config: config.clone(),
            environment: None,
            runtime_errors: Vec::new(),
            cases,
            equivalence: Vec::new(),
            profiles,
        }),
    };
    state.report.as_mut().unwrap().directory = state.artifacts.root().to_owned();
    state.save()?;
    let mut source = None;
    let mut environments = BTreeMap::new();
    let mut offset = 0;
    let mut source_valid = true;
    for (entry, resolved) in plan.profiles.iter().zip(&effective.profiles) {
        let id = &entry.profile.id;
        let range = offset..offset + entry.suite.cases.len();
        offset = range.end;
        let backend = resolved.environment.profile.backend;
        // Backend-only exclusions need neither installation nor a running server.
        for (index, case) in entry.suite.cases.iter().enumerate() {
            let requirements = entry
                .profile
                .requires
                .combine(&case.requires)
                .map_err(RunError::Config)?;
            let reason = requirements.unavailable(backend, Some(u32::MAX));
            state.report.as_mut().unwrap().cases[range.start + index].unavailable = reason;
        }
        if state.report.as_ref().unwrap().cases[range.clone()]
            .iter()
            .all(|c| c.unavailable.is_some())
        {
            state
                .report
                .as_mut()
                .unwrap()
                .profiles
                .get_mut(id)
                .unwrap()
                .state = "unavailable".into();
            continue;
        }
        tracing::info!(profile = %id, mode = %entry.suite.output_mode, "Preparing profile");
        if source.is_none() {
            match environment::prepare_source(
                &resolved.environment,
                state.artifacts.root(),
                Duration::from_secs(config.environment.setup_timeout_secs),
            )
            .await
            {
                Ok(prepared) => source = Some(prepared),
                Err(error) => {
                    state
                        .report
                        .as_mut()
                        .unwrap()
                        .runtime_errors
                        .push(format!("source preparation: {error}"));
                    break;
                }
            }
        }
        let env_key = &resolved.environment.environment_dir;
        if !environments.contains_key(env_key) {
            let output = if legacy_paths {
                state.artifacts.root().to_owned()
            } else {
                state.artifacts.directory(
                    PathBuf::from("environments")
                        .join(env_key.file_name().expect("environment key")),
                )?
            };
            let result = environment::prepare_environment(
                config,
                &resolved.environment,
                &output,
                source.as_ref().unwrap().clone(),
            )
            .await;
            environments.insert(env_key.clone(), result);
        }
        let prepared = match &environments[env_key] {
            Ok(prepared) => prepared,
            Err(error) => {
                let report = state.report.as_mut().unwrap();
                report
                    .runtime_errors
                    .push(format!("profile {id} environment preparation: {error}"));
                let profile = report.profiles.get_mut(id).unwrap();
                profile.state = "failed".into();
                profile.diagnostics.push(error.clone());
                if source.as_ref().unwrap().verify().is_err() {
                    source_valid = false;
                    break;
                }
                state.save()?;
                continue;
            }
        };
        if !state.verify_source(prepared) {
            source_valid = false;
            break;
        }
        let devices = prepared
            .record
            .pointer("/probe/backend/device_count")
            .and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok());
        let report = state.report.as_mut().unwrap();
        if report.environment.is_none() {
            report.environment = Some(prepared.record.clone());
        }
        let profile = report.profiles.get_mut(id).unwrap();
        profile.environment = Some(prepared.record_path.clone());
        profile.state = "running".into();
        report.state = "running".into();
        for (index, case) in entry.suite.cases.iter().enumerate() {
            report.cases[range.start + index].unavailable = entry
                .profile
                .requires
                .combine(&case.requires)
                .map_err(RunError::Config)?
                .unavailable(backend, devices);
        }
        state.save()?;
        let (safe, valid) = execute_profile(
            &mut state,
            &client,
            config,
            entry,
            prepared,
            range.start,
            legacy_paths,
        )
        .await?;
        source_valid &= valid;
        let report = state.report.as_mut().unwrap();
        report.profiles.get_mut(id).unwrap().state = if safe && source_valid {
            "complete"
        } else {
            "interrupted"
        }
        .into();
        state.save()?;
        if !safe || !source_valid {
            break;
        }
    }
    let report = state.report.as_mut().unwrap();
    let mut start = 0;
    for entry in &plan.profiles {
        let end = start + entry.suite.cases.len();
        compare_profile(report, &entry.suite, start..end, source_valid);
        start = end;
    }
    // Planned but unavailable/incomplete checks never look like passing coverage.
    for case in &mut report.cases {
        if case.parity.status == Status::NotRun {
            case.parity = skipped();
        }
        if !source_valid {
            case.parity = skipped();
        }
    }
    report.state = "complete".into();
    state.save()?;
    Ok(state.report.take().unwrap())
}

/// Run one profile's services; request capture and comparison stay shared.
async fn execute_profile<P: ResponsePolicy>(
    state: &mut RunArtifacts,
    client: &reqwest::Client,
    config: &RunConfig,
    entry: &ProfilePlan<P>,
    prepared: &PreparedEnvironment,
    offset: usize,
    legacy_paths: bool,
) -> Result<(bool, bool), RunError> {
    let id = &entry.profile.id;
    let mut source_valid = true;
    let server = prepared.server(&entry.profile.server);
    let mut safe = true;
    for implementation in Implementation::ALL {
        let side = implementation.as_str();
        let side_dir = if legacy_paths {
            state.artifacts.directory(side)?
        } else {
            state.artifacts.directory(format!("profiles/{id}/{side}"))?
        };
        let mut active: Option<(SglangProcess, PathBuf)> = None;
        let mut starts = 0;
        'cases: for (local, case) in entry.suite.cases.iter().enumerate() {
            let index = offset + local;
            if state.report.as_ref().unwrap().cases[index]
                .unavailable
                .is_some()
            {
                continue;
            }
            for repeat in 1..=2 {
                if case.isolation == Isolation::FreshProcess
                    && !stop(&mut active, state, id, side).await?
                {
                    safe = false;
                    break;
                }
                if active.is_none() {
                    if !state.verify_source(prepared) {
                        source_valid = false;
                        safe = false;
                        break;
                    }
                    starts += 1;
                    let log = if legacy_paths && starts == 1 {
                        side_dir.join("server.log")
                    } else {
                        let logs = side_dir.join("logs");
                        std::fs::create_dir_all(&logs)?;
                        logs.join(format!("server-{starts}.log"))
                    };
                    match track(
                        &format!("{id}/{side}: starting server"),
                        &log,
                        SglangProcess::start(
                            &server,
                            implementation,
                            &log,
                            Duration::from_secs(config.startup_timeout_secs),
                            Duration::from_secs(config.shutdown_timeout_secs),
                        ),
                    )
                    .await
                    {
                        Ok(process) => {
                            tracing::info!(profile = %id, implementation = side, "Server ready");
                            active = Some((process, log));
                        }
                        Err(error) => {
                            tracing::error!(profile = %id, implementation = side, %error, "Server startup failed");
                            state
                                .report
                                .as_mut()
                                .unwrap()
                                .runtime_errors
                                .push(format!("{id}/{side} startup: {error}"));
                            break 'cases;
                        }
                    }
                }
                let (process, log) = active.as_ref().unwrap();
                let directory = side_dir.join(&case.name).join(repeat.to_string());
                std::fs::create_dir_all(&directory)?;
                let attempt = Attempt::pending(directory.clone(), log.clone());
                state.report.as_mut().unwrap().cases[index]
                    .implementations
                    .get_mut(side)
                    .unwrap()
                    .attempts
                    .push(attempt);
                state.save()?;
                let capture = Capture {
                    client,
                    artifacts: &state.artifacts,
                    policy: &entry.policy,
                    rules: &entry.suite.comparison,
                    timeout: Duration::from_secs(config.request_timeout_secs),
                };
                let mut ready = true;
                for (step, request) in case.before_each.iter().enumerate() {
                    let step_dir = directory.join("before_each").join((step + 1).to_string());
                    let result = capture
                        .request(&request.as_case(), &process.base_url(), &step_dir, log)
                        .await?;
                    ready = result.valid();
                    state.report.as_mut().unwrap().cases[index]
                        .implementations
                        .get_mut(side)
                        .unwrap()
                        .attempts
                        .last_mut()
                        .unwrap()
                        .before_each
                        .push(result);
                    state.save()?;
                    if !ready {
                        break;
                    }
                }
                let mut attempt = if ready {
                    track(
                        &format!(
                            "{id}/{side}: case {}/{} {}, repeat {repeat}/2",
                            local + 1,
                            entry.suite.cases.len(),
                            case.name
                        ),
                        log,
                        capture.request(case, &process.base_url(), &directory, log),
                    )
                    .await?
                } else {
                    let mut attempt = Attempt::pending(directory, log.clone());
                    attempt.violations.push(Violation::new(
                        "",
                        "prerequisite failed; measured request was not sent",
                    ));
                    attempt
                };
                attempt.before_each = std::mem::take(
                    &mut state.report.as_mut().unwrap().cases[index]
                        .implementations
                        .get_mut(side)
                        .unwrap()
                        .attempts
                        .last_mut()
                        .unwrap()
                        .before_each,
                );
                *state.report.as_mut().unwrap().cases[index]
                    .implementations
                    .get_mut(side)
                    .unwrap()
                    .attempts
                    .last_mut()
                    .unwrap() = attempt;
                state.save()?;
                if case.isolation == Isolation::FreshProcess
                    && !stop(&mut active, state, id, side).await?
                {
                    safe = false;
                    break;
                }
            }
            let side_result = state.report.as_mut().unwrap().cases[index]
                .implementations
                .get_mut(side)
                .unwrap();
            side_result.repeatability = match side_result.attempts.as_slice() {
                [left, right] => match (&left.comparison, &right.comparison) {
                    (Some(left), Some(right)) => check(left, right, Status::Unstable),
                    _ => skipped(),
                },
                _ => skipped(),
            };
            state.save()?;
            if !safe {
                break;
            }
        }
        safe &= stop(&mut active, state, id, side).await?;
        source_valid &= state.verify_source(prepared);
        if !safe || !source_valid {
            break;
        }
    }
    Ok((safe, source_valid))
}

impl Attempt {
    fn pending(directory: PathBuf, server_log: PathBuf) -> Self {
        Self {
            directory,
            server_log: Some(server_log),
            observation: None,
            final_json: None,
            origins: BTreeMap::new(),
            violations: Vec::new(),
            assertions: Vec::new(),
            before_each: Vec::new(),
            comparison: None,
        }
    }
    fn valid(&self) -> bool {
        self.observation
            .as_ref()
            .is_some_and(|o| o.transport_error.is_none())
            && self.violations.is_empty()
    }
}

struct Capture<'a, P> {
    client: &'a reqwest::Client,
    artifacts: &'a Artifacts,
    policy: &'a P,
    rules: &'a crate::compare::ComparisonRules,
    timeout: Duration,
}

impl<P: ResponsePolicy> Capture<'_, P> {
    async fn request(
        &self,
        case: &HttpCase,
        base_url: &str,
        directory: &std::path::Path,
        log: &std::path::Path,
    ) -> Result<Attempt, RunError> {
        std::fs::create_dir_all(directory)?;
        let request = serde_json::to_vec_pretty(&case.body).map_err(std::io::Error::from)?;
        std::fs::write(directory.join("request.json"), &request)?;
        let mut attempt = Attempt::pending(directory.to_owned(), log.to_owned());
        let observation = http::capture(
            self.client,
            base_url,
            case,
            &request,
            &directory.join("response.body"),
            self.timeout,
        )
        .await?;
        if case.capture == CaptureMode::Sse {
            self.artifacts
                .write_json(&directory.join("events.json"), &observation.events)?;
        }
        attempt.violations = observation.violations.clone();
        if observation.transport_error.is_none() && attempt.violations.is_empty() {
            match self.policy.prepare(case, &observation) {
                Ok(prepared) => {
                    let path = directory.join("final.json");
                    self.artifacts.write_json(&path, &prepared.value)?;
                    attempt.final_json = Some(path);
                    attempt.origins = prepared.origins;
                    attempt.assertions = prepared.assertions;
                    if !case.assertions.is_empty() {
                        let actual: std::collections::BTreeSet<_> =
                            attempt.assertions.iter().map(|a| &a.name).collect();
                        if actual.len() != attempt.assertions.len()
                            || actual != case.assertions.iter().collect()
                        {
                            attempt.violations.push(Violation::new("", "response policy returned a different set of scenario assertions than declared"));
                        }
                    }
                    // Prerequisites are protocol-checked, not compared or subjected
                    // to measured-case assertions and value-exception requirements.
                    if !case.name.is_empty() && attempt.violations.is_empty() {
                        match prepare_comparison(&prepared.value, case.comparison_scope, self.rules)
                        {
                            Ok(value) => attempt.comparison = Some(value),
                            Err(errors) => attempt.violations.extend(errors),
                        }
                    }
                }
                Err(errors) => {
                    attempt.violations.extend(errors);
                    if attempt.violations.is_empty() {
                        attempt.violations.push(Violation::new(
                            "",
                            "response policy rejected the response without diagnostics",
                        ));
                    }
                }
            }
        }
        attempt.observation = Some(observation);
        Ok(attempt)
    }
}

async fn stop(
    active: &mut Option<(SglangProcess, PathBuf)>,
    state: &mut RunArtifacts,
    profile: &str,
    side: &str,
) -> Result<bool, RunError> {
    let Some((mut process, log)) = active.take() else {
        return Ok(true);
    };
    if let Err(error) = track(
        &format!("{profile}/{side}: stopping server"),
        &log,
        process.shutdown(),
    )
    .await
    {
        state
            .report
            .as_mut()
            .unwrap()
            .runtime_errors
            .push(format!("{profile}/{side} shutdown: {error}"));
        state.save()?;
        return Ok(false);
    }
    Ok(true)
}

fn skipped() -> Check {
    Check {
        status: Status::Skipped,
        ..Check::default()
    }
}

fn compare_profile(
    report: &mut Report,
    suite: &HttpSuite,
    range: std::ops::Range<usize>,
    source_valid: bool,
) {
    for case in &mut report.cases[range.clone()] {
        case.parity = match (
            source_valid,
            stable_value(&case.implementations["python"]),
            stable_value(&case.implementations["rust"]),
        ) {
            (true, Some(left), Some(right)) => check(left, right, Status::Fail),
            _ => skipped(),
        };
    }
    let mut groups: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (index, case) in suite.cases.iter().enumerate() {
        if let Some(group) = &case.equivalence_group {
            groups.entry(group).or_default().push(range.start + index);
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
                    _ => skipped(),
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
