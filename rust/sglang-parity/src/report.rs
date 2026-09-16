//! Human-readable views of recorded results, without reevaluating comparisons.
//!
//! A view reads evidence from the supplied artifact directory, so an entire run
//! can be moved without changing its JSON report. Missing evidence is diagnostic;
//! it never changes the recorded verdict or becomes a missing response field.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::{Component, Path, PathBuf};

use serde_json::Value;

use crate::CheckTarget;
use crate::artifacts::write_atomic;
use crate::compare::{ComparisonScope, DifferenceKind};
use crate::http::CaptureMode;
use crate::runner::{Attempt, CaseResult, Check, HttpSuite, Report, Status};

const PREVIEW_LIMIT: usize = 64 * 1024;
const STATUSES: [Status; 5] = [
    Status::Pass,
    Status::Fail,
    Status::Unstable,
    Status::Skipped,
    Status::NotRun,
];
type JsonFiles = BTreeMap<PathBuf, Result<Value, String>>;

#[derive(Clone, Copy)]
struct Evidence<'a> {
    semantic: bool,
    case: &'a str,
    implementation: &'a str,
    repeat: usize,
    attempt: Option<&'a Attempt>,
}

impl Evidence<'_> {
    fn file(&self) -> Option<&Path> {
        let attempt = self.attempt?;
        if self.semantic
            && let Some(evidence) = &attempt.equivalence
        {
            return Some(&evidence.file);
        }
        attempt.prepared_path()
    }

    fn projected(&self) -> bool {
        self.semantic && self.attempt.is_some_and(|a| a.equivalence.is_some())
    }

    fn output(&self) -> bool {
        !self.projected() && self.attempt.is_some_and(|a| a.output_json.is_some())
    }

    fn value_label(&self) -> &'static str {
        if self.projected() {
            "Semantic equivalence value"
        } else if self.output() {
            "Generated output value (before exceptions)"
        } else {
            "Reconstructed value (before exceptions)"
        }
    }

    fn label(&self) -> String {
        format!(
            "{} / {} / attempt {}",
            self.implementation, self.case, self.repeat
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CheckKind {
    Parity,
    Repeatability,
    Equivalence,
}

struct Comparison<'a> {
    kind: CheckKind,
    category: String,
    name: String,
    id: String,
    check: &'a Check,
    left: Evidence<'a>,
    right: Evidence<'a>,
}

impl Comparison<'_> {
    fn labels(&self) -> [String; 2] {
        match self.kind {
            CheckKind::Parity => ["Python".into(), "Rust".into()],
            CheckKind::Repeatability => ["Attempt 1".into(), "Attempt 2".into()],
            CheckKind::Equivalence => [self.left.case.into(), self.right.case.into()],
        }
    }

    fn reason(&self, kind: DifferenceKind) -> String {
        let [left, right] = self.labels();
        match kind {
            DifferenceKind::MissingLeft => format!("{left} is missing fields present in {right}"),
            DifferenceKind::MissingRight => format!("{right} is missing fields present in {left}"),
            DifferenceKind::TypeMismatch => "Field types differ".into(),
            DifferenceKind::ValueMismatch => "Field values differ".into(),
        }
    }

    fn difference_counts(&self) -> impl Iterator<Item = (DifferenceKind, usize)> + '_ {
        [
            DifferenceKind::MissingLeft,
            DifferenceKind::MissingRight,
            DifferenceKind::TypeMismatch,
            DifferenceKind::ValueMismatch,
        ]
        .into_iter()
        .filter_map(|kind| {
            let count = self
                .check
                .differences
                .iter()
                .filter(|d| d.kind == kind)
                .count();
            (count > 0).then_some((kind, count))
        })
    }

    fn reasons(&self) -> Vec<String> {
        self.difference_counts()
            .map(|(kind, count)| format!("{} ({count} differences)", self.reason(kind)))
            .collect()
    }

    fn short_reason(&self) -> String {
        let [left, right] = self.labels();
        self.difference_counts()
            .map(|(kind, count)| match kind {
                DifferenceKind::MissingLeft => format!("{count} missing in {left}"),
                DifferenceKind::MissingRight => format!("{count} missing in {right}"),
                DifferenceKind::TypeMismatch => format!("{count} type mismatches"),
                DifferenceKind::ValueMismatch => format!("{count} value mismatches"),
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[derive(Default)]
struct Validation {
    valid: usize,
    invalid: usize,
    pending: usize,
}

impl Validation {
    fn for_cases<'a>(cases: impl Iterator<Item = &'a CaseResult>) -> Self {
        let mut result = Self::default();
        for case in cases {
            let mut attempted = 0;
            for attempt in case
                .implementations
                .values()
                .flat_map(|side| &side.attempts)
            {
                attempted += 1;
                if !attempt.violations.is_empty() {
                    result.invalid += 1;
                } else if attempt.prepared_path().is_some()
                    && attempt
                        .observation
                        .as_ref()
                        .is_some_and(|o| o.transport_error.is_none())
                {
                    result.valid += 1;
                } else {
                    result.pending += 1;
                }
            }
            result.pending += 4_usize.saturating_sub(attempted);
        }
        result
    }

    fn status(&self) -> Status {
        if self.invalid > 0 {
            Status::Fail
        } else if self.pending > 0 || self.valid == 0 {
            Status::NotRun
        } else {
            Status::Pass
        }
    }

    fn summary(&self) -> String {
        format!(
            "{}/{} passed · {} invalid · {} unavailable/pending",
            self.valid,
            self.valid + self.invalid + self.pending,
            self.invalid,
            self.pending
        )
    }
}

/// Terminal and HTML presentations of one saved or freshly completed run.
///
/// Construction reads only the saved suite metadata. Full response evidence is
/// loaded when rendering HTML or expanding a particular case in the terminal.
pub struct ReportView<'a> {
    report: &'a Report,
    directory: PathBuf,
    suite: Result<HttpSuite, String>,
    profile_suites: BTreeMap<String, HttpSuite>,
    comparisons: Vec<Comparison<'a>>,
}

impl<'a> ReportView<'a> {
    pub fn new(report: &'a Report, directory: &Path) -> Self {
        let mut view = Self {
            report,
            directory: directory.to_owned(),
            suite: Err("Saved suite is unavailable".into()),
            profile_suites: BTreeMap::new(),
            comparisons: Vec::new(),
        };
        match view.read_json(&report.effective_suite) {
            Ok(value) if value.get("profiles").is_some() => {
                let parsed = (|| {
                    let profiles = value["profiles"]
                        .as_array()
                        .ok_or("saved profiles must be an array")?;
                    for entry in profiles {
                        let id = entry["profile"]["id"]
                            .as_str()
                            .ok_or("saved profile has no id")?;
                        let suite: HttpSuite = serde_json::from_value(entry["suite"].clone())
                            .map_err(|e| e.to_string())?;
                        view.profile_suites.insert(id.into(), suite);
                    }
                    view.profile_suites
                        .values()
                        .next()
                        .cloned()
                        .ok_or("saved plan has no profiles".into())
                })();
                view.suite = parsed;
            }
            Ok(value) => {
                view.suite =
                    serde_json::from_value(value.get("suite").cloned().unwrap_or(Value::Null))
                        .map_err(|error| format!("Saved suite is unavailable: {error}"))
            }
            Err(error) => view.suite = Err(error),
        }
        for (index, case) in report.cases.iter().enumerate() {
            view.comparisons.push(Comparison {
                kind: CheckKind::Parity,
                category: view.parity_label().into(),
                name: case.name.clone(),
                id: format!("parity-{index}"),
                check: &case.parity,
                left: evidence(case, "python", 1),
                right: evidence(case, "rust", 1),
            });
            for (side, result) in &case.implementations {
                view.comparisons.push(Comparison {
                    kind: CheckKind::Repeatability,
                    category: format!("{side} repeatability"),
                    name: case.name.clone(),
                    id: format!("repeat-{index}-{}", view.comparisons.len()),
                    check: &result.repeatability,
                    left: evidence(case, side, 1),
                    right: evidence(case, side, 2),
                });
            }
        }
        for (index, result) in report.equivalence.iter().enumerate() {
            let side = result.implementation.as_str();
            let lookup = |name: &'a str| {
                report
                    .cases
                    .iter()
                    .find(|case| case.name == name)
                    .map(|case| {
                        let mut value = evidence(case, side, 1);
                        value.semantic = true;
                        value
                    })
                    .unwrap_or(Evidence {
                        semantic: false,
                        case: name,
                        implementation: side,
                        repeat: 1,
                        attempt: None,
                    })
            };
            view.comparisons.push(Comparison {
                kind: CheckKind::Equivalence,
                category: format!("{side} case equivalence"),
                name: format!("{}: {} <-> {}", result.group, result.left, result.right),
                id: format!("equivalence-{index}"),
                check: &result.check,
                left: lookup(&result.left),
                right: lookup(&result.right),
            });
        }
        view
    }

    /// Render each case's checks, optionally selecting one case with full evidence.
    ///
    /// # Errors
    /// Returns an error if the requested case does not exist in this report.
    pub fn terminal(&self, case: Option<&str>, color: bool) -> Result<String, String> {
        let case = case
            .map(|name| {
                if self.report.cases.iter().any(|c| c.name == name) {
                    return Ok(name);
                }
                let matches: Vec<_> = self
                    .report
                    .cases
                    .iter()
                    .filter(|c| c.name.rsplit('/').next() == Some(name))
                    .collect();
                match matches.as_slice() {
                    [only] => Ok(only.name.as_str()),
                    [] => Err(format!("unknown report case {name:?}")),
                    _ => Err(format!(
                        "ambiguous case {name:?}; select {}",
                        matches
                            .iter()
                            .map(|c| c.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )),
                }
            })
            .transpose()?;
        let mut out = String::new();
        writeln!(
            out,
            "SGLang Parity — {}\nState: {} · exit code {}",
            terminal_status(self.verdict(), color),
            plain(&self.report.state),
            self.report.exit_code()
        )
        .unwrap();
        for (label, value) in self.metadata() {
            writeln!(out, "{label}: {}", plain(&value)).unwrap();
        }
        out.push('\n');
        for (label, value) in self.totals() {
            writeln!(out, "{label:<26} {value}").unwrap();
        }
        out.push_str("\nUse --case <name> or HTML for full differences and evidence.\n");
        if case.is_some() {
            out.push_str("<missing> = absent field; null = present null; <unavailable> = missing evidence.\n");
        }
        for message in self.run_diagnostics() {
            writeln!(out, "\nRun diagnostic: {}", plain(&message)).unwrap();
        }
        let files = case.map(|name| self.final_files(Some(name)));
        let mut profile = None;
        for (index, result) in self.report.cases.iter().enumerate() {
            if case.is_some_and(|name| name != result.name) {
                continue;
            }
            if !self.profile_suites.is_empty() && profile != Some(&result.profile_id) {
                writeln!(
                    out,
                    "\nPROFILE {} · {}",
                    plain(&result.profile_id),
                    plain(
                        self.profile_suites
                            .get(&result.profile_id)
                            .map_or("Unavailable", |s| s.output_mode.as_str())
                    )
                )
                .unwrap();
                profile = Some(&result.profile_id);
            }
            let Some(files) = &files else {
                self.terminal_case_summary(&mut out, result, index, color);
                continue;
            };
            let title = plain(&result.name);
            writeln!(
                out,
                "\n{}\n{}",
                if color {
                    format!("\x1b[1m{title}\x1b[0m")
                } else {
                    title
                },
                plain(&self.case_description(&result.name))
            )
            .unwrap();
            let comparisons = self.case_comparisons(&result.name);
            let width = comparisons
                .iter()
                .map(|c| plain(&self.check_label(c, &result.name)).chars().count())
                .max()
                .unwrap_or(0)
                .max(25);
            let validation = Validation::for_cases(std::iter::once(result));
            writeln!(
                out,
                "\n  {:<width$} {}  {}",
                self.validation_label(),
                terminal_status(status(validation.status()), color),
                validation.summary()
            )
            .unwrap();
            for comparison in &comparisons {
                writeln!(
                    out,
                    "  {:<width$} {}  {} differences",
                    plain(&self.check_label(comparison, &result.name)),
                    terminal_status(status(comparison.check.status), color),
                    comparison.check.differences.len()
                )
                .unwrap();
            }
            for message in self.case_diagnostics(result) {
                writeln!(out, "\n  {}", plain(&message)).unwrap();
            }
            for comparison in comparisons {
                self.terminal_comparison(&mut out, comparison, files);
            }
            writeln!(
                out,
                "\n  Details: {}#case-{index}",
                plain(&self.directory.join("report.html").display().to_string())
            )
            .unwrap();
        }
        writeln!(
            out,
            "\nDetails: {}\nData:    {}",
            plain(&self.directory.join("report.html").display().to_string()),
            plain(&self.directory.join("report.json").display().to_string())
        )
        .unwrap();
        Ok(out)
    }

    fn terminal_case_summary(
        &self,
        out: &mut String,
        case: &CaseResult,
        index: usize,
        color: bool,
    ) {
        let title = plain(&case.name);
        writeln!(
            out,
            "\n{} · {}",
            if color {
                format!("\x1b[1m{title}\x1b[0m")
            } else {
                title
            },
            plain(&self.case_description(&case.name))
        )
        .unwrap();
        let repeat = |side: &str| {
            terminal_status(
                status(
                    case.implementations
                        .get(side)
                        .map(|s| s.repeatability.status)
                        .unwrap_or(Status::NotRun),
                ),
                color,
            )
        };
        let validation = Validation::for_cases(std::iter::once(case));
        writeln!(
            out,
            "  {} {} · Repeat Python {} / Rust {}",
            if self.generated_content() {
                "Output integrity"
            } else {
                "Response"
            },
            terminal_status(status(validation.status()), color),
            repeat("python"),
            repeat("rust")
        )
        .unwrap();
        let comparisons = self.case_comparisons(&case.name);
        let reason = case.unavailable.clone().unwrap_or_else(|| {
            comparisons
                .iter()
                .find(|c| c.kind == CheckKind::Parity)
                .map(|c| {
                    self.check_diagnostic(c)
                        .map(|s| {
                            s.strip_prefix("Comparison skipped: ")
                                .unwrap_or(s)
                                .to_owned()
                        })
                        .unwrap_or_else(|| c.short_reason())
                })
                .unwrap_or_default()
        });
        writeln!(
            out,
            "  {} {}{}",
            self.compact_parity_label(),
            terminal_status(status(case.parity.status), color),
            if reason.is_empty() {
                String::new()
            } else {
                format!(": {}", clipped(&reason, 45))
            }
        )
        .unwrap();
        let mut equivalent = comparisons
            .iter()
            .filter(|c| c.kind == CheckKind::Equivalence)
            .peekable();
        let mut summary = if equivalent.peek().is_none() {
            "not configured".into()
        } else {
            counts(equivalent.map(|c| c.check.status))
        };
        for value in STATUSES {
            summary = summary.replace(status(value), &terminal_status(status(value), color));
        }
        if let Some(assertions) = self.assertion_summary(case) {
            writeln!(out, "  {assertions} · Equivalence: {summary}").unwrap();
        } else {
            writeln!(out, "  Equivalence: {summary}").unwrap();
        }
        writeln!(
            out,
            "  Details: --case {} · report.html#case-{index}",
            plain(&case.name)
        )
        .unwrap();
    }

    fn terminal_comparison(
        &self,
        out: &mut String,
        comparison: &Comparison<'_>,
        files: &JsonFiles,
    ) {
        writeln!(
            out,
            "\n  {} · {}",
            plain(&comparison.category),
            plain(&comparison.name)
        )
        .unwrap();
        if let Some(reason) = self.check_diagnostic(comparison) {
            writeln!(out, "  {reason}").unwrap();
        }
        for reason in comparison.reasons() {
            writeln!(out, "  {}", plain(&reason)).unwrap();
        }
        writeln!(
            out,
            "  Compared: {} <-> {}",
            plain(&comparison.left.label()),
            plain(&comparison.right.label())
        )
        .unwrap();
        let labels = comparison.labels();
        for difference in &comparison.check.differences {
            writeln!(out, "\n  {}", plain(&difference.path)).unwrap();
            for ((side, value), label) in [
                (comparison.left, difference.left.as_ref()),
                (comparison.right, difference.right.as_ref()),
            ]
            .into_iter()
            .zip(&labels)
            {
                let rule = value
                    .filter(|v| !v.is_null())
                    .and_then(|_| self.comparison_reason(side, &difference.path));
                let text = plain(&value_text(value));
                writeln!(
                    out,
                    "    {}: {text}{}",
                    plain(label),
                    if rule.is_some() {
                        " [value exception applied]"
                    } else {
                        ""
                    }
                )
                .unwrap();
                writeln!(
                    out,
                    "      {}: {}",
                    if side.projected() {
                        "semantic equivalence"
                    } else if side.output() {
                        "generated output"
                    } else {
                        "reconstructed"
                    },
                    plain(&self.original(side, &difference.path, files))
                )
                .unwrap();
                if let Some(indices) = Self::sources(side, &difference.path) {
                    writeln!(out, "      source event indices (zero-based): {indices:?}").unwrap();
                }
                if let Some(rule) = rule {
                    writeln!(out, "      value exception: {}", plain(rule)).unwrap();
                }
            }
        }
        for side in [comparison.left, comparison.right] {
            for (label, path) in self.evidence_paths(side) {
                if let Some(relative) = self.relative(&path) {
                    let path = self.directory.join(relative);
                    writeln!(
                        out,
                        "  {} {label}: {}{}",
                        plain(&side.label()),
                        plain(&path.display().to_string()),
                        if path.is_file() { "" } else { " (unavailable)" }
                    )
                    .unwrap();
                }
            }
        }
    }

    fn metadata(&self) -> Vec<(&'static str, String)> {
        let mut result = vec![
            (
                "Suite",
                self.suite
                    .as_ref()
                    .map(|s| s.name.as_str())
                    .unwrap_or("Unavailable")
                    .into(),
            ),
            (
                "Streaming mode",
                self.suite
                    .as_ref()
                    .map(|s| s.output_mode.as_str())
                    .unwrap_or("Unavailable")
                    .into(),
            ),
        ];
        if !self.profile_suites.is_empty() {
            result[1].1 = self
                .profile_suites
                .values()
                .map(|suite| suite.output_mode.as_str())
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
                .join("; ");
        }
        result.push((
            "Check",
            if self.generated_content() {
                "Generated content parity"
            } else {
                "Full response parity"
            }
            .into(),
        ));
        if self.generated_content() {
            result.push(("Metadata", "Not checked".into()));
        }
        for (label, pointer) in [
            ("Commit", "/plan/commit"),
            ("Backend", "/plan/profile/backend"),
        ] {
            result.push((
                label,
                self.report
                    .environment
                    .as_ref()
                    .and_then(|e| e.pointer(pointer))
                    .and_then(Value::as_str)
                    .unwrap_or("Unavailable")
                    .into(),
            ));
        }
        result
    }

    fn generated_content(&self) -> bool {
        self.report.check == CheckTarget::GeneratedContent
    }

    fn validation_label(&self) -> &'static str {
        if self.generated_content() {
            "Output integrity"
        } else {
            "Response validation"
        }
    }

    fn parity_label(&self) -> &'static str {
        if self.generated_content() {
            "Content parity"
        } else {
            "Python <-> Rust parity"
        }
    }

    fn compact_parity_label(&self) -> &'static str {
        if self.generated_content() {
            "Content parity"
        } else {
            "Parity"
        }
    }

    fn case_suite(&self, name: &str) -> Option<(&HttpSuite, &crate::http::HttpCase)> {
        let (suite, name) = if let Some((profile, case)) = name.split_once('/') {
            (self.profile_suites.get(profile)?, case)
        } else {
            (self.suite.as_ref().ok()?, name)
        };
        Some((suite, suite.cases.iter().find(|c| c.name == name)?))
    }

    fn case_description(&self, name: &str) -> String {
        let Some((suite, case)) = self.case_suite(name) else {
            return "Request specification unavailable".into();
        };
        let response = match case.capture {
            CaptureMode::Json => "JSON".into(),
            CaptureMode::Sse => format!("SSE · {}", suite.output_mode),
        };
        format!(
            "{} {} · {response} · expected HTTP {}",
            case.method, case.path, case.expect_status
        )
    }

    fn case_comparisons(&self, name: &str) -> Vec<&Comparison<'a>> {
        let mut comparisons: Vec<_> = self
            .comparisons
            .iter()
            .filter(|c| c.left.case == name || c.right.case == name)
            .collect();
        comparisons.sort_by_key(|c| match c.kind {
            CheckKind::Repeatability => 0,
            CheckKind::Parity => 1,
            CheckKind::Equivalence => 2,
        });
        comparisons
    }

    fn check_label(&self, comparison: &Comparison<'_>, case: &str) -> String {
        if comparison.kind == CheckKind::Equivalence {
            let other = if comparison.left.case == case {
                comparison.right.case
            } else {
                comparison.left.case
            };
            format!("With {other} ({})", comparison.left.implementation)
        } else {
            comparison.category.clone()
        }
    }

    /// Atomically save a standalone HTML view beside the report's evidence.
    ///
    /// # Errors
    /// Returns an artifact write error. Unavailable supporting files are shown
    /// inside the report and do not prevent viewing the recorded results.
    pub fn write_html(&self) -> std::io::Result<()> {
        write_atomic(&self.directory.join("report.html"), self.html().as_bytes())
    }

    fn verdict(&self) -> &'static str {
        match self.report.exit_code() {
            0 => "PASS",
            1 => "FAIL",
            _ => "ERROR",
        }
    }

    fn assertion_summary(&self, case: &CaseResult) -> Option<String> {
        if self
            .case_suite(&case.name)
            .is_none_or(|(_, c)| c.assertions.is_empty())
            && !case
                .implementations
                .values()
                .flat_map(|s| &s.attempts)
                .any(|a| !a.assertions.is_empty())
        {
            return None;
        }
        let side_status = |name| {
            let Some(side) = case.implementations.get(name) else {
                return Status::NotRun;
            };
            if side
                .attempts
                .iter()
                .flat_map(|a| &a.assertions)
                .any(|a| !a.violations.is_empty())
            {
                Status::Fail
            } else if side.attempts.len() == 2
                && side.attempts.iter().all(|a| !a.assertions.is_empty())
            {
                Status::Pass
            } else {
                Status::NotRun
            }
        };
        Some(format!(
            "Scenario Python {} / Rust {}",
            status(side_status("python")),
            status(side_status("rust"))
        ))
    }

    fn totals(&self) -> Vec<(String, String)> {
        let mut totals = vec![(
            self.validation_label().into(),
            Validation::for_cases(self.report.cases.iter()).summary(),
        )];
        for side in ["python", "rust"] {
            totals.push((
                format!("{side} repeatability"),
                counts(self.report.cases.iter().map(|case| {
                    case.implementations
                        .get(side)
                        .map(|r| r.repeatability.status)
                        .unwrap_or(Status::NotRun)
                })),
            ));
        }
        totals.push((
            self.parity_label().into(),
            format!(
                "{} · {} differences",
                counts(self.report.cases.iter().map(|case| case.parity.status)),
                self.report
                    .cases
                    .iter()
                    .map(|case| case.parity.differences.len())
                    .sum::<usize>()
            ),
        ));
        totals.push((
            "Case equivalence".into(),
            counts(
                self.report
                    .equivalence
                    .iter()
                    .map(|result| result.check.status),
            ),
        ));
        totals
    }

    fn run_diagnostics(&self) -> Vec<String> {
        let mut messages = self.report.runtime_errors.clone();
        if let Err(error) = &self.suite {
            messages.push(error.clone());
        }
        messages
    }

    fn case_diagnostics(&self, case: &CaseResult) -> Vec<String> {
        let mut messages: Vec<_> = case
            .unavailable
            .iter()
            .map(|s| format!("Not covered: {s}"))
            .collect();
        if let Some(summary) = self.assertion_summary(case) {
            messages.push(summary);
        }
        for (side, result) in &case.implementations {
            for (index, attempt) in result.attempts.iter().enumerate() {
                let label = format!("{side} / {} / attempt {}", case.name, index + 1);
                if let Some(error) = attempt
                    .observation
                    .as_ref()
                    .and_then(|o| o.transport_error.as_ref())
                {
                    messages.push(format!("{label}: transport error: {error}"));
                }
                for assertion in &attempt.assertions {
                    for violation in &assertion.violations {
                        messages.push(format!(
                            "{label}: scenario {} failed at {}: {}",
                            assertion.name, violation.path, violation.message
                        ));
                    }
                }
                for (step, prerequisite) in attempt.before_each.iter().enumerate() {
                    for violation in &prerequisite.violations {
                        messages.push(format!(
                            "{label}: prerequisite {}: {}: {}",
                            step + 1,
                            violation.path,
                            violation.message
                        ));
                    }
                    if let Some(error) = prerequisite
                        .observation
                        .as_ref()
                        .and_then(|o| o.transport_error.as_ref())
                    {
                        messages.push(format!(
                            "{label}: prerequisite {} transport error: {error}",
                            step + 1
                        ));
                    }
                }
                for violation in &attempt.violations {
                    messages.push(format!(
                        "{label}: {}{}: {}",
                        violation.path,
                        violation
                            .event
                            .map(|event| format!(" (event {event})"))
                            .unwrap_or_default(),
                        violation.message
                    ));
                }
            }
        }
        messages
    }

    fn check_diagnostic(&self, comparison: &Comparison<'_>) -> Option<&'static str> {
        match comparison.check.status {
            Status::Pass | Status::Fail => return None,
            Status::Unstable => {
                return Some("Repeated responses differ; this implementation is not repeatable");
            }
            Status::NotRun => return Some("Check was not run"),
            Status::Skipped => {}
        }
        for evidence in [comparison.left, comparison.right] {
            let Some(attempt) = evidence.attempt else {
                return Some("Comparison skipped: a required attempt was not recorded");
            };
            if attempt
                .observation
                .as_ref()
                .is_none_or(|o| o.transport_error.is_some())
            {
                return Some(
                    "Comparison skipped: a required response was not captured successfully",
                );
            }
            if !attempt.violations.is_empty() {
                return Some(if self.generated_content() {
                    "Comparison skipped: required output failed integrity checks"
                } else {
                    "Comparison skipped: a required response failed validation"
                });
            }
            if self
                .report
                .cases
                .iter()
                .find(|case| case.name == evidence.case)
                .and_then(|case| case.implementations.get(evidence.implementation))
                .is_some_and(|side| side.repeatability.status == Status::Unstable)
            {
                return Some("Comparison skipped: an implementation's repeated responses differ");
            }
        }
        Some("Comparison skipped; see run diagnostics for source verification or incomplete work")
    }

    fn relative<'p>(&self, saved: &'p Path) -> Option<&'p Path> {
        let relative = if saved.is_absolute() {
            saved.strip_prefix(&self.report.directory).ok()?
        } else {
            saved
        };
        relative
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
            .then_some(relative)
    }

    fn read_json(&self, saved: &Path) -> Result<Value, String> {
        let relative = self.relative(saved).ok_or_else(|| {
            format!(
                "Evidence outside the run directory is unavailable: {}",
                saved.display()
            )
        })?;
        let path = self.directory.join(relative);
        let bytes = std::fs::read(&path)
            .map_err(|error| format!("Unavailable {}: {error}", path.display()))?;
        serde_json::from_slice(&bytes)
            .map_err(|error| format!("Unavailable JSON {}: {error}", path.display()))
    }

    fn final_files(&self, selected: Option<&str>) -> JsonFiles {
        let mut files = BTreeMap::new();
        for comparison in &self.comparisons {
            if selected
                .is_some_and(|name| comparison.left.case != name && comparison.right.case != name)
            {
                continue;
            }
            for side in [comparison.left, comparison.right] {
                if let Some(path) = side.file() {
                    files
                        .entry(path.to_owned())
                        .or_insert_with(|| self.read_json(path));
                }
            }
        }
        files
    }

    fn original(&self, side: Evidence<'_>, pointer: &str, files: &JsonFiles) -> String {
        match side.file().and_then(|p| files.get(p)) {
            Some(Ok(value)) => value_text(value.pointer(pointer)),
            _ => "<unavailable>".into(),
        }
    }

    fn comparison_reason(&self, side: Evidence<'_>, pointer: &str) -> Option<&str> {
        let (suite, case) = self.case_suite(side.case)?;
        let relative = match if side.projected() {
            ComparisonScope::Root
        } else {
            case.comparison_scope
        } {
            ComparisonScope::Root => pointer,
            ComparisonScope::TopLevelArrayItems => {
                let (index, _) = pointer.strip_prefix('/')?.split_once('/')?;
                index.parse::<usize>().ok()?;
                &pointer[index.len() + 1..]
            }
        };
        let numeric = suite
            .comparison
            .per_result_numeric_rules
            .iter()
            .find(|rule| rule.matches(relative))
            .map(|rule| rule.reason.as_str());
        if numeric.is_some() || side.projected() {
            return numeric;
        }
        suite
            .comparison
            .per_result_value_exceptions
            .iter()
            .find(|rule| rule.path == relative)
            .map(|rule| rule.reason.as_str())
    }

    fn sources<'b>(side: Evidence<'b>, mut pointer: &str) -> Option<&'b [usize]> {
        let attempt = side.attempt?;
        let origins = if side.projected() {
            &attempt.equivalence.as_ref()?.origins
        } else {
            &attempt.origins
        };
        loop {
            if let Some(indices) = origins.get(pointer) {
                return Some(indices);
            }
            pointer = pointer.rsplit_once('/')?.0;
        }
    }

    fn source_events(side: Evidence<'_>, indices: &[usize]) -> Result<Value, String> {
        let observation = side
            .attempt
            .and_then(|a| a.observation.as_ref())
            .ok_or("Source events unavailable")?;
        let events: Result<Vec<_>, _> = indices.iter().map(|index| {
            let event = observation.events.get(*index).ok_or_else(|| format!("Source event {index} unavailable"))?;
            let data = serde_json::from_str::<Value>(&event.data).unwrap_or_else(|_| Value::String(event.data.clone()));
            Ok(serde_json::json!({"index": index, "event": event.event, "id": event.id, "data": data}))
        }).collect();
        events.map(Value::Array)
    }

    fn evidence_paths(&self, side: Evidence<'_>) -> Vec<(&'static str, PathBuf)> {
        let Some(attempt) = side.attempt else {
            return Vec::new();
        };
        let mut paths = vec![("request", attempt.directory.join("request.json"))];
        if let Some(path) = &attempt.final_json {
            paths.push(("reconstructed response", path.clone()));
        }
        if let Some(path) = &attempt.output_json {
            paths.push(("generated output", path.clone()));
        }
        if let Some(observation) = &attempt.observation {
            paths.push(("raw response", observation.raw_body.clone()));
        }
        if let Some(evidence) = &attempt.equivalence {
            paths.push(("semantic equivalence value", evidence.file.clone()));
        }
        let events = attempt.directory.join("events.json");
        if self
            .relative(&events)
            .is_some_and(|p| self.directory.join(p).exists())
        {
            paths.push(("SSE events", events));
        }
        paths.push((
            "server log",
            attempt.server_log.clone().unwrap_or_else(|| {
                self.report
                    .directory
                    .join(side.implementation)
                    .join("server.log")
            }),
        ));
        paths
    }

    fn link(&self, label: &str, saved: &Path) -> String {
        match self.relative(saved) {
            Some(relative) if self.directory.join(relative).is_file() => {
                format!("<a href=\"{}\">{}</a>", url_path(relative), escape(label))
            }
            _ => format!(
                "<span class=\"muted\">{} (unavailable)</span>",
                escape(label)
            ),
        }
    }

    fn html(&self) -> String {
        let files = self.final_files(None);
        let metadata = self.metadata();
        let mut out = format!(
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>SGLang Parity — {} · {} — {}</title><style>{STYLE}</style></head><body><main><header><h1>SGLang Parity <span class=\"status {}\">{}</span></h1><p>State: {} · Exit code: {}</p><dl class=\"metadata\">",
            escape(&metadata[0].1),
            escape(&metadata[1].1),
            self.verdict(),
            if self.report.exit_code() == 0 {
                "PASS"
            } else {
                "FAIL"
            },
            self.verdict(),
            escape(&self.report.state),
            self.report.exit_code()
        );
        for (label, value) in metadata {
            write!(
                out,
                "<div><dt>{label}</dt><dd><code>{}</code></dd></div>",
                escape(&value)
            )
            .unwrap();
        }
        out.push_str("</dl></header><section class=\"totals\" aria-label=\"Run totals\">");
        for (label, value) in self.totals() {
            write!(
                out,
                "<div><h2>{}</h2><p>{}</p></div>",
                escape(&label),
                escape(&value)
            )
            .unwrap();
        }
        out.push_str("</section><p class=\"muted\">");
        if self.generated_content() {
            out.push_str("Output integrity checks that generated content can be reconstructed. Content parity compares Python with Rust. Metadata is not checked. Generated output values come from output.json before value exceptions. ");
        } else {
            out.push_str("Response validation checks each response. Parity compares Python with Rust. Reconstructed values come from final.json before value exceptions. ");
        }
        out.push_str("Repeatability compares two runs of one implementation. Case equivalence compares declared related cases. Differences count occurrences, not independent bugs.</p><p class=\"muted\">Comparison values include declared replacements. &lt;missing&gt; means an absent field; null is a present JSON value; &lt;unavailable&gt; means evidence could not be read. This view preserves recorded verdicts and does not recompute comparisons.</p>");
        let diagnostics = self.run_diagnostics();
        if !diagnostics.is_empty() {
            out.push_str("<section><h2>Run diagnostics</h2><ul>");
            for message in diagnostics {
                write!(out, "<li>{}</li>", escape(&message)).unwrap();
            }
            out.push_str("</ul></section>");
        }
        out.push_str("<nav aria-label=\"Test cases\"><h2>Cases</h2><ul class=\"case-index\">");
        for (index, case) in self.report.cases.iter().enumerate() {
            write!(
                out,
                "<li><a href=\"#case-{index}\">{}</a><span>{} {} · {} differences</span></li>",
                escape(&case.name),
                self.compact_parity_label(),
                badge(case.parity.status),
                case.parity.differences.len()
            )
            .unwrap();
        }
        out.push_str("</ul></nav>");
        let mut profile = None;
        for (index, case) in self.report.cases.iter().enumerate() {
            if !self.profile_suites.is_empty() && profile != Some(&case.profile_id) {
                write!(
                    out,
                    "<section><h2>Profile {}</h2><p>{}</p>",
                    escape(&case.profile_id),
                    escape(
                        self.profile_suites
                            .get(&case.profile_id)
                            .map_or("Unavailable", |s| s.output_mode.as_str())
                    )
                )
                .unwrap();
                if let Ok(plan) = self.read_json(&self.report.effective_suite)
                    && let Some(entry) = plan["profiles"].as_array().and_then(|ps| {
                        ps.iter()
                            .find(|p| p["profile"]["id"].as_str() == Some(&case.profile_id))
                    })
                {
                    self.html_preview(
                        &mut out,
                        "Effective startup settings and requirements",
                        &Ok(entry["profile"].clone()),
                    );
                }
                if let Some(path) = self
                    .report
                    .profiles
                    .get(&case.profile_id)
                    .and_then(|p| p.environment.as_ref())
                {
                    write!(out, "<p>{}</p>", self.link("Verified environment", path)).unwrap();
                }
                out.push_str("</section>");
                profile = Some(&case.profile_id);
            }
            write!(out, "<section id=\"case-{index}\" class=\"case\"><h2>{}</h2><p>{}</p><dl class=\"checks\">", escape(&case.name), escape(&self.case_description(&case.name))).unwrap();
            let validation = Validation::for_cases(std::iter::once(case));
            write!(
                out,
                "<div><dt>{}</dt><dd>{} {}</dd></div>",
                self.validation_label(),
                badge(validation.status()),
                validation.summary()
            )
            .unwrap();
            let comparisons = self.case_comparisons(&case.name);
            for comparison in &comparisons {
                write!(
                    out,
                    "<div><dt><a href=\"#{}\">{}</a></dt><dd>{} {} differences</dd></div>",
                    comparison.id,
                    escape(&self.check_label(comparison, &case.name)),
                    badge(comparison.check.status),
                    comparison.check.differences.len()
                )
                .unwrap();
            }
            out.push_str("</dl>");
            let diagnostics = self.case_diagnostics(case);
            if !diagnostics.is_empty() {
                out.push_str(if self.generated_content() {
                    "<h3>Output integrity diagnostics</h3><ul>"
                } else {
                    "<h3>Response diagnostics</h3><ul>"
                });
                for message in diagnostics {
                    write!(out, "<li>{}</li>", escape(&message)).unwrap();
                }
                out.push_str("</ul>");
            }
            for comparison in comparisons {
                if comparison.kind != CheckKind::Equivalence || comparison.left.case == case.name {
                    self.html_comparison(&mut out, comparison, &files);
                }
            }
            write!(
                out,
                "<p><a href=\"#rules\">Recorded comparison and response rules</a> · {}</p>",
                self.link("Effective suite", &self.report.effective_suite)
            )
            .unwrap();
            self.html_evidence(&mut out, case, &files);
            out.push_str("</section>");
        }
        out.push_str("<section id=\"rules\"><h2>Recorded rules</h2>");
        let suites: Vec<_> = if self.profile_suites.is_empty() {
            self.suite
                .as_ref()
                .ok()
                .map(|s| ("default", s))
                .into_iter()
                .collect()
        } else {
            self.profile_suites
                .iter()
                .map(|(id, s)| (id.as_str(), s))
                .collect()
        };
        for (id, suite) in suites {
            self.html_preview(
                &mut out,
                &format!("Recorded comparison rules — {id}"),
                &Ok(serde_json::to_value(&suite.comparison).expect("serializable rules")),
            );
            if let Some(policy) = &suite.response_policy {
                self.html_preview(
                    &mut out,
                    &format!("Recorded response policy and scenario assertions — {id}"),
                    &Ok(policy.clone()),
                );
            }
        }
        write!(
            out,
            "</section><footer>{} · {} · {}</footer></main></body></html>",
            self.link(
                "Machine-readable report",
                &self.report.directory.join("report.json")
            ),
            self.link("Effective suite", &self.report.effective_suite),
            self.link("Setup log", &self.report.directory.join("setup.log"))
        )
        .unwrap();
        out
    }

    fn html_comparison(&self, out: &mut String, comparison: &Comparison<'_>, files: &JsonFiles) {
        write!(out, "<details id=\"{}\"{}><summary>{} {} · {} differences</summary><div class=\"comparison\"><p>Compared: {} &harr; {}</p>",
            comparison.id, if comparison.check.status == Status::Pass { "" } else { " open" }, badge(comparison.check.status),
            escape(&format!("{} · {}", comparison.category, comparison.name)), comparison.check.differences.len(),
            escape(&comparison.left.label()), escape(&comparison.right.label())).unwrap();
        if let Some(reason) = self.check_diagnostic(comparison) {
            write!(out, "<p>{}</p>", escape(reason)).unwrap();
        }
        for reason in comparison.reasons() {
            write!(out, "<p><strong>{}</strong></p>", escape(&reason)).unwrap();
        }
        for side in [comparison.left, comparison.right] {
            write!(out, "<p>{}: ", escape(&side.label())).unwrap();
            for (index, (label, path)) in self.evidence_paths(side).iter().enumerate() {
                if index > 0 {
                    out.push_str(" · ");
                }
                out.push_str(&self.link(label, path));
            }
            out.push_str("</p>");
        }
        if !comparison.check.differences.is_empty() {
            let labels = comparison.labels();
            write!(out, "<table class=\"differences\"><thead><tr><th scope=\"col\">Field / reason</th><th scope=\"col\">{}</th><th scope=\"col\">{}</th></tr></thead><tbody>", escape(&labels[0]), escape(&labels[1])).unwrap();
            for difference in &comparison.check.differences {
                write!(
                    out,
                    "<tr><th scope=\"row\"><code>{}</code><p>{}</p></th>",
                    escape(&difference.path),
                    escape(&comparison.reason(difference.kind))
                )
                .unwrap();
                for ((side, value), label) in [
                    (comparison.left, difference.left.as_ref()),
                    (comparison.right, difference.right.as_ref()),
                ]
                .into_iter()
                .zip(&labels)
                {
                    write!(out, "<td><strong class=\"side-label\" aria-hidden=\"true\">{}</strong><small>Comparison value</small>", escape(label)).unwrap();
                    let text = value
                        .map(|v| serde_json::to_string_pretty(v).expect("serializable JSON"))
                        .unwrap_or_else(|| "<missing>".into());
                    html_pre(out, &text);
                    if let Some(reason) = value
                        .filter(|v| !v.is_null())
                        .and_then(|_| self.comparison_reason(side, &difference.path))
                    {
                        write!(
                            out,
                            "<p class=\"muted\">Value exception applied: {}</p>",
                            escape(reason)
                        )
                        .unwrap();
                    }
                    write!(out, "<details><summary>{}</summary>", side.value_label()).unwrap();
                    html_pre(out, &self.original(side, &difference.path, files));
                    out.push_str("</details>");
                    if let Some(indices) = Self::sources(side, &difference.path) {
                        self.html_preview(
                            out,
                            "Source events (zero-based indices)",
                            &Self::source_events(side, indices),
                        );
                    }
                    out.push_str("</td>");
                }
                out.push_str("</tr>");
            }
            out.push_str("</tbody></table>");
        }
        out.push_str("</div></details>");
    }

    fn html_evidence(&self, out: &mut String, case: &CaseResult, files: &JsonFiles) {
        out.push_str("<details><summary>Requests, original responses and logs</summary>");
        for (implementation, result) in &case.implementations {
            for (index, attempt) in result.attempts.iter().enumerate() {
                let side = Evidence {
                    semantic: false,
                    case: &case.name,
                    implementation,
                    repeat: index + 1,
                    attempt: Some(attempt),
                };
                write!(out, "<h3>{}</h3><p>", escape(&side.label())).unwrap();
                for (label, path) in self.evidence_paths(side) {
                    write!(out, "{} · ", self.link(label, &path)).unwrap();
                }
                out.push_str("</p>");
                for (step, prerequisite) in attempt.before_each.iter().enumerate() {
                    write!(
                        out,
                        "<p>Prerequisite {}: {} · {}</p>",
                        step + 1,
                        self.link("request", &prerequisite.directory.join("request.json")),
                        self.link("response", &prerequisite.directory.join("response.body"))
                    )
                    .unwrap();
                }
                self.html_preview(
                    out,
                    "Request JSON",
                    &self.read_json(&attempt.directory.join("request.json")),
                );
                if let Some(path) = attempt.prepared_path()
                    && let Some(value) = files.get(path)
                {
                    self.html_preview(
                        out,
                        if attempt.output_json.is_some() {
                            "Generated output JSON (before value exceptions)"
                        } else {
                            "Reconstructed JSON (before value exceptions)"
                        },
                        value,
                    );
                }
            }
        }
        out.push_str("</details>");
    }

    fn html_preview(&self, out: &mut String, label: &str, value: &Result<Value, String>) {
        let text = match value {
            Ok(value) => serde_json::to_string_pretty(value).expect("serializable JSON"),
            Err(error) => error.clone(),
        };
        write!(out, "<details><summary>{}</summary>", escape(label)).unwrap();
        html_pre(out, &text);
        out.push_str("</details>");
    }
}

fn html_pre(out: &mut String, text: &str) {
    let mut end = text.len().min(PREVIEW_LIMIT);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    write!(out, "<pre>{}</pre>", escape(&text[..end])).unwrap();
    if end < text.len() {
        out.push_str("<p>Preview truncated at 64 KiB. Use the complete artifact link above.</p>");
    }
}

fn evidence<'a>(case: &'a CaseResult, implementation: &'a str, repeat: usize) -> Evidence<'a> {
    Evidence {
        semantic: false,
        case: &case.name,
        implementation,
        repeat,
        attempt: case
            .implementations
            .get(implementation)
            .and_then(|side| side.attempts.get(repeat - 1)),
    }
}

fn status(value: Status) -> &'static str {
    match value {
        Status::Pass => "PASS",
        Status::Fail => "FAIL",
        Status::Unstable => "UNSTABLE",
        Status::Skipped => "SKIPPED",
        Status::NotRun => "NOT_RUN",
    }
}

fn counts(values: impl Iterator<Item = Status>) -> String {
    let mut totals = [0; 5];
    for value in values {
        totals[STATUSES.iter().position(|status| *status == value).unwrap()] += 1;
    }
    let mut out = format!("{}/{} passed", totals[0], totals.iter().sum::<usize>());
    for (index, value) in STATUSES.iter().enumerate().skip(1) {
        if totals[index] > 0 {
            write!(out, " · {} {}", totals[index], status(*value)).unwrap();
        }
    }
    out
}

fn terminal_status(label: &str, color: bool) -> String {
    let code = match label {
        "PASS" => "32",
        "FAIL" | "ERROR" => "31",
        "UNSTABLE" => "33",
        _ => return label.into(),
    };
    if color {
        format!("\x1b[{code}m{label}\x1b[0m")
    } else {
        label.into()
    }
}

fn value_text(value: Option<&Value>) -> String {
    value
        .map(Value::to_string)
        .unwrap_or_else(|| "<missing>".into())
}

fn plain(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for character in value.chars() {
        if character.is_control() {
            write!(out, "{}", character.escape_default()).unwrap();
        } else {
            out.push(character);
        }
    }
    out
}

fn clipped(value: &str, limit: usize) -> String {
    let value = plain(value);
    if value.chars().count() > limit {
        format!(
            "{}… [truncated]",
            value.chars().take(limit).collect::<String>()
        )
    } else {
        value
    }
}

fn escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn url_path(path: &Path) -> String {
    let mut out = String::new();
    for byte in path.to_string_lossy().bytes() {
        if byte.is_ascii_alphanumeric() || b"/-_.~".contains(&byte) {
            out.push(char::from(byte));
        } else {
            write!(out, "%{byte:02X}").unwrap();
        }
    }
    out
}

fn badge(value: Status) -> String {
    format!(
        "<span class=\"status {}\">{}</span>",
        status(value),
        status(value)
    )
}

const STYLE: &str = "
:root {
    color-scheme: light dark;
    --bg: #f6f8fa; --panel: #fff; --text: #1f2328; --muted: #59636e;
    --line: #d1d9e0; --link: #0969da;
    --pass: #116329; --pass-bg: #dafbe1; --fail: #a40e26; --fail-bg: #ffebe9;
    --unstable: #7d4e00; --unstable-bg: #fff8c5; --neutral-bg: #eff2f5;
}
@media (prefers-color-scheme: dark) {
    :root {
        --bg: #0d1117; --panel: #151b23; --text: #f0f6fc; --muted: #b1bac4;
        --line: #3d444d; --link: #79c0ff;
        --pass: #7ee787; --pass-bg: #12261e; --fail: #ffa198; --fail-bg: #3b181c;
        --unstable: #e3b341; --unstable-bg: #30270c; --neutral-bg: #212830;
    }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 1rem/1.6 system-ui, -apple-system, 'Segoe UI', sans-serif; }
main { max-width: 1200px; margin: auto; padding: 2rem 24px; }
h1, h2, h3 { font-weight: 600; line-height: 1.4; }
h1 { font-size: 1.75rem; margin: 0; }
h2 { font-size: 1.25rem; margin: 0 0 1rem; }
h3 { font-size: 1rem; }
p { margin: .75rem 0; }
a { color: var(--link); text-decoration: underline; }
a, summary, li, dt, dd, h2 { overflow-wrap: anywhere; }
a:focus-visible, summary:focus-visible { outline: 3px solid var(--link); outline-offset: 3px; }
section, nav { background: var(--panel); border: 1px solid var(--line); border-radius: .5rem; padding: 1rem; margin: 1.5rem 0; }
details { border: 1px solid var(--line); border-radius: .375rem; margin: 1rem 0; padding: .75rem; min-width: 0; }
summary { cursor: pointer; font-weight: 600; }
summary .status { margin: 0 .5rem 0 .25rem; }
.totals { display: flex; flex-wrap: wrap; gap: 1rem; background: none; border: 0; padding: 0; }
.totals > div { flex: 1 1 200px; background: var(--panel); border: 1px solid var(--line); border-radius: .5rem; padding: 1rem; }
.totals h2 { font-size: .875rem; margin: 0; }
.totals p { margin-bottom: 0; }
.metadata { display: flex; flex-wrap: wrap; gap: .75rem 2rem; font-size: .875rem; }
.metadata div { min-width: 0; }
dt { font-weight: 600; }
dd { margin: 0; }
.metadata dt, .muted, small { color: var(--muted); }
.muted, small { font-size: .875rem; }
small { display: block; }
.checks > div { display: flex; flex-wrap: wrap; gap: .25rem 1rem; padding: .375rem 0; }
.checks dt { flex: 0 1 270px; }
.checks dd { flex: 1 1 260px; }
.case-index { padding: 0; margin: 0; list-style: none; }
.case-index li { display: flex; flex-wrap: wrap; justify-content: space-between; gap: .5rem 1rem; padding: .5rem 0; border-bottom: 1px solid var(--line); }
.case-index li:last-child { border: 0; }
.case-index span { font-size: .875rem; }
table { border-collapse: collapse; width: 100%; table-layout: fixed; text-align: left; }
th, td { padding: .75rem; vertical-align: top; border-bottom: 1px solid var(--line); overflow-wrap: anywhere; }
th { font-size: .875rem; font-weight: 600; }
.differences tbody th p { font-weight: 400; }
pre, code { font-family: ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace; font-size: .875rem; line-height: 1.55; white-space: pre-wrap; overflow-wrap: anywhere; }
pre { max-height: 440px; overflow: auto; margin: .5rem 0; font-weight: 400; }
.side-label { display: none; }
.status { display: inline-block; border-radius: .25rem; padding: .125rem .5rem; font-size: .875rem; font-weight: 600; color: var(--muted); background: var(--neutral-bg); }
.PASS { color: var(--pass); background: var(--pass-bg); }
.FAIL { color: var(--fail); background: var(--fail-bg); }
.UNSTABLE { color: var(--unstable); background: var(--unstable-bg); }
footer { margin: 2rem 0; font-size: .875rem; }
:target { outline: 2px solid var(--link); scroll-margin-top: 1rem; }
@media (max-width: 759px) {
    main { padding: 1rem 16px; }
    .differences, .differences tbody, .differences tr, .differences th, .differences td { display: block; width: 100%; }
    .differences thead { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); }
    .differences tr { border-top: 1px solid var(--line); padding: .75rem 0; }
    .differences th, .differences td { border: 0; padding: .5rem 0; }
    .side-label { display: block; margin-bottom: .25rem; }
}
";

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn recorded() -> Report {
        let cases: Vec<_> = ["json", "stream"].iter().map(|name| {
            let sides: serde_json::Map<String, Value> = ["python", "rust"].iter().map(|side| {
                let attempts: Vec<_> = (1..=2).map(|repeat| {
                    let directory = format!("/original/run/{side}/{name}/{repeat}");
                    json!({
                        "directory": directory, "final_json": format!("{directory}/final.json"),
                        "observation": {"status": 200, "headers": {}, "raw_body": format!("{directory}/response.body"), "json": null, "events": [], "transport_error": null, "violations": []},
                        "violations": []
                    })
                }).collect();
                (side.to_string(), json!({"attempts": attempts, "repeatability": {"status": "PASS", "differences": []}}))
            }).collect();
            json!({"name": name, "implementations": sides, "parity": {"status": "FAIL", "differences": [
                {"path": "/time", "kind": "missing_right", "left": 0},
                {"path": "/nullable", "kind": "missing_right", "left": null}
            ]}})
        }).collect();
        serde_json::from_value(json!({
            "state": "complete", "directory": "/original/run", "effective_suite": "/original/run/effective_suite.json",
            "config": {"server": {"model": "fixture"}}, "environment": null, "runtime_errors": [], "cases": cases,
            "equivalence": [{"group": "formats", "implementation": "python", "left": "json", "right": "stream", "check": {"status": "FAIL", "differences": [{"path": "/time", "kind": "missing_right", "left": 0}]}}]
        })).unwrap()
    }

    #[test]
    fn cases_keep_check_categories_evidence_and_equivalence_counts_distinct() {
        let report = recorded();
        assert_eq!(report.check, CheckTarget::FullResponse);
        assert!(
            report.cases[0].implementations["python"].attempts[0]
                .output_json
                .is_none()
        );
        let directory = tempfile::tempdir().unwrap();
        let view = ReportView::new(&report, directory.path());
        let text = view.terminal(None, false).unwrap();
        for expected in [
            "8/8 passed",
            "2/2 passed",
            "0/2 passed · 2 FAIL",
            "0/1 passed · 1 FAIL",
            "2 FAIL · 4 differences",
            "Parity FAIL: 2 missing in Rust",
            "Equivalence: 0/1 passed · 1 FAIL",
            "Streaming mode: Unavailable",
            "Check: Full response parity",
        ] {
            assert!(text.contains(expected), "missing {expected:?} in {text}");
        }
        assert!(!text.contains('\x1b'));
        assert!(!text.contains("WHY CHECKS DIFFER"));
        assert!(!text.contains("reconstructed: <unavailable>"));
        assert!(!text.contains("\n  /time\n"));
        for case in &report.cases {
            let block = text
                .split("\n\n")
                .find(|block| block.starts_with(&format!("{} · ", case.name)))
                .unwrap();
            assert_eq!(block.lines().count(), 5, "{block}");
        }
        let html = view.html();
        assert_eq!(html.matches("id=\"equivalence-0\"").count(), 1);
        assert_eq!(html.matches("href=\"#equivalence-0\"").count(), 2);
        assert!(html.find("id=\"case-0\"").unwrap() < html.find("id=\"equivalence-0\"").unwrap());
        assert!(html.find("id=\"equivalence-0\"").unwrap() < html.find("id=\"case-1\"").unwrap());
        assert!(html.contains("id=\"parity-0\" open"));
        assert!(!html.contains("id=\"repeat-0-1\" open"));
        assert!(view.terminal(None, true).unwrap().contains("\x1b[31mFAIL"));
        assert!(view.terminal(Some("unknown"), false).is_err());
        let expanded = view.terminal(Some("stream"), false).unwrap();
        assert!(!expanded.contains("\njson\n"));
        assert!(
            expanded
                .contains("Compared: python / json / attempt 1 <-> python / stream / attempt 1")
        );
        assert!(
            expanded.contains("Compared: rust / stream / attempt 1 <-> rust / stream / attempt 2")
        );
        assert!(expanded.contains("reconstructed: <unavailable>"));
        assert!(expanded.contains("Python: null\n"));
        assert!(expanded.contains("stream is missing fields present in json (1 differences)"));
        for (kind, message) in [
            (DifferenceKind::ValueMismatch, "1 value mismatches"),
            (DifferenceKind::TypeMismatch, "1 type mismatches"),
            (DifferenceKind::MissingLeft, "1 missing in Python"),
        ] {
            let mut report = recorded();
            report.cases[0].parity.differences[0].kind = kind;
            assert!(
                ReportView::new(&report, directory.path())
                    .terminal(None, false)
                    .unwrap()
                    .contains(message)
            );
        }
    }

    #[test]
    fn generated_content_reports_use_output_evidence_and_preserve_recorded_checks() {
        let mut report = recorded();
        report.check = CheckTarget::GeneratedContent;
        let directory = tempfile::tempdir().unwrap();
        for case in &mut report.cases {
            for side in case.implementations.values_mut() {
                for attempt in &mut side.attempts {
                    let path = attempt.directory.join("output.json");
                    let local = directory
                        .path()
                        .join(path.strip_prefix(&report.directory).unwrap());
                    std::fs::create_dir_all(local.parent().unwrap()).unwrap();
                    std::fs::write(&local, r#"{"time":123.25,"nullable":null}"#).unwrap();
                    attempt.output_json = Some(path);
                    attempt.final_json = None;
                    attempt.origins = [("/time".into(), vec![0])].into();
                    attempt
                        .observation
                        .as_mut()
                        .unwrap()
                        .events
                        .push(crate::sse::SseEvent {
                            event: "message".into(),
                            id: None,
                            data: r#"{"time":123.25}"#.into(),
                        });
                }
            }
        }
        // Keep a different full-response artifact to prove content evidence wins.
        let attempt = &mut report.cases[0]
            .implementations
            .get_mut("python")
            .unwrap()
            .attempts[0];
        attempt.final_json = Some(attempt.directory.join("final.json"));
        std::fs::write(
            directory.path().join("python/json/1/final.json"),
            r#"{"time":999.75}"#,
        )
        .unwrap();
        let recorded_checks = serde_json::to_value(&report).unwrap();
        let view = ReportView::new(&report, directory.path());
        let compact = view.terminal(None, false).unwrap();
        for expected in [
            "Check: Generated content parity",
            "Metadata: Not checked",
            "Output integrity",
            "8/8 passed",
            "Content parity FAIL: 2 missing in Rust",
        ] {
            assert!(
                compact.contains(expected),
                "missing {expected:?} in {compact}"
            );
        }
        assert!(!compact.contains("Response validation"));
        for case in &report.cases {
            let block = compact
                .split("\n\n")
                .find(|block| block.starts_with(&format!("{} · ", case.name)))
                .unwrap();
            assert_eq!(block.lines().count(), 5, "{block}");
        }
        let expanded = view.terminal(Some("json"), false).unwrap();
        for expected in [
            "Python: 0",
            "generated output: 123.25",
            "source event indices (zero-based): [0]",
            "generated output:",
            "output.json",
        ] {
            assert!(expanded.contains(expected), "missing {expected:?}");
        }
        assert!(!expanded.contains("999.75"));
        let html = view.html();
        for expected in [
            "<dt>Check</dt><dd><code>Generated content parity</code>",
            "<dt>Metadata</dt><dd><code>Not checked</code>",
            "Generated output value (before exceptions)",
            "Generated output JSON (before value exceptions)",
            "href=\"python/json/1/output.json\"",
            "Source events (zero-based indices)",
            "does not recompute comparisons",
        ] {
            assert!(html.contains(expected), "missing {expected:?}");
        }
        assert!(!html.contains("Response validation"));
        assert!(!html.contains("999.75"));
        assert_eq!(serde_json::to_value(&report).unwrap(), recorded_checks);
        std::fs::remove_file(directory.path().join("python/json/1/output.json")).unwrap();
        let expanded = view.terminal(Some("json"), false).unwrap();
        assert!(expanded.contains("generated output: <unavailable>"));
        assert!(!expanded.contains("999.75"));
        assert_eq!(report.exit_code(), 1);
    }

    #[test]
    fn moved_evidence_distinguishes_original_values_exceptions_and_unavailable_files() {
        let mut report = recorded();
        let attempt = &mut report.cases[0]
            .implementations
            .get_mut("python")
            .unwrap()
            .attempts[0];
        attempt.origins = [("/time".into(), vec![0]), ("/nullable".into(), vec![99])].into();
        attempt
            .observation
            .as_mut()
            .unwrap()
            .events
            .push(crate::sse::SseEvent {
                event: "message".into(),
                id: None,
                data: r#"{"time":123.25,"unsafe":"<script>"}"#.into(),
            });
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let attempt = root.join("python/json/1");
        std::fs::create_dir_all(&attempt).unwrap();
        std::fs::write(
            attempt.join("final.json"),
            r#"{"time":123.25,"nullable":null,"text":"<script>alert(1)</script>"}"#,
        )
        .unwrap();
        std::fs::write(
            attempt.join("request.json"),
            format!("{{\"text\":\"{}\"}}", "語".repeat(PREVIEW_LIMIT)),
        )
        .unwrap();
        let cases: Vec<_> = ["json", "stream"].iter().map(|name| json!({"name": name, "method": "POST", "path": "/example", "body": {}, "expect_status": 200, "capture": "json", "comparison_scope": "root"})).collect();
        let specification = json!({"suite": {"name": "example", "response_implementation": "example", "output_mode": "example", "cases": cases, "comparison": {"base": "exact_json", "per_result_value_exceptions": [{"path": "/time", "presence": "optional", "require": "non_negative_number", "reason": "Clock value varies"}]}}});
        std::fs::write(root.join("effective_suite.json"), specification.to_string()).unwrap();
        let mut view = ReportView::new(&report, root);
        let side = view.comparisons[0].left;
        assert_eq!(
            ReportView::sources(side, "/time/nested/0"),
            Some([0].as_slice())
        );
        assert_eq!(ReportView::sources(side, "/timestamp"), None);
        let expanded = view.terminal(Some("json"), false).unwrap();
        assert!(expanded.contains("source event indices (zero-based): [0]"));
        let html = view.html();
        for expected in [
            "123.25",
            "Clock value varies",
            "<title>SGLang Parity — example · example — FAIL</title>",
            "&lt;missing&gt;",
            "&lt;unavailable&gt;",
            "&lt;script&gt;",
            "Preview truncated at 64 KiB",
            "href=\"python/json/1/final.json\"",
            "id=\"equivalence-0\"",
            "Source events (zero-based indices)",
            "Source event 99 unavailable",
        ] {
            assert!(html.contains(expected), "missing {expected}");
        }
        assert!(!html.contains("<script>"));
        assert!(!html.contains("href=\"/original/run"));
        assert_eq!(
            view.comparison_reason(side, "/time"),
            Some("Clock value varies")
        );
        view.suite.as_mut().unwrap().cases[0].comparison_scope =
            ComparisonScope::TopLevelArrayItems;
        assert_eq!(
            view.comparison_reason(side, "/12/time"),
            Some("Clock value varies")
        );
        assert_eq!(view.comparison_reason(side, "/12/nested/time"), None);
        assert_eq!(view.comparison_reason(side, "/time"), None);
        for mode in ["cumulative", "incremental", "custom-output"] {
            view.suite.as_mut().unwrap().output_mode = mode.into();
            view.suite.as_mut().unwrap().cases[0].capture = CaptureMode::Sse;
            assert!(
                view.terminal(None, false)
                    .unwrap()
                    .contains(&format!("Streaming mode: {mode}"))
            );
            assert!(
                view.html()
                    .contains(&format!("example · {mode} — FAIL</title>"))
            );
            assert!(
                view.case_description("json")
                    .contains(&format!("SSE · {mode}"))
            );
        }
        view.write_html().unwrap();
        assert!(root.join("report.html").is_file());
        assert!(!root.join("report.html.pending").exists());
        std::fs::remove_file(attempt.join("final.json")).unwrap();
        let html = view.html();
        assert!(html.contains("reconstructed response (unavailable)"));
        assert_eq!(report.exit_code(), 1);
    }

    #[test]
    fn compact_summaries_and_complete_selected_values_keep_safe_styling() {
        let mut report = recorded();
        let long = "語".repeat(160);
        report.cases[0].parity.differences = (0..5)
            .map(|index| {
                serde_json::from_value(json!({"path": format!("/{index}/a~1b/{long}"),
                "kind": "value_mismatch", "left": long, "right": "\u{1b}[31m\n"}))
                .unwrap()
            })
            .collect();
        report.cases[1].name = "stream\u{1b}[31m\n".into();
        let directory = tempfile::tempdir().unwrap();
        let view = ReportView::new(&report, directory.path());
        let text = view.terminal(None, false).unwrap();
        let selected = view.terminal(Some("json"), false).unwrap();
        for difference in &report.cases[0].parity.differences {
            assert!(!text.contains(&difference.path));
            assert!(selected.contains(&difference.path));
        }
        assert!(text.contains("5 value mismatches"));
        assert!(!selected.contains("[truncated]"));
        assert!(selected.contains(&format!("Python: \"{long}\"")));
        assert!(!text.contains('\x1b'));
        assert!(text.contains("stream\\u{1b}[31m\\n"));
        let colored = view.terminal(None, true).unwrap();
        let mut stripped = colored;
        for sequence in ["\x1b[0m", "\x1b[1m", "\x1b[31m", "\x1b[32m", "\x1b[33m"] {
            stripped = stripped.replace(sequence, "");
        }
        assert_eq!(stripped, text);
        for (label, expected) in [
            ("PASS", "\x1b[32mPASS\x1b[0m"),
            ("FAIL", "\x1b[31mFAIL\x1b[0m"),
            ("UNSTABLE", "\x1b[33mUNSTABLE\x1b[0m"),
            ("SKIPPED", "SKIPPED"),
            ("NOT_RUN", "NOT_RUN"),
        ] {
            assert_eq!(terminal_status(label, true), expected);
        }
    }

    #[test]
    fn incomplete_checks_and_validation_errors_never_look_successful() {
        let mut report = recorded();
        report.state = "interrupted".into();
        report.runtime_errors.push("service stopped".into());
        let side = report.cases[0].implementations.get_mut("python").unwrap();
        side.repeatability.status = Status::Unstable;
        side.attempts[0]
            .violations
            .push(crate::Violation::new("/value", "bad type"));
        side.attempts[0].observation.as_mut().unwrap().violations =
            side.attempts[0].violations.clone();
        side.attempts.pop();
        report.cases[0].parity.status = Status::Skipped;
        report.cases[1].parity.status = Status::NotRun;
        let directory = tempfile::tempdir().unwrap();
        let text = ReportView::new(&report, directory.path())
            .terminal(None, false)
            .unwrap();
        for expected in [
            "SGLang Parity — ERROR",
            "6/8 passed · 1 invalid · 1 unavailable/pending",
            "1 UNSTABLE",
            "1 SKIPPED",
            "1 NOT_RUN",
            "service stopped",
        ] {
            assert!(text.contains(expected), "missing {expected}");
        }
        assert!(text.contains("a required response failed validation"));
        let view = ReportView::new(&report, directory.path());
        assert_eq!(
            view.terminal(Some("json"), false)
                .unwrap()
                .matches("bad type")
                .count(),
            1
        );
        for state in STATUSES {
            report.cases[0].parity.status = state;
            let text = ReportView::new(&report, directory.path())
                .terminal(None, false)
                .unwrap();
            let block = text
                .split("\n\n")
                .find(|block| block.starts_with("json · "))
                .unwrap();
            assert_eq!(block.lines().count(), 5, "{block}");
            assert!(block.contains(&format!("Parity {}", status(state))));
            assert!(block.contains("Response FAIL"));
            assert!(block.contains("Python UNSTABLE"));
        }
        assert_eq!(url_path(Path::new("a #?\"/b.json")), "a%20%23%3F%22/b.json");
        assert_eq!(plain("case\x1b[31m\n"), "case\\u{1b}[31m\\n");
        assert!(clipped(&"語".repeat(121), 120).ends_with("[truncated]"));
    }
}
