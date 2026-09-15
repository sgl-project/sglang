//! Human-readable views of recorded results, without reevaluating comparisons.
//!
//! A view reads evidence from the supplied artifact directory, so an entire run
//! can be moved without changing its JSON report. Missing evidence is diagnostic;
//! it never changes the recorded verdict or becomes a missing response field.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::path::{Component, Path, PathBuf};

use serde_json::Value;

use crate::artifacts::write_atomic;
use crate::compare::{ComparisonScope, Difference, DifferenceKind};
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
    case: &'a str,
    implementation: &'a str,
    repeat: usize,
    attempt: Option<&'a Attempt>,
}

impl Evidence<'_> {
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

struct Group<'a> {
    category: String,
    reason: &'static str,
    differences: Vec<(usize, &'a Difference)>,
}

impl Group<'_> {
    fn check_count(&self) -> usize {
        self.differences
            .iter()
            .map(|(i, _)| i)
            .collect::<BTreeSet<_>>()
            .len()
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
    comparisons: Vec<Comparison<'a>>,
    groups: Vec<Group<'a>>,
}

impl<'a> ReportView<'a> {
    pub fn new(report: &'a Report, directory: &Path) -> Self {
        let mut view = Self {
            report,
            directory: directory.to_owned(),
            suite: Err("Saved suite is unavailable".into()),
            comparisons: Vec::new(),
            groups: Vec::new(),
        };
        view.suite = view.read_json(&report.effective_suite).and_then(|value| {
            serde_json::from_value(value.get("suite").cloned().unwrap_or(Value::Null))
                .map_err(|error| format!("Saved suite is unavailable: {error}"))
        });
        for (index, case) in report.cases.iter().enumerate() {
            view.comparisons.push(Comparison {
                kind: CheckKind::Parity,
                category: "Python <-> Rust parity".into(),
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
                    .map(|case| evidence(case, side, 1))
                    .unwrap_or(Evidence {
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
        let mut groups = BTreeMap::<(String, &str), Group<'a>>::new();
        for (index, comparison) in view.comparisons.iter().enumerate() {
            for difference in &comparison.check.differences {
                let reason = match difference.kind {
                    DifferenceKind::MissingLeft if comparison.kind == CheckKind::Parity => {
                        "Python is missing fields present in Rust"
                    }
                    DifferenceKind::MissingRight if comparison.kind == CheckKind::Parity => {
                        "Rust is missing fields present in Python"
                    }
                    kind => difference_reason(kind),
                };
                groups
                    .entry((comparison.category.clone(), reason))
                    .or_insert_with(|| Group {
                        category: comparison.category.clone(),
                        reason,
                        differences: Vec::new(),
                    })
                    .differences
                    .push((index, difference));
            }
        }
        view.groups = groups.into_values().collect();
        view
    }

    /// Render an overview, optionally followed by one case's complete diagnostics.
    ///
    /// # Errors
    /// Returns an error if the requested case does not exist in this report.
    pub fn terminal(&self, case: Option<&str>, color: bool) -> Result<String, String> {
        if let Some(name) = case
            && !self.report.cases.iter().any(|case| case.name == name)
        {
            return Err(format!("unknown report case {name:?}"));
        }
        let mut out = String::new();
        let verdict = self.verdict();
        let styled = if color {
            format!(
                "\x1b[{}m{verdict}\x1b[0m",
                if self.report.exit_code() == 0 {
                    "32"
                } else {
                    "31"
                }
            )
        } else {
            verdict.into()
        };
        writeln!(
            out,
            "SGLang Parity — {styled}\nState: {} · exit code {}\n",
            plain(&self.report.state),
            self.report.exit_code()
        )
        .unwrap();
        for (label, value) in self.totals() {
            writeln!(out, "{label:<26} {value}").unwrap();
        }
        out.push_str("\nRepeatability compares two runs of the same implementation.\nResponse validation checks each response; parity compares Python with Rust.\n");
        if !self.groups.is_empty() {
            out.push_str("\nWHY CHECKS DIFFER\n");
            for group in &self.groups {
                writeln!(
                    out,
                    "\n{} — {}\n  {} checks affected · {} field differences",
                    group.category,
                    group.reason,
                    group.check_count(),
                    group.differences.len()
                )
                .unwrap();
                let mut paths = BTreeSet::new();
                for &(index, difference) in &group.differences {
                    if !paths.insert(&difference.path) {
                        continue;
                    }
                    if paths.len() > 3 {
                        break;
                    }
                    let comparison = &self.comparisons[index];
                    writeln!(
                        out,
                        "  {} [{}]\n    left: {}\n    right: {}\n    comparison values: {} | {}",
                        plain(&difference.path),
                        plain(&comparison.name),
                        plain(&comparison.left.label()),
                        plain(&comparison.right.label()),
                        short_value(difference.left.as_ref()),
                        short_value(difference.right.as_ref())
                    )
                    .unwrap();
                }
            }
            out.push_str("\nCounts are difference occurrences, not independent bugs.\nComparison values include declared replacements; use --case for original values and rules.\n");
        }
        let diagnostics = self.diagnostics(case);
        if !diagnostics.is_empty() {
            out.push_str("\nVALIDATION / EXECUTION DIAGNOSTICS\n");
            for diagnostic in diagnostics {
                writeln!(out, "  {}", plain(&diagnostic)).unwrap();
            }
        }
        out.push_str("\nCASES\n");
        let width = self
            .report
            .cases
            .iter()
            .map(|case| plain(&case.name).chars().count())
            .max()
            .unwrap_or(4)
            .max(4);
        writeln!(
            out,
            "{:<width$}  {:<10}  {:<11}  {:<9} Parity diffs",
            "Case", "Py repeat", "Rust repeat", "Parity"
        )
        .unwrap();
        for case in &self.report.cases {
            let repeat = |side| {
                case.implementations
                    .get(side)
                    .map(|side| status(side.repeatability.status))
                    .unwrap_or("NOT_RUN")
            };
            writeln!(
                out,
                "{:<width$}  {:<10}  {:<11}  {:<9} {}",
                plain(&case.name),
                repeat("python"),
                repeat("rust"),
                status(case.parity.status),
                case.parity.differences.len()
            )
            .unwrap();
        }
        if !self.report.equivalence.is_empty() {
            out.push_str("\nCASE EQUIVALENCE\n");
            for comparison in self
                .comparisons
                .iter()
                .filter(|c| c.kind == CheckKind::Equivalence)
            {
                writeln!(
                    out,
                    "  {} · {}: {} ({} differences)",
                    comparison.category,
                    plain(&comparison.name),
                    status(comparison.check.status),
                    comparison.check.differences.len()
                )
                .unwrap();
            }
        }
        if let Some(name) = case {
            let files = self.final_files(Some(name));
            out.push_str(
                "\nCASE DETAILS — values below are comparison values unless marked original\n",
            );
            for comparison in self
                .comparisons
                .iter()
                .filter(|c| c.left.case == name || c.right.case == name)
            {
                writeln!(
                    out,
                    "\n{} · {}: {}\n  left: {}\n  right: {}",
                    comparison.category,
                    plain(&comparison.name),
                    status(comparison.check.status),
                    plain(&comparison.left.label()),
                    plain(&comparison.right.label())
                )
                .unwrap();
                for difference in &comparison.check.differences {
                    writeln!(
                        out,
                        "  {}: {}\n    {} | {}",
                        plain(&difference.path),
                        difference_reason(difference.kind),
                        short_value(difference.left.as_ref()),
                        short_value(difference.right.as_ref())
                    )
                    .unwrap();
                    for side in [comparison.left, comparison.right] {
                        writeln!(
                            out,
                            "    {} original: {}",
                            plain(side.implementation),
                            clipped(&self.original(side, &difference.path, &files), 120)
                        )
                        .unwrap();
                        if let Some(rule) = self.exception(side.case, &difference.path) {
                            writeln!(out, "    value exception: {}", plain(rule)).unwrap();
                        }
                    }
                }
                for side in [comparison.left, comparison.right] {
                    for (label, path) in self.evidence_paths(side) {
                        if let Some(relative) = self.relative(&path) {
                            writeln!(
                                out,
                                "  {} {label}: {}",
                                plain(&side.label()),
                                plain(&self.directory.join(relative).display().to_string())
                            )
                            .unwrap();
                        }
                    }
                }
            }
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

    fn totals(&self) -> Vec<(String, String)> {
        let mut valid = 0;
        let mut invalid = 0;
        let mut attempted = 0;
        for case in &self.report.cases {
            for side in case.implementations.values() {
                for attempt in &side.attempts {
                    attempted += 1;
                    if !attempt.violations.is_empty() {
                        invalid += 1;
                    } else if attempt.final_json.is_some()
                        && attempt
                            .observation
                            .as_ref()
                            .is_some_and(|o| o.transport_error.is_none())
                    {
                        valid += 1;
                    }
                }
            }
        }
        let expected = self.report.cases.len() * 4;
        let mut totals = vec![(
            "Response validation".into(),
            format!(
                "{valid}/{expected} passed · {invalid} invalid · {} unavailable/pending",
                expected.max(attempted) - valid - invalid
            ),
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
            "Python <-> Rust parity".into(),
            counts(self.report.cases.iter().map(|case| case.parity.status)),
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

    fn diagnostics(&self, selected: Option<&str>) -> Vec<String> {
        let mut messages = self.report.runtime_errors.clone();
        if let Err(error) = &self.suite {
            messages.push(error.clone());
        }
        for case in &self.report.cases {
            if selected.is_some_and(|name| name != case.name) {
                continue;
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
        }
        for comparison in &self.comparisons {
            if selected
                .is_some_and(|name| comparison.left.case != name && comparison.right.case != name)
            {
                continue;
            }
            let Some(reason) = self.check_diagnostic(comparison) else {
                continue;
            };
            messages.push(format!(
                "{} / {}: {reason}",
                comparison.category, comparison.name
            ));
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
                return Some("Comparison skipped: a required response failed validation");
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
                if let Some(path) = side.attempt.and_then(|a| a.final_json.as_ref()) {
                    files
                        .entry(path.clone())
                        .or_insert_with(|| self.read_json(path));
                }
            }
        }
        files
    }

    fn original(&self, side: Evidence<'_>, pointer: &str, files: &JsonFiles) -> String {
        match side
            .attempt
            .and_then(|a| a.final_json.as_ref())
            .and_then(|p| files.get(p))
        {
            Some(Ok(value)) => value_text(value.pointer(pointer)),
            _ => "<unavailable>".into(),
        }
    }

    fn exception(&self, case: &str, pointer: &str) -> Option<&str> {
        let suite = self.suite.as_ref().ok()?;
        let case = suite.cases.iter().find(|c| c.name == case)?;
        let relative = match case.comparison_scope {
            ComparisonScope::Root => pointer,
            ComparisonScope::TopLevelArrayItems => {
                let (index, _) = pointer.strip_prefix('/')?.split_once('/')?;
                index.parse::<usize>().ok()?;
                &pointer[index.len() + 1..]
            }
        };
        suite
            .comparison
            .per_result_value_exceptions
            .iter()
            .find(|rule| rule.path == relative)
            .map(|rule| rule.reason.as_str())
    }

    fn evidence_paths(&self, side: Evidence<'_>) -> Vec<(&'static str, PathBuf)> {
        let Some(attempt) = side.attempt else {
            return Vec::new();
        };
        let mut paths = vec![("request", attempt.directory.join("request.json"))];
        if let Some(path) = &attempt.final_json {
            paths.push(("final response", path.clone()));
        }
        if let Some(observation) = &attempt.observation {
            paths.push(("raw response", observation.raw_body.clone()));
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
            self.report
                .directory
                .join(side.implementation)
                .join("server.log"),
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
        let mut out = format!(
            "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>SGLang Parity — {}</title><style>{STYLE}</style><body><main><header><p class=\"eyebrow\">SGLANG · PARITY REPORT</p><h1>{}</h1><p>State: {} · Exit code: {}</p></header>",
            self.verdict(),
            self.verdict(),
            escape(&self.report.state),
            self.report.exit_code()
        );
        if let Some(environment) = &self.report.environment {
            for (label, pointer) in [
                ("Commit", "/plan/commit"),
                ("Backend", "/plan/profile/backend"),
            ] {
                if let Some(value) = environment.pointer(pointer) {
                    write!(
                        out,
                        "<p>{label}: <code>{}</code></p>",
                        escape(&value_text(Some(value)))
                    )
                    .unwrap();
                }
            }
        }
        if let Ok(suite) = &self.suite {
            write!(out, "<p>Suite: <code>{}</code></p>", escape(&suite.name)).unwrap();
        }
        out.push_str("<section class=\"totals\">");
        for (label, value) in self.totals() {
            write!(
                out,
                "<div><h3>{}</h3><p>{}</p></div>",
                escape(&label),
                escape(&value)
            )
            .unwrap();
        }
        out.push_str("</section><p class=\"muted\">Response validation checks each response. Repeatability compares two runs of one implementation. Parity compares Python with Rust. Case equivalence compares declared related cases.</p><h2>Why checks differ</h2>");
        if self.groups.is_empty() {
            out.push_str("<p>No recorded field differences. Check validation and execution diagnostics below for skipped or incomplete work.</p>");
        }
        for group in &self.groups {
            write!(
                out,
                "<section><h3>{}</h3><p>{}</p><p>{} checks affected · {} field differences</p><ul>",
                escape(group.reason),
                escape(&group.category),
                group.check_count(),
                group.differences.len()
            )
            .unwrap();
            for index in group
                .differences
                .iter()
                .map(|(i, _)| *i)
                .collect::<BTreeSet<_>>()
            {
                let comparison = &self.comparisons[index];
                write!(
                    out,
                    "<li><a href=\"#{}\">{}</a></li>",
                    comparison.id,
                    escape(&comparison.name)
                )
                .unwrap();
            }
            out.push_str("</ul></section>");
        }
        out.push_str(
            "<p class=\"muted\">Counts are difference occurrences, not independent bugs.</p>",
        );
        let diagnostics = self.diagnostics(None);
        if !diagnostics.is_empty() {
            out.push_str("<h2>Validation and execution diagnostics</h2><ul>");
            for message in diagnostics {
                write!(out, "<li>{}</li>", escape(&message)).unwrap();
            }
            out.push_str("</ul>");
        }
        out.push_str("<h2>Cases</h2><div class=\"scroll\"><table><thead><tr><th>Case</th><th>Python repeatability</th><th>Rust repeatability</th><th>Parity</th><th>Parity diffs</th></tr></thead><tbody>");
        for (index, case) in self.report.cases.iter().enumerate() {
            write!(
                out,
                "<tr><td><a href=\"#case-{index}\">{}</a></td>",
                escape(&case.name)
            )
            .unwrap();
            for side in ["python", "rust"] {
                write!(
                    out,
                    "<td>{}</td>",
                    badge(
                        case.implementations
                            .get(side)
                            .map(|s| s.repeatability.status)
                            .unwrap_or(Status::NotRun)
                    )
                )
                .unwrap();
            }
            write!(
                out,
                "<td>{}</td><td>{}</td></tr>",
                badge(case.parity.status),
                case.parity.differences.len()
            )
            .unwrap();
        }
        out.push_str("</tbody></table></div>");
        for (index, case) in self.report.cases.iter().enumerate() {
            write!(
                out,
                "<section id=\"case-{index}\"><h2>{}</h2>",
                escape(&case.name)
            )
            .unwrap();
            if let Ok(suite) = &self.suite
                && let Some(spec) = suite.cases.iter().find(|c| c.name == case.name)
            {
                write!(
                    out,
                    "<p><code>{} {}</code> · expected HTTP {} · {:?}</p>",
                    escape(&spec.method),
                    escape(&spec.path),
                    spec.expect_status,
                    spec.capture
                )
                .unwrap();
            }
            for comparison in self
                .comparisons
                .iter()
                .filter(|c| c.left.case == case.name && c.kind != CheckKind::Equivalence)
            {
                self.html_comparison(&mut out, comparison, &files);
            }
            for comparison in self.comparisons.iter().filter(|c| {
                c.kind == CheckKind::Equivalence
                    && (c.left.case == case.name || c.right.case == case.name)
            }) {
                write!(
                    out,
                    "<p>Related: <a href=\"#{}\">{} · {}</a></p>",
                    comparison.id,
                    escape(&comparison.category),
                    escape(&comparison.name)
                )
                .unwrap();
            }
            out.push_str("<details><summary>Requests, original responses and logs</summary>");
            for (implementation, result) in &case.implementations {
                for (index, attempt) in result.attempts.iter().enumerate() {
                    let side = Evidence {
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
                    self.html_preview(
                        &mut out,
                        "Request JSON",
                        &self.read_json(&attempt.directory.join("request.json")),
                    );
                    if let Some(path) = &attempt.final_json
                        && let Some(value) = files.get(path)
                    {
                        self.html_preview(&mut out, "Original final JSON", value);
                    }
                }
            }
            out.push_str("</details></section>");
        }
        out.push_str("<h2>Case equivalence</h2>");
        for comparison in self
            .comparisons
            .iter()
            .filter(|c| c.kind == CheckKind::Equivalence)
        {
            self.html_comparison(&mut out, comparison, &files);
        }
        out.push_str("<h2>Recorded comparison rules</h2>");
        if let Ok(suite) = &self.suite {
            write!(
                out,
                "<pre>{}</pre>",
                escape(
                    &serde_json::to_string_pretty(&suite.comparison).expect("serializable rules")
                )
            )
            .unwrap();
        } else {
            out.push_str("<p>Saved comparison rules unavailable.</p>");
        }
        write!(
            out,
            "<footer>{} · {} · {}</footer></main></body></html>",
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
        write!(out, "<details id=\"{}\"><summary>{} {} · {} · {} differences</summary><div class=\"comparison\"><p><strong>Left:</strong> {}<br><strong>Right:</strong> {}</p>", comparison.id, badge(comparison.check.status), escape(&comparison.category), escape(&comparison.name), comparison.check.differences.len(), escape(&comparison.left.label()), escape(&comparison.right.label())).unwrap();
        if let Some(reason) = self.check_diagnostic(comparison) {
            write!(out, "<p>{}</p>", escape(reason)).unwrap();
        }
        for side in [comparison.left, comparison.right] {
            if let Some(path) = side.attempt.and_then(|a| a.final_json.as_ref()) {
                write!(
                    out,
                    "<p>{}</p>",
                    self.link(&format!("{} — original final response", side.label()), path)
                )
                .unwrap();
            }
        }
        if !comparison.check.differences.is_empty() {
            out.push_str("<p class=\"muted\">Comparison values may contain declared replacements. &lt;missing&gt; means absent; null is a present JSON value. Original values come from final.json.</p><div class=\"scroll\"><table><thead><tr><th>Path / reason</th><th>Left</th><th>Right</th></tr></thead><tbody>");
            for difference in &comparison.check.differences {
                write!(
                    out,
                    "<tr><td><code>{}</code><p>{}</p></td>",
                    escape(&difference.path),
                    difference_reason(difference.kind)
                )
                .unwrap();
                for (side, value) in [
                    (comparison.left, difference.left.as_ref()),
                    (comparison.right, difference.right.as_ref()),
                ] {
                    write!(out, "<td><small>Comparison value</small><pre>{}</pre><details><summary>Original value</summary><pre>{}</pre></details>", escape(&value_text(value)), escape(&self.original(side, &difference.path, files))).unwrap();
                    if let Some(reason) = self.exception(side.case, &difference.path) {
                        write!(
                            out,
                            "<p class=\"muted\">Declared value exception: {}</p>",
                            escape(reason)
                        )
                        .unwrap();
                    }
                    out.push_str("</td>");
                }
                out.push_str("</tr>");
            }
            out.push_str("</tbody></table></div>");
        }
        out.push_str("</div></details>");
    }

    fn html_preview(&self, out: &mut String, label: &str, value: &Result<Value, String>) {
        let text = match value {
            Ok(value) => serde_json::to_string_pretty(value).expect("serializable JSON"),
            Err(error) => error.clone(),
        };
        let mut end = text.len().min(PREVIEW_LIMIT);
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        write!(
            out,
            "<details><summary>{}</summary><pre>{}</pre>",
            escape(label),
            escape(&text[..end])
        )
        .unwrap();
        if end < text.len() {
            out.push_str(
                "<p>Preview truncated at 64 KiB. Use the complete artifact link above.</p>",
            );
        }
        out.push_str("</details>");
    }
}

fn evidence<'a>(case: &'a CaseResult, implementation: &'a str, repeat: usize) -> Evidence<'a> {
    Evidence {
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

fn difference_reason(kind: DifferenceKind) -> &'static str {
    match kind {
        DifferenceKind::MissingLeft => "Field missing on the left",
        DifferenceKind::MissingRight => "Field missing on the right",
        DifferenceKind::TypeMismatch => "Field types differ",
        DifferenceKind::ValueMismatch => "Field values differ",
    }
}

fn value_text(value: Option<&Value>) -> String {
    value
        .map(Value::to_string)
        .unwrap_or_else(|| "<missing>".into())
}

fn short_value(value: Option<&Value>) -> String {
    clipped(&value_text(value), 120)
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
:root{color-scheme:light dark;--bg:#f5f7fa;--panel:#fff;--text:#182335;--muted:#566477;--line:#dce3ec;--link:#1657a1}
@media(prefers-color-scheme:dark){:root{--bg:#111821;--panel:#1a2431;--text:#e2e9f2;--muted:#acb9ca;--line:#354255;--link:#8abfff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,sans-serif}main{max-width:1250px;margin:auto;padding:40px 28px}h1{font-size:42px;margin:0}h2{margin-top:32px}h3{font-size:16px}.eyebrow{font-size:12px;letter-spacing:.15em;color:var(--muted)}a{color:var(--link);overflow-wrap:anywhere}section,details{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px;margin:12px 0}details details{padding:8px}summary{cursor:pointer;font-weight:600;overflow-wrap:anywhere}.totals{display:flex;flex-wrap:wrap;gap:12px;background:none;border:0;padding:0}.totals>div{flex:1 1 210px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 16px}.totals h3{margin:0}.muted,small{color:var(--muted)}table{border-collapse:collapse;width:100%;text-align:left}th,td{padding:12px;vertical-align:top;border-bottom:1px solid var(--line)}th{font-size:13px}td{min-width:120px}.scroll{overflow-x:auto}pre,code{font-family:ui-monospace,monospace;font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}pre{max-height:440px;overflow:auto}td pre{max-width:440px}.status{display:inline-block;border-radius:4px;padding:1px 7px;background:var(--bg);font-size:12px}.PASS{color:#16804a}.FAIL,.UNSTABLE{color:#ce4b42}.SKIPPED,.NOT_RUN{color:var(--muted)}footer{margin:32px 0;color:var(--muted)}:target{outline:2px solid var(--link);scroll-margin-top:16px}li{overflow-wrap:anywhere}
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
    fn overview_groups_occurrences_and_keeps_check_categories_distinct() {
        let report = recorded();
        let directory = tempfile::tempdir().unwrap();
        let view = ReportView::new(&report, directory.path());
        let text = view.terminal(None, false).unwrap();
        for expected in [
            "8/8 passed",
            "2/2 passed",
            "0/2 passed · 2 FAIL",
            "0/1 passed · 1 FAIL",
            "2 checks affected · 4 field differences",
            "1 checks affected · 1 field differences",
            "null | <missing>",
            "Parity diffs",
        ] {
            assert!(text.contains(expected), "missing {expected:?} in {text}");
        }
        assert!(!text.contains('\x1b'));
        assert!(view.terminal(None, true).unwrap().contains("\x1b[31mFAIL"));
        assert!(view.terminal(Some("unknown"), false).is_err());
        let expanded = view.terminal(Some("stream"), false).unwrap();
        assert!(expanded.contains("left: python / json / attempt 1"));
        assert!(expanded.contains("right: python / stream / attempt 1"));
        assert!(expanded.contains("right: rust / stream / attempt 2"));
        assert!(expanded.contains("original: <unavailable>"));
        for (kind, message) in [
            (DifferenceKind::ValueMismatch, "Field values differ"),
            (DifferenceKind::TypeMismatch, "Field types differ"),
            (
                DifferenceKind::MissingLeft,
                "Python is missing fields present in Rust",
            ),
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
    fn moved_evidence_distinguishes_original_values_exceptions_and_unavailable_files() {
        let report = recorded();
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
        let html = view.html();
        for expected in [
            "123.25",
            "Clock value varies",
            "&lt;missing&gt;",
            "&lt;unavailable&gt;",
            "&lt;script&gt;",
            "Preview truncated at 64 KiB",
            "href=\"python/json/1/final.json\"",
            "id=\"equivalence-0\"",
        ] {
            assert!(html.contains(expected), "missing {expected}");
        }
        assert!(!html.contains("<script>"));
        assert!(!html.contains("href=\"/original/run"));
        assert_eq!(view.exception("json", "/time"), Some("Clock value varies"));
        view.suite.as_mut().unwrap().cases[0].comparison_scope =
            ComparisonScope::TopLevelArrayItems;
        assert_eq!(
            view.exception("json", "/12/time"),
            Some("Clock value varies")
        );
        assert_eq!(view.exception("json", "/12/nested/time"), None);
        assert_eq!(view.exception("json", "/time"), None);
        view.write_html().unwrap();
        assert!(root.join("report.html").is_file());
        assert!(!root.join("report.html.pending").exists());
        std::fs::remove_file(attempt.join("final.json")).unwrap();
        let html = view.html();
        assert!(html.contains("final response (unavailable)"));
        assert_eq!(report.exit_code(), 1);
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
        assert_eq!(text.matches("bad type").count(), 1);
        assert_eq!(url_path(Path::new("a #?\"/b.json")), "a%20%23%3F%22/b.json");
        assert_eq!(plain("case\x1b[31m\n"), "case\\u{1b}[31m\\n");
        assert!(clipped(&"語".repeat(121), 120).ends_with("[truncated]"));
    }
}
