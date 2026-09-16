//! Public-library integration tests with managed HTTP processes and real artifacts.

mod support;

use std::fs;
use std::net::TcpListener;
use std::process::{Command, Stdio};

use serde_json::{Value, json};
use sglang_parity::report::ReportView;
use sglang_parity::runner::{RunError, Status};
use sglang_parity::{RunConfig, Violation, describe, run};

use support::{
    EchoPolicy, Fixture, assert_process_stopped, case, git, read_json, suite, wait_until,
};

#[tokio::test]
async fn managed_json_and_sse_run_matches_describe_requests_and_artifacts() {
    let fixture = Fixture::new();
    let mut unary = case("unary", "/json", "normal");
    let mut streaming = case("streaming", "/sse", "normal");
    unary["equivalence_group"] = json!("same_request");
    streaming["equivalence_group"] = json!("same_request");
    let mut suite = suite(vec![unary, streaming]);
    suite.response_policy = Some(json!({"terminator": "[FIN]"}));
    let effective = serde_json::to_value(describe(&fixture.config, &suite).unwrap()).unwrap();
    assert!(fixture.lifecycle().is_empty());
    assert!(!fixture.config.output_dir.exists());

    let report = run(&fixture.config, &suite, &EchoPolicy).await.unwrap();
    assert_eq!(report.exit_code(), 0);
    assert_eq!(report.state, "complete");
    assert_eq!(fixture.preparations().len(), 1);
    assert_eq!(
        read_json(report.directory.join("environment.json")),
        *report.environment.as_ref().unwrap()
    );
    assert_eq!(read_json(&report.effective_suite), effective);
    assert_eq!(
        read_json(report.directory.join("report.json")),
        serde_json::to_value(&report).unwrap()
    );
    let html = fs::read_to_string(report.directory.join("report.html")).unwrap();
    assert!(html.contains(&format!(
        "<title>SGLang Parity — {} · {} — PASS</title>",
        suite.name, suite.output_mode
    )));
    assert!(html.contains("Recorded response policy"));
    // Re-render a moved run with no tools on PATH; the lifecycle must stay unchanged.
    let moved = fixture.directory.path().join("moved report");
    fs::rename(&report.directory, &moved).unwrap();
    let lifecycle_before = fixture.lifecycle();
    let preparations_before = fixture.preparations();
    let output = Command::new(env!("CARGO_BIN_EXE_sglang-parity"))
        .arg("--report")
        .arg(moved.join("report.json"))
        .args(["--case", "unary"])
        .env("PATH", "/nonexistent")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("8/8 passed"));
    assert!(text.contains("Python <-> Rust parity"));
    assert!(text.contains(&format!("Streaming mode: {}", suite.output_mode)));
    assert!(text.contains("\nunary\n"));
    assert!(!text.contains("\nstreaming\n"));
    assert!(!text.contains('\x1b'));
    assert_eq!(fixture.lifecycle(), lifecycle_before);
    assert_eq!(fixture.preparations(), preparations_before);
    let html = fs::read_to_string(moved.join("report.html")).unwrap();
    assert!(html.contains("href=\"python/unary/1/final.json\""));
    assert!(html.contains("id=\"case-1\""));
    assert_eq!(html.matches("id=\"equivalence-0\"").count(), 1);
    fs::rename(&moved, &report.directory).unwrap();
    assert_eq!(report.equivalence.len(), 2);
    assert!(
        report
            .equivalence
            .iter()
            .all(|result| result.check.status == Status::Pass)
    );

    let lifecycle = fixture.lifecycle();
    let kinds: Vec<_> = lifecycle
        .iter()
        .map(|entry| entry["kind"].as_str().unwrap())
        .collect();
    assert_eq!(
        kinds,
        [
            "start", "request", "request", "request", "request", "stop", "start", "request",
            "request", "request", "request", "stop"
        ]
    );
    for (offset, side) in [(0, "0"), (6, "1")] {
        let start = &lifecycle[offset];
        assert_eq!(start["implementation"], side);
        assert_eq!(start["deterministic"], "1");
        assert_eq!(start["shared"], "same-for-both");
        let args = start["argv"].as_array().unwrap();
        assert_eq!(
            &args[..4],
            &[
                json!("-m"),
                json!("sglang.launch_server"),
                json!("--model-path"),
                json!("fixture-model")
            ]
        );
        assert!(args.contains(&json!("--enable-deterministic-inference")));
        assert!(args.contains(&json!("--disable-radix-cache")));
        let seed = args.iter().position(|arg| arg == "--random-seed").unwrap();
        assert_eq!(args[seed + 1], "42");
    }
    let requests: Vec<_> = lifecycle
        .iter()
        .filter(|entry| entry["kind"] == "request")
        .collect();
    assert_eq!(requests.len(), 8);
    for (case_index, case) in suite.cases.iter().enumerate() {
        let expected = serde_json::to_string_pretty(&case.body).unwrap();
        for index in [
            case_index * 2,
            case_index * 2 + 1,
            case_index * 2 + 4,
            case_index * 2 + 5,
        ] {
            assert_eq!(requests[index]["raw"], expected);
            assert_eq!(requests[index]["path"], case.path);
        }
        for (side, result) in &report.cases[case_index].implementations {
            assert_eq!(result.repeatability.status, Status::Pass);
            assert_eq!(result.attempts.len(), 2);
            for attempt in &result.attempts {
                assert_eq!(
                    fs::read_to_string(attempt.directory.join("request.json")).unwrap(),
                    expected
                );
                let observation = attempt.observation.as_ref().unwrap();
                assert!(observation.transport_error.is_none());
                assert!(!fs::read(&observation.raw_body).unwrap().is_empty());
                let final_json = read_json(attempt.final_json.as_ref().unwrap());
                assert_ne!(final_json["trace"], "<dynamic>");
                assert_eq!(
                    final_json["unknown"],
                    json!({"keep": [1, null, {"nested": true}]})
                );
                if case.name == "streaming" {
                    assert_eq!(attempt.origins.get(""), Some(&vec![0]));
                    assert_eq!(
                        read_json(attempt.directory.join("events.json"))[1]["data"],
                        "[FIN]"
                    );
                } else {
                    assert!(attempt.origins.is_empty());
                }
            }
            assert!(
                fs::read_to_string(report.directory.join(side).join("server.log"))
                    .unwrap()
                    .contains("fixture server ready")
            );
        }
    }
    let socket = TcpListener::bind(("127.0.0.1", fixture.config.server.port)).unwrap();
    drop(socket);
}

#[tokio::test]
async fn precise_differences_invalid_responses_and_instability_remain_distinct() {
    let fixture = Fixture::new();
    let suite = suite(vec![
        case("different", "/json", "different"),
        case("invalid", "/json", "invalid"),
        case("unstable", "/json", "unstable"),
        case("empty_rejection", "/json", "empty_rejection"),
    ]);
    let mut report = run(&fixture.config, &suite, &EchoPolicy).await.unwrap();
    assert_eq!(
        report.exit_code(),
        2,
        "instability takes precedence over parity failure"
    );
    let differences = &report.cases[0].parity;
    assert_eq!(differences.status, Status::Fail);
    assert_eq!(
        differences
            .differences
            .iter()
            .map(|difference| difference.path.as_str())
            .collect::<Vec<_>>(),
        ["/unexpected", "/value/items/1"]
    );
    assert!(differences.differences[0].left.is_none());
    assert_eq!(differences.differences[1].left, Some(Value::Null));
    for side in ["python", "rust"] {
        let invalid = &report.cases[1].implementations[side];
        assert_eq!(invalid.repeatability.status, Status::Skipped);
        assert!(
            invalid
                .attempts
                .iter()
                .all(|attempt| !attempt.violations.is_empty() && attempt.final_json.is_none())
        );
        let unstable = &report.cases[2].implementations[side];
        assert_eq!(unstable.repeatability.status, Status::Unstable);
        assert_eq!(unstable.repeatability.differences[0].path, "/value");
        for attempt in &report.cases[3].implementations[side].attempts {
            assert!(attempt.final_json.is_none());
            assert_eq!(
                attempt.violations,
                [Violation::new(
                    "",
                    "response policy rejected the response without diagnostics"
                )]
            );
        }
    }
    assert_eq!(report.cases[1].parity.status, Status::Skipped);
    assert_eq!(report.cases[2].parity.status, Status::Skipped);

    let summary = ReportView::new(&report, &report.directory)
        .terminal(None, false)
        .unwrap();
    assert!(summary.contains("Python is missing fields present in Rust"));
    assert!(summary.contains("Field types differ"));
    assert!(summary.contains("response policy rejected the response without diagnostics"));
    // Reuse the captured results to check each exit category without masking by others.
    let cases = std::mem::take(&mut report.cases);
    for (case, expected) in cases.iter().zip([1, 1, 2, 1]) {
        report.cases = vec![case.clone()];
        assert_eq!(report.exit_code(), expected, "isolated {}", case.name);
    }
    report.cases = vec![cases[0].clone()];
    report
        .runtime_errors
        .push("deliberate runtime failure".into());
    assert_eq!(
        report.exit_code(),
        2,
        "runtime errors override parity failure"
    );
}

#[tokio::test]
async fn equivalence_requires_matching_stable_valid_members_even_when_parity_passes() {
    let fixture = Fixture::new();
    let mut cases = vec![
        case("reference", "/json", "normal"),
        case("different_value", "/sse", "normal"),
        case("invalid", "/json", "invalid"),
        case("unstable", "/sse", "unstable"),
    ];
    cases[1]["body"]["value"]["items"][1] = json!("changed");
    for case in &mut cases {
        case["equivalence_group"] = json!("same_request");
    }
    let mut report = run(&fixture.config, &suite(cases), &EchoPolicy)
        .await
        .unwrap();
    for case in &report.cases[..2] {
        assert_eq!(case.parity.status, Status::Pass);
        for side in case.implementations.values() {
            assert_eq!(side.repeatability.status, Status::Pass);
        }
    }
    assert_eq!(report.equivalence.len(), 6);
    for result in &report.equivalence {
        assert_eq!(result.group, "same_request");
        assert_eq!(result.left, "reference");
        if result.right == "different_value" {
            assert_eq!(result.check.status, Status::Fail);
            assert_eq!(result.check.differences.len(), 1);
            assert_eq!(result.check.differences[0].path, "/value/items/1");
        } else {
            assert!(matches!(result.right.as_str(), "invalid" | "unstable"));
            assert_eq!(result.check.status, Status::Skipped);
            assert!(result.check.differences.is_empty());
        }
    }
    // The stable members pass individually, but their equivalence failure still exits 1.
    report.cases.truncate(2);
    report
        .equivalence
        .retain(|result| result.right == "different_value");
    assert_eq!(report.exit_code(), 1);
}

#[tokio::test]
async fn changing_declared_rules_changes_execution_without_hidden_exceptions() {
    let fixture = Fixture::new();
    let mut suite = suite(vec![case("same", "/json", "normal")]);
    let passing = run(&fixture.config, &suite, &EchoPolicy).await.unwrap();
    assert_eq!(passing.exit_code(), 0, "{passing:#?}");
    suite.comparison.per_result_value_exceptions.remove(0);
    let effective = serde_json::to_value(describe(&fixture.config, &suite).unwrap()).unwrap();
    let failing = run(&fixture.config, &suite, &EchoPolicy).await.unwrap();
    assert_eq!(failing.exit_code(), 2);
    assert_eq!(read_json(&failing.effective_suite), effective);
    assert_ne!(read_json(&passing.effective_suite), effective);
    for side in failing.cases[0].implementations.values() {
        assert_eq!(side.repeatability.status, Status::Unstable);
        assert_eq!(side.repeatability.differences[0].path, "/trace");
    }
}

#[tokio::test]
async fn explicitly_expected_errors_use_their_own_strict_contract() {
    let fixture = Fixture::new();
    let mut negative = case("bad_request", "/json", "http_error");
    negative["expect_status"] = json!(400);
    let mut suite = suite(vec![negative]);
    suite.comparison.per_result_value_exceptions.clear();
    let report = run(&fixture.config, &suite, &EchoPolicy).await.unwrap();
    assert_eq!(report.exit_code(), 0, "{report:#?}");
    for side in report.cases[0].implementations.values() {
        for attempt in &side.attempts {
            assert_eq!(
                read_json(attempt.final_json.as_ref().unwrap()),
                json!({"error": "deliberate fixture failure"})
            );
        }
    }
}

#[tokio::test]
async fn invalid_http_status_media_type_and_json_keep_complete_raw_artifacts() {
    let fixture = Fixture::new();
    let checks = [
        ("status", "/json", "wrong_status", "/status"),
        ("json_type", "/json", "wrong_type", "/headers/content-type"),
        (
            "json_no_type",
            "/json",
            "missing_type",
            "/headers/content-type",
        ),
        ("sse_type", "/sse", "wrong_type", "/headers/content-type"),
        (
            "sse_no_type",
            "/sse",
            "missing_type",
            "/headers/content-type",
        ),
        ("malformed", "/json", "malformed_json", "/body"),
    ];
    let suite = suite(
        checks
            .iter()
            .map(|(name, path, behavior, _)| case(name, path, behavior))
            .collect(),
    );
    let report = run(&fixture.config, &suite, &EchoPolicy).await.unwrap();
    assert_eq!(report.exit_code(), 1);
    for (case, (_, path, behavior, violation_path)) in report.cases.iter().zip(checks) {
        assert_eq!(case.parity.status, Status::Skipped);
        for side in case.implementations.values() {
            assert_eq!(side.repeatability.status, Status::Skipped);
            for attempt in &side.attempts {
                let observation = attempt.observation.as_ref().unwrap();
                assert!(observation.transport_error.is_none());
                assert_eq!(
                    observation.status,
                    Some(if behavior == "wrong_status" { 400 } else { 200 })
                );
                assert_eq!(
                    observation.headers.get("content-type").map(String::as_str),
                    match behavior {
                        "wrong_type" => Some("text/plain"),
                        "missing_type" => None,
                        _ => Some("application/json"),
                    }
                );
                assert_eq!(
                    observation
                        .violations
                        .iter()
                        .map(|violation| violation.path.as_str())
                        .collect::<Vec<_>>(),
                    [violation_path],
                    "{}",
                    case.name
                );
                assert_eq!(attempt.violations, observation.violations);
                assert!(attempt.final_json.is_none());
                let raw = fs::read(&observation.raw_body).unwrap();
                assert_eq!(raw.len().to_string(), observation.headers["content-length"]);
                if behavior == "malformed_json" {
                    assert_eq!(raw, b"{\"value\":");
                    assert!(observation.json.is_none());
                } else if path == "/sse" {
                    assert_eq!(observation.events.len(), 2);
                    assert!(raw.ends_with(b"data: [FIN]\n\n"));
                } else {
                    assert_eq!(
                        serde_json::from_slice::<Value>(&raw).unwrap(),
                        *observation.json.as_ref().unwrap()
                    );
                }
            }
        }
    }
}

#[tokio::test]
async fn malformed_truncated_and_timed_out_streams_keep_partial_artifacts() {
    let mut fixture = Fixture::new();
    fixture.config.request_timeout_secs = 1;
    let suite = suite(vec![
        case("malformed", "/sse", "malformed_stream"),
        case("truncated", "/sse", "truncate"),
        case("timeout", "/sse", "timeout"),
    ]);
    let report = run(&fixture.config, &suite, &EchoPolicy).await.unwrap();
    assert_eq!(report.exit_code(), 2);
    for case in &report.cases {
        assert_eq!(case.parity.status, Status::Skipped);
        for side in case.implementations.values() {
            for attempt in &side.attempts {
                let observation = attempt.observation.as_ref().unwrap();
                let raw = fs::read_to_string(&observation.raw_body).unwrap();
                assert!(raw.starts_with("event: snapshot\ndata: "));
                assert!(attempt.final_json.is_none());
                assert_eq!(
                    read_json(attempt.directory.join("events.json"))
                        .as_array()
                        .unwrap()
                        .len(),
                    1
                );
                match case.name.as_str() {
                    "malformed" => {
                        assert!(observation.transport_error.is_none());
                        assert!(!attempt.violations.is_empty());
                        assert!(raw.ends_with("data: unfinished"));
                    }
                    "timeout" => assert!(
                        observation
                            .transport_error
                            .as_ref()
                            .unwrap()
                            .contains("exceeded")
                    ),
                    "truncated" => assert!(observation.transport_error.is_some()),
                    _ => unreachable!(),
                }
            }
        }
    }
}

#[tokio::test]
async fn cancelling_a_library_run_cleans_the_process_tree_and_saves_partial_report() {
    let mut fixture = Fixture::new();
    fixture.config.request_timeout_secs = 30;
    fixture
        .config
        .server
        .env
        .insert("PARITY_FIXTURE_CHILD".into(), "1".into());
    let suite = suite(vec![case("pending", "/sse", "timeout")]);
    let config = fixture.config.clone();
    let task = tokio::spawn(async move { run(&config, &suite, &EchoPolicy).await });
    let received = wait_until(|| {
        if !fixture.config.output_dir.exists() {
            return false;
        }
        let directory = fixture.only_run_directory();
        fs::read(directory.join("python/pending/1/response.body"))
            .is_ok_and(|bytes| !bytes.is_empty())
    })
    .await;
    assert!(
        received,
        "fixture did not stream: lifecycle={:#?}; report={:#?}; server log={:?}",
        fixture.lifecycle(),
        read_json(fixture.only_run_directory().join("report.json")),
        fs::read_to_string(fixture.only_run_directory().join("python/server.log"))
    );
    let started = fixture
        .lifecycle()
        .into_iter()
        .find(|entry| entry["kind"] == "start")
        .unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert_process_stopped(started["pid"].as_i64().unwrap() as i32).await;
    assert_process_stopped(started["worker"].as_i64().unwrap() as i32).await;
    let directory = fixture.only_run_directory();
    let partial = read_json(directory.join("report.json"));
    assert_eq!(partial["state"], "interrupted");
    assert!(
        fs::read_to_string(directory.join("report.html"))
            .unwrap()
            .contains("State: interrupted")
    );
    assert!(!partial["runtime_errors"].as_array().unwrap().is_empty());
    assert_eq!(
        partial["cases"][0]["implementations"]["python"]["attempts"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        partial["cases"][0]["implementations"]["rust"]["attempts"],
        json!([])
    );
    assert_ne!(partial["cases"][0]["parity"]["status"], "PASS");
    assert!(
        fs::read_to_string(directory.join("python/pending/1/response.body"))
            .unwrap()
            .contains("event: snapshot")
    );
    assert_eq!(
        fixture
            .lifecycle()
            .iter()
            .filter(|entry| entry["kind"] == "start")
            .count(),
        1
    );
    let socket = TcpListener::bind(("127.0.0.1", fixture.config.server.port)).unwrap();
    drop(socket);
}

#[tokio::test]
async fn artifact_write_failure_after_startup_cleans_process_tree_and_saves_partial_report() {
    let mut fixture = Fixture::new();
    fixture
        .config
        .server
        .env
        .insert("PARITY_FIXTURE_CHILD".into(), "1".into());
    let suite = suite(vec![case("blocked", "/json", "artifact_io_failure")]);
    let error = run(&fixture.config, &suite, &EchoPolicy).await.unwrap_err();
    assert!(matches!(error, RunError::Io(_)), "{error}");

    let lifecycle = fixture.lifecycle();
    let starts: Vec<_> = lifecycle
        .iter()
        .filter(|entry| entry["kind"] == "start")
        .collect();
    assert_eq!(
        starts.len(),
        1,
        "must stop before starting the second implementation"
    );
    assert_eq!(
        lifecycle
            .iter()
            .filter(|entry| entry["kind"] == "request")
            .count(),
        1
    );
    assert_process_stopped(starts[0]["pid"].as_i64().unwrap() as i32).await;
    assert_process_stopped(starts[0]["worker"].as_i64().unwrap() as i32).await;
    let directory = fixture.only_run_directory();
    assert!(
        read_json(directory.join("python/blocked/1/response.body"))
            .get("value")
            .is_some()
    );
    assert!(directory.join("python/blocked/1/final.json").is_dir());
    let partial = read_json(directory.join("report.json"));
    assert_eq!(partial["state"], "interrupted");
    assert!(!partial["runtime_errors"].as_array().unwrap().is_empty());
    assert_eq!(
        partial["cases"][0]["implementations"]["rust"]["attempts"],
        json!([])
    );
    let socket = TcpListener::bind(("127.0.0.1", fixture.config.server.port)).unwrap();
    drop(socket);
}

#[tokio::test]
async fn invalid_configuration_and_startup_failures_cannot_be_passing_runs() {
    let mut fixture = Fixture::new();
    let suite = suite(vec![case("ordinary", "/json", "normal")]);
    fixture.config.server.model.clear();
    assert!(run(&fixture.config, &suite, &EchoPolicy).await.is_err());
    assert!(fixture.lifecycle().is_empty());
    assert!(!fixture.config.output_dir.exists());
    fixture.config.server.model = "fixture-model".into();
    fixture
        .config
        .server
        .env
        .insert("PARITY_FIXTURE_EARLY_EXIT".into(), "1".into());
    let report = run(&fixture.config, &suite, &EchoPolicy).await.unwrap();
    assert_eq!(report.exit_code(), 2);
    assert_eq!(report.runtime_errors.len(), 2);
    assert!(
        report
            .runtime_errors
            .iter()
            .all(|error| error.contains("startup"))
    );
    assert!(
        report.cases[0]
            .implementations
            .values()
            .all(|side| side.attempts.is_empty())
    );
    assert_eq!(
        read_json(report.directory.join("report.json")),
        serde_json::to_value(&report).unwrap()
    );
}

#[test]
fn describe_rejects_unsafe_or_ambiguous_configuration_without_processes() {
    let fixture = Fixture::new();
    let mut config = fixture.config.clone();
    config.server.python = Some(fixture.directory.path().join("does-not-exist"));
    let valid = suite(vec![case("valid", "/json", "normal")]);
    describe(&config, &valid).unwrap();
    for name in ["../escape", "nested/path", ""] {
        let mut invalid = valid.clone();
        invalid.cases[0].name = name.into();
        assert!(describe(&config, &invalid).is_err(), "accepted {name:?}");
    }
    let mut duplicate = valid.clone();
    duplicate.cases.push(duplicate.cases[0].clone());
    assert!(describe(&config, &duplicate).is_err());
    for path in [
        "https://example.invalid/json",
        "//example.invalid/json",
        "/json#fragment",
    ] {
        let mut invalid = valid.clone();
        invalid.cases[0].path = path.into();
        assert!(describe(&config, &invalid).is_err(), "accepted {path:?}");
    }
    let mut singleton = valid.clone();
    singleton.cases[0].equivalence_group = Some("alone".into());
    assert!(describe(&config, &singleton).is_err());
    let mut no_deadline = config.clone();
    no_deadline.request_timeout_secs = 0;
    assert!(describe(&no_deadline, &valid).is_err());
    let mut unknown = serde_json::to_value(&config).unwrap();
    unknown["hidden_comparison_override"] = json!(true);
    assert!(serde_json::from_value::<RunConfig>(unknown).is_err());
    assert!(!config.output_dir.exists());
    assert!(fixture.preparations().is_empty());
}

#[test]
fn cli_describe_uses_the_same_default_and_external_spec_without_starting_python() {
    let mut fixture = Fixture::new();
    fixture.config.server.python = Some(fixture.directory.path().join("missing-python"));
    let directory = &fixture.directory;
    let config_path = directory.path().join("run.json");
    fs::write(&config_path, serde_json::to_vec(&fixture.config).unwrap()).unwrap();
    let execute = |external: Option<&std::path::Path>| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_sglang-parity"));
        command
            .env_remove("RUST_LOG")
            .arg("--config")
            .arg(&config_path)
            .arg("--describe");
        if let Some(path) = external {
            command.arg("--suite-file").arg(path);
        }
        command.output().unwrap()
    };
    let default = execute(None);
    assert!(
        default.status.success(),
        "{}",
        String::from_utf8_lossy(&default.stderr)
    );
    let default_json: Value = serde_json::from_slice(&default.stdout).unwrap();
    assert!(
        default.stderr.is_empty(),
        "describe must not emit run progress"
    );
    assert_eq!(default_json["suite"]["cases"].as_array().unwrap().len(), 10);
    assert_eq!(default_json["repeats_per_implementation"], 2);
    let rules = default_json["suite"]["comparison"]["per_result_value_exceptions"]
        .as_array()
        .unwrap();
    assert_eq!(
        rules
            .iter()
            .map(|rule| rule["presence"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["required", "required", "optional"]
    );
    assert_eq!(rules[2]["path"], "/meta_info/response_sent_to_client_ts");
    let external_path = directory.path().join("suite.json");
    fs::write(
        &external_path,
        include_str!("../suites/native_generate/suite.json"),
    )
    .unwrap();
    let external = execute(Some(&external_path));
    assert!(external.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&external.stdout).unwrap(),
        default_json
    );
    let mut invalid = read_json(&external_path);
    invalid["comparison"]["numeric_tolerance"] = json!(0.1);
    fs::write(&external_path, serde_json::to_vec(&invalid).unwrap()).unwrap();
    let rejected = execute(Some(&external_path));
    assert_eq!(rejected.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("unknown field"));
    assert!(!fixture.config.output_dir.exists());
    assert!(fixture.preparations().is_empty());
}

#[tokio::test]
async fn dirty_and_untracked_sources_are_rejected_before_preparation() {
    for (relative, diagnostic) in [
        ("python/fixture-version.txt", "uncommitted changes"),
        ("python/untracked.py", "untracked source files"),
    ] {
        let fixture = Fixture::new();
        fs::write(fixture.source().join(relative), "local edit").unwrap();
        let error = run(
            &fixture.config,
            &suite(vec![case("ordinary", "/json", "normal")]),
            &EchoPolicy,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains(diagnostic), "{error}");
        assert!(fixture.preparations().is_empty());
        assert!(fixture.lifecycle().is_empty());
        assert!(!fixture.config.output_dir.exists());
    }
}

#[tokio::test]
async fn both_servers_keep_the_prepared_head_when_original_checkout_changes() {
    let mut fixture = Fixture::new();
    let original_commit = git(&fixture.source(), &["rev-parse", "HEAD"]);
    let release = fixture.directory.path().join("release");
    fixture.config.server.env.insert(
        "PARITY_FIXTURE_RELEASE".into(),
        release.to_string_lossy().into_owned(),
    );
    let config = fixture.config.clone();
    let suite = suite(vec![case("held", "/json", "wait_for_release")]);
    let task = tokio::spawn(async move { run(&config, &suite, &EchoPolicy).await });
    assert!(
        wait_until(|| fixture
            .lifecycle()
            .iter()
            .any(|entry| entry["kind"] == "request"))
        .await
    );
    git(
        &fixture.source(),
        &["switch", "--quiet", "-c", "changed-head"],
    );
    let marker = fixture.source().join("python/fixture-version.txt");
    fs::write(&marker, "new HEAD").unwrap();
    git(
        &fixture.source(),
        &["commit", "--quiet", "-am", "advance original"],
    );
    assert_ne!(
        git(&fixture.source(), &["rev-parse", "HEAD"]),
        original_commit
    );
    fs::write(&marker, "uncommitted after switch").unwrap();
    fs::write(fixture.source().join("python/later.py"), "untracked later").unwrap();
    fs::write(&release, "resume").unwrap();

    let report = task.await.unwrap().unwrap();
    assert_eq!(report.exit_code(), 0, "{report:#?}");
    let environment = report.environment.as_ref().unwrap();
    assert_eq!(environment["plan"]["commit"], original_commit);
    let snapshot = std::path::Path::new(environment["plan"]["source_snapshot"].as_str().unwrap());
    assert_eq!(git(snapshot, &["rev-parse", "HEAD"]), original_commit);
    assert!(git(snapshot, &["status", "--porcelain"]).is_empty());
    let starts: Vec<_> = fixture
        .lifecycle()
        .into_iter()
        .filter(|entry| entry["kind"] == "start")
        .collect();
    assert_eq!(starts.len(), 2);
    for start in starts {
        assert_eq!(start["python_path"], json!(snapshot.join("python")));
        assert_eq!(start["source_version"], "initial HEAD");
    }
    assert_eq!(
        fs::read_to_string(marker).unwrap(),
        "uncommitted after switch"
    );
}

#[tokio::test]
async fn failed_environment_probe_retains_evidence_without_starting_servers() {
    let mut fixture = Fixture::new();
    fixture
        .config
        .server
        .env
        .insert("PARITY_FIXTURE_PROBE_FAIL".into(), "1".into());
    let report = run(
        &fixture.config,
        &suite(vec![case("ordinary", "/json", "normal")]),
        &EchoPolicy,
    )
    .await
    .unwrap();
    assert_eq!(report.exit_code(), 2);
    assert_eq!(report.runtime_errors.len(), 1);
    assert!(report.runtime_errors[0].contains("environment preparation"));
    assert!(
        fs::read_to_string(report.directory.join("report.html"))
            .unwrap()
            .contains("environment preparation")
    );
    assert!(report.environment.is_none());
    assert!(fixture.lifecycle().is_empty());
    assert_eq!(fixture.preparations().len(), 1);
    assert!(
        report.cases[0]
            .implementations
            .values()
            .all(|side| side.attempts.is_empty())
    );
    assert_eq!(
        read_json(report.directory.join("environment-probe.json"))["error"],
        "deliberate fixture probe failure"
    );
    assert!(
        fs::read_to_string(report.directory.join("setup.log"))
            .unwrap()
            .contains("deliberate fixture probe failure")
    );
    assert!(report.directory.join("environment.lock").is_file());
    assert_eq!(
        read_json(report.directory.join("report.json")),
        serde_json::to_value(report).unwrap()
    );
}

#[tokio::test]
async fn cli_sigterm_cleans_managed_descendants_and_retains_an_interrupted_report() {
    let mut fixture = Fixture::new();
    fixture.config.request_timeout_secs = 30;
    fixture
        .config
        .server
        .env
        .insert("PARITY_FIXTURE_CHILD".into(), "1".into());
    let config_path = fixture.directory.path().join("cli-run.json");
    fs::write(&config_path, serde_json::to_vec(&fixture.config).unwrap()).unwrap();
    let spec_path = fixture.directory.path().join("native-suite.json");
    let mut spec: Value =
        serde_json::from_str(include_str!("../suites/native_generate/suite.json")).unwrap();
    spec["cases"] = json!([{
        "name": "pending",
        "body": {
            "text": "fixture request",
            "sampling_params": {"temperature": 0, "max_new_tokens": 8, "sampling_seed": 42},
            "stream": true,
            "behavior": "timeout",
            "value": "pending"
        },
        "expect_status": 200
    }]);
    fs::write(&spec_path, serde_json::to_vec(&spec).unwrap()).unwrap();
    let log_path = fixture.directory.path().join("cli.log");
    let log = fs::File::create(&log_path).unwrap();
    let stdout_path = fixture.directory.path().join("cli.stdout");
    let mut cli = Command::new(env!("CARGO_BIN_EXE_sglang-parity"))
        .env_remove("RUST_LOG")
        .arg("--config")
        .arg(config_path)
        .arg("--suite-file")
        .arg(spec_path)
        .stdout(Stdio::from(fs::File::create(&stdout_path).unwrap()))
        .stderr(Stdio::from(log))
        .spawn()
        .unwrap();
    let received = wait_until(|| {
        fixture
            .lifecycle()
            .iter()
            .any(|entry| entry["kind"] == "request")
    })
    .await;
    // SAFETY: this PID belongs to the live CLI child spawned immediately above.
    let signal_result = unsafe { libc::kill(cli.id() as i32, libc::SIGTERM) };
    let finished = wait_until(|| cli.try_wait().unwrap().is_some()).await;
    if !finished {
        let _ = cli.kill();
    }
    let status = cli.wait().unwrap();
    let progress = fs::read_to_string(&log_path).unwrap();
    for expected in [
        "Run artifacts and logs",
        "Environment ready",
        "Server ready",
        "case 1/1 pending, repeat 1/2",
        "server.log",
    ] {
        assert!(
            progress.contains(expected),
            "missing {expected}: {progress}"
        );
    }
    assert!(
        fs::read_to_string(stdout_path).unwrap().is_empty(),
        "progress belongs on stderr"
    );
    assert!(
        received,
        "CLI did not reach its request: {:?}; lifecycle={:#?}",
        fs::read_to_string(&log_path),
        fixture.lifecycle()
    );
    assert_eq!(signal_result, 0);
    assert!(finished, "CLI did not stop after SIGTERM");
    assert_eq!(
        status.code(),
        Some(2),
        "CLI log: {:?}",
        fs::read_to_string(&log_path)
    );
    let starts: Vec<_> = fixture
        .lifecycle()
        .into_iter()
        .filter(|entry| entry["kind"] == "start")
        .collect();
    assert_eq!(starts.len(), 1);
    assert_process_stopped(starts[0]["pid"].as_i64().unwrap() as i32).await;
    assert_process_stopped(starts[0]["worker"].as_i64().unwrap() as i32).await;
    let report = read_json(fixture.only_run_directory().join("report.json"));
    assert_eq!(report["state"], "interrupted");
    assert_eq!(
        report["cases"][0]["implementations"]["rust"]["attempts"],
        json!([])
    );
    assert_ne!(report["cases"][0]["parity"]["status"], "PASS");
}
