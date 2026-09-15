//! Public-library integration tests with managed HTTP processes and real artifacts.

mod support;

use std::fs;
use std::net::TcpListener;
use std::process::{Command, Stdio};

use serde_json::{Value, json};
use sglang_parity::runner::Status;
use sglang_parity::{RunConfig, describe, run};

use support::{EchoPolicy, Fixture, assert_process_stopped, case, read_json, suite, wait_until};

#[tokio::test]
async fn managed_json_and_sse_run_matches_describe_requests_and_artifacts() {
    let fixture = Fixture::new();
    let mut unary = case("unary", "/json", "normal");
    let mut streaming = case("streaming", "/sse", "normal");
    unary["equivalence_group"] = json!("same_request");
    streaming["equivalence_group"] = json!("same_request");
    let suite = suite(vec![unary, streaming]);
    let effective = serde_json::to_value(describe(&fixture.config, &suite).unwrap()).unwrap();
    assert!(fixture.lifecycle().is_empty());
    assert!(!fixture.config.output_dir.exists());

    let report = run(&fixture.config, &suite, &EchoPolicy).await.unwrap();
    assert_eq!(report.exit_code(), 0);
    assert_eq!(report.state, "complete");
    assert_eq!(read_json(&report.effective_suite), effective);
    assert_eq!(
        read_json(report.directory.join("report.json")),
        serde_json::to_value(&report).unwrap()
    );
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
                    assert_eq!(
                        read_json(attempt.directory.join("events.json"))[1]["data"],
                        "[FIN]"
                    );
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
    ]);
    let report = run(&fixture.config, &suite, &EchoPolicy).await.unwrap();
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
    }
    assert_eq!(report.cases[1].parity.status, Status::Skipped);
    assert_eq!(report.cases[2].parity.status, Status::Skipped);
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
    let directory = tempfile::tempdir().unwrap();
    let config: RunConfig = serde_json::from_value(json!({
        "server": {"python": directory.path().join("does-not-exist"), "model": "fixture"},
        "output_dir": directory.path().join("output")
    }))
    .unwrap();
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
}

#[test]
fn cli_describe_uses_the_same_default_and_external_spec_without_starting_python() {
    let directory = tempfile::tempdir().unwrap();
    let config_path = directory.path().join("run.json");
    let output_dir = directory.path().join("output");
    fs::write(
        &config_path,
        serde_json::to_vec(&json!({
            "server": {"python": directory.path().join("missing-python"), "model": "fixture"},
            "output_dir": output_dir
        }))
        .unwrap(),
    )
    .unwrap();
    let execute = |external: Option<&std::path::Path>| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_sglang-parity"));
        command.arg("--config").arg(&config_path).arg("--describe");
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
    assert_eq!(default_json["suite"]["cases"].as_array().unwrap().len(), 10);
    assert_eq!(default_json["repeats_per_implementation"], 2);
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
    assert!(!output_dir.exists());
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
    let mut cli = Command::new(env!("CARGO_BIN_EXE_sglang-parity"))
        .arg("--config")
        .arg(config_path)
        .arg("--suite-file")
        .arg(spec_path)
        .stdout(Stdio::from(log.try_clone().unwrap()))
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
