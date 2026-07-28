use std::net::TcpListener;
use std::process::{Command, Output, Stdio};

const PEER_BINARY: &str = env!("CARGO_BIN_EXE_pd_data_mock_peer");

#[test]
fn two_process_cpu_data_plane_copies_multi_room_bytes_and_commits_once() {
    let (prefill, decode) = run_pair("positive", 45);
    assert_success(&prefill, "prefill positive");
    assert_success(&decode, "decode positive");
    for output in [&prefill, &decode] {
        let stdout = stdout(output);
        assert!(stdout.contains("DATA_PLANE_COMPLETE"));
        assert!(stdout.contains("rooms=4"));
        assert!(stdout.contains("room_zero=true"));
        assert!(stdout.contains("orders=both"));
        assert!(stdout.contains("fragmented=true"));
        assert!(stdout.contains("capacity=true"));
    }
}

#[test]
fn two_process_safe_aux_failure_stops_later_stages_and_restores_capacity() {
    let (prefill, decode) = run_pair("safe_failure", 30);
    assert_success(&prefill, "prefill safe failure");
    assert_success(&decode, "decode safe failure");
    for output in [&prefill, &decode] {
        let stdout = stdout(output);
        assert!(stdout.contains("DATA_PLANE_FAILURE"));
        assert!(stdout.contains("phase=aux"));
        assert!(stdout.contains("later_submits=0"));
        assert!(stdout.contains("baseline=true"));
    }
}

#[test]
fn two_process_timeout_quarantines_until_late_native_terminal_then_recovers() {
    let (prefill, decode) = run_pair("timeout_recovery", 30);
    assert_success(&prefill, "prefill timeout recovery");
    assert_success(&decode, "decode timeout recovery");
    for output in [&prefill, &decode] {
        let stdout = stdout(output);
        assert!(stdout.contains("DATA_PLANE_TIMEOUT"));
        assert!(stdout.contains("quarantined=true"));
        assert!(stdout.contains("recovered=true"));
        assert!(stdout.contains("baseline=true"));
    }
}

#[test]
fn two_process_disconnect_after_native_safety_releases_without_quarantine() {
    let (prefill, decode) = run_pair("disconnect", 30);
    assert_success(&prefill, "prefill disconnect");
    assert_success(&decode, "decode disconnect");
    for output in [&prefill, &decode] {
        let stdout = stdout(output);
        assert!(stdout.contains("DATA_PLANE_DISCONNECT"));
        assert!(stdout.contains("safe_release=true"));
        assert!(stdout.contains("quarantine=0"));
    }
}

fn run_pair(scenario: &str, timeout_seconds: u64) -> (Output, Output) {
    let address = free_address();
    let prefill = Command::new("timeout")
        .arg(format!("{timeout_seconds}s"))
        .arg(PEER_BINARY)
        .arg("prefill")
        .arg(&address)
        .arg(scenario)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn prefill data peer");
    let decode = Command::new("timeout")
        .arg(format!("{timeout_seconds}s"))
        .arg(PEER_BINARY)
        .arg("decode")
        .arg(&address)
        .arg(scenario)
        .output()
        .expect("run decode data peer");
    let prefill = prefill
        .wait_with_output()
        .expect("wait for prefill data peer");
    (prefill, decode)
}

fn assert_success(output: &Output, context: &str) {
    assert!(
        output.status.success(),
        "{context} failed with {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout(output),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn free_address() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("reserve loopback port");
    listener.local_addr().expect("loopback address").to_string()
}
