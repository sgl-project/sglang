use std::fs::{self, OpenOptions};
use std::io::Write;
use std::net::TcpListener;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const PEER_BINARY: &str = env!("CARGO_BIN_EXE_pd_control_mock_peer");

#[test]
fn two_independent_processes_complete_pair_ready_and_multi_room_suite() {
    let temp = temp_directory("positive");
    let psk = create_psk(&temp, "control.psk", 7);
    let address = free_address();
    let prefill = spawn_peer("prefill", &address, &psk, "positive", 30);
    let decode = run_peer("decode", &address, &psk, "positive", 30);
    let prefill = prefill.wait_with_output().expect("wait for prefill peer");

    assert_success(&decode, "decode positive");
    assert_success(&prefill, "prefill positive");
    assert!(stdout(&decode).contains("PAIR_READY role=decode rooms=8"));
    assert!(stdout(&prefill).contains("PAIR_READY role=prefill rooms=8"));
    fs::remove_dir_all(temp).expect("remove process harness temp directory");
}

#[test]
fn two_process_wrong_psk_fails_before_pair_ready() {
    let temp = temp_directory("wrong-psk");
    let server_psk = create_psk(&temp, "server.psk", 7);
    let client_psk = create_psk(&temp, "client.psk", 8);
    let address = free_address();
    let prefill = spawn_peer("prefill", &address, &server_psk, "auth_failure", 30);
    let decode = run_peer("decode", &address, &client_psk, "auth_failure", 30);
    let prefill = prefill.wait_with_output().expect("wait for prefill peer");

    assert_success(&decode, "decode auth failure");
    assert_success(&prefill, "prefill auth failure");
    assert!(stdout(&decode).contains("AUTH_REJECTED role=decode"));
    assert!(stdout(&prefill).contains("AUTH_REJECTED role=prefill"));
    assert!(!stdout(&decode).contains("PAIR_READY"));
    assert!(!stdout(&prefill).contains("PAIR_READY"));
    fs::remove_dir_all(temp).expect("remove process harness temp directory");
}

#[test]
fn disconnect_then_new_decode_process_performs_a_full_reconnect() {
    let temp = temp_directory("reconnect");
    let psk = create_psk(&temp, "control.psk", 7);
    let address = free_address();
    let prefill = spawn_peer("prefill", &address, &psk, "reconnect", 45);

    let first_decode = run_peer("decode", &address, &psk, "disconnect", 30);
    assert_success(&first_decode, "first decode disconnect");
    assert!(stdout(&first_decode).contains("DISCONNECTED role=decode ready=true"));

    let second_decode = run_peer("decode", &address, &psk, "positive", 30);
    assert_success(&second_decode, "second decode reconnect");
    assert!(stdout(&second_decode).contains("PAIR_READY role=decode rooms=8"));

    let prefill = prefill
        .wait_with_output()
        .expect("wait for reconnecting prefill");
    assert_success(&prefill, "prefill reconnect");
    assert!(stdout(&prefill).contains("PEER_LOST role=prefill ready=false"));
    assert!(stdout(&prefill).contains("RECONNECTED role=prefill sessions=2"));
    fs::remove_dir_all(temp).expect("remove process harness temp directory");
}

fn spawn_peer(
    role: &str,
    address: &str,
    psk: &Path,
    scenario: &str,
    timeout_seconds: u64,
) -> std::process::Child {
    Command::new("timeout")
        .arg(format!("{timeout_seconds}s"))
        .arg(PEER_BINARY)
        .arg(role)
        .arg(address)
        .arg(psk)
        .arg(scenario)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn PD mock peer")
}

fn run_peer(role: &str, address: &str, psk: &Path, scenario: &str, timeout_seconds: u64) -> Output {
    Command::new("timeout")
        .arg(format!("{timeout_seconds}s"))
        .arg(PEER_BINARY)
        .arg(role)
        .arg(address)
        .arg(psk)
        .arg(scenario)
        .output()
        .expect("run PD mock peer")
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

fn create_psk(directory: &Path, name: &str, byte: u8) -> PathBuf {
    let path = directory.join(name);
    let mut options = OpenOptions::new();
    options.write(true).create_new(true).mode(0o400);
    let mut file = options.open(&path).expect("create process PSK");
    file.write_all(&[byte; 32]).expect("write process PSK");
    drop(file);
    fs::set_permissions(&path, fs::Permissions::from_mode(0o400)).expect("set process PSK mode");
    path
}

fn temp_directory(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "sglang-pd-control-{label}-{}",
        uuid::Uuid::new_v4()
    ));
    fs::create_dir(&path).expect("create process harness temp directory");
    path
}
