//! Exercise managed setup with real Git snapshots and controlled tool processes.

use super::*;
use std::os::unix::fs::PermissionsExt;

struct Fixture {
    directory: tempfile::TempDir,
    config: RunConfig,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let source = root.join("source");
        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let lock = if cfg!(target_os = "macos") {
            "mlx.lock"
        } else {
            "cuda.lock"
        };
        for relative in [
            "python/pyproject.toml".to_owned(),
            "python/pyproject_other.toml".to_owned(),
            "rust/sglang-parity/environments/profiles.json".to_owned(),
            "rust/sglang-parity/environments/probe.py".to_owned(),
            format!("rust/sglang-parity/environments/{lock}"),
        ] {
            let destination = source.join(&relative);
            fs::create_dir_all(destination.parent().unwrap()).unwrap();
            fs::copy(repository.join(relative), destination).unwrap();
        }
        git(&source, &["init", "--quiet"]).unwrap();
        commit(&source);
        let tools = root.join("tools");
        fs::create_dir(&tools).unwrap();
        for name in ["uv", "cc", "cargo", "rustc", "sw_vers", "getconf"] {
            let executable = tools.join(name);
            fs::write(&executable, TOOL).unwrap();
            fs::set_permissions(&executable, fs::Permissions::from_mode(0o755)).unwrap();
        }
        let mut paths = vec![tools];
        paths.extend(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        ));
        let config = serde_json::from_value(json!({
            "server": {"model": "fixture", "env": {
                "PATH": std::env::join_paths(paths).unwrap().to_str().unwrap(),
                "PARITY_SETUP_FIXTURE": root,
            }},
            "environment": {"source_root": source, "cache_dir": root.join("cache"), "setup_timeout_secs": 15}
        })).unwrap();
        Self { directory, config }
    }

    fn output(&self, name: &str) -> PathBuf {
        let path = self.directory.path().join(name);
        fs::create_dir(&path).unwrap();
        path
    }

    fn mode(&self, value: &str) {
        fs::write(self.directory.path().join("mode"), value).unwrap();
    }

    fn phases(&self) -> Vec<String> {
        fs::read_to_string(self.directory.path().join("phases"))
            .unwrap_or_default()
            .lines()
            .map(str::to_owned)
            .collect()
    }

    async fn wait_sync(&self) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while fs::read_to_string(self.directory.path().join("pid"))
            .ok()
            .and_then(|pid| pid.parse::<i32>().ok())
            .is_none()
        {
            assert!(
                Instant::now() < deadline,
                "setup did not reach dependency installation"
            );
            sleep(Duration::from_millis(10)).await;
        }
    }
}

fn commit(repo: &Path) {
    git(repo, &["add", "."]).unwrap();
    git(
        repo,
        &[
            "-c",
            "user.name=Parity fixture",
            "-c",
            "user.email=parity@localhost",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "--quiet",
            "-m",
            "fixture source",
        ],
    )
    .unwrap();
}

const TOOL: &str = r#"#!/usr/bin/env python3
import json, os, pathlib, shutil, sys, time
root = pathlib.Path(os.environ['PARITY_SETUP_FIXTURE'])
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
if name in ('sw_vers', 'getconf'):
    print('14.0' if name == 'sw_vers' else 'glibc 2.31')
    sys.exit(0)
if name != 'python' and args == ['--version']:
    print('uv 0.11.14' if name == 'uv' else name + ' fixture')
    sys.exit(0)
if name == 'python' and '--library-paths' in args:
    pathlib.Path(args[-1]).write_text('[]')
    sys.exit(0)
phase = 'probe' if name == 'python' else args[0] if args[0] == 'venv' else args[1]
with (root / 'phases').open('a') as log:
    log.write(phase + '\n')
mode = (root / 'mode').read_text() if (root / 'mode').exists() else ''
if mode == 'fail-' + phase:
    print('injected ' + phase + ' failure', flush=True)
    sys.exit(23)
if mode == 'wait' and phase == 'sync':
    (root / 'pid').write_text(str(os.getpid()))
    time.sleep(120)
if phase == 'venv':
    executable = pathlib.Path(args[-1]) / 'bin' / 'python'
    executable.parent.mkdir(parents=True)
    shutil.copyfile(sys.argv[0], executable)
    executable.chmod(0o755)
elif phase == 'probe':
    source = pathlib.Path(args[args.index('--source') + 1])
    assert os.environ['PYTHONPATH'] == str(source / 'python')
    assert os.environ['SGLANG_RUST_BUILD_MODE'] == 'auto'
    output = pathlib.Path(args[args.index('--output') + 1])
    arguments = dict(zip(args[1::2], args[2::2]))
    result = {
        'status': 'passed', 'source': str(source.resolve()), 'dependencies_only': False,
        'lock': str(pathlib.Path(arguments['--lock']).resolve()),
        'python_version': arguments['--python-version'],
        'python_executable': arguments['--python'], 'backend': {'name': arguments['--backend']},
    }
    if mode == 'wrong-probe':
        result['source'] = '/stale-checkout'
    output.write_text(json.dumps(result))
"#;

#[test]
fn cache_identity_tracks_source_and_build_configuration() {
    let fixture = Fixture::new();
    let original = describe(&fixture.config).unwrap();
    let mut changed = fixture.config.clone();
    changed.server.seed += 1;
    changed.server.model = "different-model".into();
    assert_eq!(
        describe(&changed).unwrap().environment_dir,
        original.environment_dir
    );
    changed
        .server
        .env
        .insert("RUSTFLAGS".into(), "-C opt-level=1".into());
    assert_ne!(
        describe(&changed).unwrap().environment_dir,
        original.environment_dir
    );
    fs::write(
        original.source_root.join("python/version.txt"),
        "next version",
    )
    .unwrap();
    commit(&original.source_root);
    let next = describe(&fixture.config).unwrap();
    assert_ne!(next.environment_dir, original.environment_dir);
    assert_ne!(next.source_snapshot, original.source_snapshot);
    assert!(
        !original.cache_dir.exists(),
        "describing created installation state"
    );
}

#[tokio::test]
async fn cached_environments_are_reverified_and_incomplete_installations_are_rebuilt() {
    let fixture = Fixture::new();
    let plan = describe(&fixture.config).unwrap();
    let ready = plan.environment_dir.join("parity-ready.json");
    let first = prepare(&fixture.config, &plan, &fixture.output("first"))
        .await
        .unwrap();
    assert_eq!(first.record["reused"], false);
    assert_eq!(first.server.python.as_ref(), Some(&plan.python));
    assert_eq!(fixture.phases(), ["venv", "sync", "install", "probe"]);
    drop(first);
    let marker = fs::read(&ready).unwrap();
    let reused = prepare(&fixture.config, &plan, &fixture.output("reused"))
        .await
        .unwrap();
    assert_eq!(reused.record["reused"], true);
    assert_eq!(
        fixture.phases(),
        ["venv", "sync", "install", "probe", "probe"]
    );
    assert_eq!(fs::read(&ready).unwrap(), marker);
    drop(reused);

    fixture.mode("wrong-probe");
    assert!(
        prepare(&fixture.config, &plan, &fixture.output("wrong-probe"))
            .await
            .err()
            .unwrap()
            .contains("/source")
    );
    fixture.mode("fail-probe");
    let output = fixture.output("invalid-dependencies");
    assert!(prepare(&fixture.config, &plan, &output).await.is_err());
    assert!(
        fs::read_to_string(output.join("setup.log"))
            .unwrap()
            .contains("injected probe failure")
    );
    assert_eq!(
        fixture
            .phases()
            .iter()
            .filter(|phase| *phase == "sync")
            .count(),
        1
    );
    assert_eq!(fs::read(&ready).unwrap(), marker);

    fixture.mode("");
    fs::remove_file(&ready).unwrap();
    let partial = plan.environment_dir.join("partial-installation");
    fs::write(&partial, "incomplete").unwrap();
    let rebuilt = prepare(&fixture.config, &plan, &fixture.output("rebuilt"))
        .await
        .unwrap();
    assert_eq!(rebuilt.record["reused"], false);
    assert!(!partial.exists());
    assert_eq!(
        fixture
            .phases()
            .iter()
            .filter(|phase| *phase == "sync")
            .count(),
        2
    );
}

#[tokio::test]
async fn invalid_completion_markers_fail_without_reinstalling() {
    let fixture = Fixture::new();
    let plan = describe(&fixture.config).unwrap();
    drop(
        prepare(&fixture.config, &plan, &fixture.output("first"))
            .await
            .unwrap(),
    );
    let ready = plan.environment_dir.join("parity-ready.json");
    let mut wrong_commit: Value = serde_json::from_slice(&fs::read(&ready).unwrap()).unwrap();
    wrong_commit["plan"]["commit"] = json!("another commit");
    for (name, content) in [
        ("malformed", "{".to_owned()),
        ("mismatched", wrong_commit.to_string()),
    ] {
        fs::write(&ready, content).unwrap();
        let error = prepare(&fixture.config, &plan, &fixture.output(name))
            .await
            .err()
            .unwrap();
        assert!(error.contains("completion marker"), "{error}");
    }
    assert_eq!(fixture.phases(), ["venv", "sync", "install", "probe"]);
}

#[tokio::test]
async fn a_prepared_environment_holds_its_lease_until_dropped() {
    let fixture = Fixture::new();
    let plan = describe(&fixture.config).unwrap();
    let first = prepare(&fixture.config, &plan, &fixture.output("first"))
        .await
        .unwrap();
    let config = fixture.config.clone();
    let output = fixture.output("second");
    let second_output = output.clone();
    let mut second = tokio::spawn(async move { prepare(&config, &plan, &output).await });
    let deadline = Instant::now() + Duration::from_secs(10);
    while !second_output.join("uv-version.log").is_file() {
        assert!(
            Instant::now() < deadline,
            "second setup did not finish preflight"
        );
        sleep(Duration::from_millis(10)).await;
    }
    assert!(
        tokio::time::timeout(Duration::from_millis(500), &mut second)
            .await
            .is_err()
    );
    assert_eq!(fixture.phases(), ["venv", "sync", "install", "probe"]);
    drop(first);
    let second = tokio::time::timeout(Duration::from_secs(10), second)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(second.record["reused"], true);
    assert_eq!(
        fixture.phases(),
        ["venv", "sync", "install", "probe", "probe"]
    );
}

#[tokio::test]
async fn setup_failure_timeout_and_cancellation_never_complete_an_environment() {
    for (mode, expected) in [("fail-sync", "exited with"), ("wait", "timed out")] {
        let mut fixture = Fixture::new();
        if mode == "wait" {
            fixture.config.environment.setup_timeout_secs = 5;
        }
        fixture.mode(mode);
        let plan = describe(&fixture.config).unwrap();
        let output = fixture.output("failed");
        let error = prepare(&fixture.config, &plan, &output)
            .await
            .err()
            .unwrap();
        assert!(
            error.contains(expected) && error.contains("setup.log"),
            "{error}"
        );
        assert!(!plan.environment_dir.join("parity-ready.json").exists());
        assert!(fixture.phases().iter().any(|phase| phase == "sync"));
        assert!(output.join("setup.log").is_file());
        assert!(output.join("environment.lock").is_file());
        assert!(!output.join("environment.json").exists());
    }
    let fixture = Fixture::new();
    fixture.mode("wait");
    let plan = describe(&fixture.config).unwrap();
    let ready = plan.environment_dir.join("parity-ready.json");
    let config = fixture.config.clone();
    let output = fixture.output("cancelled");
    let task = tokio::spawn(async move { prepare(&config, &plan, &output).await });
    fixture.wait_sync().await;
    task.abort();
    assert!(task.await.err().unwrap().is_cancelled());
    assert!(!ready.exists());
    let pid: i32 = fs::read_to_string(fixture.directory.path().join("pid"))
        .unwrap()
        .parse()
        .unwrap();
    // SAFETY: signal zero only observes the fixture's already reaped direct child.
    assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH)
    );
}
