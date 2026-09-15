//! Derive backend dependencies from repository metadata and maintain hashed locks.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::process::run_command;

const DIRECTORY: &str = "rust/sglang-parity/environments";

/// Supported dependency profiles; automatic selection uses the host platform.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    #[default]
    Auto,
    Mlx,
    Cuda,
}

impl Backend {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Mlx => "mlx",
            Self::Cuda => "cuda",
        }
    }
}

/// Pinned interpreter, resolver and wheel compatibility contract for one backend.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub backend: Backend,
    pub python: String,
    pub uv: String,
    pub platform: String,
    pub macos_deployment_target: Option<String>,
    pub indexes: Vec<String>,
    pub torch_backend: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct LockSpec {
    pub path: PathBuf,
    pub sha256: String,
    pub input_digest: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Profiles {
    python: String,
    uv: String,
    profiles: BTreeMap<String, Platform>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Platform {
    platform: String,
    macos_deployment_target: Option<String>,
    indexes: Vec<String>,
    torch_backend: Option<String>,
}

pub(crate) fn load_profile(repo: &Path, backend: Backend) -> Result<Profile, String> {
    let backend = match backend {
        Backend::Auto => match (std::env::consts::OS, std::env::consts::ARCH) {
            ("macos", "aarch64") => Backend::Mlx,
            ("linux", "x86_64") => Backend::Cuda,
            host => return Err(format!("no automatic environment profile for {host:?}")),
        },
        backend => backend,
    };
    let path = repo.join(DIRECTORY).join("profiles.json");
    let mut profiles: Profiles = serde_json::from_slice(&read(&path)?)
        .map_err(|error| format!("invalid {}: {error}", path.display()))?;
    let platform = profiles
        .profiles
        .remove(backend.name())
        .ok_or_else(|| format!("missing {} environment profile", backend.name()))?;
    for version in [&profiles.python, &profiles.uv] {
        if version.split('.').count() != 3
            || version
                .split('.')
                .any(|part| part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()))
        {
            return Err(format!(
                "profile requires an exact three-part version: {version:?}"
            ));
        }
    }
    let expected = match backend {
        Backend::Mlx => ("aarch64-apple-darwin", Some("14.0"), None),
        Backend::Cuda => ("x86_64-manylinux_2_31", None, Some("cu130")),
        Backend::Auto => unreachable!(),
    };
    if (
        platform.platform.as_str(),
        platform.macos_deployment_target.as_deref(),
        platform.torch_backend.as_deref(),
    ) != expected
    {
        return Err(format!(
            "unsupported platform/backend in {} profile",
            backend.name()
        ));
    }
    let expected_links: &[&str] = match backend {
        Backend::Mlx => &[],
        Backend::Cuda => &["https://pypi.nvidia.com/"],
        Backend::Auto => unreachable!(),
    };
    if platform.indexes != expected_links {
        return Err(format!(
            "unsupported wheel sources in {} profile",
            backend.name()
        ));
    }
    Ok(Profile {
        backend,
        python: profiles.python,
        uv: profiles.uv,
        platform: platform.platform,
        macos_deployment_target: platform.macos_deployment_target,
        indexes: platform.indexes,
        torch_backend: platform.torch_backend,
    })
}

#[derive(Deserialize)]
struct Metadata {
    project: Project,
    #[serde(rename = "build-system")]
    build: BuildSystem,
}

#[derive(Deserialize)]
struct Project {
    name: String,
    dependencies: Vec<String>,
    #[serde(default, rename = "optional-dependencies")]
    extras: BTreeMap<String, Vec<String>>,
}

#[derive(Deserialize)]
struct BuildSystem {
    requires: Vec<String>,
}

fn metadata(path: &Path) -> Result<Metadata, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    toml::from_str(&text).map_err(|error| format!("invalid {}: {error}", path.display()))
}

fn package_name(requirement: &str) -> String {
    requirement
        .trim_start()
        .chars()
        .take_while(|character| character.is_ascii_alphanumeric() || "-_.".contains(*character))
        .map(|character| match character {
            '_' | '.' => '-',
            character => character.to_ascii_lowercase(),
        })
        .collect()
}

fn with_marker(requirement: &str, marker: Option<&str>) -> String {
    match marker {
        None => requirement.to_owned(),
        Some(marker) => match requirement.split_once(';') {
            Some((body, own)) => format!("{body}; ({}) and ({marker})", own.trim()),
            None => format!("{requirement}; ({marker})"),
        },
    }
}

fn expand(
    project: &Project,
    requirement: &str,
    inherited_marker: Option<&str>,
    stack: &mut Vec<String>,
    output: &mut BTreeSet<String>,
) -> Result<(), String> {
    if package_name(requirement) != package_name(&project.name) {
        output.insert(with_marker(requirement, inherited_marker));
        return Ok(());
    }
    let requirement = with_marker(requirement, inherited_marker);
    let (body, marker) = match requirement.split_once(';') {
        Some((body, marker)) => (body.trim(), Some(marker.trim())),
        None => (requirement.trim(), None),
    };
    let Some((name, extras)) = body.split_once('[') else {
        return Err(format!(
            "local requirement must select extras: {requirement:?}"
        ));
    };
    let Some(extras) = extras.strip_suffix(']') else {
        return Err(format!("unsupported local requirement: {requirement:?}"));
    };
    if name.trim() != project.name {
        return Err(format!("unsupported local requirement: {requirement:?}"));
    }
    for extra in extras.split(',').map(str::trim) {
        if stack.iter().any(|entry| entry == extra) {
            return Err(format!(
                "cyclic local extra: {} -> {extra}",
                stack.join(" -> ")
            ));
        }
        let dependencies = project
            .extras
            .get(extra)
            .ok_or_else(|| format!("missing local extra {extra:?}"))?;
        stack.push(extra.to_owned());
        for dependency in dependencies {
            expand(project, dependency, marker, stack, output)?;
        }
        stack.pop();
    }
    Ok(())
}

pub(crate) fn resolve_requirements(repo: &Path, profile: &Profile) -> Result<Vec<String>, String> {
    let default = metadata(&repo.join("python/pyproject.toml"))?;
    let other;
    let project = match profile.backend {
        Backend::Mlx => {
            other = metadata(&repo.join("python/pyproject_other.toml"))?;
            &other.project
        }
        Backend::Cuda => &default.project,
        Backend::Auto => {
            return Err("resolve the automatic backend before deriving dependencies".into());
        }
    };
    if project.name != "sglang" || default.project.name != "sglang" {
        return Err("dependency metadata must describe the local sglang project".into());
    }
    let mut output = BTreeSet::new();
    for requirement in &project.dependencies {
        expand(project, requirement, None, &mut Vec::new(), &mut output)?;
    }
    if profile.backend == Backend::Mlx {
        expand(
            project,
            "sglang[srt_mps]",
            None,
            &mut Vec::new(),
            &mut output,
        )?;
        let constraints: Vec<_> = default
            .project
            .dependencies
            .iter()
            .filter(|requirement| package_name(requirement) == "tokenizers")
            .collect();
        if constraints.len() != 1 {
            return Err("expected one tokenizers constraint in python/pyproject.toml".into());
        }
        output.insert(constraints[0].clone());
    }
    for requirement in default.build.requires {
        expand(
            &default.project,
            &requirement,
            None,
            &mut Vec::new(),
            &mut output,
        )?;
    }
    Ok(output.into_iter().collect())
}

pub(crate) fn inputs_digest(repo: &Path, profile: &Profile) -> Result<String, String> {
    let value = serde_json::json!({
        "extractor_version": 1,
        "profile": profile,
        "requirements": resolve_requirements(repo, profile)?,
    });
    Ok(sha256(
        &serde_json::to_vec(&value).map_err(|error| error.to_string())?,
    ))
}

fn header(profile: &Profile, input_digest: &str) -> String {
    format!(
        "# SGLang parity dependency lock v1\n# inputs-sha256: {input_digest}\n# python: {}\n# uv: {}\n# platform: {}\n# macos-deployment-target: {}\n# torch-backend: {}\n# additional-indexes: {}\n\n",
        profile.python,
        profile.uv,
        profile.platform,
        profile.macos_deployment_target.as_deref().unwrap_or("none"),
        profile.torch_backend.as_deref().unwrap_or("none"),
        if profile.indexes.is_empty() {
            "none".into()
        } else {
            profile.indexes.join(", ")
        },
    )
}

fn validate_packages(contents: &str) -> Result<(), String> {
    let mut packages = BTreeSet::new();
    let mut current = None;
    let mut has_hash = false;
    for line in contents.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(hash) = line.strip_prefix("--hash=sha256:") {
            let hash = hash.trim_end_matches('\\').trim_end();
            if current.is_none()
                || hash.len() != 64
                || !hash.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                return Err("invalid dependency SHA-256 hash in lock".into());
            }
            has_hash = true;
            continue;
        }
        if current.is_some() && !has_hash {
            return Err("dependency has no distribution hashes in lock".into());
        }
        let Some((name, version)) = line.split_once("==") else {
            return Err(format!("dependency is not exactly pinned: {line:?}"));
        };
        if name.is_empty()
            || name != package_name(name)
            || name == "sglang"
            || version.trim().is_empty()
            || version.starts_with('=')
            || version.contains('*')
            || !packages.insert(name)
        {
            return Err(format!("invalid or duplicate dependency pin: {line:?}"));
        }
        current = Some(name);
        has_hash = false;
    }
    if current.is_none() || !has_hash {
        return Err("lock must contain pinned dependencies with distribution hashes".into());
    }
    Ok(())
}

pub(crate) fn inspect_lock(repo: &Path, profile: &Profile) -> Result<LockSpec, String> {
    let path = repo
        .join(DIRECTORY)
        .join(format!("{}.lock", profile.backend.name()));
    let bytes = read(&path)?;
    let input_digest = inputs_digest(repo, profile)?;
    let text = std::str::from_utf8(&bytes).map_err(|error| error.to_string())?;
    let expected = header(profile, &input_digest);
    let packages = text.strip_prefix(&expected).ok_or_else(|| {
        format!(
            "{} is stale; run sglang-parity --update-env-lock --backend {}",
            path.display(),
            profile.backend.name()
        )
    })?;
    validate_packages(packages)?;
    Ok(LockSpec {
        path,
        sha256: sha256(&bytes),
        input_digest,
    })
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read(path: &Path) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))
}

struct TemporaryDirectory(PathBuf);

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn uv_command(directory: &Path) -> Command {
    let mut command = Command::new("uv");
    command.current_dir(directory).arg("--no-config");
    // Resolver overrides must not silently change the committed profile.
    for (name, _) in std::env::vars_os() {
        if name.to_str().is_some_and(|name| {
            name.starts_with("UV_")
                || name.starts_with("PIP_")
                || name.starts_with("PYTHON")
                || matches!(name, "VIRTUAL_ENV" | "CONDA_PREFIX")
        }) {
            command.env_remove(name);
        }
    }
    command
}

pub(crate) async fn update_lock(
    repo: &Path,
    backend: Backend,
    log: &Path,
    timeout: Duration,
) -> Result<PathBuf, String> {
    let deadline = tokio::time::Instant::now()
        .checked_add(timeout)
        .ok_or("lock update timeout is too large")?;
    let profile = load_profile(repo, backend)?;
    let input_digest = inputs_digest(repo, &profile)?;
    let directory = repo.join(DIRECTORY);
    let staging = TemporaryDirectory(directory.join(format!(".lock-{}", uuid::Uuid::new_v4())));
    fs::create_dir(&staging.0).map_err(|error| error.to_string())?;
    let version_log = staging.0.join("uv-version.log");
    run_command(
        uv_command(&staging.0).arg("--version"),
        &version_log,
        timeout,
    )
    .await?;
    let version = fs::read_to_string(&version_log).map_err(|error| error.to_string())?;
    if version.split_whitespace().take(2).collect::<Vec<_>>() != ["uv", profile.uv.as_str()] {
        return Err(format!(
            "expected uv {}, found {}",
            profile.uv,
            version.trim()
        ));
    }
    let requirements = staging.0.join("requirements.in");
    fs::write(
        &requirements,
        resolve_requirements(repo, &profile)?.join("\n") + "\n",
    )
    .map_err(|error| error.to_string())?;
    let output = staging.0.join("requirements.txt");
    let mut command = uv_command(&staging.0);
    command
        .args([
            "pip",
            "compile",
            "--generate-hashes",
            "--only-binary",
            ":all:",
            "--python-version",
            &profile.python,
            "--python-platform",
            &profile.platform,
            "--no-header",
            "--no-annotate",
            "--default-index",
            "https://pypi.org/simple",
            "--index-strategy",
            "first-index",
            "--output-file",
        ])
        .arg(&output)
        .arg(&requirements);
    if let Some(backend) = &profile.torch_backend {
        command.args(["--torch-backend", backend]);
    }
    for link in &profile.indexes {
        command.args(["--index", link]);
    }
    command.env_remove("MACOSX_DEPLOYMENT_TARGET");
    if let Some(target) = &profile.macos_deployment_target {
        command.env("MACOSX_DEPLOYMENT_TARGET", target);
    }
    run_command(
        &mut command,
        log,
        deadline.saturating_duration_since(tokio::time::Instant::now()),
    )
    .await?;
    let packages = fs::read_to_string(&output).map_err(|error| error.to_string())?;
    validate_packages(&packages)?;
    if inputs_digest(repo, &load_profile(repo, profile.backend)?)? != input_digest {
        return Err("dependency inputs changed while resolving the lock; retry update-lock".into());
    }
    fs::write(&output, header(&profile, &input_digest) + &packages)
        .map_err(|error| error.to_string())?;
    let destination = directory.join(format!("{}.lock", profile.backend.name()));
    fs::rename(output, &destination).map_err(|error| error.to_string())?;
    Ok(destination)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(extras: BTreeMap<String, Vec<String>>) -> Project {
        Project {
            name: "sglang".into(),
            dependencies: Vec::new(),
            extras,
        }
    }

    #[test]
    fn local_extras_expand_recursively_without_dropping_constraints_or_markers() {
        let project = project(BTreeMap::from([
            (
                "base".into(),
                vec!["package[extra]>=1; python_version >= '3.11'".into()],
            ),
            (
                "runtime".into(),
                vec!["sglang[base]".into(), "package<3".into()],
            ),
        ]));
        let mut output = BTreeSet::new();
        expand(
            &project,
            "sglang[runtime]; sys_platform == 'darwin'",
            None,
            &mut Vec::new(),
            &mut output,
        )
        .unwrap();
        assert_eq!(
            output,
            BTreeSet::from([
                "package<3; (sys_platform == 'darwin')".into(),
                "package[extra]>=1; (python_version >= '3.11') and ((sys_platform == 'darwin'))"
                    .into(),
            ])
        );
    }

    #[test]
    fn invalid_local_extras_and_unhashed_locks_fail_closed() {
        let project = project(BTreeMap::from([(
            "cycle".into(),
            vec!["sglang[cycle]".into()],
        )]));
        for requirement in [
            "sglang[missing]",
            "sglang[cycle]",
            "sglang[cycle]>=1",
            "sglang",
        ] {
            assert!(
                expand(
                    &project,
                    requirement,
                    None,
                    &mut Vec::new(),
                    &mut BTreeSet::new()
                )
                .is_err()
            );
        }
        let hash = "a".repeat(64);
        let good = format!("package==1.2.3 \\\n    --hash=sha256:{hash}\n");
        validate_packages(&good).unwrap();
        for contents in [
            String::new(),
            "package>=1".into(),
            "package==1".into(),
            good.replace("package", "sglang"),
            good.repeat(2),
            good.replace(&hash, "bad"),
        ] {
            assert!(validate_packages(&contents).is_err(), "{contents}");
        }
    }

    #[test]
    fn dependency_changes_invalidate_locks_but_unrelated_metadata_does_not() {
        let directory = tempfile::tempdir().unwrap();
        let repo = directory.path();
        fs::create_dir_all(repo.join("python")).unwrap();
        fs::create_dir_all(repo.join(DIRECTORY)).unwrap();
        fs::write(
            repo.join(DIRECTORY).join("profiles.json"),
            include_str!("../../environments/profiles.json"),
        )
        .unwrap();
        let default = r#"
[build-system]
requires = ["build==1", "torch==2"]
[project]
name = "sglang"
dependencies = ["cuda==1", "tokenizers==2", "torch==2", "torchaudio==1"]
"#;
        fs::write(repo.join("python/pyproject.toml"), default).unwrap();
        fs::write(
            repo.join("python/pyproject_other.toml"),
            r#"
[build-system]
requires = ["unused-build"]
[project]
name = "sglang"
dependencies = ["base==1"]
[project.optional-dependencies]
srt_mps = ["sglang[runtime]", "mlx>=1"]
runtime = ["torch==2", "torchaudio==1"]
"#,
        )
        .unwrap();
        let mlx = load_profile(repo, Backend::Mlx).unwrap();
        assert_eq!(
            resolve_requirements(repo, &mlx).unwrap(),
            [
                "base==1",
                "build==1",
                "mlx>=1",
                "tokenizers==2",
                "torch==2",
                "torchaudio==1"
            ]
        );
        let cuda = load_profile(repo, Backend::Cuda).unwrap();
        assert_eq!(
            resolve_requirements(repo, &cuda).unwrap(),
            [
                "build==1",
                "cuda==1",
                "tokenizers==2",
                "torch==2",
                "torchaudio==1"
            ]
        );
        let digest = inputs_digest(repo, &mlx).unwrap();
        let lock = repo.join(DIRECTORY).join("mlx.lock");
        fs::write(
            &lock,
            header(&mlx, &digest) + &format!("base==1 \\\n    --hash=sha256:{}\n", "a".repeat(64)),
        )
        .unwrap();
        let inspected = inspect_lock(repo, &mlx).unwrap();
        assert_eq!(inspected.sha256, sha256(&fs::read(&lock).unwrap()));
        fs::write(
            repo.join("python/pyproject.toml"),
            format!("{default}\ndescription = 'unrelated metadata'\n# comment\n"),
        )
        .unwrap();
        assert_eq!(inputs_digest(repo, &mlx).unwrap(), digest);
        inspect_lock(repo, &mlx).unwrap();
        fs::write(
            repo.join("python/pyproject.toml"),
            default.replace("tokenizers==2", "tokenizers==3"),
        )
        .unwrap();
        assert!(inspect_lock(repo, &mlx).unwrap_err().contains("stale"));
    }

    #[test]
    fn committed_locks_match_the_declared_dependency_profiles() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for backend in [Backend::Mlx, Backend::Cuda] {
            let profile = load_profile(&repo, backend).unwrap();
            let spec = inspect_lock(&repo, &profile).unwrap();
            assert_eq!(spec.input_digest, inputs_digest(&repo, &profile).unwrap());
        }
    }
}
