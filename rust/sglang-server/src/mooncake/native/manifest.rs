use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::mooncake::EngineError;

pub const VERSION: &str = "0.3.11.post1";
pub const COMMIT: &str = "e9c61075720039bcfc5fffd19f847608402be3d0";
pub const HEADER_SHA256: &str = "1c128925bc63839fca0fce3cfacd84a400f10a7891bdf9fa86840261ee6e299d";

pub const REQUIRED_SYMBOLS: &[&str] = &[
    "allocateBatchID",
    "closeSegment",
    "createTransferEngine",
    "destroyTransferEngine",
    "freeBatchID",
    "getLocalIpAndPort",
    "getTransferStatus",
    "installTransport",
    "openSegment",
    "registerLocalMemory",
    "submitTransfer",
    "uninstallTransport",
    "unregisterLocalMemory",
];

const REQUIRED_FLAGS: &[(&str, &str)] = &[
    ("BUILD_BENCHMARK", "OFF"),
    ("BUILD_EXAMPLES", "OFF"),
    ("BUILD_SHARED_LIBS", "ON"),
    ("BUILD_UNIT_TESTS", "OFF"),
    ("ENABLE_MULTI_PROTOCOL", "OFF"),
    ("USE_3FS", "OFF"),
    ("USE_BAREX", "OFF"),
    ("USE_CUDA", "ON"),
    ("USE_CXL", "OFF"),
    ("USE_EFA", "OFF"),
    ("USE_ETCD", "OFF"),
    ("USE_HTTP", "OFF"),
    ("USE_INTRA_NVLINK", "OFF"),
    ("USE_MLX5DV", "OFF"),
    ("USE_MNNVL", "OFF"),
    ("USE_NVMEOF", "OFF"),
    ("USE_REDIS", "OFF"),
    ("USE_TCP", "OFF"),
    ("USE_TENT", "OFF"),
    ("USE_UB", "OFF"),
    ("WITH_EP", "OFF"),
    ("WITH_METRICS", "OFF"),
    ("WITH_P2P_STORE", "OFF"),
    ("WITH_STORE", "OFF"),
    ("WITH_STORE_GO", "OFF"),
    ("WITH_STORE_RUST", "OFF"),
    ("WITH_TE", "ON"),
];

#[derive(Debug)]
pub struct ValidatedArtifact {
    pub library_path: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactManifest {
    schema_version: u32,
    version: String,
    commit: String,
    header_sha256: String,
    library_sha256: String,
    bundled_libraries: Vec<BundledLibrary>,
    license: LicenseIdentity,
    build: BuildIdentity,
    abi: AbiIdentity,
    required_symbols: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BundledLibrary {
    name: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LicenseIdentity {
    spdx: String,
    source: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BuildIdentity {
    compiler: String,
    cmake: String,
    cmake_arguments: Vec<String>,
    flags: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AbiIdentity {
    pointer_width_bits: usize,
    transfer_request_size: usize,
    transfer_request_align: usize,
    transfer_status_size: usize,
    transfer_status_align: usize,
}

pub fn validate_artifact(directory: &Path) -> Result<ValidatedArtifact, EngineError> {
    let library_path = directory.join("libtransfer_engine.so");
    if !library_path.is_file() {
        return Err(EngineError::LibraryMissing { path: library_path });
    }
    reject_symlink(&library_path, "library")?;

    let manifest_path = directory.join("abi-manifest.json");
    if !manifest_path.is_file() {
        return Err(EngineError::ManifestMissing {
            path: manifest_path,
        });
    }
    reject_symlink(&manifest_path, "manifest")?;
    let manifest_bytes = fs::read(&manifest_path).map_err(|error| EngineError::LoaderFailure {
        path: manifest_path.clone(),
        detail: error.to_string(),
    })?;
    let manifest: ArtifactManifest =
        serde_json::from_slice(&manifest_bytes).map_err(|error| EngineError::AbiMismatch {
            detail: format!("invalid manifest JSON: {error}"),
        })?;

    compare("schema_version", "1", &manifest.schema_version.to_string())?;
    compare("version", VERSION, &manifest.version)?;
    compare("commit", COMMIT, &manifest.commit)?;
    compare("header_sha256", HEADER_SHA256, &manifest.header_sha256)?;

    if manifest.build.compiler.trim().is_empty() || manifest.build.cmake.trim().is_empty() {
        return Err(EngineError::AbiMismatch {
            detail: "compiler and CMake identities must be non-empty".into(),
        });
    }
    if manifest.build.cmake_arguments.is_empty() {
        return Err(EngineError::AbiMismatch {
            detail: "complete CMake argument list is missing".into(),
        });
    }
    for (flag, expected) in REQUIRED_FLAGS {
        let actual = manifest
            .build
            .flags
            .get(*flag)
            .map(String::as_str)
            .unwrap_or("<missing>");
        compare("build.flags", expected, actual).map_err(|_| EngineError::AbiMismatch {
            detail: format!("build flag {flag} must be {expected}, got {actual}"),
        })?;
    }

    let mut symbols = manifest.required_symbols.clone();
    symbols.sort();
    let expected_symbols: Vec<_> = REQUIRED_SYMBOLS
        .iter()
        .map(|value| value.to_string())
        .collect();
    if symbols != expected_symbols {
        return Err(EngineError::AbiMismatch {
            detail: format!(
                "required symbol manifest differs: expected {expected_symbols:?}, got {symbols:?}"
            ),
        });
    }

    let expected_abi = super::ffi::abi_layout();
    let actual_abi = &manifest.abi;
    if actual_abi.pointer_width_bits != expected_abi.pointer_width_bits
        || actual_abi.transfer_request_size != expected_abi.transfer_request_size
        || actual_abi.transfer_request_align != expected_abi.transfer_request_align
        || actual_abi.transfer_status_size != expected_abi.transfer_status_size
        || actual_abi.transfer_status_align != expected_abi.transfer_status_align
    {
        return Err(EngineError::AbiMismatch {
            detail: format!(
                "C/Rust layout differs: manifest={actual_abi:?}, rust={expected_abi:?}"
            ),
        });
    }

    let header_path = directory.join("transfer_engine_c.h");
    validate_hashed_file(
        &header_path,
        "header",
        HEADER_SHA256,
        &manifest.header_sha256,
    )?;
    validate_hashed_file(
        &library_path,
        "library",
        &manifest.library_sha256,
        &manifest.library_sha256,
    )?;
    let mut bundled_names: Vec<_> = manifest
        .bundled_libraries
        .iter()
        .map(|library| library.name.as_str())
        .collect();
    bundled_names.sort_unstable();
    if bundled_names != ["libasio.so", "libmooncake_common.so"] {
        return Err(EngineError::AbiMismatch {
            detail: format!(
                "bundled library set differs: expected libasio.so and libmooncake_common.so, got {bundled_names:?}"
            ),
        });
    }
    for library in &manifest.bundled_libraries {
        validate_hashed_file(
            &directory.join(&library.name),
            "bundled_library",
            &library.sha256,
            &library.sha256,
        )?;
    }

    compare("license.spdx", "Apache-2.0", &manifest.license.spdx)?;
    compare(
        "license.source",
        "Mooncake/LICENSE-APACHE",
        &manifest.license.source,
    )?;
    let license_path = directory.join("LICENSE-APACHE");
    validate_hashed_file(
        &license_path,
        "license",
        &manifest.license.sha256,
        &manifest.license.sha256,
    )?;

    Ok(ValidatedArtifact { library_path })
}

fn validate_hashed_file(
    path: &Path,
    field: &'static str,
    expected: &str,
    manifest_value: &str,
) -> Result<(), EngineError> {
    if !path.is_file() {
        return Err(EngineError::ArtifactMismatch {
            field,
            expected: expected.into(),
            actual: "<missing>".into(),
        });
    }
    reject_symlink(path, field)?;
    compare(field, expected, manifest_value)?;
    let actual = sha256_file(path)?;
    compare(field, expected, &actual)
}

fn sha256_file(path: &Path) -> Result<String, EngineError> {
    let bytes = fs::read(path).map_err(|error| EngineError::LoaderFailure {
        path: path.to_path_buf(),
        detail: error.to_string(),
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn reject_symlink(path: &Path, field: &'static str) -> Result<(), EngineError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| EngineError::LoaderFailure {
        path: path.to_path_buf(),
        detail: error.to_string(),
    })?;
    if metadata.file_type().is_symlink() {
        return Err(EngineError::ArtifactMismatch {
            field,
            expected: "regular file".into(),
            actual: "symlink".into(),
        });
    }
    Ok(())
}

fn compare(field: &'static str, expected: &str, actual: &str) -> Result<(), EngineError> {
    if expected == actual {
        Ok(())
    } else {
        Err(EngineError::ArtifactMismatch {
            field,
            expected: expected.into(),
            actual: actual.into(),
        })
    }
}
