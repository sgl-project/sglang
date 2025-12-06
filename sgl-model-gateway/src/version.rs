//! Version information module
//!
//! Provides version information including version number, build time, and Git metadata.

macro_rules! build_env {
    ($name:ident) => {
        env!(concat!("SGL_MODEL_GATEWAY_", stringify!($name)))
    };
}

pub const PROJECT_NAME: &str = build_env!(PROJECT_NAME);
pub const VERSION: &str = build_env!(VERSION);
pub const BUILD_TIME: &str = build_env!(BUILD_TIME);
pub const GIT_BRANCH: &str = build_env!(GIT_BRANCH);
pub const GIT_COMMIT: &str = build_env!(GIT_COMMIT);
pub const GIT_STATUS: &str = build_env!(GIT_STATUS);
pub const RUSTC_VERSION: &str = build_env!(RUSTC_VERSION);
pub const CARGO_VERSION: &str = build_env!(CARGO_VERSION);
pub const TARGET_TRIPLE: &str = build_env!(TARGET_TRIPLE);
pub const BUILD_MODE: &str = build_env!(BUILD_MODE);

/// Get simple version string (default for --version)
pub fn get_version_string() -> String {
    format!("{} {}", PROJECT_NAME, VERSION)
}

/// Get verbose version information string with full build details (for --version-verbose)
pub fn get_verbose_version_string() -> String {
    format!(
        "{}\n\n\
Build Information:\n\
  Build Time: {}\n\
  Build Mode: {}\n\
  Platform: {}\n\n\
Version Control:\n\
  Git Branch: {}\n\
  Git Commit: {}\n\
  Git Status: {}\n\n\
Compiler:\n\
  {}\n\
  {}",
        get_version_string(),
        BUILD_TIME,
        BUILD_MODE,
        TARGET_TRIPLE,
        GIT_BRANCH,
        GIT_COMMIT,
        GIT_STATUS,
        RUSTC_VERSION,
        CARGO_VERSION
    )
}

/// Get version number only
pub fn get_version() -> &'static str {
    VERSION
}
