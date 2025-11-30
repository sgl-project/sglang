//! Version information module
//!
//! Provides version information including version number, build time, and Git metadata.

/// Project name from pyproject.toml (set at compile time)
pub const PROJECT_NAME: &str = env!("SGL_ROUTER_PROJECT_NAME");

/// Version string from pyproject.toml (set at compile time)
pub const VERSION: &str = env!("SGL_ROUTER_VERSION");

/// Build time in UTC format (set at compile time)
pub const BUILD_TIME: &str = env!("SGL_ROUTER_BUILD_TIME");

/// Git branch name (set at compile time, "unknown" if not available)
pub const GIT_BRANCH: &str = env!("SGL_ROUTER_GIT_BRANCH");

/// Git commit hash (short) (set at compile time, "unknown" if not available)
pub const GIT_COMMIT: &str = env!("SGL_ROUTER_GIT_COMMIT");

/// Git repository status (clean/dirty) (set at compile time)
pub const GIT_STATUS: &str = env!("SGL_ROUTER_GIT_STATUS");

/// Rustc version (set at compile time)
pub const RUSTC_VERSION: &str = env!("SGL_ROUTER_RUSTC_VERSION");

/// Cargo version (set at compile time)
pub const CARGO_VERSION: &str = env!("SGL_ROUTER_CARGO_VERSION");

/// Target triple (platform) (set at compile time)
pub const TARGET_TRIPLE: &str = env!("SGL_ROUTER_TARGET_TRIPLE");

/// Build mode (debug/release) (set at compile time)
pub const BUILD_MODE: &str = env!("SGL_ROUTER_BUILD_MODE");

/// Get formatted version information string with structured format
pub fn get_version_string() -> String {
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
        get_title(),
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

/// Get version title line
pub fn get_title() -> String {
    format!("{} version {}", PROJECT_NAME, VERSION)
}

/// Get version number only
pub fn get_version() -> &'static str {
    VERSION
}

/// Get short version information string
pub fn get_short_version_string() -> String {
    format!("{} version {}, build {}", PROJECT_NAME, VERSION, GIT_COMMIT)
}
