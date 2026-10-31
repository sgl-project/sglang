//! Where WASM modules may be loaded from.
//!
//! Module registration takes a caller-supplied filesystem path, so it needs a
//! rule for which paths are acceptable. That rule used to be a deny-list of
//! sensitive system directories (`/etc/`, `/proc/`, ...) compared with
//! `str::starts_with`. Three things were wrong with it:
//!
//! 1. A deny-list of sensitive directories can never be complete. The old list
//!    did not cover `~/.ssh`, `~/.aws`, `/srv` or `/home/*` even on Linux, and
//!    on Windows — where paths look like `C:\Windows\System32\` — it matched
//!    nothing at all, so the control was silently inert on that platform.
//! 2. Comparing paths as raw strings is not the same as comparing them as
//!    paths: `/srv/modules-evil/x.wasm` textually starts with `/srv/modules`.
//! 3. Reporting *which* rule rejected a path leaks information back to the
//!    caller, which is precisely what the check exists to prevent.
//!
//! This module replaces that with an allow-list: a module must resolve to a
//! location inside one of the configured roots. Deny-by-default means there is
//! nothing to forget to block, and because the roots come from configuration
//! there are no hardcoded paths to get wrong per platform.

use std::path::{Path, PathBuf};

use thiserror::Error;

/// Directories a WASM module may be loaded from.
///
/// The roots are canonicalised once, when the gateway starts, so that every
/// later comparison has both sides in the same normalised form. On Windows that
/// form is the `\\?\C:\...` extended-length path with the on-disk casing, which
/// is why comparing a canonicalised candidate against a raw configured string
/// would not work.
#[derive(Debug, Clone)]
pub struct ModuleRoots {
    roots: Vec<PathBuf>,
}

/// Failure to establish the roots at startup.
#[derive(Debug, Error)]
pub enum ModuleRootsError {
    #[error("WASM is enabled but no module root is configured (set --wasm-module-root)")]
    NoRootsConfigured,

    #[error("configured WASM module root {path} is not usable: {source}")]
    UnusableRoot {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

/// Why a requested module path was refused.
///
/// `Display` is the rendering that crosses the API boundary, so the distinction
/// the caller must not learn is kept out of it *structurally* rather than by
/// remembering to collapse it at each call site: [`Unavailable`] covers both
/// "no such file" and "resolved, but outside every root", and its message
/// interpolates nothing. Telling those two apart would turn module registration
/// into an existence oracle for arbitrary filesystem paths — precisely the
/// disclosure the containment check exists to prevent.
///
/// The cause is not discarded; it rides along in a field that only `Debug`
/// renders, so server-side logs keep full detail.
///
/// [`Unavailable`]: ModulePathError::Unavailable
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ModulePathError {
    #[error("module file is not available at the requested path")]
    Unavailable(Unavailability),

    /// Reachable only by a file that already resolved *inside* a configured
    /// root, so naming this case tells the caller nothing about paths it could
    /// not already reach. Kept distinct because collapsing it would leave a
    /// legitimate operator with no hint about a genuine mistake.
    #[error("module file must have a .wasm extension")]
    NotWasm,
}

/// The server-side cause behind [`ModulePathError::Unavailable`].
///
/// Deliberately not `Display`: giving it a user-facing rendering is exactly how
/// the distinction would escape to the caller again. Logs read it via `Debug`.
#[derive(Debug, PartialEq, Eq)]
pub enum Unavailability {
    /// `canonicalize` failed — the path does not exist, or is not readable.
    Unresolvable,
    /// The path resolved to a real location, but outside every configured root.
    OutsideRoots,
}

impl ModuleRoots {
    /// Canonicalise the configured roots.
    ///
    /// Fails closed: enabling WASM without naming a root is a configuration
    /// mistake, not permission to read the whole filesystem.
    pub fn load(raw: &[String]) -> Result<Self, ModuleRootsError> {
        if raw.is_empty() {
            return Err(ModuleRootsError::NoRootsConfigured);
        }

        let roots = raw
            .iter()
            .map(|path| {
                std::fs::canonicalize(path).map_err(|source| ModuleRootsError::UnusableRoot {
                    path: path.clone(),
                    source,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self { roots })
    }

    /// Resolve a caller-supplied path to a real location inside a root.
    ///
    /// `canonicalize` collapses `.` and `..`, resolves symlinks and junctions,
    /// and normalises casing and 8.3 short names. A single call therefore
    /// subsumes the separate absolute-path check, the `..`/`.` component scan
    /// and the post-symlink re-check that the old validation performed by hand.
    pub async fn resolve(&self, requested: &str) -> Result<PathBuf, ModulePathError> {
        let canonical = tokio::fs::canonicalize(requested)
            .await
            .map_err(|_| ModulePathError::Unavailable(Unavailability::Unresolvable))?;

        if !self.contains(&canonical) {
            return Err(ModulePathError::Unavailable(Unavailability::OutsideRoots));
        }

        if !has_wasm_extension(&canonical) {
            return Err(ModulePathError::NotWasm);
        }

        Ok(canonical)
    }

    /// Whether an already-canonical path lies inside one of the roots.
    ///
    /// `Path::starts_with` compares whole path components. That is the entire
    /// point: a textual comparison would accept `<root>-evil/x.wasm`, because
    /// those characters really do start with `<root>`.
    fn contains(&self, canonical: &Path) -> bool {
        self.roots.iter().any(|root| canonical.starts_with(root))
    }

    /// The canonicalised roots, for logging and diagnostics.
    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }
}

/// Whether a path ends in `.wasm`, ignoring case.
///
/// This is hygiene rather than a security boundary — containment above is the
/// boundary — but it keeps obvious mistakes from reaching the WASM compiler.
fn has_wasm_extension(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("wasm"))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;

    /// Lays out:
    ///   <base>/modules/ok.wasm
    ///   <base>/modules/sub/deep.wasm
    ///   <base>/modules/notes.txt
    ///   <base>/modules-evil/evil.wasm   <- sibling sharing a textual prefix
    ///   <base>/outside.wasm
    fn fixture() -> (TempDir, ModuleRoots) {
        let base = TempDir::new().expect("temp dir");
        let modules = base.path().join("modules");
        fs::create_dir_all(modules.join("sub")).unwrap();
        fs::create_dir_all(base.path().join("modules-evil")).unwrap();
        fs::write(modules.join("ok.wasm"), b"\0asm").unwrap();
        fs::write(modules.join("sub/deep.wasm"), b"\0asm").unwrap();
        fs::write(modules.join("notes.txt"), b"hi").unwrap();
        fs::write(base.path().join("modules-evil/evil.wasm"), b"\0asm").unwrap();
        fs::write(base.path().join("outside.wasm"), b"\0asm").unwrap();

        let roots = ModuleRoots::load(&[modules.to_string_lossy().into_owned()]).unwrap();
        (base, roots)
    }

    #[tokio::test]
    async fn accepts_a_module_directly_inside_a_root() {
        let (base, roots) = fixture();
        let path = base.path().join("modules/ok.wasm");
        let resolved = roots.resolve(&path.to_string_lossy()).await.unwrap();
        assert!(resolved.ends_with("ok.wasm"));
    }

    #[tokio::test]
    async fn accepts_a_module_nested_below_a_root() {
        let (base, roots) = fixture();
        let path = base.path().join("modules/sub/deep.wasm");
        assert!(roots.resolve(&path.to_string_lossy()).await.is_ok());
    }

    /// The case a textual prefix comparison gets wrong.
    #[tokio::test]
    async fn rejects_a_sibling_directory_sharing_a_textual_prefix() {
        let (base, roots) = fixture();
        let path = base.path().join("modules-evil/evil.wasm");
        assert_eq!(
            roots.resolve(&path.to_string_lossy()).await,
            Err(ModulePathError::Unavailable(Unavailability::OutsideRoots))
        );
    }

    #[tokio::test]
    async fn rejects_an_escape_through_parent_directory_components() {
        let (base, roots) = fixture();
        let path = base.path().join("modules/../outside.wasm");
        assert_eq!(
            roots.resolve(&path.to_string_lossy()).await,
            Err(ModulePathError::Unavailable(Unavailability::OutsideRoots))
        );
    }

    #[tokio::test]
    async fn rejects_a_path_outside_every_root() {
        let (base, roots) = fixture();
        let path = base.path().join("outside.wasm");
        assert_eq!(
            roots.resolve(&path.to_string_lossy()).await,
            Err(ModulePathError::Unavailable(Unavailability::OutsideRoots))
        );
    }

    #[tokio::test]
    async fn rejects_a_non_wasm_file_inside_a_root() {
        let (base, roots) = fixture();
        let path = base.path().join("modules/notes.txt");
        assert_eq!(
            roots.resolve(&path.to_string_lossy()).await,
            Err(ModulePathError::NotWasm)
        );
    }

    #[tokio::test]
    async fn rejects_a_path_that_does_not_exist() {
        let (base, roots) = fixture();
        let path = base.path().join("modules/absent.wasm");
        assert_eq!(
            roots.resolve(&path.to_string_lossy()).await,
            Err(ModulePathError::Unavailable(Unavailability::Unresolvable))
        );
    }

    /// The property the error type exists to hold: the caller cannot tell "no
    /// such file" from "exists, but outside the roots". If these ever render
    /// differently, module registration is an existence oracle again.
    #[tokio::test]
    async fn does_not_reveal_whether_a_rejected_path_exists() {
        let (base, roots) = fixture();

        let absent = roots
            .resolve(&base.path().join("modules/absent.wasm").to_string_lossy())
            .await
            .unwrap_err();
        let outside = roots
            .resolve(&base.path().join("outside.wasm").to_string_lossy())
            .await
            .unwrap_err();

        // The server still knows which is which...
        assert_eq!(
            absent,
            ModulePathError::Unavailable(Unavailability::Unresolvable)
        );
        assert_eq!(
            outside,
            ModulePathError::Unavailable(Unavailability::OutsideRoots)
        );
        assert_ne!(absent, outside);

        // ...but the caller is told one and the same thing.
        assert_eq!(absent.to_string(), outside.to_string());
    }

    /// Windows counterpart to the Unix test below.
    ///
    /// This is the case the old deny-list never covered: containment on Windows
    /// was untested because nothing ran there. Creating a symlink needs
    /// `SeCreateSymbolicLinkPrivilege` (an elevated shell, or Developer Mode);
    /// where that is unavailable the test reports why and returns, rather than
    /// failing for a reason that has nothing to do with the code under test.
    #[cfg(windows)]
    #[tokio::test]
    async fn rejects_a_symlink_escaping_a_root() {
        let (base, roots) = fixture();
        let link = base.path().join("modules/escape.wasm");

        if std::os::windows::fs::symlink_file(base.path().join("outside.wasm"), &link).is_err() {
            eprintln!(
                "skipped: creating a symlink needs Developer Mode or an elevated shell \
                 on Windows"
            );
            return;
        }

        assert_eq!(
            roots.resolve(&link.to_string_lossy()).await,
            Err(ModulePathError::Unavailable(Unavailability::OutsideRoots))
        );
    }

    /// A symlink inside a root that points outside it must not grant access.
    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_a_symlink_escaping_a_root() {
        let (base, roots) = fixture();
        let link = base.path().join("modules/escape.wasm");
        std::os::unix::fs::symlink(base.path().join("outside.wasm"), &link).unwrap();
        assert_eq!(
            roots.resolve(&link.to_string_lossy()).await,
            Err(ModulePathError::Unavailable(Unavailability::OutsideRoots))
        );
    }

    #[test]
    fn refuses_to_start_with_no_roots_configured() {
        assert!(matches!(
            ModuleRoots::load(&[]),
            Err(ModuleRootsError::NoRootsConfigured)
        ));
    }

    #[test]
    fn refuses_to_start_when_a_root_does_not_exist() {
        let missing = TempDir::new().unwrap().path().join("nope");
        assert!(matches!(
            ModuleRoots::load(&[missing.to_string_lossy().into_owned()]),
            Err(ModuleRootsError::UnusableRoot { .. })
        ));
    }

    #[tokio::test]
    async fn honours_every_configured_root() {
        let base = TempDir::new().unwrap();
        let a = base.path().join("a");
        let b = base.path().join("b");
        fs::create_dir_all(&a).unwrap();
        fs::create_dir_all(&b).unwrap();
        fs::write(a.join("x.wasm"), b"\0asm").unwrap();
        fs::write(b.join("y.wasm"), b"\0asm").unwrap();

        let roots = ModuleRoots::load(&[
            a.to_string_lossy().into_owned(),
            b.to_string_lossy().into_owned(),
        ])
        .unwrap();

        assert!(roots
            .resolve(&a.join("x.wasm").to_string_lossy())
            .await
            .is_ok());
        assert!(roots
            .resolve(&b.join("y.wasm").to_string_lossy())
            .await
            .is_ok());
    }
}
