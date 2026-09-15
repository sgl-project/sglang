//! Environment variables with the semantics of Python `sglang.srt.environ`:
//! unset -> default, invalid -> warn + default (never an error). Variables are
//! declared once in [`envs`]; call sites read them with `.get()`, which hits
//! the process environment on every call.

pub mod envs;

use std::fmt::Debug;
use std::str::FromStr;

/// A declared variable: its name and the default used when unset or invalid.
pub struct EnvField<T> {
    name: &'static str,
    default: T,
}

pub type EnvBool = EnvField<bool>;
pub type EnvInt = EnvField<i64>;
pub type EnvU64 = EnvField<u64>;
pub type EnvUsize = EnvField<usize>;

impl<T: Copy> EnvField<T> {
    pub const fn new(name: &'static str, default: T) -> Self {
        Self { name, default }
    }

    /// The declared default, for call sites that also expose it as a constant.
    pub const fn default_value(&self) -> T {
        self.default
    }
}

impl EnvBool {
    pub fn get(&self) -> bool {
        self.read(parse_bool)
    }
}

impl EnvInt {
    pub fn get(&self) -> i64 {
        self.read(parse)
    }
}

impl EnvU64 {
    pub fn get(&self) -> u64 {
        self.read(parse)
    }
}

impl EnvUsize {
    pub fn get(&self) -> usize {
        self.read(parse)
    }
}

impl<T: Copy + Debug> EnvField<T> {
    /// Read with an extra call-site rule (such as "must be positive"); a value
    /// the rule rejects warns and falls back exactly like a parse failure.
    pub fn get_with(&self, parse: impl Fn(&str) -> Option<T>) -> T {
        self.read(parse)
    }

    fn read(&self, parse: impl Fn(&str) -> Option<T>) -> T {
        let Ok(raw) = std::env::var(self.name) else {
            return self.default;
        };
        parse(&raw).unwrap_or_else(|| {
            tracing::warn!(
                name = self.name,
                value = %raw,
                default = ?self.default,
                "invalid env value; using default"
            );
            self.default
        })
    }
}

/// Python `EnvBool.parse`: true = `true/1/yes/y`, false = `false/0/no/n`
/// (case-insensitive); anything else is invalid.
fn parse_bool(raw: &str) -> Option<bool> {
    match raw.to_lowercase().as_str() {
        "true" | "1" | "yes" | "y" => Some(true),
        "false" | "0" | "no" | "n" => Some(false),
        _ => None,
    }
}

fn parse<T: FromStr>(raw: &str) -> Option<T> {
    raw.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parity pins copied from Python `EnvBool.parse`, not this crate's invention.
    #[test]
    fn parse_bool_matches_python_envbool_parse() {
        for raw in ["true", "1", "YES", "y"] {
            assert_eq!(parse_bool(raw), Some(true), "{raw:?}");
        }
        for raw in ["false", "0", "No", "n"] {
            assert_eq!(parse_bool(raw), Some(false), "{raw:?}");
        }
        for raw in ["off", "2", ""] {
            assert_eq!(parse_bool(raw), None, "{raw:?}");
        }
    }

    const VAR: &str = "SGLANG_TEST_ENV";
    static AS_INT: EnvInt = EnvInt::new(VAR, 20);
    static AS_BOOL: EnvBool = EnvBool::new(VAR, true);
    static AS_SIZE: EnvUsize = EnvUsize::new(VAR, 7);

    /// The only test that touches the process environment - keep it that way,
    /// or `set_var` races the other tests in this binary.
    #[test]
    fn get_reads_the_environment_on_every_call() {
        unsafe { std::env::remove_var(VAR) };
        assert_eq!(AS_INT.get(), 20);

        unsafe { std::env::set_var(VAR, "-45") };
        assert_eq!(AS_INT.get(), -45);
        assert_eq!(AS_SIZE.get(), 7);
        assert!(AS_BOOL.get());
        let positive = |raw: &str| raw.parse().ok().filter(|&n| n > 0);
        assert_eq!(AS_SIZE.get_with(positive), 7);

        // Re-read, not a value cached in the declaration.
        unsafe { std::env::set_var(VAR, "1") };
        assert_eq!(AS_INT.get(), 1);
        assert_eq!(AS_SIZE.get_with(positive), 1);

        unsafe { std::env::remove_var(VAR) };
    }
}
