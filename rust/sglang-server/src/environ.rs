//! Env-var parsing with the semantics of Python `sglang.srt.environ.EnvField`:
//! unset → default, invalid → warn + default (never an error). One shared
//! parser per type — call sites pass their variable name + default instead of
//! each hand-rolling a reader.

/// Python `EnvBool.parse`: true = `true/1/yes/y`, false = `false/0/no/n`
/// (case-insensitive); anything else is invalid.
pub fn env_bool(name: &str, default: bool) -> bool {
    read(name, default, |raw| match raw.to_lowercase().as_str() {
        "true" | "1" | "yes" | "y" => Some(true),
        "false" | "0" | "no" | "n" => Some(false),
        _ => None,
    })
}

/// Python `EnvInt.parse`: any `int()`-parsable string.
pub fn env_u64(name: &str, default: u64) -> u64 {
    read(name, default, |raw| raw.parse().ok())
}

/// Shared read-or-default: unset → default; a set-but-unparsable value warns
/// and falls back to the default (mirrors `EnvField.get`'s `warnings.warn`).
fn read<T: Copy + std::fmt::Debug>(name: &str, default: T, parse: impl Fn(&str) -> Option<T>) -> T {
    let Ok(raw) = std::env::var(name) else {
        return default;
    };
    match parse(&raw) {
        Some(v) => v,
        None => {
            tracing::warn!(name, value = %raw, ?default, "invalid env value; using default");
            default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The accepted literal sets are copied from Python `EnvBool.parse`
    /// (`true/1/yes/y` / `false/0/no/n`, case-insensitive; invalid → default) —
    /// parity pins, not this crate's invention.
    #[test]
    fn env_bool_matches_python_envbool_parse() {
        // Unique var name per case: tests in this binary run concurrently and
        // share the process environment.
        for (raw, want) in [
            ("true", true),
            ("1", true),
            ("YES", true),
            ("y", true),
            ("false", false),
            ("0", false),
            ("No", false),
            ("n", false),
            // Invalid → default (here: true), matching the warn-and-default path.
            ("off", true),
            ("2", true),
        ] {
            let name = format!("SGLANG_TEST_ENV_BOOL_{raw}");
            unsafe { std::env::set_var(&name, raw) };
            assert_eq!(env_bool(&name, true), want, "value {raw:?}");
        }
        assert!(env_bool("SGLANG_TEST_ENV_BOOL_UNSET", true));
        assert!(!env_bool("SGLANG_TEST_ENV_BOOL_UNSET", false));
    }

    /// `env_u64`: parsable → value, unset/invalid → default.
    #[test]
    fn env_u64_parses_or_defaults() {
        unsafe { std::env::set_var("SGLANG_TEST_ENV_U64_OK", "45") };
        assert_eq!(env_u64("SGLANG_TEST_ENV_U64_OK", 20), 45);
        unsafe { std::env::set_var("SGLANG_TEST_ENV_U64_BAD", "20s") };
        assert_eq!(env_u64("SGLANG_TEST_ENV_U64_BAD", 20), 20);
        assert_eq!(env_u64("SGLANG_TEST_ENV_U64_UNSET", 20), 20);
    }
}
