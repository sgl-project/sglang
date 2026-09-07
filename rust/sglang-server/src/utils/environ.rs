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

/// Signed integer parser. Accepts the `i64::from_str` grammar, including
/// negative values, while invalid or out-of-range values warn and use the
/// default.
pub fn env_i64(name: &str, default: i64) -> i64 {
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

    /// `env_i64`: strict `i64::from_str` grammar, including negative values;
    /// everything else falls back to the default.
    #[test]
    fn env_i64_parses_or_defaults() {
        for (i, (raw, want)) in [
            ("45", 45),
            ("+45", 45),
            ("-1", -1),
            ("-9223372036854775808", i64::MIN),
            ("9223372036854775807", i64::MAX),
            // Invalid → default.
            ("20s", 20),
            ("", 20),
            (" 45 ", 20),
            ("4_5", 20),
            ("9223372036854775808", 20),
            ("-9223372036854775809", 20),
            ("١٢", 20), // non-ASCII digits
        ]
        .into_iter()
        .enumerate()
        {
            let name = format!("SGLANG_TEST_ENV_I64_{i}");
            unsafe { std::env::set_var(&name, raw) };
            assert_eq!(env_i64(&name, 20), want, "value {raw:?}");
        }
        assert_eq!(env_i64("SGLANG_TEST_ENV_I64_UNSET", 20), 20);
    }
}
