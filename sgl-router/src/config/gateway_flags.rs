// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! sgl-model-gateway flags that sgl-router does not define. They fail at startup
//! with what to pass instead, or a note that the feature is not supported,
//! rather than with clap's generic unexpected-argument error.

use std::ffi::OsString;
use std::fmt;

use clap::error::ErrorKind;
use clap::{CommandFactory, Parser};

use crate::config::Cli;

/// What an sgl-model-gateway flag becomes on sgl-router.
#[derive(Clone, Copy, Debug)]
enum Migration {
    /// Pass this instead.
    Use(&'static str),
    /// Remove the flag; the text says what sgl-router does in its place.
    Omit(&'static str),
    /// No sgl-router equivalent; the text, if any, says what sgl-router does instead.
    Unsupported(Option<&'static str>),
}

use Migration::{Omit, Unsupported, Use};

impl fmt::Display for Migration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Use(replacement) => write!(f, "use {replacement}"),
            Omit(reason) => write!(f, "remove it; {reason}"),
            Unsupported(None) => f.write_str("not supported by sgl-router"),
            Unsupported(Some(note)) => write!(f, "not supported by sgl-router; {note}"),
        }
    }
}

const PRESSURE_GUARD: &str = "the cache_aware pressure guard compares token loads, \
     set with --pressure-abs-threshold-tokens and --pressure-rel-threshold";
const KV_EVENT_TREE: &str =
    "cache_aware matches prefixes against the engines' KV events, evictions included";
const EVICTION_SWEEP: &str = "cache_aware follows the engines' KV events, evictions included, \
     and --policy sticky sets its sweep with --sticky-eviction-interval-secs";
const WORKERS_REGISTER_LIVE: &str =
    "workers register as they come up, and /readyz answers 503 until they can serve";
const METRICS_ON_MAIN_PORT: &str = "sgl-router serves /metrics on --host and --port";
const NO_HEALTH_CHECKS: &str = "it runs no worker health checks, \
     and its circuit breaker takes failing workers out of routing";
const ENGINE_PARSERS: &str =
    "sgl-router uses the parser each engine reports in /server_info, so set it on the engines";
const ROUTES_TO_SGLANG_ONLY: &str = "sgl-router routes to SGLang engines only";

/// Flags of both sgl-model-gateway launchers, the `smg` binary and
/// `python -m sglang_router.launch_router`, that sgl-router does not define.
/// Flags both routers define are left to clap.
const GATEWAY_ONLY_FLAGS: &[(&str, Migration)] = &[
    // Routing policy
    (
        "--cache-threshold",
        Unsupported(Some(
            "cache_aware sets its minimum prefix match with \
             --cache-affinity-min-matched-tokens and --cache-affinity-min-match-ratio",
        )),
    ),
    ("--balance-abs-threshold", Unsupported(Some(PRESSURE_GUARD))),
    ("--balance-rel-threshold", Unsupported(Some(PRESSURE_GUARD))),
    (
        "--bucket-adjust-interval-secs",
        Unsupported(Some("buckets are static, from --bucket-config")),
    ),
    ("--eviction-interval", Omit(EVICTION_SWEEP)),
    ("--eviction-interval-secs", Omit(EVICTION_SWEEP)),
    ("--max-tree-size", Omit(KV_EVENT_TREE)),
    (
        "--max-idle-secs",
        Use("--sticky-idle-secs with --policy sticky"),
    ),
    (
        "--assignment-mode",
        Use(
            "--sticky-fallback-policy random (or load_based for min_load) with --policy sticky; \
             min_group has no counterpart",
        ),
    ),
    ("--prefix-token-count", Unsupported(None)),
    ("--prefix-hash-load-factor", Unsupported(None)),
    (
        "--enable-igw",
        Unsupported(Some("a router serves one model, --model-id")),
    ),
    // PD disaggregation
    (
        "--pd-disaggregation",
        Omit("sgl-router reads each worker's PD role from its /server_info"),
    ),
    (
        "--prefill",
        Use("--worker-urls; each worker's PD role and bootstrap port come from its /server_info"),
    ),
    (
        "--decode",
        Use("--worker-urls; each worker's PD role comes from its /server_info"),
    ),
    ("--prefill-policy", Use("--policy")),
    (
        "--worker-startup-timeout-secs",
        Unsupported(Some(WORKERS_REGISTER_LIVE)),
    ),
    (
        "--worker-startup-check-interval",
        Unsupported(Some(WORKERS_REGISTER_LIVE)),
    ),
    ("--mini-lb", Unsupported(None)),
    ("--test-external-dp-routing", Unsupported(None)),
    // Kubernetes service discovery
    (
        "--service-discovery-port",
        Omit("each worker's port comes from its EndpointSlice"),
    ),
    (
        "--bootstrap-port-annotation",
        Omit("each prefill worker's bootstrap port comes from its /server_info"),
    ),
    // Logging and metrics
    ("--log-dir", Unsupported(Some("logs go to stdout"))),
    ("--json-log", Use("--log-format json")),
    ("--prometheus-port", Omit(METRICS_ON_MAIN_PORT)),
    ("--prometheus-host", Omit(METRICS_ON_MAIN_PORT)),
    ("--prometheus-duration-buckets", Unsupported(None)),
    // Request handling and upstream HTTP client
    (
        "--request-id-headers",
        Unsupported(Some("the access log reads x-request-id")),
    ),
    (
        "--shutdown-grace-period-secs",
        Unsupported(Some(
            "the in-flight drain has no time limit, \
             and --shutdown-drain-secs sets the readiness pause before it",
        )),
    ),
    (
        "--max-payload-size",
        Use("the SGLANG_ROUTER_MAX_BODY_BYTES environment variable, in bytes"),
    ),
    ("--cors-allowed-origins", Unsupported(None)),
    ("--pool-idle-timeout-secs", Unsupported(None)),
    ("--connect-timeout-secs", Unsupported(None)),
    ("--pool-max-idle-per-host", Unsupported(None)),
    ("--tcp-keepalive-secs", Unsupported(None)),
    // Rate limiting
    ("--max-concurrent-requests", Unsupported(None)),
    ("--queue-size", Unsupported(None)),
    ("--queue-timeout-secs", Unsupported(None)),
    ("--rate-limit-tokens-per-second", Unsupported(None)),
    // Retries and circuit breaker
    (
        "--retry-max-retries",
        Use("--retry-max-attempts; both count the first attempt"),
    ),
    (
        "--retry-backoff-multiplier",
        Unsupported(Some("the backoff doubles per retry")),
    ),
    ("--retry-jitter-factor", Unsupported(None)),
    (
        "--disable-retries",
        Omit("sgl-router retries only when --retry-max-attempts is above 1"),
    ),
    ("--cb-failure-threshold", Use("--cb-threshold")),
    (
        "--cb-success-threshold",
        Unsupported(Some("one successful probe closes the breaker")),
    ),
    ("--cb-timeout-duration-secs", Use("--cb-cool-down-secs")),
    (
        "--cb-window-duration-secs",
        Omit("the breaker counts consecutive failures, with no time window"),
    ),
    // Health checks
    (
        "--health-failure-threshold",
        Unsupported(Some(NO_HEALTH_CHECKS)),
    ),
    (
        "--health-success-threshold",
        Unsupported(Some(NO_HEALTH_CHECKS)),
    ),
    (
        "--health-check-timeout-secs",
        Unsupported(Some(NO_HEALTH_CHECKS)),
    ),
    (
        "--health-check-interval-secs",
        Unsupported(Some(NO_HEALTH_CHECKS)),
    ),
    (
        "--health-check-endpoint",
        Unsupported(Some(NO_HEALTH_CHECKS)),
    ),
    (
        "--disable-health-check",
        Omit("sgl-router runs no worker health checks"),
    ),
    // Tokenizer and parsers
    (
        "--model-path",
        Use("--tokenizer-path, which defaults to --model-id"),
    ),
    (
        "--chat-template",
        Unsupported(Some(
            "set the template on the engines and pass --disable-input-ids-forwarding",
        )),
    ),
    ("--tokenizer-cache-enable-l0", Unsupported(None)),
    ("--tokenizer-cache-l0-max-entries", Unsupported(None)),
    (
        "--tokenizer-cache-enable-l1",
        Use("--tokenizer-l1-cache-mb with a size above 0"),
    ),
    (
        "--tokenizer-cache-l1-max-memory",
        Use("--tokenizer-l1-cache-mb, in MiB rather than bytes"),
    ),
    ("--reasoning-parser", Omit(ENGINE_PARSERS)),
    ("--tool-call-parser", Omit(ENGINE_PARSERS)),
    ("--mcp-config-path", Unsupported(None)),
    // Backend and history storage
    ("--backend", Omit(ROUTES_TO_SGLANG_ONLY)),
    ("--runtime", Omit(ROUTES_TO_SGLANG_ONLY)),
    ("--history-backend", Unsupported(None)),
    ("--enable-wasm", Unsupported(None)),
    ("--oracle-wallet-path", Unsupported(None)),
    ("--oracle-tns-alias", Unsupported(None)),
    ("--oracle-dsn", Unsupported(None)),
    ("--oracle-connect-descriptor", Unsupported(None)),
    ("--oracle-user", Unsupported(None)),
    ("--oracle-username", Unsupported(None)),
    ("--oracle-password", Unsupported(None)),
    ("--oracle-pool-min", Unsupported(None)),
    ("--oracle-pool-max", Unsupported(None)),
    ("--oracle-pool-timeout-secs", Unsupported(None)),
    ("--postgres-db-url", Unsupported(None)),
    ("--postgres-pool-max-size", Unsupported(None)),
    ("--postgres-pool-max", Unsupported(None)),
    ("--redis-url", Unsupported(None)),
    ("--redis-pool-max-size", Unsupported(None)),
    ("--redis-pool-max", Unsupported(None)),
    ("--redis-retention-days", Unsupported(None)),
    // TLS and tracing
    ("--tls-cert-path", Unsupported(None)),
    ("--tls-key-path", Unsupported(None)),
    ("--client-cert-path", Unsupported(None)),
    ("--client-key-path", Unsupported(None)),
    ("--ca-cert-paths", Unsupported(None)),
    ("--enable-trace", Unsupported(None)),
    ("--otlp-traces-endpoint", Unsupported(None)),
    // Authentication
    (
        "--api-key",
        Unsupported(Some(
            "it does not check client keys and passes the caller's Authorization header \
             on to the engines, which check it when launched with an API key; \
             --worker-api-key is the key for its own requests to workers",
        )),
    ),
    ("--jwt-issuer", Unsupported(None)),
    ("--jwt-audience", Unsupported(None)),
    ("--jwt-jwks-uri", Unsupported(None)),
    ("--jwt-role-claim", Unsupported(None)),
    ("--jwt-role-mapping", Unsupported(None)),
    ("--control-plane-api-keys", Unsupported(None)),
    ("--disable-audit-logging", Unsupported(None)),
    ("--control-plane-audit-enabled", Unsupported(None)),
    // Mesh
    ("--enable-mesh", Unsupported(None)),
    ("--mesh-server-name", Unsupported(None)),
    ("--mesh-host", Unsupported(None)),
    ("--mesh-port", Unsupported(None)),
    ("--mesh-peer-urls", Unsupported(None)),
];

/// The table entries for the gateway-only flags in `args` (program name first),
/// in order of first use; `--` ends the flags, as it does for clap.
fn gateway_only_flags_in(args: &[OsString]) -> Vec<(&'static str, Migration)> {
    let mut found: Vec<(&'static str, Migration)> = Vec::new();
    for arg in args
        .iter()
        .skip(1)
        .filter_map(|arg| arg.to_str())
        .take_while(|arg| *arg != "--")
    {
        let name = arg.split_once('=').map_or(arg, |(name, _)| name);
        let Some(&entry) = GATEWAY_ONLY_FLAGS.iter().find(|(flag, _)| *flag == name) else {
            continue;
        };
        if !found.iter().any(|(flag, _)| *flag == entry.0) {
            found.push(entry);
        }
    }
    found
}

fn rejection_message(found: &[(&str, Migration)]) -> String {
    let lines: Vec<String> = found
        .iter()
        .map(|(flag, migration)| format!("  {flag}: {migration}"))
        .collect();
    format!(
        "sgl-model-gateway flags that sgl-router does not accept:\n{}",
        lines.join("\n")
    )
}

impl Cli {
    /// [`Parser::try_parse_from`], except that sgl-model-gateway flags fail with
    /// what to use on sgl-router instead of clap's unexpected-argument error.
    pub fn try_parse_rejecting_gateway_flags<I, T>(args: I) -> Result<Self, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString>,
    {
        let args: Vec<OsString> = args.into_iter().map(Into::into).collect();
        let found = gateway_only_flags_in(&args);
        if !found.is_empty() {
            return Err(
                Self::command().error(ErrorKind::UnknownArgument, rejection_message(&found))
            );
        }
        Self::try_parse_from(args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn parse(extra: &[&str]) -> Result<Cli, clap::Error> {
        let base = [
            "sgl-router",
            "--model-id",
            "m",
            "--worker-urls",
            "http://w:1",
        ];
        Cli::try_parse_rejecting_gateway_flags(base.iter().chain(extra))
    }

    fn sgl_router_flags() -> HashSet<String> {
        let command = Cli::command();
        command
            .get_arguments()
            .flat_map(|arg| {
                arg.get_long()
                    .into_iter()
                    .chain(arg.get_all_aliases().unwrap_or_default())
            })
            .map(|name| format!("--{name}"))
            .collect()
    }

    #[test]
    fn each_gateway_only_flag_fails_with_its_migration() {
        for &(flag, migration) in GATEWAY_ONLY_FLAGS {
            let with_value = format!("{flag}=x");
            for arg in [flag, with_value.as_str()] {
                let error = parse(&[arg]).expect_err(arg);
                assert_eq!(error.kind(), ErrorKind::UnknownArgument, "{arg}");
                let message = error.to_string();
                assert!(
                    message.contains(&format!("\n  {flag}: {migration}\n")),
                    "{arg}: {message}"
                );
            }
        }
    }

    #[test]
    fn no_table_entry_is_an_sgl_router_flag() {
        let live = sgl_router_flags();
        let mut seen = HashSet::new();
        for (flag, _) in GATEWAY_ONLY_FLAGS {
            assert!(
                !live.contains(*flag),
                "{flag} is an sgl-router flag; remove it from GATEWAY_ONLY_FLAGS"
            );
            assert!(seen.insert(flag), "{flag} is listed twice");
        }
    }

    #[test]
    fn migrations_name_only_sgl_router_flags() {
        let live = sgl_router_flags();
        for (flag, migration) in GATEWAY_ONLY_FLAGS {
            let advice = migration.to_string();
            let named = advice
                .split(|c: char| !(c.is_ascii_alphanumeric() || c == '-'))
                .filter(|word| word.starts_with("--"));
            for name in named {
                assert!(
                    live.contains(name),
                    "{flag}: names {name}, not an sgl-router flag"
                );
            }
        }
    }

    #[test]
    fn sgl_router_flags_that_share_a_gateway_prefix_still_parse() {
        parse(&[
            "--prefill-selector",
            "app=prefill",
            "--decode-selector",
            "app=decode",
            "--decode-policy",
            "power_of_two",
            "--decode-dispatch-limit",
            "4",
        ])
        .unwrap();
    }

    #[test]
    fn all_gateway_flags_are_reported_together_once_each() {
        let message = parse(&[
            "--pd-disaggregation",
            "--prefill",
            "http://p:1",
            "9001",
            "--decode",
            "http://d:1",
            "--prefill",
            "http://p:2",
        ])
        .unwrap_err()
        .to_string();
        let flags: Vec<&str> = message
            .lines()
            .filter_map(|line| line.strip_prefix("  "))
            .filter_map(|line| line.split_once(':'))
            .map(|(flag, _)| flag)
            .collect();
        assert_eq!(
            flags,
            ["--pd-disaggregation", "--prefill", "--decode"],
            "{message}"
        );
    }
}
