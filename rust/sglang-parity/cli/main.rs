//! Select a compiled suite and present the library's execution and review views.

use std::path::PathBuf;
use std::process::ExitCode;

use sglang_parity::{RunConfig, describe, run};

#[path = "../suites/native_generate/mod.rs"]
mod native_generate;

const USAGE: &str = "Usage: sglang-parity --config <run.json> [--suite native_generate] [--suite-file <suite.json>] [--describe]\n\n--describe validates and prints the effective specification without starting services.";

#[derive(Default)]
struct Arguments {
    config: Option<PathBuf>,
    suite_file: Option<PathBuf>,
    describe: bool,
}

fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Option<Arguments>, String> {
    let mut result = Arguments::default();
    let mut seen = std::collections::BTreeSet::new();
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        if argument == "--help" || argument == "-h" {
            return Ok(None);
        }
        if !seen.insert(argument.clone()) {
            return Err(format!("duplicate option {argument}"));
        }
        match argument.as_str() {
            "--describe" => result.describe = true,
            "--config" | "--suite" | "--suite-file" => {
                let value = arguments
                    .next()
                    .filter(|value| !value.starts_with("--"))
                    .ok_or_else(|| format!("{argument} requires a value"))?;
                match argument.as_str() {
                    "--config" => result.config = Some(value.into()),
                    "--suite-file" => result.suite_file = Some(value.into()),
                    _ if value != "native_generate" => {
                        return Err(format!("unsupported suite {value:?}"));
                    }
                    _ => {}
                }
            }
            _ => return Err(format!("unknown option {argument}")),
        }
    }
    if result.config.is_none() {
        return Err("--config is required".into());
    }
    Ok(Some(result))
}

async fn execute(arguments: Arguments) -> Result<i32, Box<dyn std::error::Error>> {
    let config: RunConfig = serde_json::from_slice(&std::fs::read(arguments.config.unwrap())?)?;
    let external = arguments
        .suite_file
        .map(std::fs::read_to_string)
        .transpose()?;
    let (suite, policy) = native_generate::load(
        external.as_deref().unwrap_or(native_generate::DEFAULT_SPEC),
        &config,
    )
    .map_err(std::io::Error::other)?;
    if arguments.describe {
        println!(
            "{}",
            serde_json::to_string_pretty(&describe(&config, &suite)?)?
        );
        return Ok(0);
    }
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let report = tokio::select! {
        result = run(&config, &suite, &policy) => result?,
        signal = async {
            tokio::select! {
                signal = tokio::signal::ctrl_c() => signal,
                _ = terminate.recv() => Ok(()),
            }
        } => {
            signal?;
            eprintln!("Run interrupted; managed services have been stopped. Partial artifacts remain in {}.", config.output_dir.display());
            return Ok(2);
        }
    };
    for case in &report.cases {
        let violations: usize = case
            .implementations
            .values()
            .flat_map(|side| &side.attempts)
            .map(|attempt| attempt.violations.len())
            .sum();
        println!(
            "{}: parity={:?}, python={:?}, rust={:?}, violations={violations}",
            case.name,
            case.parity.status,
            case.implementations["python"].repeatability.status,
            case.implementations["rust"].repeatability.status
        );
    }
    for error in &report.runtime_errors {
        eprintln!("{error}");
    }
    println!("Report: {}", report.directory.join("report.json").display());
    Ok(report.exit_code())
}

#[tokio::main]
async fn main() -> ExitCode {
    match parse(std::env::args().skip(1)) {
        Ok(None) => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Some(arguments)) => match execute(arguments).await {
            Ok(code) => ExitCode::from(code as u8),
            Err(error) => {
                eprintln!("{error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("{error}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_ambiguous_or_obsolete_options() {
        for args in [
            vec!["--config", "--describe"],
            vec!["--cases", "cases.json"],
            vec!["--config", "one", "--config", "two"],
            vec!["--config", "one", "--suite", "grpc"],
        ] {
            assert!(parse(args.into_iter().map(String::from)).is_err());
        }
    }
}
