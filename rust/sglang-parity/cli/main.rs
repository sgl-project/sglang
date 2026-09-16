//! Select a compiled suite and present the library's execution and review views.

use std::io::IsTerminal;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use sglang_parity::environment::{self, Backend};
use sglang_parity::report::ReportView;
use sglang_parity::{Report, RunConfig, describe, describe_plan, run, run_plan};

#[path = "../suites/native_generate/mod.rs"]
mod native_generate;

const USAGE: &str = "Usage: sglang-parity --config <run.json> [--suite native_generate] [--suite-file <suite.json>] [--describe]\n       sglang-parity --report <report.json> [--case <name>]\n       sglang-parity --update-env-lock --backend <mlx|cuda>\n\n--describe validates and prints the effective specification without installing environments or starting services.";

#[derive(Default)]
struct Arguments {
    config: Option<PathBuf>,
    report: Option<PathBuf>,
    case: Option<String>,
    suite_file: Option<PathBuf>,
    describe: bool,
    update_env_lock: bool,
    backend: Option<Backend>,
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
            "--update-env-lock" => result.update_env_lock = true,
            "--config" | "--suite" | "--suite-file" | "--backend" | "--report" | "--case" => {
                let value = arguments
                    .next()
                    .filter(|value| !value.starts_with("--"))
                    .ok_or_else(|| format!("{argument} requires a value"))?;
                match argument.as_str() {
                    "--config" => result.config = Some(value.into()),
                    "--report" => result.report = Some(value.into()),
                    "--case" => result.case = Some(value),
                    "--suite-file" => result.suite_file = Some(value.into()),
                    "--backend" => {
                        result.backend = Some(match value.as_str() {
                            "mlx" => Backend::Mlx,
                            "cuda" => Backend::Cuda,
                            _ => return Err(format!("unsupported environment backend {value:?}")),
                        })
                    }
                    _ if value != "native_generate" => {
                        return Err(format!("unsupported suite {value:?}"));
                    }
                    _ => {}
                }
            }
            _ => return Err(format!("unknown option {argument}")),
        }
    }
    if result.case.is_some() && result.report.is_none() {
        return Err("--case requires --report".into());
    }
    if result.report.is_some() {
        if seen
            .iter()
            .any(|option| !matches!(option.as_str(), "--report" | "--case"))
        {
            return Err("--report cannot be combined with run or environment options".into());
        }
    } else if result.update_env_lock {
        if ["--config", "--describe", "--suite", "--suite-file"]
            .iter()
            .any(|option| seen.contains(*option))
        {
            return Err("--update-env-lock cannot be combined with run or suite options".into());
        }
        if result.backend.is_none() {
            return Err("--update-env-lock requires --backend mlx or cuda".into());
        }
    } else {
        if result.backend.is_some() {
            return Err("--backend requires --update-env-lock".into());
        }
        if result.config.is_none() {
            return Err("--config is required".into());
        }
    }
    Ok(Some(result))
}

async fn execute(arguments: Arguments) -> Result<i32, Box<dyn std::error::Error>> {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    tokio::select! {
        result = execute_inner(arguments) => result,
        signal = async {
            tokio::select! {
                signal = tokio::signal::ctrl_c() => signal,
                _ = terminate.recv() => Ok(()),
            }
        } => {
            signal?;
            eprintln!("Operation interrupted; managed process groups have been stopped. Partial artifacts and setup logs are retained.");
            Ok(2)
        }
    }
}

async fn execute_inner(arguments: Arguments) -> Result<i32, Box<dyn std::error::Error>> {
    let color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    if let Some(path) = arguments.report {
        let path = std::fs::canonicalize(path)?;
        let report: Report = serde_json::from_slice(&std::fs::read(&path)?)?;
        let view = ReportView::new(&report, path.parent().expect("report file has a parent"));
        let summary = view
            .terminal(arguments.case.as_deref(), color)
            .map_err(std::io::Error::other)?;
        view.write_html()?;
        print!("{summary}");
        return Ok(report.exit_code());
    }
    if arguments.update_env_lock {
        let backend = arguments.backend.unwrap();
        let name = match backend {
            Backend::Mlx => "mlx",
            Backend::Cuda => "cuda",
            Backend::Auto => unreachable!(),
        };
        let repo = environment::source_root(None).map_err(std::io::Error::other)?;
        let log = repo.join(format!("rust/target/parity-env-lock-{name}.log"));
        std::fs::create_dir_all(log.parent().unwrap())?;
        let path = environment::update_lock(&repo, backend, &log, Duration::from_secs(1800))
            .await
            .map_err(std::io::Error::other)?;
        println!("Lock: {}", path.display());
        return Ok(0);
    }
    let config: RunConfig = serde_json::from_slice(&std::fs::read(arguments.config.unwrap())?)?;
    let external = arguments
        .suite_file
        .map(std::fs::read_to_string)
        .transpose()?;
    let plan = native_generate::load_plan(
        external.as_deref().unwrap_or(native_generate::DEFAULT_SPEC),
        &config,
    )
    .map_err(std::io::Error::other)?;
    // Preserve the original single-profile review format and artifact layout.
    let single = config.profiles.is_empty()
        && plan.profiles.len() == 1
        && plan.profiles[0].profile.id == "default";
    if arguments.describe {
        let value = if single {
            serde_json::to_value(describe(&config, &plan.profiles[0].suite)?)?
        } else {
            serde_json::to_value(describe_plan(&config, &plan)?)?
        };
        println!("{}", serde_json::to_string_pretty(&value)?);
        return Ok(0);
    }
    let report = if single {
        run(&config, &plan.profiles[0].suite, &plan.profiles[0].policy).await?
    } else {
        run_plan(&config, &plan).await?
    };
    print!(
        "{}",
        ReportView::new(&report, &report.directory)
            .terminal(None, color)
            .map_err(std::io::Error::other)?
    );
    Ok(report.exit_code())
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sglang_parity=info".into()),
        )
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal())
        .with_target(false)
        .init();
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
            vec!["--report", "report.json", "--describe"],
            vec!["--report", "report.json", "--config", "run.json"],
            vec![
                "--report",
                "report.json",
                "--update-env-lock",
                "--backend",
                "mlx",
            ],
            vec!["--config", "run.json", "--case", "one"],
            vec!["--report"],
            vec!["--cases", "cases.json"],
            vec!["--config", "one", "--config", "two"],
            vec!["--config", "one", "--suite", "grpc"],
            vec!["--update-env-lock"],
            vec!["--update-env-lock", "--backend", "auto"],
            vec!["--update-env-lock", "--backend", "mlx", "--describe"],
            vec!["--update-env-lock", "--backend", "cuda", "--config", "one"],
            vec![
                "--update-env-lock",
                "--backend",
                "mlx",
                "--suite",
                "native_generate",
            ],
            vec![
                "--update-env-lock",
                "--backend",
                "mlx",
                "--suite-file",
                "one",
            ],
            vec!["--config", "one", "--backend", "mlx"],
        ] {
            assert!(parse(args.into_iter().map(String::from)).is_err());
        }
        for (name, backend) in [("mlx", Backend::Mlx), ("cuda", Backend::Cuda)] {
            let arguments = parse(
                ["--update-env-lock", "--backend", name]
                    .into_iter()
                    .map(String::from),
            )
            .unwrap()
            .unwrap();
            assert!(arguments.update_env_lock);
            assert_eq!(arguments.backend, Some(backend));
            assert!(arguments.config.is_none());
        }
    }
}
