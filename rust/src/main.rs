// src/main.rs
use clap::Parser;
use clap::ValueEnum;
// declare child modules
mod router;
mod server;
mod tree;

use crate::router::PolicyConfig;

#[derive(Debug, Clone, ValueEnum)]
pub enum PolicyType {
    Random,
    RoundRobin,
    ApproxTree,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(
        long,
        default_value = "127.0.0.1",
        help = "Host address to bind the server to"
    )]
    host: String,

    #[arg(long, default_value_t = 3001, help = "Port number to listen on")]
    port: u16,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Comma-separated list of worker URLs to distribute requests to"
    )]
    worker_urls: Vec<String>,

    #[arg(
        long,
        default_value_t = PolicyType::RoundRobin,
        value_enum,
        help = "Load balancing policy to use: random, round_robin, or approx_tree"
    )]
    policy: PolicyType,

    #[arg(
        long,
        requires = "policy",
        required_if_eq("policy", "approx_tree"),
        help = "Path to the tokenizer file, required when using approx_tree policy"
    )]
    tokenizer_path: Option<String>,

    #[arg(
        long,
        default_value = "0.50",
        requires = "policy",
        required_if_eq("policy", "approx_tree"),
        help = "Cache threshold (0.0-1.0) for approx_tree routing. Routes to cached worker if match rate exceeds threshold, otherwise routes to shortest queue worker"
    )]
    cache_threshold: Option<f32>,
}

impl Args {
    fn get_policy_config(&self) -> PolicyConfig {
        match self.policy {
            PolicyType::Random => PolicyConfig::RandomConfig,
            PolicyType::RoundRobin => PolicyConfig::RoundRobinConfig,
            PolicyType::ApproxTree => PolicyConfig::ApproxTreeConfig {
                tokenizer_path: self
                    .tokenizer_path
                    .clone()
                    .expect("tokenizer_path is required for approx_tree policy"),
                cache_threshold: self
                    .cache_threshold
                    .expect("cache_threshold is required for approx_tree policy"),
            },
        }
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let policy_config = args.get_policy_config();
    server::startup(args.host, args.port, args.worker_urls, policy_config).await
}
