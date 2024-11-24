use clap::Parser;
use clap::ValueEnum;

use sglang_router_rs::{router::PolicyConfig, server};

#[derive(Debug, Clone, ValueEnum)]
pub enum PolicyType {
    Random,
    RoundRobin,
    CacheAware,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(
        long,
        default_value = "127.0.0.1",
        help = "Host address to bind the router server to. Default: 127.0.0.1"
    )]
    host: String,

    #[arg(
        long,
        default_value_t = 3001,
        help = "Port number to bind the router server to. Default: 3001"
    )]
    port: u16,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Comma-separated list of worker URLs that will handle the requests. Each URL should include the protocol, host, and port (e.g., http://worker1:8000,http://worker2:8000)"
    )]
    worker_urls: Vec<String>,

    #[arg(
        long,
        default_value_t = PolicyType::CacheAware,
        value_enum,
        help = "Load balancing policy to use for request distribution:\n\
              - random: Randomly select workers\n\
              - round_robin: Distribute requests in round-robin fashion\n\
              - cache_aware: Distribute requests in cache-aware fashion\n"
    )]
    policy: PolicyType,

    #[arg(
        long,
        default_value_t = 0.5,
        requires = "policy",
        required_if_eq("policy", "cache_aware"),
        help = "Cache threshold (0.0-1.0) for cache-aware routing. Routes to cached worker if the match rate exceeds threshold, otherwise routes to the worker with the smallest tree. Default: 0.5"
    )]
    cache_threshold: f32,

    #[arg(
        long,
        default_value_t = 2.0,
        requires = "policy",
        required_if_eq("policy", "cache_aware"),
        help = "Threshold for load imbalance (>= 1.0). Load balancing is used when max_load > min_load * threshold. Default: 2.0 meaning load balancing triggers when any worker has more than double the load of the least loaded worker"
    )]
    imbalance_threshold: f32,

    #[arg(
        long,
        default_value_t = 60,
        requires = "policy",
        required_if_eq("policy", "cache_aware"),
        help = "Interval in seconds between cache eviction operations in cache-aware routing. Default: 60"
    )]
    eviction_interval_secs: u64,

    #[arg(
        long,
        default_value_t = 2usize.pow(24),
        requires = "policy",
        required_if_eq("policy", "cache_aware"),
        help = "Maximum size of the approximation tree for cache-aware routing. Default: 2^24"
    )]
    max_tree_size: usize,
}

impl Args {
    fn get_policy_config(&self) -> PolicyConfig {
        match self.policy {
            PolicyType::Random => PolicyConfig::RandomConfig,
            PolicyType::RoundRobin => PolicyConfig::RoundRobinConfig,
            PolicyType::CacheAware => PolicyConfig::CacheAwareConfig {
                cache_threshold: self.cache_threshold,
                imbalance_threshold: self.imbalance_threshold,
                eviction_interval_secs: self.eviction_interval_secs,
                max_tree_size: self.max_tree_size,
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
