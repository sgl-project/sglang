// src/main.rs
use clap::Parser;
use clap::builder::PossibleValuesParser;
// declare child modules
mod server;
mod router;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value_t = 3001)]
    port: u16,

    #[arg(long, value_delimiter = ',')]
    worker_urls: Vec<String>,

    #[arg(long, default_value = "round_robin", value_parser = PossibleValuesParser::new(&["round_robin", "random"]))]
    policy: String,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Args::parse();
    server::startup(args.host, args.port, args.worker_urls, args.policy).await
}