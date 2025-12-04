use clap::Parser;
use sim::{Config, Simulator};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "LLM Inference Simulator", long_about = None)]
struct Args {
    /// Path to the TOML configuration file
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    println!("LLM Inference Simulator");
    println!("Loading configuration from: {:?}\n", args.config);

    // Load configuration
    let config = match Config::from_file(&args.config) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Error loading configuration: {}", e);
            std::process::exit(1);
        }
    };

    println!("Configuration loaded successfully!");
    println!("  Hardware: {}", config.hardware.name);
    println!("  Model: {}", config.model.name);
    println!(
        "  Compute-bound threshold: {} tokens",
        config.hardware.compute_bound_threshold
    );
    println!(
        "  Max batched tokens: {}",
        config.scheduler.max_num_batched_tokens
    );
    if config.scheduler.enable_chunked_prefill {
        println!(
            "  Chunked prefill enabled: {} tokens/chunk",
            config.scheduler.long_prefill_token_threshold
        );
    }
    println!("  Arrival rate: {} req/sec", config.workload.arrival_rate);
    println!(
        "  Number of requests: {}",
        config
            .workload
            .num_requests
            .map(|n| n.to_string())
            .unwrap_or_else(|| "unlimited".to_string())
    );
    println!();

    // Create and run simulator
    let mut simulator = match Simulator::new(config) {
        Ok(sim) => sim,
        Err(e) => {
            eprintln!("Error creating simulator: {}", e);
            std::process::exit(1);
        }
    };

    simulator.run();
}
