use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

#[wasm_bindgen]
pub fn run_simulation(config_json: &str) -> Result<JsValue, JsValue> {
    // Set up better panic messages
    console_error_panic_hook::set_once();

    let mut config: crate::Config = serde_json::from_str(config_json)
        .map_err(|e| JsValue::from_str(&format!("Config parse error: {}", e)))?;

    // Compute derived fields (same as from_file)
    config.hardware.compute_threshold();
    config.model.compute_kv_cache_size(config.hardware.bytes_per_param);

    // Compute KV cache capacity if not explicitly set
    let model_size_bytes = config.model.num_parameters * config.hardware.bytes_per_param as u64;
    config.hardware.compute_kv_cache_capacity(model_size_bytes);

    config.scheduler.set_default_prefill_threshold(config.model.max_seq_len);

    let mut simulator = crate::Simulator::new(config)
        .map_err(|e| JsValue::from_str(&format!("Simulator error: {}", e)))?;

    // Run simulation
    simulator.run();

    // Extract metrics
    let summary = simulator.get_metrics_summary();
    let time_series = simulator.get_time_series_data();
    let input_lengths = simulator.get_input_lengths();
    let output_lengths = simulator.get_output_lengths();
    let total_time = simulator.get_current_time();

    let result = SimulationResult {
        metrics: MetricsData {
            ttft_mean: summary.ttft_mean,
            ttft_p50: summary.ttft_p50,
            ttft_p90: summary.ttft_p90,
            ttft_p99: summary.ttft_p99,
            e2e_mean: summary.e2e_mean,
            e2e_p50: summary.e2e_p50,
            e2e_p90: summary.e2e_p90,
            e2e_p99: summary.e2e_p99,
            per_token_mean: summary.per_token_mean,
            per_token_p50: summary.per_token_p50,
            per_token_p90: summary.per_token_p90,
            per_token_p99: summary.per_token_p99,
            input_tokens_per_sec: summary.input_tokens_per_sec,
            output_tokens_per_sec: summary.output_tokens_per_sec,
            requests_per_sec: summary.requests_per_sec,
            avg_kv_cache_util: summary.avg_kv_cache_util,
            avg_flops_util: summary.avg_flops_util,
            avg_bandwidth_util: summary.avg_bandwidth_util,
            total_preemptions: summary.total_preemptions,
            avg_preemptions_per_request: summary.preemptions_per_request_mean,
            completed_requests: summary.completed_requests,
            total_requests: summary.total_requests,
            total_time,
        },
        time_series: TimeSeriesData {
            times: time_series.iter().map(|p| p.time).collect(),
            arrivals: time_series.iter().map(|p| p.arrivals).collect(),
            running: time_series.iter().map(|p| p.running).collect(),
            waiting: time_series.iter().map(|p| p.waiting).collect(),
            kv_cache_util: time_series.iter().map(|p| p.kv_cache_util).collect(),
            num_prefilling: time_series.iter().map(|p| p.num_prefilling).collect(),
            num_decoding: time_series.iter().map(|p| p.num_decoding).collect(),
            prefill_tokens: time_series.iter().map(|p| p.prefill_tokens).collect(),
            decode_tokens: time_series.iter().map(|p| p.decode_tokens).collect(),
        },
        distributions: DistributionData {
            input_lengths: input_lengths.to_vec(),
            output_lengths: output_lengths.to_vec(),
        },
    };

    Ok(serde_wasm_bindgen::to_value(&result)?)
}

#[derive(Serialize, Deserialize)]
struct SimulationResult {
    metrics: MetricsData,
    time_series: TimeSeriesData,
    distributions: DistributionData,
}

#[derive(Serialize, Deserialize)]
struct MetricsData {
    ttft_mean: f64,
    ttft_p50: f64,
    ttft_p90: f64,
    ttft_p99: f64,
    e2e_mean: f64,
    e2e_p50: f64,
    e2e_p90: f64,
    e2e_p99: f64,
    per_token_mean: f64,
    per_token_p50: f64,
    per_token_p90: f64,
    per_token_p99: f64,
    input_tokens_per_sec: f64,
    output_tokens_per_sec: f64,
    requests_per_sec: f64,
    avg_kv_cache_util: f64,
    avg_flops_util: f64,
    avg_bandwidth_util: f64,
    total_preemptions: u64,
    avg_preemptions_per_request: f64,
    completed_requests: u64,
    total_requests: u64,
    total_time: f64,
}

#[derive(Serialize, Deserialize)]
struct TimeSeriesData {
    times: Vec<f64>,
    arrivals: Vec<u64>,
    running: Vec<usize>,
    waiting: Vec<usize>,
    kv_cache_util: Vec<f64>,
    num_prefilling: Vec<usize>,
    num_decoding: Vec<usize>,
    prefill_tokens: Vec<u32>,
    decode_tokens: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
struct DistributionData {
    input_lengths: Vec<u32>,
    output_lengths: Vec<u32>,
}
