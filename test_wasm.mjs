import { readFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import init, { run_simulation } from './pkg/sim.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Test the WASM module
async function test() {
    try {
        // Initialize WASM with the file buffer
        console.log("Loading WASM file...");
        const wasmPath = join(__dirname, 'pkg', 'sim_bg.wasm');
        const wasmBuffer = await readFile(wasmPath);
        
        console.log("Initializing WASM...");
        await init(wasmBuffer);
        console.log("WASM initialized!\n");
        
        // Test config
        const config = {
            hardware: {
                name: "H100",
                compute_flops: 1.513e15,
                memory_bandwidth: 3.35e12,
                memory_capacity: 85899345920,
                kv_cache_capacity: 68719476736,
                bytes_per_param: 2
            },
            model: {
                name: "Llama-3-70B",
                num_parameters: 70000000000,
                num_layers: 80,
                hidden_dim: 8192,
                num_heads: 64,
                max_seq_len: 8192
            },
            scheduler: {
                max_num_batched_tokens: 8192,
                max_num_seqs: 256,
                policy: "fcfs",
                enable_chunked_prefill: true,
                block_size: 16
            },
            workload: {
                arrival_pattern: "poisson",
                arrival_rate: 5.0,
                num_requests: 10,  // Small number for testing
                seed: 42,
                input_len_dist: {
                    type: "lognormal",
                    mean: 6.9,
                    std_dev: 0.7
                },
                output_len_dist: {
                    type: "lognormal",
                    mean: 5.3,
                    std_dev: 0.8
                }
            },
            simulation: {
                log_interval: 5
            }
        };
        
        console.log("Running simulation with 10 requests...");
        const configJson = JSON.stringify(config);
        const result = run_simulation(configJson);
        console.log("\n✓ Success!");
        console.log("\nMetrics Summary:");
        console.log("  TTFT Mean:", (result.metrics.ttft_mean * 1000).toFixed(2), "ms");
        console.log("  E2E Mean:", (result.metrics.e2e_mean * 1000).toFixed(2), "ms");
        console.log("  Input tokens/s:", result.metrics.input_tokens_per_sec.toFixed(2));
        console.log("  Output tokens/s:", result.metrics.output_tokens_per_sec.toFixed(2));
        console.log("  Requests/s:", result.metrics.requests_per_sec.toFixed(2));
        console.log("  KV Cache Util:", (result.metrics.avg_kv_cache_util * 100).toFixed(1), "%");
        console.log("  FLOPS Util:", (result.metrics.avg_flops_util * 100).toFixed(1), "%");
        console.log("  Preemptions:", result.metrics.total_preemptions);
        console.log("  Completed:", result.metrics.completed_requests, "/", result.metrics.total_requests);
        
    } catch (err) {
        console.error("\n✗ Error:", err.message);
        if (err.stack) {
            console.error("\nStack trace:");
            console.error(err.stack);
        }
    }
}

test();
