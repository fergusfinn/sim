const fs = require('fs');
const path = require('path');

// Load the WASM module
async function test() {
    try {
        // Import the JS bindings
        const { run_simulation } = require('./pkg/sim.js');
        
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
        
        console.log("Running simulation...");
        const configJson = JSON.stringify(config);
        const result = run_simulation(configJson);
        console.log("Success!");
        console.log(JSON.stringify(result, null, 2));
        
    } catch (err) {
        console.error("Error:", err);
        console.error("Stack:", err.stack);
    }
}

test();
