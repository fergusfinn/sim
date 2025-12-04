# Closed-Loop Workload Pattern

## Overview

The closed-loop pattern simulates the workload model used in the [inference-arithmetic blog post](https://blog.doubleword.ai/resources/inference-arithmetic) and InferenceMAX benchmarks:

**N concurrent users, each immediately sending a new request when their previous one completes.**

This maintains constant concurrency and matches real-world inference benchmark setups better than Poisson arrivals.

## How It Works

1. **Initial State**: N users each send a request at time 0
2. **Steady State**: When a request completes, that "user slot" immediately generates a new request
3. **Termination**: Stops after generating `num_requests` total requests

## Configuration

### TOML Config

```toml
[workload]
arrival_pattern = "closed_loop"
arrival_rate = 1.0                  # Not used, but required by config schema
num_concurrent_users = 64           # Number of concurrent users
num_requests = 640                  # Total requests (e.g., 10 per user)
seed = 42

[workload.input_len_dist]
type = "fixed"
value = 1024

[workload.output_len_dist]
type = "fixed"
value = 1024
```

### Web UI

In the web interface:
1. Open the "Workload" section
2. Select "Closed Loop (N concurrent users)" from arrival pattern dropdown
3. Set "Concurrent Users" field (default: 64)
4. Set "Number of Requests" for total requests to generate

## Example Configs

### Blog Post Default (H100 TP=2, 64 concurrent users)

See `test_blog.toml` for the full configuration matching the blog post's benchmark parameters:
- H100 TP=2, FP8 (3.958 TFLOPS)
- Llama-3-70B
- ISL=1024, OSL=1024
- 64 concurrent users

Run with:
```bash
cargo build --release
./target/release/sim --config test_blog.toml
```

### Debugging (4 concurrent users, 12 requests)

For quick testing:
```bash
./target/release/sim --config /tmp/test_closed_loop_simple.toml
```

## Comparison to Other Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **closed_loop** | N concurrent users, constant concurrency | Benchmark calibration, matches blog post |
| poisson | Random arrivals following Poisson process | Realistic traffic modeling |
| batched | All requests arrive at once | Batch processing scenarios |

## Metrics Interpretation

With closed-loop workloads:

- **Throughput**: Limited by `num_concurrent_users` and system capacity
- **Latency**: Includes queueing effects from maintaining constant load
- **Utilization**: Should be high (70-100%) as system stays saturated

This pattern is ideal for comparing against theoretical hardware limits from the blog post.
