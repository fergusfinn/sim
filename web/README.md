# LLM Inference Simulator - Web UI

Interactive web interface for the LLM Inference Simulator, compiled to WebAssembly for in-browser execution.

## Features

- **Interactive Configuration**: Adjust all simulation parameters in real-time
  - Hardware specs (H100, A100, etc.)
  - Model configuration (Llama-3-70B, etc.)
  - Scheduler settings (FCFS, Priority, chunked prefill)
  - Workload patterns (Poisson arrivals, lognormal distributions)

- **Comprehensive Metrics**: View detailed performance metrics
  - Latency: TTFT, E2E, per-token (mean, p50, p90, p99)
  - Throughput: Input/output tokens per second, requests per second
  - Resource utilization: KV cache, FLOPS, bandwidth
  - Preemption statistics

- **Live Visualizations**: Interactive charts powered by Chart.js
  - Queue dynamics (running vs waiting requests)
  - KV cache utilization over time

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Rust toolchain with wasm-pack (if rebuilding WASM)

### Installation

\`\`\`bash
# Install dependencies
npm install

# Start development server
npm run dev
\`\`\`

The application will be available at http://localhost:5173/

### Building for Production

\`\`\`bash
# Build the web app
npm run build

# Preview production build
npm run preview
\`\`\`

## Rebuilding the WASM Module

If you make changes to the Rust simulator code:

\`\`\`bash
# From the parent directory (sim/)
wasm-pack build --target web --no-default-features

# Then reinstall the package in web/
cd web
npm install ../pkg
\`\`\`

## Usage

1. **Configure the simulation**: Adjust hardware, model, scheduler, and workload parameters in the left panel
2. **Run the simulation**: Click "Run Simulation" to execute
3. **View results**: Metrics and charts will appear in the right panel

### Default Configuration

The default configuration simulates:
- **Hardware**: H100 GPU (1513 TFLOPS, 3.35 TB/s, 64GB KV cache)
- **Model**: Llama-3-70B (70B parameters, 80 layers)
- **Workload**: 400 requests at 5 req/s with lognormal input/output distributions
- **Scheduler**: FCFS with chunked prefill enabled

### Example Scenarios

**Low Load Test**:
- Arrival Rate: 1-2 req/s
- Num Requests: 100

**High Load Test**:
- Arrival Rate: 10+ req/s
- Num Requests: 1000

**Memory Pressure Test**:
- Reduce KV Cache Capacity to 32GB
- Increase arrival rate to observe preemptions

## Architecture

- **Frontend**: React + TypeScript + Vite
- **Visualization**: Chart.js via react-chartjs-2
- **Compute**: Rust simulator compiled to WASM via wasm-bindgen
- **Data Flow**: JSON config → WASM → JSON results → React state

## Performance Notes

- Simulations run entirely in the browser (no server required)
- Large simulations (1000+ requests) may take 10-30 seconds
- The simulator uses synchronous execution (consider using Web Workers for future enhancement)

## Troubleshooting

**WASM fails to load**:
- Ensure the \`sim\` package is installed: \`npm install ../pkg\`
- Check browser console for CORS or MIME type errors

**Simulation runs slowly**:
- Reduce \`num_requests\` or increase \`log_interval\`
- Complex workloads with many preemptions are computationally intensive

**Charts not rendering**:
- Ensure Chart.js is installed: \`npm install chart.js react-chartjs-2\`
- Check for console errors related to canvas rendering

## License

Same as parent project.
