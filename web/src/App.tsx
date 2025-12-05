import { useState, useEffect, useRef, useMemo } from 'react'
import { Line, Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import './App.css'

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

// Import WASM module (for type definitions)
import type { run_simulation } from 'sim'

interface SimulationConfig {
  hardware: {
    name: string
    compute_flops: number
    memory_bandwidth: number
    memory_capacity: number
    kv_cache_capacity: number
    gpu_memory_utilization: number
    bytes_per_param: number
    tensor_parallel: number
    vram_per_gpu: number
  }
  model: {
    name: string
    num_parameters: number
    num_active_parameters?: number
    num_layers: number
    hidden_dim: number
    num_heads: number
    num_kv_heads?: number
    max_seq_len: number
    sliding_window?: number
    num_sliding_layers?: number
  }
  scheduler: {
    max_num_batched_tokens: number
    max_num_seqs: number
    policy: string
    enable_chunked_prefill: boolean
    block_size: number
  }
  workload: {
    arrival_pattern: string
    arrival_rate: number
    num_requests: number
    num_concurrent_users?: number
    seed: number
    input_len_dist: {
      type: string
      value?: number
      mean?: number
      std_dev?: number
      min?: number
      max?: number
    }
    output_len_dist: {
      type: string
      value?: number
      mean?: number
      std_dev?: number
      min?: number
      max?: number
    }
  }
  simulation: {
    log_interval: number
  }
}

interface SimulationResults {
  metrics: {
    ttft_mean: number
    ttft_p50: number
    ttft_p90: number
    ttft_p99: number
    e2e_mean: number
    e2e_p50: number
    e2e_p90: number
    e2e_p99: number
    per_token_mean: number
    per_token_p50: number
    per_token_p90: number
    per_token_p99: number
    input_tokens_per_sec: number
    input_tokens_per_sec_p50: number
    input_tokens_per_sec_p90: number
    input_tokens_per_sec_p99: number
    output_tokens_per_sec: number
    output_tokens_per_sec_p50: number
    output_tokens_per_sec_p90: number
    output_tokens_per_sec_p99: number
    requests_per_sec: number
    requests_per_sec_p50: number
    requests_per_sec_p90: number
    requests_per_sec_p99: number
    avg_kv_cache_util: number
    avg_flops_util: number
    avg_bandwidth_util: number
    total_preemptions: number
    avg_preemptions_per_request: number
    completed_requests: number
    total_requests: number
    total_time: number
  }
  time_series: {
    times: number[]
    arrivals: number[]
    running: number[]
    waiting: number[]
    kv_cache_util: number[]
    num_prefilling: number[]
    num_decoding: number[]
    prefill_tokens: number[]
    decode_tokens: number[]
  }
  distributions: {
    input_lengths: number[]
    output_lengths: number[]
  }
  latency_samples: {
    ttft_samples: number[]
    e2e_samples: number[]
    tpot_samples: number[]
    ttft_timestamps: number[]
    e2e_timestamps: number[]
    tpot_timestamps: number[]
  }
}

// FLOPS values per bitwidth (from blog post)
const HARDWARE_SPECS: Record<string, {
  name: string
  memory_bandwidth: number
  vram_per_gpu: number
  compute_fp4?: number  // TFLOPS (optional, not all GPUs support)
  compute_fp8: number   // TFLOPS
  compute_fp16: number  // TFLOPS
}> = {
  'MI355X': {
    name: "AMD MI355X",
    memory_bandwidth: 8.0e12,
    vram_per_gpu: 268435456000, // 250GB (est)
    compute_fp4: 10066.4,
    compute_fp8: 5033.2,
    compute_fp16: 2516.6,
  },
  'B200': {
    name: "NVIDIA B200",
    memory_bandwidth: 8.0e12,
    vram_per_gpu: 206158430208, // 192GB
    compute_fp4: 9000,
    compute_fp8: 4500,
    compute_fp16: 2250,
  },
  'MI325X': {
    name: "AMD MI325X",
    memory_bandwidth: 6.0e12,
    vram_per_gpu: 268435456000, // 250GB (est)
    compute_fp4: 5229.8,
    compute_fp8: 2614.9,
    compute_fp16: 1307.4,
  },
  'MI300X': {
    name: "AMD MI300X",
    memory_bandwidth: 5.3e12,
    vram_per_gpu: 206158430208, // 192GB
    compute_fp4: 5229.8,
    compute_fp8: 2614.9,
    compute_fp16: 1307.4,
  },
  'H200': {
    name: "NVIDIA H200",
    memory_bandwidth: 4.8e12,
    vram_per_gpu: 154618822656, // 144GB
    compute_fp8: 1979,
    compute_fp16: 989.5,
  },
  'H100': {
    name: "NVIDIA H100",
    memory_bandwidth: 3.35e12,
    vram_per_gpu: 85899345920, // 80GB
    compute_fp8: 1979,
    compute_fp16: 989.5,
  },
}

const HARDWARE_PRESETS: Record<string, Omit<SimulationConfig['hardware'], 'kv_cache_capacity' | 'memory_capacity'>> = {
  'MI355X': {
    name: "MI355X",
    compute_flops: 5033.2e12, // FP8 default
    memory_bandwidth: 8.0e12,
    vram_per_gpu: 268435456000, // 250GB
    gpu_memory_utilization: 0.9,
    bytes_per_param: 1,
    tensor_parallel: 1,
  },
  'B200': {
    name: "B200",
    compute_flops: 4500e12, // FP8 default
    memory_bandwidth: 8.0e12,
    vram_per_gpu: 206158430208, // 192GB
    gpu_memory_utilization: 0.9,
    bytes_per_param: 1,
    tensor_parallel: 1,
  },
  'MI325X': {
    name: "MI325X",
    compute_flops: 2614.9e12, // FP8 default
    memory_bandwidth: 6.0e12,
    vram_per_gpu: 268435456000, // 250GB
    gpu_memory_utilization: 0.9,
    bytes_per_param: 1,
    tensor_parallel: 1,
  },
  'MI300X': {
    name: "MI300X",
    compute_flops: 2614.9e12, // FP8 default
    memory_bandwidth: 5.3e12,
    vram_per_gpu: 206158430208, // 192GB
    gpu_memory_utilization: 0.9,
    bytes_per_param: 1,
    tensor_parallel: 1,
  },
  'H200': {
    name: "H200",
    compute_flops: 1979e12, // FP8 default
    memory_bandwidth: 4.8e12,
    vram_per_gpu: 154618822656, // 144GB
    gpu_memory_utilization: 0.9,
    bytes_per_param: 1,
    tensor_parallel: 1,
  },
  'H100': {
    name: "H100",
    compute_flops: 1979e12, // FP8 default
    memory_bandwidth: 3.35e12,
    vram_per_gpu: 85899345920, // 80GB
    gpu_memory_utilization: 0.9,
    bytes_per_param: 1,
    tensor_parallel: 1,
  },
}

const MODEL_PRESETS: Record<string, SimulationConfig['model']> = {
  'Llama-3-70B': {
    name: "Llama-3-70B",
    num_parameters: 70000000000,
    num_layers: 80,
    hidden_dim: 8192,
    num_heads: 64,
    max_seq_len: 8192,
  },
  'Llama-3-8B': {
    name: "Llama-3-8B",
    num_parameters: 8000000000,
    num_layers: 32,
    hidden_dim: 4096,
    num_heads: 32,
    max_seq_len: 8192,
  },
  'Llama-3.1-405B': {
    name: "Llama-3.1-405B",
    num_parameters: 405000000000,
    num_layers: 126,
    hidden_dim: 16384,
    num_heads: 128,
    max_seq_len: 8192,
  },
  'Qwen3-30B-A3B': {
    name: "Qwen3-30B-A3B",
    num_parameters: 30500000000,
    num_active_parameters: 3340000000,
    num_layers: 48,
    hidden_dim: 2048,
    num_heads: 32,
    num_kv_heads: 4,
    max_seq_len: 32768,
  },
  'GPT-OSS-120B': {
    name: "GPT-OSS-120B (MoE, 4/128 experts)",
    num_parameters: 120000000000, // Total params with all 128 experts
    num_active_parameters: 6000000000, // Active params with 4 experts per token
    num_layers: 36,
    hidden_dim: 2880,
    num_heads: 64,
    num_kv_heads: 8,
    max_seq_len: 131072,
    sliding_window: 128, // Sliding window size for 18 layers
    num_sliding_layers: 18, // Half the layers use sliding window
  },
}

// Helper function to compute effective hardware config with TP
// Note: kv_cache_capacity is set to 0 and will be calculated by backend using gpu_memory_utilization
function computeEffectiveHardware(
  hardware: Omit<SimulationConfig['hardware'], 'kv_cache_capacity' | 'memory_capacity'>,
  _model: SimulationConfig['model']
): SimulationConfig['hardware'] {
  const tp = hardware.tensor_parallel
  const totalVRAM = hardware.vram_per_gpu * tp

  return {
    ...hardware,
    compute_flops: hardware.compute_flops * tp,
    memory_bandwidth: hardware.memory_bandwidth * tp,
    memory_capacity: totalVRAM,
    kv_cache_capacity: 0, // Backend will calculate using gpu_memory_utilization
  }
}

// Create base config with TP=2, FP8
const DEFAULT_HARDWARE_BASE = {
  name: "H100",
  compute_flops: 1979e12, // FP8
  memory_bandwidth: 3.35e12,
  vram_per_gpu: 85899345920,
  gpu_memory_utilization: 0.9,
  bytes_per_param: 1,
  tensor_parallel: 2,
}

const DEFAULT_MODEL = {
  name: "Llama-3-70B",
  num_parameters: 70000000000,
  num_layers: 80,
  hidden_dim: 8192,
  num_heads: 64,
  num_kv_heads: 8,
  max_seq_len: 8192,
}

const DEFAULT_CONFIG: SimulationConfig = {
  hardware: computeEffectiveHardware(DEFAULT_HARDWARE_BASE, DEFAULT_MODEL),
  model: DEFAULT_MODEL,
  scheduler: {
    max_num_batched_tokens: 8192,
    max_num_seqs: 256,
    policy: "fcfs",
    enable_chunked_prefill: true,
    block_size: 16,
  },
  workload: {
    arrival_pattern: "closed_loop",
    arrival_rate: 5.0,
    num_requests: 400,
    num_concurrent_users: 64,
    seed: 42,
    input_len_dist: {
      type: "lognormal",
      mean: 6.9,
      std_dev: 0.7,
    },
    output_len_dist: {
      type: "lognormal",
      mean: 5.3,
      std_dev: 0.8,
    },
  },
  simulation: {
    log_interval: 5,
  },
}

// Helper function to format latency values intelligently
function formatLatency(ms: number): { display: string, full: string, unit: string } {
  if (ms >= 1000) {
    const seconds = ms / 1000
    return {
      display: seconds.toFixed(1),
      full: `${ms.toFixed(2)} ms (${seconds.toFixed(3)} s)`,
      unit: 's'
    }
  } else {
    return {
      display: ms.toFixed(0),
      full: `${ms.toFixed(2)} ms`,
      unit: 'ms'
    }
  }
}

// Helper function to format bytes intelligently
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'

  const absBytes = Math.abs(bytes)
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const k = 1024
  const i = Math.floor(Math.log(absBytes) / Math.log(k))
  const value = bytes / Math.pow(k, i)

  // Use appropriate decimal places based on size
  const decimals = i >= 3 ? 1 : 0 // GB and above: 1 decimal, MB and below: 0 decimals

  return `${value.toFixed(decimals)} ${units[i]}`
}

// Helper function to parse formatted bytes back to number
function parseBytes(formatted: string): number {
  const match = formatted.trim().match(/^(-?[\d.]+)\s*([KMGT]?B)$/i)
  if (!match) return parseFloat(formatted)

  const value = parseFloat(match[1])
  const unit = match[2].toUpperCase()

  const units: Record<string, number> = {
    'B': 1,
    'KB': 1024,
    'MB': 1024 * 1024,
    'GB': 1024 * 1024 * 1024,
    'TB': 1024 * 1024 * 1024 * 1024,
  }

  return value * (units[unit] || 1)
}

// Helper function to create histogram data from values
function createHistogram(values: number[], numBins: number = 30): { bins: string[], counts: number[] } {
  if (values.length === 0) return { bins: [], counts: [] }

  const min = Math.min(...values)
  const max = Math.max(...values)

  // Handle degenerate case where all values are the same (fixed distribution)
  if (min === max) {
    return {
      bins: [`${Math.round(min)}`],
      counts: [values.length]
    }
  }

  const binSize = (max - min) / numBins

  const counts = new Array(numBins).fill(0)
  values.forEach(val => {
    const binIndex = Math.min(Math.floor((val - min) / binSize), numBins - 1)
    counts[binIndex]++
  })

  const bins = Array.from({ length: numBins }, (_, i) => {
    const binStart = min + i * binSize
    const binEnd = min + (i + 1) * binSize
    return `${Math.round(binStart)}-${Math.round(binEnd)}`
  })

  return { bins, counts }
}

// Generate TOML config string from SimulationConfig
function generateTOML(config: SimulationConfig): string {
  const hardware = config.hardware
  const model = config.model
  const scheduler = config.scheduler
  const workload = config.workload

  // Format input/output length distribution
  const formatLengthDist = (dist: any) => {
    if (dist.type === 'fixed') {
      return `type = "fixed"\nvalue = ${dist.value}`
    } else if (dist.type === 'uniform') {
      return `type = "uniform"\nmin = ${dist.min}\nmax = ${dist.max}`
    } else if (dist.type === 'lognormal') {
      return `type = "lognormal"\nmean = ${dist.mean}\nstd_dev = ${dist.std_dev}`
    } else if (dist.type === 'normal') {
      return `type = "normal"\nmean = ${dist.mean}\nstd_dev = ${dist.std_dev}`
    }
    return ''
  }

  return `# Configuration generated from web UI
# This config reproduces the simulation experiment

[hardware]
name = "${hardware.name}"
compute_flops = ${hardware.compute_flops.toExponential(3)}
memory_bandwidth = ${hardware.memory_bandwidth.toExponential(3)}
memory_capacity = ${hardware.memory_capacity}
gpu_memory_utilization = ${hardware.gpu_memory_utilization}
bytes_per_param = ${hardware.bytes_per_param}

[model]
name = "${model.name}"
num_parameters = ${model.num_parameters}${model.num_active_parameters ? `\nnum_active_parameters = ${model.num_active_parameters}` : ''}
num_layers = ${model.num_layers}
hidden_dim = ${model.hidden_dim}
num_heads = ${model.num_heads}${model.num_kv_heads ? `\nnum_kv_heads = ${model.num_kv_heads}` : ''}
max_seq_len = ${model.max_seq_len}${model.sliding_window ? `\nsliding_window = ${model.sliding_window}` : ''}${model.num_sliding_layers ? `\nnum_sliding_layers = ${model.num_sliding_layers}` : ''}

[scheduler]
max_num_batched_tokens = ${scheduler.max_num_batched_tokens}
max_num_seqs = ${scheduler.max_num_seqs}
policy = "${scheduler.policy}"
enable_chunked_prefill = ${scheduler.enable_chunked_prefill}
block_size = ${scheduler.block_size}

[workload]
arrival_pattern = "${workload.arrival_pattern}"
arrival_rate = ${workload.arrival_rate}${workload.num_concurrent_users ? `\nnum_concurrent_users = ${workload.num_concurrent_users}` : ''}
num_requests = ${workload.num_requests}
seed = ${workload.seed}

[workload.input_len_dist]
${formatLengthDist(workload.input_len_dist)}

[workload.output_len_dist]
${formatLengthDist(workload.output_len_dist)}

[simulation]
log_interval = 10
`
}

function App() {
  const [config, setConfig] = useState<SimulationConfig>(DEFAULT_CONFIG)
  const [results, setResults] = useState<SimulationResults | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [workerReady, setWorkerReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedHardware, setSelectedHardware] = useState<string>('H100')
  const [selectedModel, setSelectedModel] = useState<string>('Llama-3-70B')
  const [activeTab, setActiveTab] = useState<'metrics' | 'charts' | 'config'>('metrics')
  const [chartsSubTab, setChartsSubTab] = useState<'timeseries' | 'distributions'>('timeseries')
  const [expandedSections, setExpandedSections] = useState({
    hardware: false,
    model: false,
    scheduler: false,
    workload: false,
  })
  const [progressInfo, setProgressInfo] = useState<string>('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [simulationSpeed, setSimulationSpeed] = useState<number>(2) // 0=1x, 1=10x, 2=100x, 3=MAX

  // Create Web Worker on mount
  const workerRef = useRef<Worker | null>(null)
  const lastUpdateRef = useRef<number>(0)
  const updateThrottleMs = 50 // Update UI at most every 50ms (20 times/sec)

  // Histogram state: use exponential bins with lazy bin creation
  // Store bins as a Map from binIndex to count, allows dynamic expansion
  type LazyHistogram = {
    base: number  // exponential base (e.g., 1.5)
    binCounts: Map<number, number>  // binIndex -> count
    minBinIndex: number
    maxBinIndex: number
  }

  const [histograms, setHistograms] = useState<{
    ttft: LazyHistogram | null
    e2e: LazyHistogram | null
    tpot: LazyHistogram | null
    input_length: LazyHistogram | null
    output_length: LazyHistogram | null
  }>({ ttft: null, e2e: null, tpot: null, input_length: null, output_length: null })

  // Helper to calculate exponential bin index for a value
  const getExpBinIndex = (value: number, base: number): number => {
    // Handle values close to zero (use minimum of 0.1ms)
    if (value < 0.1) return Math.floor(Math.log(0.1) / Math.log(base))
    return Math.floor(Math.log(value) / Math.log(base))
  }

  // Helper function to update histogram with new samples using exponential bins
  const updateHistogramBins = (
    currentHist: LazyHistogram | null,
    newSamples: number[],
    metricType: 'ttft' | 'e2e' | 'tpot' | 'input_length' | 'output_length'
  ): LazyHistogram => {
    if (newSamples.length === 0 && currentHist) {
      return currentHist
    }
    if (newSamples.length === 0) {
      // Empty histogram
      return { base: 1.5, binCounts: new Map(), minBinIndex: 0, maxBinIndex: 0 }
    }

    // If no existing histogram, initialize
    if (!currentHist) {
      // Use exponential base of 1.5 (bins increase by 50% each time)
      // This gives good resolution across wide dynamic range
      const base = 1.5

      const binCounts = new Map<number, number>()
      let minBinIndex = Infinity
      let maxBinIndex = -Infinity

      for (const value of newSamples) {
        const binIndex = getExpBinIndex(value, base)
        binCounts.set(binIndex, (binCounts.get(binIndex) || 0) + 1)
        minBinIndex = Math.min(minBinIndex, binIndex)
        maxBinIndex = Math.max(maxBinIndex, binIndex)
      }

      return { base, binCounts, minBinIndex, maxBinIndex }
    }

    // Update existing histogram - lazily create bins as needed
    const { base, binCounts, minBinIndex, maxBinIndex } = currentHist
    const newBinCounts = new Map(binCounts)
    let newMinBinIndex = minBinIndex
    let newMaxBinIndex = maxBinIndex

    for (const value of newSamples) {
      const binIndex = getExpBinIndex(value, base)
      newBinCounts.set(binIndex, (newBinCounts.get(binIndex) || 0) + 1)
      newMinBinIndex = Math.min(newMinBinIndex, binIndex)
      newMaxBinIndex = Math.max(newMaxBinIndex, binIndex)
    }

    return {
      base,
      binCounts: newBinCounts,
      minBinIndex: newMinBinIndex,
      maxBinIndex: newMaxBinIndex
    }
  }

  // Helper to convert LazyHistogram to bins/counts arrays for rendering
  const lazyHistogramToArrays = (hist: LazyHistogram | null): { bins: string[], counts: number[] } => {
    if (!hist || hist.binCounts.size === 0) {
      return { bins: [], counts: [] }
    }

    const { base, binCounts, minBinIndex, maxBinIndex } = hist
    const bins: string[] = []
    const counts: number[] = []

    for (let i = minBinIndex; i <= maxBinIndex; i++) {
      const binStart = Math.pow(base, i)
      const binEnd = Math.pow(base, i + 1)
      bins.push(`${Math.round(binStart)}-${Math.round(binEnd)}`)
      counts.push(binCounts.get(i) || 0)
    }

    return { bins, counts }
  }

  // Convert lazy histograms to arrays for rendering
  const ttftHistogram = lazyHistogramToArrays(histograms.ttft)
  const e2eHistogram = lazyHistogramToArrays(histograms.e2e)
  const tpotHistogram = lazyHistogramToArrays(histograms.tpot)
  const inputLengthHistogram = lazyHistogramToArrays(histograms.input_length)
  const outputLengthHistogram = lazyHistogramToArrays(histograms.output_length)

  useEffect(() => {
    // Create worker
    const worker = new Worker(new URL('./simulator.worker.ts', import.meta.url), {
      type: 'module'
    })

    // Handle messages from worker
    worker.onmessage = (e) => {
      const { type, data, error: workerError } = e.data

      if (type === 'initialized') {
        setWorkerReady(true)
      } else if (type === 'progress') {
        const now = Date.now()
        setIsStreaming(true)

        // Always update the text progress info
        setProgressInfo(
          `Time: ${data.current_time.toFixed(2)}s | ` +
          `Completed: ${data.completed_requests}/${data.total_requests} | ` +
          `Running: ${data.running} | Waiting: ${data.waiting}`
        )

        // Update histogram bins with new delta samples
        if (data.latency_samples) {
          const { ttft_samples, e2e_samples, tpot_samples } = data.latency_samples

          setHistograms(prev => ({
            ttft: updateHistogramBins(prev.ttft, ttft_samples, 'ttft'),
            e2e: updateHistogramBins(prev.e2e, e2e_samples, 'e2e'),
            tpot: updateHistogramBins(prev.tpot, tpot_samples, 'tpot'),
            input_length: prev.input_length,
            output_length: prev.output_length,
          }))
        }

        // Update distribution histograms with new delta samples
        if (data.distribution_samples) {
          const { input_lengths, output_lengths } = data.distribution_samples

          setHistograms(prev => ({
            ...prev,
            input_length: updateHistogramBins(prev.input_length, input_lengths, 'input_length'),
            output_length: updateHistogramBins(prev.output_length, output_lengths, 'output_length'),
          }))
        }

        // Throttle chart updates to reduce abruptness
        if (now - lastUpdateRef.current >= updateThrottleMs) {
          // Update partial results if time_series, metrics, OR latency_samples available
          // This ensures results exists so charts can render
          if (data.time_series || data.metrics || data.latency_samples) {
            setResults({
              metrics: data.metrics || results?.metrics || {} as any,
              time_series: data.time_series || results?.time_series || {
                times: [], arrivals: [], running: [], waiting: [], kv_cache_util: [],
                num_prefilling: [], num_decoding: [], prefill_tokens: [], decode_tokens: []
              },
              distributions: results?.distributions || { input_lengths: [], output_lengths: [] },
              latency_samples: results?.latency_samples || {
                ttft_samples: [],
                e2e_samples: [],
                tpot_samples: [],
                ttft_timestamps: [],
                e2e_timestamps: [],
                tpot_timestamps: [],
              },
            })

            lastUpdateRef.current = now
          }
        }
      } else if (type === 'complete') {
        // Simulation complete, set final results
        setIsStreaming(false)
        setResults(data)

        // Histograms should already be complete from streaming deltas
        // No need to rebuild them

        setIsRunning(false)
        setProgressInfo('')
        lastUpdateRef.current = 0
      } else if (type === 'error') {
        setError(`Worker error: ${workerError}`)
        setIsRunning(false)
        setProgressInfo('')
        lastUpdateRef.current = 0
      }
    }

    worker.onerror = (err) => {
      setError(`Worker error: ${err.message}`)
      setIsRunning(false)
    }

    workerRef.current = worker

    // Initialize worker
    worker.postMessage({ type: 'init' })

    return () => {
      worker.terminate()
    }
  }, [])

  const runSimulation = async () => {
    if (!workerReady) {
      setError("Worker not initialized yet")
      return
    }

    if (isRunning) {
      return // Already running
    }

    if (!workerRef.current) {
      setError("Worker not available")
      return
    }

    setIsRunning(true)
    setError(null)
    setProgressInfo('Starting simulation...')
    setResults(null) // Clear previous results

    // Clear histogram state
    setHistograms({ ttft: null, e2e: null, tpot: null })

    try {
      // Send config to worker to start simulation
      const configJson = JSON.stringify(config)
      workerRef.current.postMessage({
        type: 'run',
        config: configJson,
        speed: simulationSpeed,
      })
    } catch (err) {
      setError(`Simulation error: ${err}`)
      setIsRunning(false)
      setProgressInfo('')
    }
  }

  const loadHardwarePreset = (hardwareName: string) => {
    const hardwarePreset = HARDWARE_PRESETS[hardwareName]
    if (hardwarePreset) {
      const hardware = computeEffectiveHardware(hardwarePreset, config.model)
      setConfig({ ...config, hardware })
      setSelectedHardware(hardwareName)
      setResults(null)
    }
  }

  const loadModelPreset = (modelName: string) => {
    const model = MODEL_PRESETS[modelName]
    if (model) {
      // Reset kv_cache_capacity to 0 so backend recalculates with new model
      const hardware = { ...config.hardware, kv_cache_capacity: 0 }
      setConfig({ ...config, model, hardware })
      setSelectedModel(modelName)
      setResults(null)
    }
  }

  const resetToDefaults = () => {
    setConfig(DEFAULT_CONFIG)
    setSelectedHardware('H100')
    setSelectedModel('Llama-3-70B')
    setResults(null)
  }

  const updateConfig = (path: string[], value: any) => {
    setConfig((prev) => {
      const newConfig = JSON.parse(JSON.stringify(prev))
      let current = newConfig
      for (let i = 0; i < path.length - 1; i++) {
        current = current[path[i]]
      }
      current[path[path.length - 1]] = value
      return newConfig
    })
  }

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }))
  }

  return (
    <div className="app">
      <header className="header">
        <h1>LLM Inference Simulator</h1>
        <p className="subtitle">High-fidelity simulation of vLLM scheduling and performance</p>
      </header>

      {error && (
        <div className="error">
          <strong>Error:</strong> {error}
        </div>
      )}

      <div className="layout">
        <div className="config-panel">
          <div className="panel-header">
            <h2>Configuration</h2>
            <button onClick={resetToDefaults} className="reset-btn-full">Reset to Defaults</button>
          </div>

          <div className="config-sections">
            <section className={`collapsible ${expandedSections.hardware ? 'expanded' : ''}`}>
              <h3 onClick={() => toggleSection('hardware')}>
                <span className="toggle-icon">{expandedSections.hardware ? '▼' : '▶'}</span>
                Hardware
              </h3>
              {expandedSections.hardware && (
                <div className="section-content">
                  <div className="input-group">
                    <label>Preset</label>
                    <select
                      onChange={(e) => loadHardwarePreset(e.target.value)}
                      value={selectedHardware}
                    >
                      {Object.keys(HARDWARE_PRESETS).map(name => (
                        <option key={name} value={name}>{name}</option>
                      ))}
                    </select>
                  </div>
                  <div className="input-group">
                    <label>Name</label>
                    <input
                      type="text"
                      value={config.hardware.name}
                      onChange={(e) => updateConfig(['hardware', 'name'], e.target.value)}
                    />
                  </div>
                  <div className="input-group">
                    <label>Bitwidth</label>
                    <select
                      value={config.hardware.bytes_per_param}
                      onChange={(e) => {
                        const bytesPerParam = parseFloat(e.target.value)
                        const hardwareName = config.hardware.name

                        // Look up BASE FLOPS based on bitwidth from HARDWARE_SPECS
                        let base_compute_flops = config.hardware.compute_flops / config.hardware.tensor_parallel
                        if (HARDWARE_SPECS[hardwareName]) {
                          const spec = HARDWARE_SPECS[hardwareName]
                          if (bytesPerParam === 0.5 && spec.compute_fp4) {
                            base_compute_flops = spec.compute_fp4 * 1e12 // Convert TFLOPS to FLOPS
                          } else if (bytesPerParam === 1) {
                            base_compute_flops = spec.compute_fp8 * 1e12
                          } else if (bytesPerParam === 2) {
                            base_compute_flops = spec.compute_fp16 * 1e12
                          }
                        }

                        // Apply TP scaling to the base FLOPS
                        const hardware = {
                          ...config.hardware,
                          bytes_per_param: bytesPerParam,
                          compute_flops: base_compute_flops * config.hardware.tensor_parallel,
                          kv_cache_capacity: 0  // Backend will recalculate
                        }
                        setConfig({ ...config, hardware })
                      }}
                    >
                      {HARDWARE_SPECS[config.hardware.name]?.compute_fp4 && (
                        <option value="0.5">FP4 (0.5 bytes/param)</option>
                      )}
                      <option value="1">FP8 (1 byte/param)</option>
                      <option value="2">FP16/BF16 (2 bytes/param)</option>
                    </select>
                  </div>
                  <div className="input-group">
                    <label>Tensor Parallel Degree</label>
                    <select
                      value={config.hardware.tensor_parallel}
                      onChange={(e) => {
                        const newTp = parseInt(e.target.value)
                        const oldTp = config.hardware.tensor_parallel
                        // Un-scale by old TP, then re-scale by new TP
                        const hardware = {
                          ...config.hardware,
                          compute_flops: (config.hardware.compute_flops / oldTp) * newTp,
                          memory_bandwidth: (config.hardware.memory_bandwidth / oldTp) * newTp,
                          memory_capacity: (config.hardware.memory_capacity / oldTp) * newTp,
                          tensor_parallel: newTp,
                          kv_cache_capacity: 0  // Backend will recalculate
                        }
                        setConfig({ ...config, hardware })
                      }}
                    >
                      <option value="1">1</option>
                      <option value="2">2</option>
                      <option value="4">4</option>
                      <option value="8">8</option>
                    </select>
                  </div>
                  <div className="input-group">
                    <label>VRAM per GPU</label>
                    <input
                      type="text"
                      value={formatBytes(config.hardware.vram_per_gpu)}
                      onChange={(e) => {
                        const vramPerGPU = parseBytes(e.target.value)
                        const tp = config.hardware.tensor_parallel
                        // Just update vram_per_gpu and recalculate total
                        const hardware = {
                          ...config.hardware,
                          vram_per_gpu: vramPerGPU,
                          memory_capacity: vramPerGPU * tp,
                          kv_cache_capacity: 0  // Backend will recalculate
                        }
                        setConfig({ ...config, hardware })
                      }}
                    />
                  </div>
                  <div className="input-group">
                    <label>GPU Memory Utilization</label>
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.05"
                      value={config.hardware.gpu_memory_utilization}
                      onChange={(e) => {
                        const gpuMemUtil = parseFloat(e.target.value)
                        // Just update the field directly, don't recompute TP scaling
                        setConfig({
                          ...config,
                          hardware: {
                            ...config.hardware,
                            gpu_memory_utilization: gpuMemUtil,
                            kv_cache_capacity: 0  // Backend will recalculate
                          }
                        })
                      }}
                    />
                    <small style={{ fontSize: '0.85em', color: 'var(--gray-600)', marginTop: '0.25rem' }}>
                      Fraction of GPU memory to use (vLLM default: 0.9)
                    </small>
                  </div>
                  <div className="input-group">
                    <label>Total VRAM</label>
                    <input
                      type="text"
                      value={formatBytes(config.hardware.memory_capacity)}
                      disabled
                      style={{ backgroundColor: 'var(--gray-100)', cursor: 'not-allowed' }}
                    />
                  </div>
                  <div className="input-group">
                    <label>Available KV Cache</label>
                    <input
                      type="text"
                      value={config.hardware.kv_cache_capacity > 0 ? formatBytes(config.hardware.kv_cache_capacity) : 'Calculated by backend'}
                      disabled
                      style={{
                        backgroundColor: 'var(--gray-100)',
                        cursor: 'not-allowed',
                      }}
                    />
                    <small style={{ fontSize: '0.85em', color: 'var(--gray-600)', marginTop: '0.25rem' }}>
                      Calculated as (Total VRAM × GPU Memory Utilization) - Model Size
                    </small>
                  </div>
                </div>
              )}
            </section>

            <section className={`collapsible ${expandedSections.model ? 'expanded' : ''}`}>
              <h3 onClick={() => toggleSection('model')}>
                <span className="toggle-icon">{expandedSections.model ? '▼' : '▶'}</span>
                Model
              </h3>
              {expandedSections.model && (
                <div className="section-content">
                  <div className="input-group">
                    <label>Preset</label>
                    <select
                      onChange={(e) => loadModelPreset(e.target.value)}
                      value={selectedModel}
                    >
                      {Object.keys(MODEL_PRESETS).map(name => (
                        <option key={name} value={name}>{name}</option>
                      ))}
                    </select>
                  </div>
                  <div className="input-group">
                    <label>Name</label>
                    <input
                      type="text"
                      value={config.model.name}
                      onChange={(e) => updateConfig(['model', 'name'], e.target.value)}
                    />
                  </div>
                  <div className="input-group">
                    <label>Parameters</label>
                    <input
                      type="number"
                      value={config.model.num_parameters}
                      onChange={(e) => updateConfig(['model', 'num_parameters'], parseInt(e.target.value))}
                    />
                  </div>
                  <div className="input-group">
                    <label>Layers</label>
                    <input
                      type="number"
                      value={config.model.num_layers}
                      onChange={(e) => updateConfig(['model', 'num_layers'], parseInt(e.target.value))}
                    />
                  </div>
                  <div className="input-group">
                    <label>Hidden Dimension</label>
                    <input
                      type="number"
                      value={config.model.hidden_dim}
                      onChange={(e) => updateConfig(['model', 'hidden_dim'], parseInt(e.target.value))}
                    />
                  </div>
                  <div className="input-group">
                    <label>Sliding Window Size</label>
                    <input
                      type="number"
                      value={config.model.sliding_window || ''}
                      placeholder="None (full attention)"
                      onChange={(e) => {
                        const value = e.target.value === '' ? undefined : parseInt(e.target.value)
                        updateConfig(['model', 'sliding_window'], value)
                      }}
                    />
                    <small style={{ fontSize: '0.85em', color: 'var(--gray-600)', marginTop: '0.25rem' }}>
                      Context window size for sliding attention layers (leave empty for full attention)
                    </small>
                  </div>
                  <div className="input-group">
                    <label>Number of Sliding Layers</label>
                    <input
                      type="number"
                      value={config.model.num_sliding_layers || ''}
                      placeholder="0"
                      onChange={(e) => {
                        const value = e.target.value === '' ? undefined : parseInt(e.target.value)
                        updateConfig(['model', 'num_sliding_layers'], value)
                      }}
                    />
                    <small style={{ fontSize: '0.85em', color: 'var(--gray-600)', marginTop: '0.25rem' }}>
                      Number of layers using sliding window (rest use full attention)
                    </small>
                  </div>
                </div>
              )}
            </section>

            <section className={`collapsible ${expandedSections.scheduler ? 'expanded' : ''}`}>
              <h3 onClick={() => toggleSection('scheduler')}>
                <span className="toggle-icon">{expandedSections.scheduler ? '▼' : '▶'}</span>
                Scheduler
              </h3>
              {expandedSections.scheduler && (
                <div className="section-content">
                  <div className="input-group">
                    <label>Max Batched Tokens</label>
                    <input
                      type="number"
                      value={config.scheduler.max_num_batched_tokens}
                      onChange={(e) => updateConfig(['scheduler', 'max_num_batched_tokens'], parseInt(e.target.value))}
                    />
                  </div>
                  <div className="input-group">
                    <label>Max Sequences</label>
                    <input
                      type="number"
                      value={config.scheduler.max_num_seqs}
                      onChange={(e) => updateConfig(['scheduler', 'max_num_seqs'], parseInt(e.target.value))}
                    />
                  </div>
                  <div className="input-group">
                    <label>Policy</label>
                    <select
                      value={config.scheduler.policy}
                      onChange={(e) => updateConfig(['scheduler', 'policy'], e.target.value)}
                    >
                      <option value="fcfs">FCFS (First Come First Serve)</option>
                      <option value="priority">Priority</option>
                      <option value="sjf">SJF (Shortest Job First)</option>
                    </select>
                  </div>
                  <div className="input-group checkbox">
                    <label>
                      <input
                        type="checkbox"
                        checked={config.scheduler.enable_chunked_prefill}
                        onChange={(e) => updateConfig(['scheduler', 'enable_chunked_prefill'], e.target.checked)}
                      />
                      Enable Chunked Prefill
                    </label>
                  </div>
                </div>
              )}
            </section>

            <section className={`collapsible ${expandedSections.workload ? 'expanded' : ''}`}>
              <h3 onClick={() => toggleSection('workload')}>
                <span className="toggle-icon">{expandedSections.workload ? '▼' : '▶'}</span>
                Workload
              </h3>
              {expandedSections.workload && (
                <div className="section-content">
                  <div className="input-group">
                    <label>Arrival Pattern</label>
                    <select
                      value={config.workload.arrival_pattern}
                      onChange={(e) => updateConfig(['workload', 'arrival_pattern'], e.target.value)}
                    >
                      <option value="poisson">Poisson (continuous arrivals)</option>
                      <option value="closed_loop">Closed Loop (N concurrent users)</option>
                    </select>
                  </div>
                  {config.workload.arrival_pattern === 'poisson' && (
                    <div className="input-group">
                      <label>Arrival Rate (req/s)</label>
                      <input
                        type="number"
                        step="0.1"
                        value={config.workload.arrival_rate}
                        onChange={(e) => updateConfig(['workload', 'arrival_rate'], parseFloat(e.target.value))}
                      />
                    </div>
                  )}
                  {config.workload.arrival_pattern === 'closed_loop' && (
                    <div className="input-group">
                      <label>Concurrent Users</label>
                      <input
                        type="number"
                        value={config.workload.num_concurrent_users || 64}
                        onChange={(e) => updateConfig(['workload', 'num_concurrent_users'], parseInt(e.target.value))}
                      />
                    </div>
                  )}
                  <div className="input-group">
                    <label>Number of Requests</label>
                    <input
                      type="number"
                      value={config.workload.num_requests}
                      onChange={(e) => updateConfig(['workload', 'num_requests'], parseInt(e.target.value))}
                    />
                  </div>
                  <div className="input-group">
                    <label>Input Length Distribution</label>
                    <select
                      value={config.workload.input_len_dist.type}
                      onChange={(e) => {
                        const type = e.target.value
                        if (type === 'fixed') {
                          updateConfig(['workload', 'input_len_dist'], { type, value: 1024 })
                        } else if (type === 'uniform') {
                          updateConfig(['workload', 'input_len_dist'], { type, min: 100, max: 2000 })
                        } else {
                          updateConfig(['workload', 'input_len_dist'], { type, mean: 6.9, std_dev: 0.7 })
                        }
                      }}
                    >
                      <option value="fixed">Fixed</option>
                      <option value="normal">Normal</option>
                      <option value="lognormal">Log-Normal</option>
                      <option value="uniform">Uniform</option>
                    </select>
                  </div>
                  {config.workload.input_len_dist.type === 'fixed' ? (
                    <div className="input-group">
                      <label>Input Length</label>
                      <input
                        type="number"
                        value={config.workload.input_len_dist.value || 1024}
                        onChange={(e) => updateConfig(['workload', 'input_len_dist', 'value'], parseInt(e.target.value))}
                      />
                    </div>
                  ) : config.workload.input_len_dist.type === 'uniform' ? (
                    <>
                      <div className="input-group">
                        <label>Input Length Min</label>
                        <input
                          type="number"
                          value={config.workload.input_len_dist.min || 100}
                          onChange={(e) => updateConfig(['workload', 'input_len_dist', 'min'], parseFloat(e.target.value))}
                        />
                      </div>
                      <div className="input-group">
                        <label>Input Length Max</label>
                        <input
                          type="number"
                          value={config.workload.input_len_dist.max || 2000}
                          onChange={(e) => updateConfig(['workload', 'input_len_dist', 'max'], parseFloat(e.target.value))}
                        />
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="input-group">
                        <label>Input Length Mean</label>
                        <input
                          type="number"
                          step="0.1"
                          value={config.workload.input_len_dist.mean || 6.9}
                          onChange={(e) => updateConfig(['workload', 'input_len_dist', 'mean'], parseFloat(e.target.value))}
                        />
                      </div>
                      <div className="input-group">
                        <label>Input Length Std Dev</label>
                        <input
                          type="number"
                          step="0.1"
                          value={config.workload.input_len_dist.std_dev || 0.7}
                          onChange={(e) => updateConfig(['workload', 'input_len_dist', 'std_dev'], parseFloat(e.target.value))}
                        />
                      </div>
                    </>
                  )}
                  <div className="input-group">
                    <label>Output Length Distribution</label>
                    <select
                      value={config.workload.output_len_dist.type}
                      onChange={(e) => {
                        const type = e.target.value
                        if (type === 'fixed') {
                          updateConfig(['workload', 'output_len_dist'], { type, value: 1024 })
                        } else if (type === 'uniform') {
                          updateConfig(['workload', 'output_len_dist'], { type, min: 50, max: 500 })
                        } else {
                          updateConfig(['workload', 'output_len_dist'], { type, mean: 5.3, std_dev: 0.8 })
                        }
                      }}
                    >
                      <option value="fixed">Fixed</option>
                      <option value="normal">Normal</option>
                      <option value="lognormal">Log-Normal</option>
                      <option value="uniform">Uniform</option>
                    </select>
                  </div>
                  {config.workload.output_len_dist.type === 'fixed' ? (
                    <div className="input-group">
                      <label>Output Length</label>
                      <input
                        type="number"
                        value={config.workload.output_len_dist.value || 1024}
                        onChange={(e) => updateConfig(['workload', 'output_len_dist', 'value'], parseInt(e.target.value))}
                      />
                    </div>
                  ) : config.workload.output_len_dist.type === 'uniform' ? (
                    <>
                      <div className="input-group">
                        <label>Output Length Min</label>
                        <input
                          type="number"
                          value={config.workload.output_len_dist.min || 50}
                          onChange={(e) => updateConfig(['workload', 'output_len_dist', 'min'], parseFloat(e.target.value))}
                        />
                      </div>
                      <div className="input-group">
                        <label>Output Length Max</label>
                        <input
                          type="number"
                          value={config.workload.output_len_dist.max || 500}
                          onChange={(e) => updateConfig(['workload', 'output_len_dist', 'max'], parseFloat(e.target.value))}
                        />
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="input-group">
                        <label>Output Length Mean</label>
                        <input
                          type="number"
                          step="0.1"
                          value={config.workload.output_len_dist.mean || 5.3}
                          onChange={(e) => updateConfig(['workload', 'output_len_dist', 'mean'], parseFloat(e.target.value))}
                        />
                      </div>
                      <div className="input-group">
                        <label>Output Length Std Dev</label>
                        <input
                          type="number"
                          step="0.1"
                          value={config.workload.output_len_dist.std_dev || 0.8}
                          onChange={(e) => updateConfig(['workload', 'output_len_dist', 'std_dev'], parseFloat(e.target.value))}
                        />
                      </div>
                    </>
                  )}
                </div>
              )}
            </section>
          </div>

          <button
            onClick={runSimulation}
            disabled={isRunning || !workerReady}
            className="run-button"
          >
            {isRunning ? (
              <>
                <span className="spinner"></span>
                Running Simulation...
              </>
            ) : workerReady ? (
              '▶ Run Simulation'
            ) : (
              'Initializing Worker...'
            )}
          </button>
        </div>

        {(results || isRunning) && (
          <div className="results-panel">
            <div className="panel-header">
              <h2>Results</h2>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <label style={{
                    fontSize: '0.75rem',
                    color: 'var(--gray-600)',
                    whiteSpace: 'nowrap',
                    fontWeight: 500
                  }}>
                    Simulation Speed:
                  </label>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                    <input
                      type="range"
                      min="0"
                      max="3"
                      step="1"
                      value={simulationSpeed}
                      onChange={(e) => setSimulationSpeed(parseInt(e.target.value))}
                      disabled={isRunning}
                      list="speed-markers"
                      style={{ width: '150px', margin: 0 }}
                    />
                    <datalist id="speed-markers">
                      <option value="0"></option>
                      <option value="1"></option>
                      <option value="2"></option>
                      <option value="3"></option>
                    </datalist>
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      width: '150px',
                      fontSize: '0.65rem',
                      color: 'var(--gray-500)',
                      fontFamily: 'monospace',
                      paddingLeft: '2px',
                      paddingRight: '2px'
                    }}>
                      <span style={{ fontWeight: simulationSpeed === 0 ? 600 : 400 }}>1x</span>
                      <span style={{ fontWeight: simulationSpeed === 1 ? 600 : 400 }}>10x</span>
                      <span style={{ fontWeight: simulationSpeed === 2 ? 600 : 400 }}>100x</span>
                      <span style={{ fontWeight: simulationSpeed === 3 ? 600 : 400 }}>MAX</span>
                    </div>
                  </div>
                </div>
                {results && (
                  <div className="completion-badge" style={{
                    position: 'relative',
                    overflow: 'hidden',
                    fontSize: '0.85rem',
                    fontFamily: 'inherit'
                  }}>
                    {isRunning && (
                      <div style={{
                        position: 'absolute',
                        left: 0,
                        top: 0,
                        bottom: 0,
                        width: `${(results.metrics.completed_requests / results.metrics.total_requests) * 100}%`,
                        background: 'rgba(255, 255, 255, 0.25)',
                        transition: 'width 0.2s ease',
                        zIndex: 0
                      }} />
                    )}
                    <span style={{ position: 'relative', zIndex: 1 }}>
                      {results.metrics.completed_requests}/{results.metrics.total_requests} requests
                      {progressInfo && (
                        <span style={{ opacity: 0.9, fontSize: '0.8em', marginLeft: '0.5rem' }}>
                          • {progressInfo.replace('Time: ', '').replace('Completed: ', '').replace(/\d+\/\d+ \| /, '')}
                        </span>
                      )}
                    </span>
                  </div>
                )}
              </div>
            </div>

            <div className="tabs">
              <div className="tabs-left">
                <button
                  className={`tab ${activeTab === 'metrics' ? 'active' : ''}`}
                  onClick={() => setActiveTab('metrics')}
                >
                  Metrics
                </button>
                <button
                  className={`tab ${activeTab === 'charts' ? 'active' : ''}`}
                  onClick={() => setActiveTab('charts')}
                >
                  Charts
                </button>
                <button
                  className={`tab ${activeTab === 'config' ? 'active' : ''}`}
                  onClick={() => setActiveTab('config')}
                >
                  Config
                </button>
              </div>

              {activeTab === 'charts' && (
                <div className="charts-toggle">
                  <button
                    className={`toggle-btn ${chartsSubTab === 'timeseries' ? 'active' : ''}`}
                    onClick={() => setChartsSubTab('timeseries')}
                  >
                    Time Series
                  </button>
                  <button
                    className={`toggle-btn ${chartsSubTab === 'distributions' ? 'active' : ''}`}
                    onClick={() => setChartsSubTab('distributions')}
                  >
                    Distributions
                  </button>
                </div>
              )}
            </div>

            {activeTab === 'metrics' && results && (
              <div className="metrics-overview">
              <div className="metrics-section">
                <h3>Latency</h3>
                <div className="metrics-grid">
                  <div className="metric-card latency-card">
                    <div className="metric-label">Time to First Token (TTFT)</div>
                    <div className="percentile-grid">
                      <div className="percentile-item">
                        <div className="percentile-label">Mean</div>
                        <div className="percentile-value">
                          {formatLatency(results.metrics.ttft_mean).display}
                          <span className="percentile-unit">{formatLatency(results.metrics.ttft_mean).unit}</span>
                        </div>
                      </div>
                      <div className="percentile-item p50">
                        <div className="percentile-label">p50</div>
                        <div className="percentile-value">
                          {formatLatency(results.metrics.ttft_p50).display}
                          <span className="percentile-unit">{formatLatency(results.metrics.ttft_p50).unit}</span>
                        </div>
                      </div>
                      <div className="percentile-item p90">
                        <div className="percentile-label">p90</div>
                        <div className="percentile-value">
                          {formatLatency(results.metrics.ttft_p90).display}
                          <span className="percentile-unit">{formatLatency(results.metrics.ttft_p90).unit}</span>
                        </div>
                      </div>
                      <div className="percentile-item p99">
                        <div className="percentile-label">p99</div>
                        <div className="percentile-value">
                          {formatLatency(results.metrics.ttft_p99).display}
                          <span className="percentile-unit">{formatLatency(results.metrics.ttft_p99).unit}</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="metric-card latency-card">
                    <div className="metric-label">End-to-End Latency (E2E)</div>
                    <div className="percentile-grid">
                      <div className="percentile-item">
                        <div className="percentile-label">Mean</div>
                        <div className="percentile-value">
                          {formatLatency(results.metrics.e2e_mean).display}
                          <span className="percentile-unit">{formatLatency(results.metrics.e2e_mean).unit}</span>
                        </div>
                      </div>
                      <div className="percentile-item p50">
                        <div className="percentile-label">p50</div>
                        <div className="percentile-value">
                          {formatLatency(results.metrics.e2e_p50).display}
                          <span className="percentile-unit">{formatLatency(results.metrics.e2e_p50).unit}</span>
                        </div>
                      </div>
                      <div className="percentile-item p90">
                        <div className="percentile-label">p90</div>
                        <div className="percentile-value">
                          {formatLatency(results.metrics.e2e_p90).display}
                          <span className="percentile-unit">{formatLatency(results.metrics.e2e_p90).unit}</span>
                        </div>
                      </div>
                      <div className="percentile-item p99">
                        <div className="percentile-label">p99</div>
                        <div className="percentile-value">
                          {formatLatency(results.metrics.e2e_p99).display}
                          <span className="percentile-unit">{formatLatency(results.metrics.e2e_p99).unit}</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="metric-card latency-card">
                    <div className="metric-label">Time per Output Token (TPOT)</div>
                    <div className="percentile-grid">
                      <div className="percentile-item">
                        <div className="percentile-label">Mean</div>
                        <div className="percentile-value">
                          {formatLatency(results.metrics.per_token_mean).display}
                          <span className="percentile-unit">{formatLatency(results.metrics.per_token_mean).unit}</span>
                        </div>
                      </div>
                      <div className="percentile-item p50">
                        <div className="percentile-label">p50</div>
                        <div className="percentile-value">
                          {formatLatency(results.metrics.per_token_p50).display}
                          <span className="percentile-unit">{formatLatency(results.metrics.per_token_p50).unit}</span>
                        </div>
                      </div>
                      <div className="percentile-item p90">
                        <div className="percentile-label">p90</div>
                        <div className="percentile-value">
                          {formatLatency(results.metrics.per_token_p90).display}
                          <span className="percentile-unit">{formatLatency(results.metrics.per_token_p90).unit}</span>
                        </div>
                      </div>
                      <div className="percentile-item p99">
                        <div className="percentile-label">p99</div>
                        <div className="percentile-value">
                          {formatLatency(results.metrics.per_token_p99).display}
                          <span className="percentile-unit">{formatLatency(results.metrics.per_token_p99).unit}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="metrics-section">
                <h3>Throughput</h3>
                <div className="metrics-grid">
                  <div className="metric-card latency-card">
                    <div className="metric-label">Input Tokens per Second</div>
                    <div className="percentile-grid">
                      <div className="percentile-item">
                        <div className="percentile-label">Mean</div>
                        <div className="percentile-value">
                          {results.metrics.input_tokens_per_sec.toFixed(0)}
                          <span className="percentile-unit">tok/s</span>
                        </div>
                      </div>
                      <div className="percentile-item p50">
                        <div className="percentile-label">p50</div>
                        <div className="percentile-value">
                          {results.metrics.input_tokens_per_sec_p50.toFixed(0)}
                          <span className="percentile-unit">tok/s</span>
                        </div>
                      </div>
                      <div className="percentile-item p90">
                        <div className="percentile-label">p90</div>
                        <div className="percentile-value">
                          {results.metrics.input_tokens_per_sec_p90.toFixed(0)}
                          <span className="percentile-unit">tok/s</span>
                        </div>
                      </div>
                      <div className="percentile-item p99">
                        <div className="percentile-label">p99</div>
                        <div className="percentile-value">
                          {results.metrics.input_tokens_per_sec_p99.toFixed(0)}
                          <span className="percentile-unit">tok/s</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="metric-card latency-card">
                    <div className="metric-label">Output Tokens per Second</div>
                    <div className="percentile-grid">
                      <div className="percentile-item">
                        <div className="percentile-label">Mean</div>
                        <div className="percentile-value">
                          {results.metrics.output_tokens_per_sec.toFixed(0)}
                          <span className="percentile-unit">tok/s</span>
                        </div>
                      </div>
                      <div className="percentile-item p50">
                        <div className="percentile-label">p50</div>
                        <div className="percentile-value">
                          {results.metrics.output_tokens_per_sec_p50.toFixed(0)}
                          <span className="percentile-unit">tok/s</span>
                        </div>
                      </div>
                      <div className="percentile-item p90">
                        <div className="percentile-label">p90</div>
                        <div className="percentile-value">
                          {results.metrics.output_tokens_per_sec_p90.toFixed(0)}
                          <span className="percentile-unit">tok/s</span>
                        </div>
                      </div>
                      <div className="percentile-item p99">
                        <div className="percentile-label">p99</div>
                        <div className="percentile-value">
                          {results.metrics.output_tokens_per_sec_p99.toFixed(0)}
                          <span className="percentile-unit">tok/s</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="metric-card latency-card">
                    <div className="metric-label">Requests per Second</div>
                    <div className="percentile-grid">
                      <div className="percentile-item">
                        <div className="percentile-label">Mean</div>
                        <div className="percentile-value">
                          {results.metrics.requests_per_sec.toFixed(2)}
                          <span className="percentile-unit">req/s</span>
                        </div>
                      </div>
                      <div className="percentile-item p50">
                        <div className="percentile-label">p50</div>
                        <div className="percentile-value">
                          {results.metrics.requests_per_sec_p50.toFixed(2)}
                          <span className="percentile-unit">req/s</span>
                        </div>
                      </div>
                      <div className="percentile-item p90">
                        <div className="percentile-label">p90</div>
                        <div className="percentile-value">
                          {results.metrics.requests_per_sec_p90.toFixed(2)}
                          <span className="percentile-unit">req/s</span>
                        </div>
                      </div>
                      <div className="percentile-item p99">
                        <div className="percentile-label">p99</div>
                        <div className="percentile-value">
                          {results.metrics.requests_per_sec_p99.toFixed(2)}
                          <span className="percentile-unit">req/s</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="metrics-section">
                <h3>Resource Utilization</h3>
                <div className="utilization-bars">
                  <div className="util-bar">
                    <div className="util-label">
                      <span>Compute (FLOPS)</span>
                      <span className="util-value">{(results.metrics.avg_flops_util * 100).toFixed(1)}%</span>
                    </div>
                    <div className="util-track">
                      <div
                        className="util-fill flops"
                        style={{ width: `${results.metrics.avg_flops_util * 100}%` }}
                      />
                    </div>
                  </div>
                  <div className="util-bar">
                    <div className="util-label">
                      <span>Memory Bandwidth</span>
                      <span className="util-value">{(results.metrics.avg_bandwidth_util * 100).toFixed(1)}%</span>
                    </div>
                    <div className="util-track">
                      <div
                        className="util-fill bandwidth"
                        style={{ width: `${results.metrics.avg_bandwidth_util * 100}%` }}
                      />
                    </div>
                  </div>
                  <div className="util-bar">
                    <div className="util-label">
                      <span>KV Cache</span>
                      <span className="util-value">{(results.metrics.avg_kv_cache_util * 100).toFixed(1)}%</span>
                    </div>
                    <div className="util-track">
                      <div
                        className="util-fill kv-cache"
                        style={{ width: `${results.metrics.avg_kv_cache_util * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              <div className="metrics-section">
                <h3>Simulation Info</h3>
                <div className="metrics-grid">
                  <div className="metric-card">
                    <div className="metric-label">Total Time</div>
                    <div className="metric-value">{(results.metrics.total_time || 0).toFixed(2)}<span className="unit">s</span></div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-label">Requests Completed</div>
                    <div className="metric-value">{results.metrics.completed_requests}<span className="unit"> / {results.metrics.total_requests}</span></div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-label">Preemptions</div>
                    <div className="metric-value">{results.metrics.total_preemptions}</div>
                    <div className="metric-detail">
                      {results.metrics.avg_preemptions_per_request.toFixed(2)} avg per request
                    </div>
                  </div>
                </div>
              </div>
            </div>
            )}

            {activeTab === 'charts' && results && (
            <div className="charts-section">
              {chartsSubTab === 'timeseries' && (
              <>
              <div className="chart-container">
                <h3>Queue Dynamics</h3>
                <Line
                  data={{
                    labels: results.time_series.times.map(t => t.toFixed(1)),
                    datasets: [
                      {
                        label: 'Running Requests',
                        data: results.time_series.running,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: true,
                        tension: 0.4,
                        borderWidth: 2,
                      },
                      {
                        label: 'Waiting Requests',
                        data: results.time_series.waiting,
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        fill: true,
                        tension: 0.4,
                        borderWidth: 2,
                      },
                    ],
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    interaction: {
                      mode: 'index',
                      intersect: false,
                    },
                    plugins: {
                      legend: {
                        position: 'top',
                        labels: {
                          usePointStyle: true,
                          padding: 15,
                        },
                      },
                    },
                    scales: {
                      x: {
                        title: {
                          display: true,
                          text: 'Time (seconds)',
                        },
                        ticks: {
                          maxTicksLimit: 10,
                        },
                      },
                      y: {
                        title: {
                          display: true,
                          text: 'Number of Requests',
                        },
                        beginAtZero: true,
                      },
                    },
                  }}
                />
              </div>

              <div className="chart-container">
                <h3>KV Cache Utilization Over Time</h3>
                <Line
                  data={{
                    labels: results.time_series.times.map(t => t.toFixed(1)),
                    datasets: [
                      {
                        label: 'KV Cache Utilization',
                        data: results.time_series.kv_cache_util.map(v => v * 100),
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        fill: true,
                        tension: 0.4,
                        borderWidth: 2,
                      },
                    ],
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    interaction: {
                      mode: 'index',
                      intersect: false,
                    },
                    plugins: {
                      legend: {
                        position: 'top',
                        labels: {
                          usePointStyle: true,
                          padding: 15,
                        },
                      },
                    },
                    scales: {
                      x: {
                        title: {
                          display: true,
                          text: 'Time (seconds)',
                        },
                        ticks: {
                          maxTicksLimit: 10,
                        },
                      },
                      y: {
                        title: {
                          display: true,
                          text: 'Utilization (%)',
                        },
                        min: 0,
                        max: 100,
                      },
                    },
                  }}
                />
              </div>

              <div className="chart-container">
                <h3>Prefill vs Decode Requests Over Time</h3>
                <Line
                  data={{
                    labels: results.time_series.times.map(t => t.toFixed(1)),
                    datasets: [
                      {
                        label: 'Prefilling Requests',
                        data: results.time_series.num_prefilling,
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.5)',
                        fill: 'origin',
                        tension: 0.4,
                        borderWidth: 1,
                      },
                      {
                        label: 'Decoding Requests',
                        data: results.time_series.num_decoding,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.5)',
                        fill: '-1',
                        tension: 0.4,
                        borderWidth: 1,
                      },
                    ],
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    interaction: {
                      mode: 'index',
                      intersect: false,
                    },
                    plugins: {
                      legend: {
                        position: 'top',
                        labels: {
                          usePointStyle: true,
                          padding: 15,
                        },
                      },
                    },
                    scales: {
                      x: {
                        title: {
                          display: true,
                          text: 'Time (seconds)',
                        },
                        ticks: {
                          maxTicksLimit: 10,
                        },
                      },
                      y: {
                        title: {
                          display: true,
                          text: 'Number of Requests',
                        },
                        beginAtZero: true,
                        stacked: true,
                      },
                    },
                  }}
                />
              </div>

              <div className="chart-container">
                <h3>Prefill vs Decode Tokens Per Iteration</h3>
                <Line
                  data={{
                    labels: results.time_series.times.map(t => t.toFixed(1)),
                    datasets: [
                      {
                        label: 'Prefill Tokens',
                        data: results.time_series.prefill_tokens,
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.5)',
                        fill: 'origin',
                        tension: 0.4,
                        borderWidth: 1,
                      },
                      {
                        label: 'Decode Tokens',
                        data: results.time_series.decode_tokens,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.5)',
                        fill: '-1',
                        tension: 0.4,
                        borderWidth: 1,
                      },
                    ],
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    interaction: {
                      mode: 'index',
                      intersect: false,
                    },
                    plugins: {
                      legend: {
                        position: 'top',
                        labels: {
                          usePointStyle: true,
                          padding: 15,
                        },
                      },
                    },
                    scales: {
                      x: {
                        title: {
                          display: true,
                          text: 'Time (seconds)',
                        },
                        ticks: {
                          maxTicksLimit: 10,
                        },
                      },
                      y: {
                        title: {
                          display: true,
                          text: 'Tokens Processed',
                        },
                        beginAtZero: true,
                        stacked: true,
                      },
                    },
                  }}
                />
              </div>

              <div className="chart-container">
                <h3>Throughput Over Time</h3>
                <Line
                  data={{
                    labels: results.time_series.times.map(t => t.toFixed(1)),
                    datasets: [
                      {
                        label: 'Input Tokens/sec',
                        data: results.time_series.input_throughput,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        fill: false,
                        tension: 0.4,
                        borderWidth: 2,
                        yAxisID: 'y-input',
                      },
                      {
                        label: 'Output Tokens/sec',
                        data: results.time_series.output_throughput,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: false,
                        tension: 0.4,
                        borderWidth: 2,
                        yAxisID: 'y-output',
                      },
                    ],
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    interaction: {
                      mode: 'index',
                      intersect: false,
                    },
                    plugins: {
                      legend: {
                        position: 'top',
                        labels: {
                          usePointStyle: true,
                          padding: 15,
                        },
                      },
                    },
                    scales: {
                      x: {
                        title: {
                          display: true,
                          text: 'Time (seconds)',
                        },
                        ticks: {
                          maxTicksLimit: 10,
                        },
                      },
                      'y-input': {
                        type: 'linear',
                        position: 'left',
                        title: {
                          display: true,
                          text: 'Input Tokens/sec',
                        },
                        beginAtZero: true,
                      },
                      'y-output': {
                        type: 'linear',
                        position: 'right',
                        title: {
                          display: true,
                          text: 'Output Tokens/sec',
                        },
                        beginAtZero: true,
                        grid: {
                          drawOnChartArea: false,
                        },
                      },
                    },
                  }}
                />
              </div>

              <div className="chart-container">
                <h3>Latency Over Time</h3>
                <Line
                  data={{
                    labels: results.time_series.times.map(t => t.toFixed(1)),
                    datasets: [
                      {
                        label: 'TTFT p50 (ms)',
                        data: results.time_series.ttft_p50,
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        fill: false,
                        tension: 0.4,
                        borderWidth: 2,
                        yAxisID: 'y-ttft',
                        spanGaps: true,
                      },
                      {
                        label: 'TPOT p50 (ms)',
                        data: results.time_series.tpot_p50,
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        fill: false,
                        tension: 0.4,
                        borderWidth: 2,
                        yAxisID: 'y-tpot',
                        spanGaps: true,
                      },
                    ],
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    interaction: {
                      mode: 'index',
                      intersect: false,
                    },
                    plugins: {
                      legend: {
                        position: 'top',
                        labels: {
                          usePointStyle: true,
                          padding: 15,
                        },
                      },
                    },
                    scales: {
                      x: {
                        title: {
                          display: true,
                          text: 'Time (seconds)',
                        },
                        ticks: {
                          maxTicksLimit: 10,
                        },
                      },
                      'y-ttft': {
                        type: 'linear',
                        position: 'left',
                        title: {
                          display: true,
                          text: 'TTFT p50 (ms)',
                        },
                        beginAtZero: true,
                      },
                      'y-tpot': {
                        type: 'linear',
                        position: 'right',
                        title: {
                          display: true,
                          text: 'TPOT p50 (ms)',
                        },
                        beginAtZero: true,
                        grid: {
                          drawOnChartArea: false,
                        },
                      },
                    },
                  }}
                />
              </div>
              </>
              )}

              {chartsSubTab === 'distributions' && (
              <>
              <div className="chart-container">
                <h3>TTFT Distribution</h3>
                <Bar
                    data={{
                      labels: ttftHistogram.bins,
                      datasets: [
                        {
                          label: 'TTFT Count',
                          data: ttftHistogram.counts,
                          backgroundColor: 'rgba(59, 130, 246, 0.7)',
                          borderColor: '#3b82f6',
                          borderWidth: 1,
                        },
                      ],
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    plugins: {
                      legend: {
                        position: 'top',
                        labels: {
                          usePointStyle: true,
                          padding: 15,
                        },
                      },
                      tooltip: {
                        callbacks: {
                          title: (items) => `TTFT: ${items[0].label} ms`,
                          label: (item) => `Count: ${item.formattedValue}`,
                        },
                      },
                    },
                    scales: {
                      x: {
                        title: {
                          display: true,
                          text: 'Time to First Token (ms)',
                        },
                        ticks: {
                          maxTicksLimit: 10,
                        },
                      },
                      y: {
                        title: {
                          display: true,
                          text: 'Count',
                        },
                        beginAtZero: true,
                      },
                    },
                  }}
                />
              </div>

              <div className="chart-container">
                <h3>E2E Latency Distribution</h3>
                <Bar
                    data={{
                      labels: e2eHistogram.bins,
                      datasets: [
                        {
                          label: 'E2E Latency Count',
                          data: e2eHistogram.counts,
                          backgroundColor: 'rgba(168, 85, 247, 0.7)',
                          borderColor: '#a855f7',
                          borderWidth: 1,
                        },
                      ],
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    plugins: {
                      legend: {
                        position: 'top',
                        labels: {
                          usePointStyle: true,
                          padding: 15,
                        },
                      },
                      tooltip: {
                        callbacks: {
                          title: (items) => `E2E Latency: ${items[0].label} ms`,
                          label: (item) => `Count: ${item.formattedValue}`,
                        },
                      },
                    },
                    scales: {
                      x: {
                        title: {
                          display: true,
                          text: 'End-to-End Latency (ms)',
                        },
                        ticks: {
                          maxTicksLimit: 10,
                        },
                      },
                      y: {
                        title: {
                          display: true,
                          text: 'Count',
                        },
                        beginAtZero: true,
                      },
                    },
                  }}
                />
              </div>

              <div className="chart-container">
                <h3>Time Per Output Token Distribution</h3>
                <Bar
                    data={{
                      labels: tpotHistogram.bins,
                      datasets: [
                        {
                          label: 'TPOT Count',
                          data: tpotHistogram.counts,
                          backgroundColor: 'rgba(236, 72, 153, 0.7)',
                          borderColor: '#ec4899',
                          borderWidth: 1,
                        },
                      ],
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    plugins: {
                      legend: {
                        position: 'top',
                        labels: {
                          usePointStyle: true,
                          padding: 15,
                        },
                      },
                      tooltip: {
                        callbacks: {
                          title: (items) => `TPOT: ${items[0].label} ms`,
                          label: (item) => `Count: ${item.formattedValue}`,
                        },
                      },
                    },
                    scales: {
                      x: {
                        title: {
                          display: true,
                          text: 'Time Per Output Token (ms)',
                        },
                        ticks: {
                          maxTicksLimit: 10,
                        },
                      },
                      y: {
                        title: {
                          display: true,
                          text: 'Count',
                        },
                        beginAtZero: true,
                      },
                    },
                  }}
                />
              </div>

              <div className="chart-container">
                <h3>Input Length Distribution</h3>
                <Bar
                    data={{
                      labels: inputLengthHistogram.bins,
                      datasets: [
                        {
                          label: 'Input Length Count',
                          data: inputLengthHistogram.counts,
                          backgroundColor: 'rgba(59, 130, 246, 0.7)',
                          borderColor: '#3b82f6',
                          borderWidth: 1,
                        },
                      ],
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    plugins: {
                      legend: {
                        position: 'top',
                        labels: {
                          usePointStyle: true,
                          padding: 15,
                        },
                      },
                    },
                    scales: {
                      x: {
                        title: {
                          display: true,
                          text: 'Input Length (tokens)',
                        },
                        ticks: {
                          maxTicksLimit: 10,
                        },
                      },
                      y: {
                        title: {
                          display: true,
                          text: 'Count',
                        },
                        beginAtZero: true,
                      },
                    },
                  }}
                />
              </div>

              <div className="chart-container">
                <h3>Output Length Distribution</h3>
                <Bar
                    data={{
                      labels: outputLengthHistogram.bins,
                      datasets: [
                        {
                          label: 'Output Length Count',
                          data: outputLengthHistogram.counts,
                          backgroundColor: 'rgba(16, 185, 129, 0.7)',
                          borderColor: '#10b981',
                          borderWidth: 1,
                        },
                      ],
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    plugins: {
                      legend: {
                        position: 'top',
                        labels: {
                          usePointStyle: true,
                          padding: 15,
                        },
                      },
                    },
                    scales: {
                      x: {
                        title: {
                          display: true,
                          text: 'Output Length (tokens)',
                        },
                        ticks: {
                          maxTicksLimit: 10,
                        },
                      },
                      y: {
                        title: {
                          display: true,
                          text: 'Count',
                        },
                        beginAtZero: true,
                      },
                    },
                  }}
                />
              </div>
              </>
              )}
            </div>
            )}

            {!results && isRunning && (
              <div style={{
                padding: '3rem',
                textAlign: 'center',
                color: 'var(--gray-500)'
              }}>
                <div className="spinner" style={{ margin: '0 auto 1rem' }}></div>
                <p>Initializing simulation...</p>
              </div>
            )}

            {activeTab === 'config' && results && (
              <div style={{ padding: '2rem' }}>
                <div style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h3 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 600, color: 'var(--gray-900)' }}>
                    Configuration File (config.toml)
                  </h3>
                  <button
                    onClick={() => {
                      const toml = generateTOML(config)
                      navigator.clipboard.writeText(toml)
                        .then(() => alert('Configuration copied to clipboard!'))
                        .catch(() => alert('Failed to copy to clipboard'))
                    }}
                    style={{
                      padding: '0.5rem 1rem',
                      background: 'var(--primary)',
                      color: 'white',
                      border: 'none',
                      borderRadius: '6px',
                      fontSize: '0.875rem',
                      fontWeight: 500,
                      cursor: 'pointer',
                      transition: 'all 0.2s'
                    }}
                    onMouseOver={(e) => e.currentTarget.style.background = 'var(--primary-dark)'}
                    onMouseOut={(e) => e.currentTarget.style.background = 'var(--primary)'}
                  >
                    📋 Copy to Clipboard
                  </button>
                </div>
                <SyntaxHighlighter
                  language="toml"
                  style={vscDarkPlus}
                  customStyle={{
                    borderRadius: '8px',
                    maxHeight: 'calc(100vh - 300px)',
                    fontSize: '0.9rem',
                    lineHeight: '1.5',
                    margin: 0,
                  }}
                  showLineNumbers
                >
                  {generateTOML(config)}
                </SyntaxHighlighter>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
