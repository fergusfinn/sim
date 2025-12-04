import { useState, useEffect } from 'react'
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

// Import WASM module
import init, { run_simulation } from 'sim'

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
    output_tokens_per_sec: number
    requests_per_sec: number
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
      type: "fixed",
      value: 1024,
    },
    output_len_dist: {
      type: "fixed",
      value: 1024,
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
max_seq_len = ${model.max_seq_len}

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
  const [wasmInitialized, setWasmInitialized] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedHardware, setSelectedHardware] = useState<string>('H100')
  const [selectedModel, setSelectedModel] = useState<string>('Llama-3-70B')
  const [activeTab, setActiveTab] = useState<'metrics' | 'charts' | 'config'>('metrics')
  const [expandedSections, setExpandedSections] = useState({
    hardware: false,
    model: false,
    scheduler: false,
    workload: false,
  })

  // Initialize WASM on mount
  useEffect(() => {
    init().then(() => {
      setWasmInitialized(true)
    }).catch((err) => {
      setError(`Failed to initialize WASM: ${err}`)
    })
  }, [])

  const runSimulation = async () => {
    if (!wasmInitialized) {
      setError("WASM not initialized yet")
      return
    }

    if (isRunning) {
      return // Already running
    }

    setIsRunning(true)
    setError(null)

    try {
      // config.hardware is already effective (TP scaled), backend will compute kv_cache_capacity
      // Small delay to ensure UI updates
      await new Promise(resolve => setTimeout(resolve, 50))
      const configJson = JSON.stringify(config)
      const result = run_simulation(configJson)
      console.log('Simulation result:', result)
      console.log('Distributions:', (result as any).distributions)
      setResults(result as SimulationResults)
    } catch (err) {
      setError(`Simulation error: ${err}`)
    } finally {
      setIsRunning(false)
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
            disabled={isRunning || !wasmInitialized}
            className="run-button"
          >
            {isRunning ? (
              <>
                <span className="spinner"></span>
                Running Simulation...
              </>
            ) : wasmInitialized ? (
              '▶ Run Simulation'
            ) : (
              'Initializing...'
            )}
          </button>
        </div>

        {results && (
          <div className="results-panel">
            <div className="panel-header">
              <h2>Results</h2>
              <div className="completion-badge">
                {results.metrics.completed_requests}/{results.metrics.total_requests} requests completed
              </div>
            </div>

            <div className="tabs">
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

            {activeTab === 'metrics' && (
              <div className="metrics-overview">
              <div className="metrics-section">
                <h3>Latency</h3>
                <div className="metrics-grid">
                  <div className="metric-card">
                    <div className="metric-label">Time to First Token</div>
                    <div className="metric-value">
                      {formatLatency(results.metrics.ttft_mean).display}
                      <span className="unit">{formatLatency(results.metrics.ttft_mean).unit}</span>
                    </div>
                    <div className="metric-detail">
                      P50: {formatLatency(results.metrics.ttft_p50).display}{formatLatency(results.metrics.ttft_p50).unit} | P90: {formatLatency(results.metrics.ttft_p90).display}{formatLatency(results.metrics.ttft_p90).unit} | P99: {formatLatency(results.metrics.ttft_p99).display}{formatLatency(results.metrics.ttft_p99).unit}
                    </div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-label">End-to-End Latency</div>
                    <div className="metric-value">
                      {formatLatency(results.metrics.e2e_mean).display}
                      <span className="unit">{formatLatency(results.metrics.e2e_mean).unit}</span>
                    </div>
                    <div className="metric-detail">
                      P50: {formatLatency(results.metrics.e2e_p50).display}{formatLatency(results.metrics.e2e_p50).unit} | P90: {formatLatency(results.metrics.e2e_p90).display}{formatLatency(results.metrics.e2e_p90).unit} | P99: {formatLatency(results.metrics.e2e_p99).display}{formatLatency(results.metrics.e2e_p99).unit}
                    </div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-label">Per-Token Latency</div>
                    <div className="metric-value">
                      {formatLatency(results.metrics.per_token_mean).display}
                      <span className="unit">{formatLatency(results.metrics.per_token_mean).unit}</span>
                    </div>
                    <div className="metric-detail">
                      P50: {formatLatency(results.metrics.per_token_p50).display}{formatLatency(results.metrics.per_token_p50).unit} | P90: {formatLatency(results.metrics.per_token_p90).display}{formatLatency(results.metrics.per_token_p90).unit} | P99: {formatLatency(results.metrics.per_token_p99).display}{formatLatency(results.metrics.per_token_p99).unit}
                    </div>
                  </div>
                </div>
              </div>

              <div className="metrics-section">
                <h3>Throughput</h3>
                <div className="metrics-grid">
                  <div className="metric-card">
                    <div className="metric-label">Input Tokens</div>
                    <div className="metric-value">{results.metrics.input_tokens_per_sec.toFixed(0)}<span className="unit">tok/s</span></div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-label">Output Tokens</div>
                    <div className="metric-value">{results.metrics.output_tokens_per_sec.toFixed(0)}<span className="unit">tok/s</span></div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-label">Request Rate</div>
                    <div className="metric-value">{results.metrics.requests_per_sec.toFixed(2)}<span className="unit">req/s</span></div>
                  </div>
                </div>
              </div>

              <div className="metrics-section">
                <h3>Resource Utilization</h3>
                <div className="utilization-bars">
                  <div className="util-bar">
                    <div className="util-label">
                      <span>GPU</span>
                      <span className="util-value">{(Math.max(results.metrics.avg_flops_util, results.metrics.avg_bandwidth_util) * 100).toFixed(1)}%</span>
                    </div>
                    <div className="util-track">
                      <div
                        className="util-fill gpu"
                        style={{ width: `${Math.max(results.metrics.avg_flops_util, results.metrics.avg_bandwidth_util) * 100}%` }}
                      />
                    </div>
                    <div className="metric-detail" style={{ marginTop: '4px', fontSize: '0.85em', opacity: 0.7 }}>
                      {results.metrics.avg_flops_util > results.metrics.avg_bandwidth_util ? 'Compute-bound' : 'Memory-bound'}
                    </div>
                  </div>
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

            {activeTab === 'charts' && (
            <div className="charts-section">
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
                <h3>Input Length Distribution</h3>
                <Bar
                    data={{
                      labels: createHistogram(results.distributions?.input_lengths || [], 30).bins,
                      datasets: [
                        {
                          label: 'Input Length Count',
                          data: createHistogram(results.distributions?.input_lengths || [], 30).counts,
                          backgroundColor: 'rgba(59, 130, 246, 0.7)',
                          borderColor: '#3b82f6',
                          borderWidth: 1,
                        },
                      ],
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
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
                      labels: createHistogram(results.distributions?.output_lengths || [], 30).bins,
                      datasets: [
                        {
                          label: 'Output Length Count',
                          data: createHistogram(results.distributions?.output_lengths || [], 30).counts,
                          backgroundColor: 'rgba(16, 185, 129, 0.7)',
                          borderColor: '#10b981',
                          borderWidth: 1,
                        },
                      ],
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
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
            </div>
            )}

            {activeTab === 'config' && (
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
                <pre style={{
                  background: 'var(--gray-900)',
                  color: '#f8f8f2',
                  padding: '1.5rem',
                  borderRadius: '8px',
                  overflow: 'auto',
                  maxHeight: 'calc(100vh - 300px)',
                  fontSize: '0.9rem',
                  lineHeight: '1.5',
                  fontFamily: "'Consolas', 'Monaco', 'Courier New', monospace"
                }}>
                  <code>{generateTOML(config)}</code>
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
