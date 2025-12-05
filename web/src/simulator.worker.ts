// Web Worker for running simulation in background
import init, { run_simulation_streaming } from 'sim'

let wasmInitialized = false

interface WorkerMessage {
  type: 'init' | 'run'
  config?: string
  speed?: number
}

interface ProgressUpdate {
  current_time: number
  completed_requests: number
  total_requests: number
  running: number
  waiting: number
  kv_cache_util: number
  time_series?: any
  metrics?: any
  latency_samples?: {
    ttft_samples: number[]
    e2e_samples: number[]
    tpot_samples: number[]
  }
}

// Initialize WASM module
async function initWasm() {
  if (!wasmInitialized) {
    await init()
    wasmInitialized = true
  }
}

// Handle messages from main thread
self.onmessage = async (e: MessageEvent<WorkerMessage>) => {
  const { type, config, speed } = e.data

  try {
    if (type === 'init') {
      await initWasm()
      self.postMessage({ type: 'initialized' })
    } else if (type === 'run') {
      if (!wasmInitialized) {
        await initWasm()
      }

      if (!config) {
        throw new Error('No config provided')
      }

      const speedSetting = speed !== undefined ? speed : 2 // Default to 100x
      // Convert slider value to speed multiplier: 0=1x, 1=10x, 2=100x, 3=MAX
      const speedMultiplier = speedSetting === 0 ? 1 : speedSetting === 1 ? 10 : speedSetting === 2 ? 100 : Infinity

      let lastSimTime = 0
      let lastRealTime = Date.now()

      // Run simulation with progress callback
      const finalResult = run_simulation_streaming(config, (progress: ProgressUpdate) => {
        // Calculate delay based on simulation speed
        if (speedMultiplier < Infinity) {
          const simTimeDelta = progress.current_time - lastSimTime
          const delayMs = (simTimeDelta * 1000) / speedMultiplier

          if (delayMs > 0) {
            // Sleep for the calculated delay
            const targetTime = lastRealTime + delayMs
            const now = Date.now()
            if (now < targetTime) {
              const sleepTime = targetTime - now
              // Busy wait for small delays (more accurate)
              const start = Date.now()
              while (Date.now() - start < sleepTime) {
                // Busy wait
              }
            }
          }

          lastRealTime = Date.now()
          lastSimTime = progress.current_time
        }

        // Send progress updates back to main thread
        self.postMessage({
          type: 'progress',
          data: progress,
        })
      })

      // Send final result
      self.postMessage({
        type: 'complete',
        data: finalResult,
      })
    }
  } catch (error) {
    self.postMessage({
      type: 'error',
      error: error instanceof Error ? error.message : String(error),
    })
  }
}
