use crate::compute::ComputeEngine;
use crate::config::Config;
use crate::kv_cache::KVCacheManager;
use crate::metrics::MetricsCollector;
use crate::request::RequestGenerator;
use crate::scheduler::Scheduler;

#[derive(Debug, Clone)]
pub struct TimeSeriesPoint {
    pub time: f64,
    pub arrivals: u64,
    pub running: usize,
    pub waiting: usize,
    pub kv_cache_util: f64,
    pub num_prefilling: usize,
    pub num_decoding: usize,
    pub prefill_tokens: u32,
    pub decode_tokens: u32,
}

pub struct Simulator {
    scheduler: Scheduler,
    compute_engine: ComputeEngine,
    request_generator: RequestGenerator,
    metrics: MetricsCollector,
    time_series_data: Vec<TimeSeriesPoint>,
    sample_interval: f64,
    next_sample_time: f64,

    current_time: f64,
    iteration: u64,
    log_interval: u64,
}

impl Simulator {
    pub fn new(config: Config) -> Result<Self, String> {
        let kv_cache_manager = KVCacheManager::new(
            config.hardware.kv_cache_capacity,
            config.scheduler.block_size,
            config.model.kv_cache_bytes_per_token,
            true, // enable_prefix_caching
        );

        let scheduler = Scheduler::new(
            config.scheduler.clone(),
            config.hardware.clone(),
            config.model.clone(),
            kv_cache_manager,
        )?;

        let compute_engine = ComputeEngine::new(config.hardware, config.model);
        let request_generator = RequestGenerator::new(config.workload);
        let metrics = MetricsCollector::new(0.0);

        Ok(Self {
            scheduler,
            compute_engine,
            request_generator,
            metrics,
            time_series_data: Vec::new(),
            sample_interval: 0.1,
            next_sample_time: 0.0,
            current_time: 0.0,
            iteration: 0,
            log_interval: config.simulation.log_interval,
        })
    }

    /// Run the simulation
    pub fn run(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        println!("Starting simulation...\n");

        loop {
            self.iteration += 1;

            // 1. Generate new arrivals up to current_time
            while let Some(request) = self.request_generator.next_if_before(self.current_time) {
                self.scheduler.add_request(request);
                self.metrics.total_requests += 1;
            }

            // 2. Run scheduler
            let decision = self.scheduler.schedule(self.current_time);

            // 3. Calculate iteration time
            let iteration_time = if decision.num_scheduled() > 0 {
                // Build batch of scheduled requests
                let running = self.scheduler.running_mut();
                let batch_requests: Vec<&_> = decision
                    .scheduled_new
                    .iter()
                    .chain(decision.scheduled_running.iter())
                    .filter_map(|&idx| running.get(idx))
                    .collect();

                let tokens_per_req: Vec<u32> = batch_requests
                    .iter()
                    .enumerate()
                    .map(|(i, _req)| {
                        let idx = if i < decision.scheduled_new.len() {
                            decision.scheduled_new[i]
                        } else {
                            decision.scheduled_running[i - decision.scheduled_new.len()]
                        };
                        *decision.tokens_per_request.get(&idx).unwrap_or(&0)
                    })
                    .collect();

                self.compute_engine
                    .calculate_iteration_time(&batch_requests, &tokens_per_req)
            } else {
                0.001 // Small time step when idle
            };

            // 4. Advance time
            self.current_time += iteration_time;

            // 5. Determine which requests were prefilling vs decoding BEFORE updating state
            let mut prefilling_reqs = std::collections::HashSet::new();
            for (&idx, &_tokens) in &decision.tokens_per_request {
                if let Some(request) = self.scheduler.running().get(idx) {
                    if request.is_prefill() {
                        prefilling_reqs.insert(idx);
                    }
                }
            }

            // 6. Update request states
            for (&idx, &tokens) in &decision.tokens_per_request {
                if let Some(request) = self.scheduler.running_mut().get_mut(idx) {
                    request.record_generated_tokens(tokens, self.current_time);
                }
            }

            // 7. Record iteration metrics (before moving completed requests)
            let kv_util = self.scheduler.kv_cache_manager().utilization();

            // Calculate bandwidth and flops utilization
            let (bandwidth_util, flops_util) = if decision.num_scheduled() > 0 {
                let running = self.scheduler.running_mut();
                let batch_requests: Vec<&_> = decision
                    .scheduled_new
                    .iter()
                    .chain(decision.scheduled_running.iter())
                    .filter_map(|&idx| running.get(idx))
                    .collect();

                let tokens_per_req: Vec<u32> = batch_requests
                    .iter()
                    .enumerate()
                    .map(|(i, _req)| {
                        let idx = if i < decision.scheduled_new.len() {
                            decision.scheduled_new[i]
                        } else {
                            decision.scheduled_running[i - decision.scheduled_new.len()]
                        };
                        *decision.tokens_per_request.get(&idx).unwrap_or(&0)
                    })
                    .collect();

                let bytes_transferred = self
                    .compute_engine
                    .calculate_bytes_transferred(&batch_requests, &tokens_per_req);
                let bandwidth_util = self
                    .compute_engine
                    .calculate_bandwidth_utilization(bytes_transferred, iteration_time);

                let flops_util = self
                    .compute_engine
                    .calculate_flops_utilization(&batch_requests, &tokens_per_req, iteration_time);

                (bandwidth_util, flops_util)
            } else {
                (0.0, 0.0)
            };

            self.metrics
                .record_iteration_metrics(kv_util, flops_util, bandwidth_util);

            // 7. Record time-series data (BEFORE handling completed requests)
            if self.current_time >= self.next_sample_time {
                // Calculate prefill vs decode breakdown
                let running = self.scheduler.running();
                let mut num_prefilling = 0;
                let mut num_decoding = 0;
                let mut prefill_tokens = 0;
                let mut decode_tokens = 0;

                for req in running {
                    if req.is_prefill() {
                        num_prefilling += 1;
                    } else {
                        num_decoding += 1;
                    }
                }

                // Count tokens scheduled in this iteration
                // Use the prefilling_reqs set we captured before updating state
                for (&idx, &tokens) in &decision.tokens_per_request {
                    if prefilling_reqs.contains(&idx) {
                        prefill_tokens += tokens;
                    } else {
                        decode_tokens += tokens;
                    }
                }

                self.time_series_data.push(TimeSeriesPoint {
                    time: self.current_time,
                    arrivals: self.metrics.total_requests,
                    running: self.scheduler.num_running(),
                    waiting: self.scheduler.num_waiting(),
                    kv_cache_util: kv_util,
                    num_prefilling,
                    num_decoding,
                    prefill_tokens,
                    decode_tokens,
                });
                self.next_sample_time = self.current_time + self.sample_interval;
            }

            // 8. Handle completed requests
            for request in decision.completed {
                // Free KV cache blocks
                self.scheduler
                    .kv_cache_manager_mut()
                    .free_blocks(&request.kv_blocks);

                self.metrics.record_request_completion(&request);

                // For closed-loop workloads, generate a new request when one completes
                self.request_generator.on_request_complete(self.current_time);
            }

            // 9. Periodic logging
            if self.iteration % self.log_interval == 0 {
                self.log_progress();
            }

            // 10. Check termination conditions
            if self.should_terminate() {
                break;
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        println!("\nSimulation complete!");
        #[cfg(not(target_arch = "wasm32"))]
        self.print_final_metrics();

        #[cfg(feature = "cli")]
        self.generate_plots();
    }

    #[cfg(feature = "cli")]
    fn generate_plots(&self) {
        use crate::visualization::TimeSeriesCollector;

        let mut collector = TimeSeriesCollector::new(self.sample_interval);
        for point in &self.time_series_data {
            collector.data_points.push(crate::visualization::TimeSeriesPoint {
                time: point.time,
                arrivals: point.arrivals,
                running: point.running,
                waiting: point.waiting,
                kv_cache_util: point.kv_cache_util,
                num_prefilling: point.num_prefilling,
                num_decoding: point.num_decoding,
                prefill_tokens: point.prefill_tokens,
                decode_tokens: point.decode_tokens,
            });
        }

        if let Err(e) = collector.generate_plots("plots") {
            eprintln!("Warning: Failed to generate plots: {}", e);
        }
    }

    pub fn get_metrics_summary(&self) -> crate::metrics::MetricsSummary {
        self.metrics.compute_summary(self.current_time)
    }

    pub fn get_time_series_data(&self) -> &[TimeSeriesPoint] {
        &self.time_series_data
    }

    pub fn get_input_lengths(&self) -> &[u32] {
        self.metrics.get_input_lengths()
    }

    pub fn get_output_lengths(&self) -> &[u32] {
        self.metrics.get_output_lengths()
    }

    pub fn get_current_time(&self) -> f64 {
        self.current_time
    }

    fn log_progress(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        println!(
            "[{:.2}s] Iteration {}: {} running, {} waiting, {:.1}% KV cache used",
            self.current_time,
            self.iteration,
            self.scheduler.num_running(),
            self.scheduler.num_waiting(),
            self.scheduler.kv_cache_manager().utilization() * 100.0,
        );
    }

    fn should_terminate(&self) -> bool {
        // Check if we've generated all requests and completed them all
        self.request_generator.is_finished()
            && self.scheduler.num_running() == 0
            && self.scheduler.num_waiting() == 0
    }

    fn print_final_metrics(&self) {
        let summary = self.metrics.compute_summary(self.current_time);
        summary.print();
    }
}
