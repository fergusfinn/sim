use plotters::prelude::*;
use std::error::Error;

/// Time-series data point
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

/// Collect and visualize simulation metrics over time
pub struct TimeSeriesCollector {
    pub data_points: Vec<TimeSeriesPoint>,
    sample_interval: f64,
    next_sample_time: f64,
}

impl TimeSeriesCollector {
    pub fn new(sample_interval: f64) -> Self {
        Self {
            data_points: Vec::new(),
            sample_interval,
            next_sample_time: 0.0,
        }
    }

    /// Check if we should sample at this time
    pub fn should_sample(&self, current_time: f64) -> bool {
        current_time >= self.next_sample_time
    }

    /// Record a data point
    pub fn record(
        &mut self,
        time: f64,
        arrivals: u64,
        running: usize,
        waiting: usize,
        kv_cache_util: f64,
    ) {
        self.data_points.push(TimeSeriesPoint {
            time,
            arrivals,
            running,
            waiting,
            kv_cache_util,
            num_prefilling: 0,
            num_decoding: 0,
            prefill_tokens: 0,
            decode_tokens: 0,
        });
        self.next_sample_time = time + self.sample_interval;
    }

    /// Generate visualization plots
    pub fn generate_plots(&self, output_dir: &str) -> Result<(), Box<dyn Error>> {
        std::fs::create_dir_all(output_dir)?;

        self.plot_arrivals(&format!("{}/arrivals.png", output_dir))?;
        self.plot_queue_depth(&format!("{}/queue_depth.png", output_dir))?;
        self.plot_kv_cache(&format!("{}/kv_cache.png", output_dir))?;
        self.plot_prefill_decode_requests(&format!("{}/prefill_decode_requests.png", output_dir))?;
        self.plot_prefill_decode_tokens(&format!("{}/prefill_decode_tokens.png", output_dir))?;

        println!("\nGenerated plots in {}:", output_dir);
        println!("  - arrivals.png: Request arrivals over time");
        println!("  - queue_depth.png: Running and waiting requests over time");
        println!("  - kv_cache.png: KV cache utilization over time");
        println!("  - prefill_decode_requests.png: Prefilling vs decoding requests over time");
        println!("  - prefill_decode_tokens.png: Prefill vs decode tokens per iteration");

        Ok(())
    }

    /// Plot cumulative arrivals over time
    fn plot_arrivals(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_time = self
            .data_points
            .last()
            .map(|p| p.time)
            .unwrap_or(1.0)
            .max(1.0);
        let max_arrivals = self
            .data_points
            .iter()
            .map(|p| p.arrivals)
            .max()
            .unwrap_or(1)
            .max(1);

        let mut chart = ChartBuilder::on(&root)
            .caption("Request Arrivals Over Time", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0..max_time, 0u64..max_arrivals)?;

        chart
            .configure_mesh()
            .x_desc("Time (s)")
            .y_desc("Cumulative Arrivals")
            .draw()?;

        chart.draw_series(LineSeries::new(
            self.data_points.iter().map(|p| (p.time, p.arrivals)),
            &BLUE,
        ))?;

        root.present()?;
        Ok(())
    }

    /// Plot running and waiting requests over time
    fn plot_queue_depth(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_time = self
            .data_points
            .last()
            .map(|p| p.time)
            .unwrap_or(1.0)
            .max(1.0);
        let max_count = self
            .data_points
            .iter()
            .map(|p| (p.running + p.waiting).max(1))
            .max()
            .unwrap_or(1);

        let mut chart = ChartBuilder::on(&root)
            .caption("Queue Depth Over Time", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0..max_time, 0..max_count)?;

        chart
            .configure_mesh()
            .x_desc("Time (s)")
            .y_desc("Number of Requests")
            .draw()?;

        // Running requests
        chart
            .draw_series(LineSeries::new(
                self.data_points.iter().map(|p| (p.time, p.running)),
                &GREEN,
            ))?
            .label("Running")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

        // Waiting requests
        chart
            .draw_series(LineSeries::new(
                self.data_points.iter().map(|p| (p.time, p.waiting)),
                &RED,
            ))?
            .label("Waiting")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
        Ok(())
    }

    /// Plot KV cache utilization over time
    fn plot_kv_cache(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_time = self
            .data_points
            .last()
            .map(|p| p.time)
            .unwrap_or(1.0)
            .max(1.0);

        let mut chart = ChartBuilder::on(&root)
            .caption("KV Cache Utilization Over Time", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0..max_time, 0.0..100.0)?;

        chart
            .configure_mesh()
            .x_desc("Time (s)")
            .y_desc("Utilization (%)")
            .draw()?;

        chart.draw_series(LineSeries::new(
            self.data_points
                .iter()
                .map(|p| (p.time, p.kv_cache_util * 100.0)),
            &MAGENTA,
        ))?;

        root.present()?;
        Ok(())
    }

    /// Plot prefilling vs decoding requests over time
    fn plot_prefill_decode_requests(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_time = self
            .data_points
            .last()
            .map(|p| p.time)
            .unwrap_or(1.0)
            .max(1.0);
        let max_count = self
            .data_points
            .iter()
            .map(|p| (p.num_prefilling + p.num_decoding).max(1))
            .max()
            .unwrap_or(1);

        let mut chart = ChartBuilder::on(&root)
            .caption("Prefill vs Decode Requests Over Time", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0..max_time, 0..max_count)?;

        chart
            .configure_mesh()
            .x_desc("Time (s)")
            .y_desc("Number of Requests")
            .draw()?;

        // Prefilling requests
        chart
            .draw_series(LineSeries::new(
                self.data_points.iter().map(|p| (p.time, p.num_prefilling)),
                &RED,
            ))?
            .label("Prefilling")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        // Decoding requests
        chart
            .draw_series(LineSeries::new(
                self.data_points.iter().map(|p| (p.time, p.num_decoding)),
                &GREEN,
            ))?
            .label("Decoding")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
        Ok(())
    }

    /// Plot prefill vs decode tokens per iteration
    fn plot_prefill_decode_tokens(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_time = self
            .data_points
            .last()
            .map(|p| p.time)
            .unwrap_or(1.0)
            .max(1.0);
        let max_tokens = self
            .data_points
            .iter()
            .map(|p| (p.prefill_tokens + p.decode_tokens).max(1))
            .max()
            .unwrap_or(1);

        let mut chart = ChartBuilder::on(&root)
            .caption("Prefill vs Decode Tokens Per Iteration", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0..max_time, 0u32..max_tokens)?;

        chart
            .configure_mesh()
            .x_desc("Time (s)")
            .y_desc("Tokens Processed")
            .draw()?;

        // Prefill tokens
        chart
            .draw_series(LineSeries::new(
                self.data_points.iter().map(|p| (p.time, p.prefill_tokens)),
                &RED,
            ))?
            .label("Prefill Tokens")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        // Decode tokens
        chart
            .draw_series(LineSeries::new(
                self.data_points.iter().map(|p| (p.time, p.decode_tokens)),
                &GREEN,
            ))?
            .label("Decode Tokens")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
        Ok(())
    }
}
