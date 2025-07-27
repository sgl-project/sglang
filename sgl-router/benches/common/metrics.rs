use std::time::Duration;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct LoadTestMetrics {
    pub total_requests: usize,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub total_time: Duration,
    pub average_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub throughput_rps: f64,
    pub worker_distribution: Vec<usize>,
    // Configuration details
    pub config: LoadTestConfig,
}

#[derive(Debug, Clone)]
pub struct LoadTestConfig {
    pub workers: usize,
    pub batch_size: usize,
    pub worker_delay_ms: u64,
    pub router_port: u16,
    pub routing_mode: String,
    pub policy: String,
    pub prefill_workers: usize,
    pub decode_workers: usize,
}

#[allow(dead_code)]
impl LoadTestMetrics {
    pub fn calculate_percentiles(avg_latency_ms: f64) -> (f64, f64, f64) {
        // Simplified percentile estimation
        let p50 = avg_latency_ms * 0.8;
        let p95 = avg_latency_ms * 1.5;
        let p99 = avg_latency_ms * 2.0;
        (p50, p95, p99)
    }

    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(80));
        println!("                          LOAD TEST RESULTS                          ");
        println!("{}", "=".repeat(80));

        println!("\n📊 Summary:");
        println!("  Total requests:       {}", self.total_requests);
        println!(
            "  Completed:           {} ({:.1}%)",
            self.successful_requests + self.failed_requests,
            ((self.successful_requests + self.failed_requests) as f64 / self.total_requests as f64)
                * 100.0
        );
        println!(
            "  Successful:          {} ({:.1}%)",
            self.successful_requests,
            if self.total_requests > 0 {
                (self.successful_requests as f64 / self.total_requests as f64) * 100.0
            } else {
                0.0
            }
        );
        println!(
            "  Failed:              {} ({:.1}%)",
            self.failed_requests,
            if self.total_requests > 0 {
                (self.failed_requests as f64 / self.total_requests as f64) * 100.0
            } else {
                0.0
            }
        );

        println!("\n⏱️  Performance:");
        println!(
            "  Total time:          {:.2}s",
            self.total_time.as_secs_f64()
        );
        println!(
            "  Throughput:          {:.0} requests/second",
            self.throughput_rps
        );
        println!("  Average latency:     {:.2}ms", self.average_latency_ms);
        println!("  Estimated P50:       {:.2}ms", self.p50_latency_ms);
        println!("  Estimated P95:       {:.2}ms", self.p95_latency_ms);
        println!("  Estimated P99:       {:.2}ms", self.p99_latency_ms);

        println!("\n🔄 Worker Distribution:");
        for (i, &count) in self.worker_distribution.iter().enumerate() {
            let percentage = if self.successful_requests > 0 {
                (count as f64 / self.successful_requests as f64) * 100.0
            } else {
                0.0
            };

            let worker_type_str = if self.config.routing_mode == "pd" {
                if i < self.config.prefill_workers {
                    " (Prefill)"
                } else {
                    " (Decode)"
                }
            } else {
                ""
            };

            println!(
                "  Worker {}{}: {} requests ({:.1}%)",
                i + 1,
                worker_type_str,
                count,
                percentage
            );
        }

        println!("\n⚙️  Configuration:");
        if self.config.routing_mode == "pd" {
            println!("  Prefill workers:     {}", self.config.prefill_workers);
            println!("  Decode workers:      {}", self.config.decode_workers);
        } else {
            println!("  Workers:             {}", self.config.workers);
        }
        println!("  Batch size:          {}", self.config.batch_size);
        println!("  Worker delay:        {}ms", self.config.worker_delay_ms);
        println!("  Router port:         {}", self.config.router_port);
        println!("  Routing mode:        {}", self.config.routing_mode);
        println!("  Policy:              {}", self.config.policy);

        println!("\n{}", "=".repeat(80));
    }
}
