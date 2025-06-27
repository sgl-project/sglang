use reqwest::{Client, ClientBuilder};
use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::sync::Semaphore;
use tracing::{debug, info};

#[derive(Debug, Clone)]
pub struct ConnectionPoolConfig {
    pub max_idle_per_host: usize,
    pub idle_timeout_secs: u64,
    pub connect_timeout_secs: u64,
    pub request_timeout_secs: u64,
    pub pool_timeout_secs: u64,
    pub tcp_keepalive_secs: Option<u64>,
    pub http2_enabled: bool,
    pub http2_keep_alive_interval_secs: Option<u64>,
    pub http2_keep_alive_timeout_secs: u64,
    pub max_concurrent_connections: Option<u64>,
    pub fd_soft_limit_threshold: f64,
    pub fd_check_threshold: u64, // Only check FD when active connections exceed this
    pub metrics_sampling_rate: f64, // Sample metrics (0.0-1.0, 1.0 = all requests)
}

impl Default for ConnectionPoolConfig {
    fn default() -> Self {
        Self {
            max_idle_per_host: 100,
            idle_timeout_secs: 90,
            connect_timeout_secs: 10,
            request_timeout_secs: 600,
            pool_timeout_secs: 30,
            tcp_keepalive_secs: Some(30),
            http2_enabled: false, // Default to HTTP/1.1 for compatibility
            http2_keep_alive_interval_secs: Some(10),
            http2_keep_alive_timeout_secs: 20,
            max_concurrent_connections: Some(25000),
            fd_soft_limit_threshold: 0.8,
            fd_check_threshold: 20_000,
            metrics_sampling_rate: 1.0, // Sample all requests by default
        }
    }
}

#[derive(Debug)]
pub struct ConnectionPool {
    client: Client,
    connection_semaphore: Option<Arc<Semaphore>>,
    active_connections: Arc<AtomicU64>,
    max_connections: Option<u64>,
    fd_stats: FdStats,
    #[allow(dead_code)]
    metrics_sampling_rate: f64,
}

impl ConnectionPool {
    pub fn new(config: ConnectionPoolConfig) -> Result<Self, String> {
        info!("Initializing connection pool with config: {:?}", config);

        let active_connections = Arc::new(AtomicU64::new(0));
        let max_connections = config.max_concurrent_connections;
        let fd_stats = FdStats::new(config.fd_soft_limit_threshold, config.fd_check_threshold);
        fd_stats.start_monitoring(active_connections.clone(), max_connections);

        let client = build_client(&config)?;
        let connection_semaphore =
            max_connections.map(|max| Arc::new(Semaphore::new(max as usize)));

        Ok(Self {
            client,
            connection_semaphore,
            active_connections,
            max_connections,
            fd_stats,
            metrics_sampling_rate: config.metrics_sampling_rate,
        })
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    #[inline]
    pub async fn request_with_metrics(
        &self,
        request_builder: reqwest::RequestBuilder,
        worker_url: &str,
    ) -> Result<reqwest::Response, reqwest::Error> {
        let start = std::time::Instant::now();

        let result = request_builder.send().await;

        // Only record metrics after request completes
        let duration = start.elapsed().as_secs_f64();
        if let Ok(response) = &result {
            let status = response.status().as_u16().to_string();
            metrics::histogram!(
                "sgl_router_outbound_request_duration_seconds",
                "worker" => worker_url.to_string(),
                "status" => status
            )
            .record(duration);
        } else {
            metrics::counter!(
                "sgl_router_outbound_requests_failed_total",
                "worker" => worker_url.to_string()
            )
            .increment(1);
        }

        result
    }

    #[inline(always)]
    pub async fn acquire_connection_permit(&self) -> Result<ConnectionPermitGuard, String> {
        // Fast path: no semaphore (unlimited connections)
        if self.connection_semaphore.is_none() {
            let count = self.active_connections.fetch_add(1, Ordering::Relaxed) + 1;
            metrics::counter!("sgl_router_inbound_connections_accepted").increment(1);
            metrics::gauge!("sgl_router_inbound_connections_active").set(count as f64);
            return Ok(ConnectionPermitGuard::new(
                None,
                self.active_connections.clone(),
            ));
        }

        // Only check FD limit when we have many active connections
        let active = self.active_connections.load(Ordering::Relaxed);
        if active > self.fd_stats.check_threshold() && self.fd_stats.is_approaching_limit() {
            metrics::counter!("sgl_router_inbound_connections_rejected", "reason" => "fd_limit")
                .increment(1);
            return Err("Server approaching file descriptor limit".to_string());
        }

        // Try to acquire semaphore permit
        let semaphore = self.connection_semaphore.as_ref().unwrap();
        match semaphore.clone().try_acquire_owned() {
            Ok(permit) => {
                let count = self.active_connections.fetch_add(1, Ordering::Relaxed) + 1;
                metrics::counter!("sgl_router_inbound_connections_accepted").increment(1);
                metrics::gauge!("sgl_router_inbound_connections_active").set(count as f64);
                Ok(ConnectionPermitGuard::new(
                    Some(permit),
                    self.active_connections.clone(),
                ))
            }
            Err(_) => {
                metrics::counter!("sgl_router_inbound_connections_rejected", "reason" => "capacity").increment(1);
                Err("Connection limit reached".to_string())
            }
        }
    }

    pub fn connection_stats(&self) -> ConnectionStats {
        ConnectionStats {
            active_connections: self.active_connections.load(Ordering::Relaxed),
            max_connections: self.max_connections,
        }
    }
}

fn build_client(config: &ConnectionPoolConfig) -> Result<Client, String> {
    let mut builder = ClientBuilder::new()
        .pool_idle_timeout(Duration::from_secs(config.idle_timeout_secs))
        .pool_max_idle_per_host(config.max_idle_per_host)
        .connect_timeout(Duration::from_secs(config.connect_timeout_secs))
        .timeout(Duration::from_secs(config.request_timeout_secs))
        .tcp_nodelay(true);

    if let Some(keepalive) = config.tcp_keepalive_secs {
        builder = builder.tcp_keepalive(Duration::from_secs(keepalive));
    }

    if config.http2_enabled {
        // Enable HTTP/2 but let it negotiate properly (no prior knowledge)
        // This allows HTTP/1.1 servers to work and HTTP/2 servers to upgrade
        builder = builder.http2_adaptive_window(true);

        if let Some(interval) = config.http2_keep_alive_interval_secs {
            builder = builder
                .http2_keep_alive_interval(Duration::from_secs(interval))
                .http2_keep_alive_timeout(Duration::from_secs(
                    config.http2_keep_alive_timeout_secs,
                ));
        }
    }

    builder
        .build()
        .map_err(|e| format!("HTTP client build failed: {}", e))
}

#[derive(Debug, Clone)]
pub struct ConnectionStats {
    pub active_connections: u64,
    pub max_connections: Option<u64>,
}

pub struct ConnectionPermitGuard {
    _permit: Option<tokio::sync::OwnedSemaphorePermit>,
    active_connections: Arc<AtomicU64>,
}

impl ConnectionPermitGuard {
    fn new(
        permit: Option<tokio::sync::OwnedSemaphorePermit>,
        active_connections: Arc<AtomicU64>,
    ) -> Self {
        Self {
            _permit: permit,
            active_connections,
        }
    }
}

impl Drop for ConnectionPermitGuard {
    fn drop(&mut self) {
        let count = self.active_connections.fetch_sub(1, Ordering::Relaxed) - 1;
        debug!("Connection released, active: {}", count);
        metrics::gauge!("sgl_router_inbound_connections_active").set(count as f64);
    }
}

#[derive(Debug)]
pub struct FdStats {
    threshold: f64,
    cached_fd_limit: AtomicU64,
    check_threshold: u64,
}

impl FdStats {
    pub fn new(threshold: f64, check_threshold: u64) -> Self {
        let mut stats = Self {
            threshold,
            cached_fd_limit: AtomicU64::new(0),
            check_threshold,
        };
        // Initialize the cached limit
        stats.update_cached_limit();
        stats
    }

    #[inline(always)]
    pub fn check_threshold(&self) -> u64 {
        self.check_threshold
    }

    #[inline]
    pub fn is_approaching_limit(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            // Use cached limit for fast check
            let limit = self.cached_fd_limit.load(Ordering::Relaxed);
            if limit == 0 {
                return false;
            }

            if let Ok(entries) = std::fs::read_dir("/proc/self/fd") {
                let current_fds = entries.count() as u64;
                return current_fds > (limit as f64 * self.threshold) as u64;
            }
        }
        false
    }

    fn update_cached_limit(&mut self) {
        #[cfg(unix)]
        {
            use libc::{getrlimit, rlimit, RLIMIT_NOFILE};
            let mut rlim = rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            unsafe {
                if getrlimit(RLIMIT_NOFILE, &mut rlim) == 0 {
                    self.cached_fd_limit.store(rlim.rlim_cur, Ordering::Relaxed);
                }
            }
        }
    }

    pub fn start_monitoring(
        &self,
        active_connections: Arc<AtomicU64>,
        max_connections: Option<u64>,
    ) {
        #[allow(unused_variables)]
        let threshold = self.threshold;
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                let active = active_connections.load(Ordering::Relaxed);
                metrics::gauge!("sgl_router_active_connections").set(active as f64);

                if let Some(max) = max_connections {
                    metrics::gauge!("sgl_router_connection_utilization")
                        .set((active as f64 / max as f64) * 100.0);
                }

                #[cfg(target_os = "linux")]
                if let Ok(entries) = std::fs::read_dir("/proc/self/fd") {
                    let fd_count = entries.count() as f64;
                    metrics::gauge!("sgl_router_process_fd_count").set(fd_count);
                }

                #[cfg(unix)]
                {
                    use libc::{getrlimit, rlimit, RLIMIT_NOFILE};
                    let mut rlim = rlimit {
                        rlim_cur: 0,
                        rlim_max: 0,
                    };
                    unsafe {
                        if getrlimit(RLIMIT_NOFILE, &mut rlim) == 0 {
                            metrics::gauge!("sgl_router_fd_limit").set(rlim.rlim_cur as f64);
                            #[cfg(target_os = "linux")]
                            if let Ok(entries) = std::fs::read_dir("/proc/self/fd") {
                                let current_fds = entries.count() as f64;
                                let utilization = (current_fds / rlim.rlim_cur as f64) * 100.0;
                                metrics::gauge!("sgl_router_fd_utilization").set(utilization);
                                if utilization > threshold * 100.0 {
                                    tracing::warn!(
                                        "High FD utilization: {:.1}% ({}/{})",
                                        utilization,
                                        current_fds,
                                        rlim.rlim_cur
                                    );
                                }
                            }
                        }
                    }
                }
            }
        });
    }
}

/// Middleware for connection limiting
pub struct ConnectionLimitMiddleware {
    connection_pool: Arc<ConnectionPool>,
}

impl ConnectionLimitMiddleware {
    pub fn new(connection_pool: Arc<ConnectionPool>) -> Self {
        Self { connection_pool }
    }
}

impl<S, B> actix_web::dev::Transform<S, actix_web::dev::ServiceRequest>
    for ConnectionLimitMiddleware
where
    S: actix_web::dev::Service<
            actix_web::dev::ServiceRequest,
            Response = actix_web::dev::ServiceResponse<B>,
            Error = actix_web::Error,
        > + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = actix_web::dev::ServiceResponse<B>;
    type Error = actix_web::Error;
    type Transform = ConnectionLimitMiddlewareService<S>;
    type InitError = ();
    type Future = std::future::Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        std::future::ready(Ok(ConnectionLimitMiddlewareService {
            service: std::rc::Rc::new(service),
            connection_pool: self.connection_pool.clone(),
        }))
    }
}

pub struct ConnectionLimitMiddlewareService<S> {
    service: std::rc::Rc<S>,
    connection_pool: Arc<ConnectionPool>,
}

impl<S, B> actix_web::dev::Service<actix_web::dev::ServiceRequest>
    for ConnectionLimitMiddlewareService<S>
where
    S: actix_web::dev::Service<
            actix_web::dev::ServiceRequest,
            Response = actix_web::dev::ServiceResponse<B>,
            Error = actix_web::Error,
        > + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = actix_web::dev::ServiceResponse<B>;
    type Error = actix_web::Error;
    type Future =
        futures_util::future::LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    actix_web::dev::forward_ready!(service);

    fn call(&self, req: actix_web::dev::ServiceRequest) -> Self::Future {
        let service = self.service.clone();
        let connection_pool = self.connection_pool.clone();
        let start_time = std::time::Instant::now();

        Box::pin(async move {
            // Try to acquire connection permit
            match connection_pool.acquire_connection_permit().await {
                Ok(_guard) => {
                    // Process the request first
                    let res = service.call(req).await?;

                    // Record basic metrics after processing
                    let duration = start_time.elapsed().as_secs_f64();
                    metrics::histogram!("sgl_router_http_request_duration_seconds")
                        .record(duration);

                    // Guard will be dropped here, releasing the connection
                    Ok(res)
                }
                Err(err) => {
                    // Record rejection
                    metrics::counter!(
                        "sgl_router_http_requests_rejected_total",
                        "reason" => if err.contains("fd_limit") { "fd_limit" } else { "capacity" }
                    )
                    .increment(1);

                    // Return 503 Service Unavailable when at capacity
                    Err(actix_web::error::ErrorServiceUnavailable(format!(
                        "Server at capacity: {}",
                        err
                    )))
                }
            }
        })
    }
}
