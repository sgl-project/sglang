use std::{
    sync::{Arc, OnceLock},
    time::Duration,
};

use reqwest::Client;
use tracing::debug;

use crate::{
    config::RouterConfig,
    core::{ConnectionMode, JobQueue, LoadMonitor, WorkerRegistry, WorkerService},
    data_connector::{
        create_storage, ConversationItemStorage, ConversationStorage, ResponseStorage,
    },
    mcp::McpManager,
    middleware::TokenBucket,
    policies::PolicyRegistry,
    reasoning_parser::ParserFactory as ReasoningParserFactory,
    routers::router_manager::RouterManager,
    tokenizer::{
        cache::{CacheConfig, CachedTokenizer},
        factory as tokenizer_factory,
        traits::Tokenizer,
    },
    tool_parser::ParserFactory as ToolParserFactory,
    wasm::{config::WasmRuntimeConfig, module_manager::WasmModuleManager},
    workflow::WorkflowEngine,
};

/// Error type for AppContext builder
#[derive(Debug)]
pub struct AppContextBuildError(&'static str);

impl std::fmt::Display for AppContextBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Missing required field: {}", self.0)
    }
}

impl std::error::Error for AppContextBuildError {}

#[derive(Clone)]
pub struct AppContext {
    pub client: Client,
    pub router_config: RouterConfig,
    pub rate_limiter: Option<Arc<TokenBucket>>,
    pub tokenizer: Option<Arc<dyn Tokenizer>>,
    pub reasoning_parser_factory: Option<ReasoningParserFactory>,
    pub tool_parser_factory: Option<ToolParserFactory>,
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub router_manager: Option<Arc<RouterManager>>,
    pub response_storage: Arc<dyn ResponseStorage>,
    pub conversation_storage: Arc<dyn ConversationStorage>,
    pub conversation_item_storage: Arc<dyn ConversationItemStorage>,
    pub load_monitor: Option<Arc<LoadMonitor>>,
    pub configured_reasoning_parser: Option<String>,
    pub configured_tool_parser: Option<String>,
    pub worker_job_queue: Arc<OnceLock<Arc<JobQueue>>>,
    pub workflow_engine: Arc<OnceLock<Arc<WorkflowEngine>>>,
    pub mcp_manager: Arc<OnceLock<Arc<McpManager>>>,
    pub wasm_manager: Option<Arc<WasmModuleManager>>,
    pub worker_service: Arc<WorkerService>,
}

pub struct AppContextBuilder {
    client: Option<Client>,
    router_config: Option<RouterConfig>,
    rate_limiter: Option<Arc<TokenBucket>>,
    tokenizer: Option<Arc<dyn Tokenizer>>,
    reasoning_parser_factory: Option<ReasoningParserFactory>,
    tool_parser_factory: Option<ToolParserFactory>,
    worker_registry: Option<Arc<WorkerRegistry>>,
    policy_registry: Option<Arc<PolicyRegistry>>,
    router_manager: Option<Arc<RouterManager>>,
    response_storage: Option<Arc<dyn ResponseStorage>>,
    conversation_storage: Option<Arc<dyn ConversationStorage>>,
    conversation_item_storage: Option<Arc<dyn ConversationItemStorage>>,
    load_monitor: Option<Arc<LoadMonitor>>,
    worker_job_queue: Option<Arc<OnceLock<Arc<JobQueue>>>>,
    workflow_engine: Option<Arc<OnceLock<Arc<WorkflowEngine>>>>,
    mcp_manager: Option<Arc<OnceLock<Arc<McpManager>>>>,
    wasm_manager: Option<Arc<WasmModuleManager>>,
}

impl AppContext {
    pub fn builder() -> AppContextBuilder {
        AppContextBuilder::new()
    }

    /// Create AppContext from config with all components initialized
    /// This is the main entry point that replaces ~194 lines of initialization in server.rs
    pub async fn from_config(
        router_config: RouterConfig,
        request_timeout_secs: u64,
    ) -> Result<Self, String> {
        AppContextBuilder::from_config(router_config, request_timeout_secs)
            .await?
            .build()
            .map_err(|e| e.to_string())
    }
}

impl AppContextBuilder {
    pub fn new() -> Self {
        Self {
            client: None,
            router_config: None,
            rate_limiter: None,
            tokenizer: None,
            reasoning_parser_factory: None,
            tool_parser_factory: None,
            worker_registry: None,
            policy_registry: None,
            router_manager: None,
            response_storage: None,
            conversation_storage: None,
            conversation_item_storage: None,
            load_monitor: None,
            worker_job_queue: None,
            workflow_engine: None,
            mcp_manager: None,
            wasm_manager: None,
        }
    }

    pub fn client(mut self, client: Client) -> Self {
        self.client = Some(client);
        self
    }

    pub fn router_config(mut self, router_config: RouterConfig) -> Self {
        self.router_config = Some(router_config);
        self
    }

    pub fn rate_limiter(mut self, rate_limiter: Option<Arc<TokenBucket>>) -> Self {
        self.rate_limiter = rate_limiter;
        self
    }

    pub fn tokenizer(mut self, tokenizer: Option<Arc<dyn Tokenizer>>) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    pub fn reasoning_parser_factory(
        mut self,
        reasoning_parser_factory: Option<ReasoningParserFactory>,
    ) -> Self {
        self.reasoning_parser_factory = reasoning_parser_factory;
        self
    }

    pub fn tool_parser_factory(mut self, tool_parser_factory: Option<ToolParserFactory>) -> Self {
        self.tool_parser_factory = tool_parser_factory;
        self
    }

    pub fn worker_registry(mut self, worker_registry: Arc<WorkerRegistry>) -> Self {
        self.worker_registry = Some(worker_registry);
        self
    }

    pub fn policy_registry(mut self, policy_registry: Arc<PolicyRegistry>) -> Self {
        self.policy_registry = Some(policy_registry);
        self
    }

    pub fn router_manager(mut self, router_manager: Option<Arc<RouterManager>>) -> Self {
        self.router_manager = router_manager;
        self
    }

    pub fn response_storage(mut self, response_storage: Arc<dyn ResponseStorage>) -> Self {
        self.response_storage = Some(response_storage);
        self
    }

    pub fn conversation_storage(
        mut self,
        conversation_storage: Arc<dyn ConversationStorage>,
    ) -> Self {
        self.conversation_storage = Some(conversation_storage);
        self
    }

    pub fn conversation_item_storage(
        mut self,
        conversation_item_storage: Arc<dyn ConversationItemStorage>,
    ) -> Self {
        self.conversation_item_storage = Some(conversation_item_storage);
        self
    }

    pub fn load_monitor(mut self, load_monitor: Option<Arc<LoadMonitor>>) -> Self {
        self.load_monitor = load_monitor;
        self
    }

    pub fn worker_job_queue(mut self, worker_job_queue: Arc<OnceLock<Arc<JobQueue>>>) -> Self {
        self.worker_job_queue = Some(worker_job_queue);
        self
    }

    pub fn workflow_engine(mut self, workflow_engine: Arc<OnceLock<Arc<WorkflowEngine>>>) -> Self {
        self.workflow_engine = Some(workflow_engine);
        self
    }

    pub fn mcp_manager(mut self, mcp_manager: Arc<OnceLock<Arc<McpManager>>>) -> Self {
        self.mcp_manager = Some(mcp_manager);
        self
    }

    pub fn wasm_manager(mut self, wasm_manager: Option<Arc<WasmModuleManager>>) -> Self {
        self.wasm_manager = wasm_manager;
        self
    }

    pub fn build(self) -> Result<AppContext, AppContextBuildError> {
        let router_config = self
            .router_config
            .ok_or(AppContextBuildError("router_config"))?;
        let configured_reasoning_parser = router_config.reasoning_parser.clone();
        let configured_tool_parser = router_config.tool_call_parser.clone();

        let worker_registry = self
            .worker_registry
            .ok_or(AppContextBuildError("worker_registry"))?;
        let worker_job_queue = self
            .worker_job_queue
            .ok_or(AppContextBuildError("worker_job_queue"))?;

        // Create WorkerService from the already-built components
        let worker_service = Arc::new(WorkerService::new(
            worker_registry.clone(),
            worker_job_queue.clone(),
            router_config.clone(),
        ));

        Ok(AppContext {
            client: self.client.ok_or(AppContextBuildError("client"))?,
            router_config,
            rate_limiter: self.rate_limiter,
            tokenizer: self.tokenizer,
            reasoning_parser_factory: self.reasoning_parser_factory,
            tool_parser_factory: self.tool_parser_factory,
            worker_registry,
            policy_registry: self
                .policy_registry
                .ok_or(AppContextBuildError("policy_registry"))?,
            router_manager: self.router_manager,
            response_storage: self
                .response_storage
                .ok_or(AppContextBuildError("response_storage"))?,
            conversation_storage: self
                .conversation_storage
                .ok_or(AppContextBuildError("conversation_storage"))?,
            conversation_item_storage: self
                .conversation_item_storage
                .ok_or(AppContextBuildError("conversation_item_storage"))?,
            load_monitor: self.load_monitor,
            configured_reasoning_parser,
            configured_tool_parser,
            worker_job_queue,
            workflow_engine: self
                .workflow_engine
                .ok_or(AppContextBuildError("workflow_engine"))?,
            mcp_manager: self
                .mcp_manager
                .ok_or(AppContextBuildError("mcp_manager"))?,
            wasm_manager: self.wasm_manager,
            worker_service,
        })
    }

    /// Initialize AppContext from config - creates ALL components
    /// This replaces ~194 lines of initialization logic from server.rs
    pub async fn from_config(
        router_config: RouterConfig,
        request_timeout_secs: u64,
    ) -> Result<Self, String> {
        Ok(Self::new()
            .with_client(&router_config, request_timeout_secs)?
            .maybe_rate_limiter(&router_config)
            .maybe_tokenizer(&router_config)?
            .maybe_reasoning_parser_factory(&router_config)
            .maybe_tool_parser_factory(&router_config)
            .with_worker_registry()
            .with_policy_registry(&router_config)
            .with_storage(&router_config)?
            .with_load_monitor(&router_config)
            .with_worker_job_queue()
            .with_workflow_engine()
            .with_mcp_manager(&router_config)
            .await?
            .with_wasm_manager(&router_config)?
            .router_config(router_config))
    }

    /// Create HTTP client with TLS/mTLS configuration
    fn with_client(mut self, config: &RouterConfig, timeout_secs: u64) -> Result<Self, String> {
        // FIXME: Current implementation creates a single HTTP client for all workers.
        // This works well for single security domain deployments where all workers share
        // the same CA and can accept the same client certificate.
        //
        // For multi-domain deployments (e.g., different model families with different CAs),
        // this architecture needs significant refactoring:
        // 1. Move client creation into worker registration workflow (per-worker clients)
        // 2. Store client per worker in WorkerRegistry
        // 3. Update PDRouter and other routers to fetch client from worker
        // 4. Add per-worker TLS spec in WorkerConfigRequest
        //
        // Current single-domain approach is sufficient for most deployments.
        //
        // Use rustls TLS backend when TLS/mTLS is configured (client cert or CA certs provided).
        // This ensures proper PKCS#8 key format support. For plain HTTP workers, use default
        // backend to avoid unnecessary TLS initialization overhead.
        let has_tls_config = config.client_identity.is_some() || !config.ca_certificates.is_empty();

        let mut client_builder = Client::builder()
            .pool_idle_timeout(Some(Duration::from_secs(50)))
            .pool_max_idle_per_host(500)
            .timeout(Duration::from_secs(timeout_secs))
            .connect_timeout(Duration::from_secs(10))
            .tcp_nodelay(true)
            .tcp_keepalive(Some(Duration::from_secs(30)));

        // Force rustls backend when TLS is configured
        if has_tls_config {
            client_builder = client_builder.use_rustls_tls();
            debug!("Using rustls TLS backend for TLS/mTLS connections");
        }

        // Configure mTLS client identity if provided (certificates already loaded during config creation)
        if let Some(identity_pem) = &config.client_identity {
            let identity = reqwest::Identity::from_pem(identity_pem)
                .map_err(|e| format!("Failed to create client identity: {}", e))?;
            client_builder = client_builder.identity(identity);
            debug!("mTLS client authentication enabled");
        }

        // Add CA certificates for verifying worker TLS (certificates already loaded during config creation)
        for ca_cert in &config.ca_certificates {
            let cert = reqwest::Certificate::from_pem(ca_cert)
                .map_err(|e| format!("Failed to add CA certificate: {}", e))?;
            client_builder = client_builder.add_root_certificate(cert);
        }
        if !config.ca_certificates.is_empty() {
            debug!(
                "Added {} CA certificate(s) for worker verification",
                config.ca_certificates.len()
            );
        }

        let client = client_builder
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        self.client = Some(client);
        Ok(self)
    }

    /// Create rate limiter based on config
    fn maybe_rate_limiter(mut self, config: &RouterConfig) -> Self {
        self.rate_limiter = match config.max_concurrent_requests {
            n if n <= 0 => None,
            n => {
                let rate_limit_tokens = config
                    .rate_limit_tokens_per_second
                    .filter(|&t| t > 0)
                    .unwrap_or(n);
                Some(Arc::new(TokenBucket::new(
                    n as usize,
                    rate_limit_tokens as usize,
                )))
            }
        };
        self
    }

    /// Create tokenizer for gRPC mode
    fn maybe_tokenizer(mut self, config: &RouterConfig) -> Result<Self, String> {
        if matches!(config.connection_mode, ConnectionMode::Grpc { .. }) {
            let tokenizer_path = config
                .tokenizer_path
                .clone()
                .or_else(|| config.model_path.clone())
                .ok_or_else(|| {
                    "gRPC mode requires either --tokenizer-path or --model-path to be specified"
                        .to_string()
                })?;

            let base_tokenizer = tokenizer_factory::create_tokenizer_with_chat_template_blocking(
                &tokenizer_path,
                config.chat_template.as_deref(),
            )
            .map_err(|e| {
                format!(
                    "Failed to create tokenizer from '{}': {}. \
                    Ensure the path is valid and points to a tokenizer file (tokenizer.json) \
                    or a HuggingFace model ID. For directories, ensure they contain tokenizer files.",
                    tokenizer_path, e
                )
            })?;

            // Conditionally wrap with caching layer if at least one cache is enabled
            self.tokenizer = if config.tokenizer_cache.enable_l0 || config.tokenizer_cache.enable_l1
            {
                let cache_config = CacheConfig {
                    enable_l0: config.tokenizer_cache.enable_l0,
                    l0_max_entries: config.tokenizer_cache.l0_max_entries,
                    enable_l1: config.tokenizer_cache.enable_l1,
                    l1_max_memory: config.tokenizer_cache.l1_max_memory,
                };
                Some(Arc::new(CachedTokenizer::new(base_tokenizer, cache_config))
                    as Arc<dyn Tokenizer>)
            } else {
                // Use base tokenizer directly without caching
                Some(base_tokenizer)
            };
        }

        Ok(self)
    }

    /// Create reasoning parser factory for gRPC mode
    fn maybe_reasoning_parser_factory(mut self, config: &RouterConfig) -> Self {
        if matches!(config.connection_mode, ConnectionMode::Grpc { .. }) {
            self.reasoning_parser_factory = Some(ReasoningParserFactory::new());
        }
        self
    }

    /// Create tool parser factory for gRPC mode
    fn maybe_tool_parser_factory(mut self, config: &RouterConfig) -> Self {
        if matches!(config.connection_mode, ConnectionMode::Grpc { .. }) {
            self.tool_parser_factory = Some(ToolParserFactory::new());
        }
        self
    }

    /// Create worker registry
    fn with_worker_registry(mut self) -> Self {
        self.worker_registry = Some(Arc::new(WorkerRegistry::new()));
        self
    }

    /// Create policy registry
    fn with_policy_registry(mut self, config: &RouterConfig) -> Self {
        self.policy_registry = Some(Arc::new(PolicyRegistry::new(config.policy.clone())));
        if config.dp_minimum_tokens_scheduler {
            self.policy_registry
                .as_ref()
                .unwrap()
                .enable_dp_minimum_tokens_scheduler();
        }
        self
    }

    /// Create all storage backends using the factory function
    fn with_storage(mut self, config: &RouterConfig) -> Result<Self, String> {
        let (response_storage, conversation_storage, conversation_item_storage) =
            create_storage(config)?;

        self.response_storage = Some(response_storage);
        self.conversation_storage = Some(conversation_storage);
        self.conversation_item_storage = Some(conversation_item_storage);

        Ok(self)
    }

    /// Create load monitor
    fn with_load_monitor(mut self, config: &RouterConfig) -> Self {
        let client = self
            .client
            .as_ref()
            .expect("client must be set before load monitor");
        self.load_monitor = Some(Arc::new(LoadMonitor::new(
            self.worker_registry
                .as_ref()
                .expect("worker_registry must be set")
                .clone(),
            self.policy_registry
                .as_ref()
                .expect("policy_registry must be set")
                .clone(),
            client.clone(),
            config.worker_load_check_interval_secs,
        )));
        self
    }

    /// Create worker job queue OnceLock container
    fn with_worker_job_queue(mut self) -> Self {
        self.worker_job_queue = Some(Arc::new(OnceLock::new()));
        self
    }

    /// Create workflow engine OnceLock container
    fn with_workflow_engine(mut self) -> Self {
        self.workflow_engine = Some(Arc::new(OnceLock::new()));
        self
    }

    /// Create and initialize MCP manager with empty config
    ///
    /// This initializes the MCP manager with an empty config and default settings.
    /// MCP servers will be registered later via the InitializeMcpServers job.
    async fn with_mcp_manager(mut self, _router_config: &RouterConfig) -> Result<Self, String> {
        // Create OnceLock container
        let mcp_manager_lock = Arc::new(OnceLock::new());

        // Always create with empty config and defaults
        debug!("Initializing MCP manager with empty config and default settings (5 min TTL, 100 max connections)");

        let empty_config = crate::mcp::McpConfig {
            servers: Vec::new(),
            pool: Default::default(),
            proxy: None,
            warmup: Vec::new(),
            inventory: Default::default(),
        };

        let manager = McpManager::with_defaults(empty_config)
            .await
            .map_err(|e| format!("Failed to initialize MCP manager with defaults: {}", e))?;

        // Store the initialized manager in the OnceLock
        mcp_manager_lock
            .set(Arc::new(manager))
            .map_err(|_| "Failed to set MCP manager in OnceLock".to_string())?;

        self.mcp_manager = Some(mcp_manager_lock);
        Ok(self)
    }

    /// Create wasm manager if enabled in config
    fn with_wasm_manager(mut self, config: &RouterConfig) -> Result<Self, String> {
        self.wasm_manager = if config.enable_wasm {
            Some(Arc::new(
                WasmModuleManager::new(WasmRuntimeConfig::default())
                    .map_err(|e| format!("Failed to initialize WASM module manager: {}", e))?,
            ))
        } else {
            None
        };
        Ok(self)
    }
}

impl Default for AppContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}
