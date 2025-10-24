use std::sync::{Arc, OnceLock};

use reqwest::Client;

use crate::{
    config::RouterConfig,
    core::{workflow::WorkflowEngine, JobQueue, LoadMonitor, WorkerRegistry},
    data_connector::{
        SharedConversationItemStorage, SharedConversationStorage, SharedResponseStorage,
    },
    middleware::TokenBucket,
    policies::PolicyRegistry,
    reasoning_parser::ParserFactory as ReasoningParserFactory,
    routers::router_manager::RouterManager,
    tokenizer::traits::Tokenizer,
    tool_parser::ParserFactory as ToolParserFactory,
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
    pub response_storage: SharedResponseStorage,
    pub conversation_storage: SharedConversationStorage,
    pub conversation_item_storage: SharedConversationItemStorage,
    pub load_monitor: Option<Arc<LoadMonitor>>,
    pub configured_reasoning_parser: Option<String>,
    pub configured_tool_parser: Option<String>,
    pub worker_job_queue: Arc<OnceLock<Arc<JobQueue>>>,
    pub workflow_engine: Arc<OnceLock<Arc<WorkflowEngine>>>,
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
    response_storage: Option<SharedResponseStorage>,
    conversation_storage: Option<SharedConversationStorage>,
    conversation_item_storage: Option<SharedConversationItemStorage>,
    load_monitor: Option<Arc<LoadMonitor>>,
    worker_job_queue: Option<Arc<OnceLock<Arc<JobQueue>>>>,
    workflow_engine: Option<Arc<OnceLock<Arc<WorkflowEngine>>>>,
}

impl AppContext {
    pub fn builder() -> AppContextBuilder {
        AppContextBuilder::new()
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

    pub fn response_storage(mut self, response_storage: SharedResponseStorage) -> Self {
        self.response_storage = Some(response_storage);
        self
    }

    pub fn conversation_storage(mut self, conversation_storage: SharedConversationStorage) -> Self {
        self.conversation_storage = Some(conversation_storage);
        self
    }

    pub fn conversation_item_storage(
        mut self,
        conversation_item_storage: SharedConversationItemStorage,
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

    pub fn build(self) -> Result<AppContext, AppContextBuildError> {
        let router_config = self
            .router_config
            .ok_or(AppContextBuildError("router_config"))?;
        let configured_reasoning_parser = router_config.reasoning_parser.clone();
        let configured_tool_parser = router_config.tool_call_parser.clone();

        Ok(AppContext {
            client: self.client.ok_or(AppContextBuildError("client"))?,
            router_config,
            rate_limiter: self.rate_limiter,
            tokenizer: self.tokenizer,
            reasoning_parser_factory: self.reasoning_parser_factory,
            tool_parser_factory: self.tool_parser_factory,
            worker_registry: self
                .worker_registry
                .ok_or(AppContextBuildError("worker_registry"))?,
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
            worker_job_queue: self
                .worker_job_queue
                .ok_or(AppContextBuildError("worker_job_queue"))?,
            workflow_engine: self
                .workflow_engine
                .ok_or(AppContextBuildError("workflow_engine"))?,
        })
    }
}

impl Default for AppContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}
