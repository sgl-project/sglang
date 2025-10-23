// These modules are used by tests and benchmarks
#![allow(dead_code)]

pub mod mock_mcp_server;
pub mod mock_openai_server;
pub mod mock_worker;
pub mod streaming_helpers;
pub mod test_app;

use std::{
    fs,
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
};

use serde_json::json;
use sglang_router_rs::{
    config::RouterConfig,
    core::{LoadMonitor, WorkerRegistry},
    data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    },
    middleware::TokenBucket,
    policies::PolicyRegistry,
    protocols::common::{Function, Tool},
    server::AppContext,
};

/// Helper function to create AppContext for tests
pub fn create_test_context(config: RouterConfig) -> Arc<AppContext> {
    let client = reqwest::Client::new();

    // Initialize rate limiter
    let rate_limiter = match config.max_concurrent_requests {
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

    // Initialize registries
    let worker_registry = Arc::new(WorkerRegistry::new());
    let policy_registry = Arc::new(PolicyRegistry::new(config.policy.clone()));

    // Initialize storage backends (Memory for tests)
    let response_storage = Arc::new(MemoryResponseStorage::new());
    let conversation_storage = Arc::new(MemoryConversationStorage::new());
    let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());

    // Initialize load monitor
    let load_monitor = Some(Arc::new(LoadMonitor::new(
        worker_registry.clone(),
        policy_registry.clone(),
        client.clone(),
        config.worker_startup_check_interval_secs,
    )));

    // Create empty OnceLock for worker job queue and workflow engine
    let worker_job_queue = Arc::new(OnceLock::new());
    let workflow_engine = Arc::new(OnceLock::new());

    let app_context = Arc::new(AppContext::new(
        config,
        client,
        rate_limiter,
        None, // tokenizer
        None, // reasoning_parser_factory
        None, // tool_parser_factory
        worker_registry,
        policy_registry,
        response_storage,
        conversation_storage,
        conversation_item_storage,
        load_monitor,
        worker_job_queue,
        workflow_engine,
    ));

    // Initialize JobQueue after AppContext is created
    let weak_context = Arc::downgrade(&app_context);
    let job_queue = sglang_router_rs::core::JobQueue::new(
        sglang_router_rs::core::JobQueueConfig::default(),
        weak_context,
    );
    app_context
        .worker_job_queue
        .set(job_queue)
        .expect("JobQueue should only be initialized once");

    // Initialize WorkflowEngine and register workflows
    use sglang_router_rs::core::workflow::{
        create_worker_registration_workflow, create_worker_removal_workflow, WorkflowEngine,
    };
    let engine = Arc::new(WorkflowEngine::new());
    engine.register_workflow(create_worker_registration_workflow());
    engine.register_workflow(create_worker_removal_workflow());
    app_context
        .workflow_engine
        .set(engine)
        .expect("WorkflowEngine should only be initialized once");

    app_context
}

// Tokenizer download configuration
const TINYLLAMA_TOKENIZER_URL: &str =
    "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json";
const CACHE_DIR: &str = ".tokenizer_cache";
const TINYLLAMA_TOKENIZER_FILENAME: &str = "tinyllama_tokenizer.json";

// Global mutex to prevent concurrent downloads
static DOWNLOAD_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

/// Downloads the TinyLlama tokenizer from HuggingFace if not already cached.
/// Returns the path to the cached tokenizer file.
///
/// This function is thread-safe and will only download the tokenizer once
/// even if called from multiple threads concurrently.
pub fn ensure_tokenizer_cached() -> PathBuf {
    // Get or initialize the mutex
    let mutex = DOWNLOAD_MUTEX.get_or_init(|| Mutex::new(()));

    // Lock to ensure only one thread downloads at a time
    let _guard = mutex.lock().unwrap();

    let cache_dir = PathBuf::from(CACHE_DIR);
    let tokenizer_path = cache_dir.join(TINYLLAMA_TOKENIZER_FILENAME);

    // Create cache directory if it doesn't exist
    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");
    }

    // Download tokenizer if not already cached
    if !tokenizer_path.exists() {
        println!("Downloading TinyLlama tokenizer from HuggingFace...");

        // Use blocking reqwest client since we're in tests/benchmarks
        let client = reqwest::blocking::Client::new();
        let response = client
            .get(TINYLLAMA_TOKENIZER_URL)
            .send()
            .expect("Failed to download tokenizer");

        if !response.status().is_success() {
            panic!("Failed to download tokenizer: HTTP {}", response.status());
        }

        let content = response.bytes().expect("Failed to read tokenizer content");

        if content.len() < 100 {
            panic!("Downloaded content too small: {} bytes", content.len());
        }

        fs::write(&tokenizer_path, content).expect("Failed to write tokenizer to cache");
        println!(
            "Tokenizer downloaded and cached successfully ({} bytes)",
            tokenizer_path.metadata().unwrap().len()
        );
    }

    tokenizer_path
}

/// Common test prompts for consistency across tests
pub const TEST_PROMPTS: [&str; 4] = [
    "deep learning is",
    "Deep learning is",
    "has anyone seen nemo lately",
    "another prompt",
];

/// Pre-computed hashes for verification
pub const EXPECTED_HASHES: [u64; 4] = [
    1209591529327510910,
    4181375434596349981,
    6245658446118930933,
    5097285695902185237,
];

/// Create a comprehensive set of test tools covering all parser test scenarios
#[allow(dead_code)]
pub fn create_test_tools() -> Vec<Tool> {
    vec![
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "search".to_string(),
                description: Some("Search for information".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather information".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "location": {"type": "string"},
                        "date": {"type": "string"},
                        "units": {"type": "string"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "calculate".to_string(),
                description: Some("Perform calculations".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "translate".to_string(),
                description: Some("Translate text".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "to": {"type": "string"},
                        "target_lang": {"type": "string"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_time".to_string(),
                description: Some("Get current time".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string"},
                        "format": {"type": "string"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_current_time".to_string(),
                description: Some("Get current time".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string"},
                        "format": {"type": "string"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "update_settings".to_string(),
                description: Some("Update settings".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "preferences": {"type": "object"},
                        "notifications": {"type": "boolean"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "ping".to_string(),
                description: Some("Ping service".to_string()),
                parameters: json!({"type": "object", "properties": {}}),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "test".to_string(),
                description: Some("Test function".to_string()),
                parameters: json!({"type": "object", "properties": {}}),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "process".to_string(),
                description: Some("Process data".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "count": {"type": "number"},
                        "rate": {"type": "number"},
                        "enabled": {"type": "boolean"},
                        "data": {"type": "object"},
                        "text": {"type": "string"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "web_search".to_string(),
                description: Some("Search the web".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "num_results": {"type": "number"},
                        "search_type": {"type": "string"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_tourist_attractions".to_string(),
                description: Some("Get tourist attractions".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "config".to_string(),
                description: Some("Configuration function".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "debug": {"type": "boolean"},
                        "verbose": {"type": "boolean"},
                        "optional": {"type": "null"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "test_func".to_string(),
                description: Some("Test function".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "bool_true": {"type": "boolean"},
                        "bool_false": {"type": "boolean"},
                        "none_val": {"type": "null"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "create".to_string(),
                description: Some("Create resource".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "add".to_string(),
                description: Some("Add operation".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "calc".to_string(),
                description: Some("Calculate".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "func1".to_string(),
                description: Some("Function 1".to_string()),
                parameters: json!({"type": "object", "properties": {}}),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "func2".to_string(),
                description: Some("Function 2".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "y": {"type": "number"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "tool1".to_string(),
                description: Some("Tool 1".to_string()),
                parameters: json!({"type": "object", "properties": {}}),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "tool2".to_string(),
                description: Some("Tool 2".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "y": {"type": "number"}
                    }
                }),
                strict: None,
            },
        },
    ]
}
