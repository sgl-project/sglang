// Essential PDLB types extracted for PD routing

use crate::core::{Worker, WorkerType};
use crate::openai_api_types::{CompletionRequest, StringOrArray};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// Custom error type for PD router operations
#[derive(Debug, thiserror::Error)]
pub enum PDRouterError {
    #[error("Worker already exists: {url}")]
    WorkerAlreadyExists { url: String },

    #[error("Worker not found: {url}")]
    WorkerNotFound { url: String },

    #[error("Lock acquisition failed: {operation}")]
    LockError { operation: String },

    #[error("Health check failed for worker: {url}")]
    HealthCheckFailed { url: String },

    #[error("Invalid worker configuration: {reason}")]
    InvalidConfiguration { reason: String },

    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Timeout waiting for worker: {url}")]
    Timeout { url: String },
}

// Helper functions for workers
pub fn api_path(url: &str, api_path: &str) -> String {
    if api_path.starts_with("/") {
        format!("{}{}", url, api_path)
    } else {
        format!("{}/{}", url, api_path)
    }
}

pub fn get_hostname(url: &str) -> String {
    // Simple hostname extraction without external dependencies
    let url = url
        .trim_start_matches("http://")
        .trim_start_matches("https://");
    url.split(':').next().unwrap_or("localhost").to_string()
}

// PD-specific routing policies
#[derive(Debug, Clone, PartialEq)]
pub enum PDSelectionPolicy {
    Random,
    PowerOfTwo,
    CacheAware {
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
    },
}
// Bootstrap types from PDLB
#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum SingleOrBatch<T> {
    Single(T),
    Batch(Vec<T>),
}

pub type InputIds = SingleOrBatch<Vec<i32>>;
pub type InputText = SingleOrBatch<String>;
pub type BootstrapHost = SingleOrBatch<String>;
pub type BootstrapPort = SingleOrBatch<Option<u16>>;
pub type BootstrapRoom = SingleOrBatch<u64>;

// Bootstrap trait for request handling
pub trait Bootstrap: Send + Sync {
    fn is_stream(&self) -> bool;
    fn get_batch_size(&self) -> Result<Option<usize>, String>;
    fn set_bootstrap_info(
        &mut self,
        bootstrap_host: BootstrapHost,
        bootstrap_port: BootstrapPort,
        bootstrap_room: BootstrapRoom,
    );

    fn add_bootstrap_info(&mut self, prefill_worker: &dyn Worker) -> Result<(), String> {
        let batch_size = self.get_batch_size()?;

        // Extract bootstrap port from prefill worker if it's a prefill type
        let bootstrap_port = match prefill_worker.worker_type() {
            WorkerType::Prefill { bootstrap_port } => bootstrap_port,
            _ => None,
        };

        let hostname = get_hostname(prefill_worker.url());

        if let Some(batch_size) = batch_size {
            self.set_bootstrap_info(
                BootstrapHost::Batch(vec![hostname; batch_size]),
                BootstrapPort::Batch(vec![bootstrap_port; batch_size]),
                // Use high-quality random numbers to minimize collision risk
                BootstrapRoom::Batch(
                    (0..batch_size)
                        .map(|_| {
                            // Generate a value in the range [0, 2^63 - 1] to match Python's random.randint(0, 2**63 - 1)
                            rand::random::<u64>() & (i64::MAX as u64)
                        })
                        .collect(),
                ),
            );
        } else {
            self.set_bootstrap_info(
                BootstrapHost::Single(hostname),
                BootstrapPort::Single(bootstrap_port),
                BootstrapRoom::Single(
                    // Generate a value in the range [0, 2^63 - 1] to match Python's random.randint(0, 2**63 - 1)
                    rand::random::<u64>() & (i64::MAX as u64),
                ),
            );
        }
        Ok(())
    }
}

// Request types
#[derive(Debug, Deserialize, Serialize)]
pub struct GenerateReqInput {
    pub text: Option<InputText>,
    pub input_ids: Option<InputIds>,
    #[serde(default)]
    pub stream: bool,
    pub bootstrap_host: Option<BootstrapHost>,
    pub bootstrap_port: Option<BootstrapPort>,
    pub bootstrap_room: Option<BootstrapRoom>,

    #[serde(flatten)]
    pub other: Value,
}

impl GenerateReqInput {
    pub fn get_batch_size(&self) -> Result<Option<usize>, String> {
        if self.text.is_some() && self.input_ids.is_some() {
            return Err("Both text and input_ids are present in the request".to_string());
        }

        // Check text batch
        if let Some(InputText::Batch(texts)) = &self.text {
            if texts.is_empty() {
                return Err("Batch text array is empty".to_string());
            }
            return Ok(Some(texts.len()));
        }

        // Check input_ids batch
        if let Some(InputIds::Batch(ids)) = &self.input_ids {
            if ids.is_empty() {
                return Err("Batch input_ids array is empty".to_string());
            }
            // Validate each sequence is not empty
            for (i, seq) in ids.iter().enumerate() {
                if seq.is_empty() {
                    return Err(format!("Input sequence at index {} is empty", i));
                }
            }
            return Ok(Some(ids.len()));
        }

        Ok(None)
    }
}

impl Bootstrap for GenerateReqInput {
    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_batch_size(&self) -> Result<Option<usize>, String> {
        self.get_batch_size()
    }

    fn set_bootstrap_info(
        &mut self,
        bootstrap_host: BootstrapHost,
        bootstrap_port: BootstrapPort,
        bootstrap_room: BootstrapRoom,
    ) {
        self.bootstrap_host = Some(bootstrap_host);
        self.bootstrap_port = Some(bootstrap_port);
        self.bootstrap_room = Some(bootstrap_room);
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatReqInput {
    #[serde(default)]
    pub stream: bool,
    pub bootstrap_host: Option<BootstrapHost>,
    pub bootstrap_port: Option<BootstrapPort>,
    pub bootstrap_room: Option<BootstrapRoom>,

    #[serde(flatten)]
    pub other: Value,
}

impl Bootstrap for ChatReqInput {
    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_batch_size(&self) -> Result<Option<usize>, String> {
        // Check if 'n' parameter is present and > 1
        if let Some(n_value) = self.other.get("n") {
            if let Some(n) = n_value.as_u64() {
                if n > 1 {
                    return Ok(Some(n as usize));
                }
            }
        }
        Ok(None)
    }

    fn set_bootstrap_info(
        &mut self,
        bootstrap_host: BootstrapHost,
        bootstrap_port: BootstrapPort,
        bootstrap_room: BootstrapRoom,
    ) {
        self.bootstrap_host = Some(bootstrap_host);
        self.bootstrap_port = Some(bootstrap_port);
        self.bootstrap_room = Some(bootstrap_room);
    }
}

// Bootstrap implementation for CompletionRequest to preserve OpenAI format
impl Bootstrap for CompletionRequest {
    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_batch_size(&self) -> Result<Option<usize>, String> {
        if let StringOrArray::Array(prompts) = &self.prompt {
            if prompts.is_empty() {
                return Err("Batch prompt array is empty".to_string());
            }
            return Ok(Some(prompts.len()));
        }

        // Single string prompt
        Ok(None)
    }

    fn set_bootstrap_info(
        &mut self,
        bootstrap_host: BootstrapHost,
        bootstrap_port: BootstrapPort,
        bootstrap_room: BootstrapRoom,
    ) {
        // Insert bootstrap_host - it serializes correctly whether Single or Batch
        if let Ok(host_value) = serde_json::to_value(&bootstrap_host) {
            self.other.insert("bootstrap_host".to_string(), host_value);
        }

        // Insert bootstrap_port - it serializes correctly whether Single or Batch
        if let Ok(port_value) = serde_json::to_value(&bootstrap_port) {
            self.other.insert("bootstrap_port".to_string(), port_value);
        }

        // Insert bootstrap_room - it serializes correctly whether Single or Batch
        if let Ok(room_value) = serde_json::to_value(&bootstrap_room) {
            self.other.insert("bootstrap_room".to_string(), room_value);
        }
    }
}

#[cfg(test)]
mod bootstrap_tests {
    use super::*;
    use crate::core::BasicWorker;
    use crate::openai_api_types::StringOrArray;

    #[test]
    fn test_completion_batch_size_with_array_prompt() {
        let req = CompletionRequest {
            model: "test".to_string(),
            prompt: StringOrArray::Array(vec!["prompt1".to_string(), "prompt2".to_string()]),
            n: None,
            other: serde_json::Map::new(),
            suffix: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: false,
            stream_options: None,
            logprobs: None,
            echo: false,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            best_of: None,
            logit_bias: None,
            user: None,
            seed: None,
        };

        // Should return batch size for array prompt
        assert_eq!(req.get_batch_size().unwrap(), Some(2));
    }

    #[test]
    fn test_completion_batch_size_with_single_prompt() {
        let req = CompletionRequest {
            model: "test".to_string(),
            prompt: StringOrArray::String("single prompt".to_string()),
            n: None,
            other: serde_json::Map::new(),
            suffix: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: false,
            stream_options: None,
            logprobs: None,
            echo: false,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            best_of: None,
            logit_bias: None,
            user: None,
            seed: None,
        };

        // Should return None for single prompt
        assert_eq!(req.get_batch_size().unwrap(), None);
    }

    #[test]
    fn test_completion_batch_size_with_n_parameter() {
        let req = CompletionRequest {
            model: "test".to_string(),
            prompt: StringOrArray::String("single prompt".to_string()),
            n: Some(3),
            other: serde_json::Map::new(),
            suffix: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: false,
            stream_options: None,
            logprobs: None,
            echo: false,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            best_of: None,
            logit_bias: None,
            user: None,
            seed: None,
        };

        // Should return None for single string prompt, even with n > 1
        // SGLang handles n parameter differently than batch requests
        assert_eq!(req.get_batch_size().unwrap(), None);
    }

    #[test]
    fn test_completion_bootstrap_single_values() {
        let mut req = CompletionRequest {
            model: "test".to_string(),
            prompt: StringOrArray::Array(vec!["prompt1".to_string(), "prompt2".to_string()]),
            n: None,
            other: serde_json::Map::new(),
            suffix: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: false,
            stream_options: None,
            logprobs: None,
            echo: false,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            best_of: None,
            logit_bias: None,
            user: None,
            seed: None,
        };

        // Set bootstrap info - should always use single values
        req.set_bootstrap_info(
            BootstrapHost::Single("test-server".to_string()),
            BootstrapPort::Single(Some(5678)),
            BootstrapRoom::Single(12345),
        );

        // Verify single values were created
        assert!(req.other.get("bootstrap_host").unwrap().is_string());
        assert!(req.other.get("bootstrap_port").unwrap().is_number());
        assert!(req.other.get("bootstrap_room").unwrap().is_number());

        assert_eq!(
            req.other.get("bootstrap_host").unwrap().as_str().unwrap(),
            "test-server"
        );
        assert_eq!(
            req.other.get("bootstrap_port").unwrap().as_u64().unwrap(),
            5678
        );
        assert_eq!(
            req.other.get("bootstrap_room").unwrap().as_u64().unwrap(),
            12345
        );
    }

    #[test]
    fn test_completion_bootstrap_array_values() {
        let mut req = CompletionRequest {
            model: "test".to_string(),
            prompt: StringOrArray::Array(vec!["prompt1".to_string(), "prompt2".to_string()]),
            n: None,
            other: serde_json::Map::new(),
            suffix: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: false,
            stream_options: None,
            logprobs: None,
            echo: false,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            best_of: None,
            logit_bias: None,
            user: None,
            seed: None,
        };

        // Set bootstrap info with arrays
        req.set_bootstrap_info(
            BootstrapHost::Batch(vec!["test-server".to_string(); 2]),
            BootstrapPort::Batch(vec![Some(5678); 2]),
            BootstrapRoom::Batch(vec![12345, 67890]),
        );

        // Verify arrays were created correctly
        assert!(req.other.get("bootstrap_host").unwrap().is_array());
        assert!(req.other.get("bootstrap_port").unwrap().is_array());
        assert!(req.other.get("bootstrap_room").unwrap().is_array());

        let hosts = req.other.get("bootstrap_host").unwrap().as_array().unwrap();
        assert_eq!(hosts.len(), 2);
        assert_eq!(hosts[0].as_str().unwrap(), "test-server");

        let ports = req.other.get("bootstrap_port").unwrap().as_array().unwrap();
        assert_eq!(ports.len(), 2);
        assert_eq!(ports[0].as_u64().unwrap(), 5678);

        let rooms = req.other.get("bootstrap_room").unwrap().as_array().unwrap();
        assert_eq!(rooms.len(), 2);
        assert_eq!(rooms[0].as_u64().unwrap(), 12345);
        assert_eq!(rooms[1].as_u64().unwrap(), 67890);
    }

    #[test]
    fn test_bootstrap_room_range() {
        // Test that bootstrap_room values are within the expected range [0, 2^63 - 1]
        let worker = BasicWorker::new(
            "http://test:8000".to_string(),
            WorkerType::Prefill {
                bootstrap_port: Some(8080),
            },
        );

        // Test single request
        let mut single_req = GenerateReqInput {
            text: Some(InputText::Single("test".to_string())),
            input_ids: None,
            stream: false,
            bootstrap_host: None,
            bootstrap_port: None,
            bootstrap_room: None,
            other: Value::Object(serde_json::Map::new()),
        };

        for _ in 0..200000 {
            single_req.add_bootstrap_info(&worker).unwrap();
            if let Some(BootstrapRoom::Single(room)) = single_req.bootstrap_room {
                // Verify the room value is within signed 64-bit range
                assert!(room <= i64::MAX as u64, "Room {} exceeds i64::MAX", room);
            } else {
                panic!("Expected single bootstrap room");
            }
        }

        // Test batch request
        let mut batch_req = GenerateReqInput {
            text: Some(InputText::Batch(vec![
                "test1".to_string(),
                "test2".to_string(),
            ])),
            input_ids: None,
            stream: false,
            bootstrap_host: None,
            bootstrap_port: None,
            bootstrap_room: None,
            other: Value::Object(serde_json::Map::new()),
        };

        for _ in 0..200000 {
            batch_req.add_bootstrap_info(&worker).unwrap();
            if let Some(BootstrapRoom::Batch(rooms)) = &batch_req.bootstrap_room {
                for room in rooms {
                    // Verify each room value is within signed 64-bit range
                    assert!(*room <= i64::MAX as u64, "Room {} exceeds i64::MAX", room);
                }
            } else {
                panic!("Expected batch bootstrap rooms");
            }
        }
    }
}
