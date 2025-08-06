// Bootstrap field injection for PD routing
// Directly injects bootstrap fields into JSON requests without intermediate type conversions

use crate::core::{Worker, WorkerType};
use crate::routers::pd_types::get_hostname;
use serde_json::{json, Value};

/// Inject bootstrap fields directly into a JSON request
/// This replaces the complex ToPdRequest -> Bootstrap trait pattern
pub fn inject_bootstrap_fields(json: &mut Value, worker: &dyn Worker) -> Result<(), String> {
    let batch_size = extract_batch_size(json)?;

    // Extract bootstrap port from prefill worker if it's a prefill type
    let bootstrap_port = match worker.worker_type() {
        WorkerType::Prefill { bootstrap_port } => bootstrap_port,
        _ => None,
    };

    let hostname = get_hostname(worker.url());

    if let Some(batch_size) = batch_size {
        // Batch scenario - create arrays of bootstrap values
        json["bootstrap_host"] = json!(vec![hostname; batch_size]);
        json["bootstrap_port"] = json!(vec![bootstrap_port; batch_size]);
        json["bootstrap_room"] = json!((0..batch_size)
            .map(|_| {
                // Generate a value in the range [0, 2^63 - 1] to match Python's random.randint(0, 2**63 - 1)
                rand::random::<u64>() & (i64::MAX as u64)
            })
            .collect::<Vec<_>>());
    } else {
        // Single scenario - create single bootstrap values
        json["bootstrap_host"] = json!(hostname);
        json["bootstrap_port"] = json!(bootstrap_port);
        json["bootstrap_room"] = json!(rand::random::<u64>() & (i64::MAX as u64));
    }

    Ok(())
}

/// Extract batch size from various JSON request formats
/// Handles chat completions, completions, and generate requests
fn extract_batch_size(json: &Value) -> Result<Option<usize>, String> {
    // Check for chat completions 'n' parameter (number of choices)
    if let Some(n) = json.get("n").and_then(|v| v.as_u64()) {
        if n > 1 {
            return Ok(Some(n as usize));
        }
    }

    // Check for array prompts (completions API)
    if let Some(prompt) = json.get("prompt") {
        if let Some(arr) = prompt.as_array() {
            if arr.is_empty() {
                return Err("Batch prompt array is empty".to_string());
            }
            return Ok(Some(arr.len()));
        }
    }

    // Check for array texts (generate API)
    if let Some(text) = json.get("text") {
        if let Some(arr) = text.as_array() {
            if arr.is_empty() {
                return Err("Batch text array is empty".to_string());
            }
            return Ok(Some(arr.len()));
        }
    }

    // Check for batch input_ids (generate API)
    if let Some(input_ids) = json.get("input_ids") {
        if let Some(arr) = input_ids.as_array() {
            if arr.is_empty() {
                return Err("Batch input_ids array is empty".to_string());
            }
            return Ok(Some(arr.len()));
        }
    }

    // No batch indicators found - single request
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BasicWorker;
    use serde_json::json;

    fn create_test_worker() -> BasicWorker {
        BasicWorker::new(
            "http://test-server:8000".to_string(),
            WorkerType::Prefill {
                bootstrap_port: Some(5678),
            },
        )
    }

    #[test]
    fn test_inject_bootstrap_single_request() {
        let worker = create_test_worker();
        let mut json = json!({
            "model": "test-model",
            "prompt": "Hello world",
            "max_tokens": 100
        });

        let result = inject_bootstrap_fields(&mut json, &worker);
        assert!(result.is_ok());

        // Verify bootstrap fields were added
        assert_eq!(json["bootstrap_host"], json!("test-server"));
        assert_eq!(json["bootstrap_port"], json!(5678));
        assert!(json["bootstrap_room"].is_number());

        // Verify original fields preserved
        assert_eq!(json["model"], json!("test-model"));
        assert_eq!(json["prompt"], json!("Hello world"));
        assert_eq!(json["max_tokens"], json!(100));
    }

    #[test]
    fn test_inject_bootstrap_batch_prompt() {
        let worker = create_test_worker();
        let mut json = json!({
            "model": "test-model",
            "prompt": ["Hello", "World"],
            "max_tokens": 100
        });

        let result = inject_bootstrap_fields(&mut json, &worker);
        assert!(result.is_ok());

        // Verify batch bootstrap fields
        assert_eq!(
            json["bootstrap_host"],
            json!(["test-server", "test-server"])
        );
        assert_eq!(json["bootstrap_port"], json!([5678, 5678]));

        let bootstrap_rooms = json["bootstrap_room"].as_array().unwrap();
        assert_eq!(bootstrap_rooms.len(), 2);
        for room in bootstrap_rooms {
            assert!(room.is_number());
            let room_val = room.as_u64().unwrap();
            assert!(room_val <= i64::MAX as u64);
        }
    }

    #[test]
    fn test_inject_bootstrap_chat_n_parameter() {
        let worker = create_test_worker();
        let mut json = json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "n": 3
        });

        let result = inject_bootstrap_fields(&mut json, &worker);
        assert!(result.is_ok());

        // Verify batch bootstrap fields for n=3
        let bootstrap_hosts = json["bootstrap_host"].as_array().unwrap();
        assert_eq!(bootstrap_hosts.len(), 3);
        assert_eq!(bootstrap_hosts[0], json!("test-server"));

        let bootstrap_ports = json["bootstrap_port"].as_array().unwrap();
        assert_eq!(bootstrap_ports.len(), 3);
        assert_eq!(bootstrap_ports[0], json!(5678));

        let bootstrap_rooms = json["bootstrap_room"].as_array().unwrap();
        assert_eq!(bootstrap_rooms.len(), 3);
    }

    #[test]
    fn test_inject_bootstrap_generate_text_array() {
        let worker = create_test_worker();
        let mut json = json!({
            "text": ["First prompt", "Second prompt"],
            "stream": false
        });

        let result = inject_bootstrap_fields(&mut json, &worker);
        assert!(result.is_ok());

        // Verify batch bootstrap fields
        let bootstrap_hosts = json["bootstrap_host"].as_array().unwrap();
        assert_eq!(bootstrap_hosts.len(), 2);

        let bootstrap_rooms = json["bootstrap_room"].as_array().unwrap();
        assert_eq!(bootstrap_rooms.len(), 2);
        // Ensure room values are different (randomness)
        assert_ne!(bootstrap_rooms[0], bootstrap_rooms[1]);
    }

    #[test]
    fn test_inject_bootstrap_input_ids_array() {
        let worker = create_test_worker();
        let mut json = json!({
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "stream": false
        });

        let result = inject_bootstrap_fields(&mut json, &worker);
        assert!(result.is_ok());

        // Verify batch bootstrap fields
        let bootstrap_hosts = json["bootstrap_host"].as_array().unwrap();
        assert_eq!(bootstrap_hosts.len(), 2);
    }

    #[test]
    fn test_extract_batch_size_empty_array_error() {
        let json = json!({
            "prompt": [],
            "model": "test"
        });

        let result = extract_batch_size(&json);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[test]
    fn test_extract_batch_size_single_requests() {
        // Single string prompt
        let json = json!({
            "prompt": "Hello world",
            "model": "test"
        });
        assert_eq!(extract_batch_size(&json).unwrap(), None);

        // Single text
        let json = json!({
            "text": "Hello world",
            "stream": false
        });
        assert_eq!(extract_batch_size(&json).unwrap(), None);

        // Chat with n=1 (default)
        let json = json!({
            "messages": [{"role": "user", "content": "Hello"}],
            "n": 1
        });
        assert_eq!(extract_batch_size(&json).unwrap(), None);

        // Chat without n parameter
        let json = json!({
            "messages": [{"role": "user", "content": "Hello"}]
        });
        assert_eq!(extract_batch_size(&json).unwrap(), None);
    }

    #[test]
    fn test_inject_bootstrap_preserves_sglang_fields() {
        let worker = create_test_worker();
        let mut json = json!({
            "model": "test-model",
            "prompt": "Hello",
            // SGLang extensions should be preserved
            "top_k": 40,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
            "regex": "test_pattern",
            "lora_path": "test.bin",
            "no_stop_trim": true,
            "ignore_eos": false
        });

        let result = inject_bootstrap_fields(&mut json, &worker);
        assert!(result.is_ok());

        // Verify bootstrap fields added
        assert!(json.get("bootstrap_host").is_some());
        assert!(json.get("bootstrap_port").is_some());
        assert!(json.get("bootstrap_room").is_some());

        // Verify all SGLang fields preserved
        assert_eq!(json["top_k"], json!(40));
        assert_eq!(json["min_p"], json!(0.05));
        assert_eq!(json["repetition_penalty"], json!(1.1));
        assert_eq!(json["regex"], json!("test_pattern"));
        assert_eq!(json["lora_path"], json!("test.bin"));
        assert_eq!(json["no_stop_trim"], json!(true));
        assert_eq!(json["ignore_eos"], json!(false));
    }

    #[test]
    fn test_bootstrap_room_range() {
        let worker = create_test_worker();

        // Test single request room generation
        for _ in 0..1000 {
            let mut json = json!({"prompt": "test"});
            inject_bootstrap_fields(&mut json, &worker).unwrap();

            let room = json["bootstrap_room"].as_u64().unwrap();
            assert!(room <= i64::MAX as u64, "Room {} exceeds i64::MAX", room);
        }

        // Test batch request room generation
        for _ in 0..100 {
            let mut json = json!({"prompt": ["test1", "test2"]});
            inject_bootstrap_fields(&mut json, &worker).unwrap();

            let rooms = json["bootstrap_room"].as_array().unwrap();
            for room_val in rooms {
                let room = room_val.as_u64().unwrap();
                assert!(room <= i64::MAX as u64, "Room {} exceeds i64::MAX", room);
            }
        }
    }

    #[test]
    fn test_worker_without_bootstrap_port() {
        let worker = BasicWorker::new(
            "http://decode-only:8000".to_string(),
            WorkerType::Decode, // No bootstrap port
        );

        let mut json = json!({
            "prompt": "Hello world"
        });

        let result = inject_bootstrap_fields(&mut json, &worker);
        assert!(result.is_ok());

        // Verify bootstrap fields with null port
        assert_eq!(json["bootstrap_host"], json!("decode-only"));
        assert_eq!(json["bootstrap_port"], json!(null));
        assert!(json["bootstrap_room"].is_number());
    }
}
