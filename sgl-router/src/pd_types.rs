// sgl-router/src/pd_types.rs

use anyhow::Error;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EngineType {
    Sglang,
    Vllm,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineInfo {
    pub engine_type: EngineType,
    pub url: String,
    pub bootstrap_port: Option<u16>,
}

#[typetag::serde(tag = "type")]
pub trait Bootstrap {
    fn is_stream(&self) -> bool;
    fn get_batch_size(&self) -> Result<Option<usize>, Error>;
    fn add_bootstrap_info(&mut self, prefill_info: &EngineInfo) -> Result<(), Error>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SingleOrBatch<T> {
    Single(T),
    Batch(Vec<T>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PDSelectionPolicy {
    Random,
    PowerOfTwo,
    CacheAware {
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
    },
}

impl Default for PDSelectionPolicy {
    fn default() -> Self {
        PDSelectionPolicy::Random
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_engine_info_serde() {
        let info = EngineInfo {
            engine_type: EngineType::Sglang,
            url: "http://localhost:8080".to_string(),
            bootstrap_port: Some(9000),
        };
        let serialized = serde_json::to_string(&info).unwrap();
        let deserialized: EngineInfo = serde_json::from_str(&serialized).unwrap();
        assert_eq!(info, deserialized);
    }

    #[test]
    fn test_pd_selection_policy_serde() {
        let policy1 = PDSelectionPolicy::Random;
        let serialized1 = serde_json::to_string(&policy1).unwrap();
        let deserialized1: PDSelectionPolicy = serde_json::from_str(&serialized1).unwrap();
        assert_eq!(policy1, deserialized1);

        let policy2 = PDSelectionPolicy::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.0,
        };
        let serialized2 = serde_json::to_string(&policy2).unwrap();
        let deserialized2: PDSelectionPolicy = serde_json::from_str(&serialized2).unwrap();
        assert_eq!(policy2, deserialized2);
    }

    #[test]
    fn test_single_or_batch_serde() {
        // Test Single
        let single: SingleOrBatch<i32> = SingleOrBatch::Single(42);
        let serialized_single = serde_json::to_string(&single).unwrap();
        assert_eq!(serialized_single, "42");
        let deserialized_single: SingleOrBatch<i32> = serde_json::from_str(&serialized_single).unwrap();
        assert_eq!(single, deserialized_single);

        // Test Batch
        let batch: SingleOrBatch<i32> = SingleOrBatch::Batch(vec![1, 2, 3]);
        let serialized_batch = serde_json::to_string(&batch).unwrap();
        assert_eq!(serialized_batch, "[1,2,3]");
        let deserialized_batch: SingleOrBatch<i32> = serde_json::from_str(&serialized_batch).unwrap();
        assert_eq!(batch, deserialized_batch);
    }

    // A mock struct for testing the Bootstrap trait
    #[derive(Serialize, Deserialize)]
    struct MockRequest {
        is_stream: bool,
        batch_size: Option<usize>,
    }

    #[typetag::serde]
    impl Bootstrap for MockRequest {
        fn is_stream(&self) -> bool {
            self.is_stream
        }

        fn get_batch_size(&self) -> Result<Option<usize>, Error> {
            Ok(self.batch_size)
        }

        fn add_bootstrap_info(&mut self, _prefill_info: &EngineInfo) -> Result<(), Error> {
            // No-op for mock
            Ok(())
        }
    }

    #[test]
    fn test_bootstrap_serde() {
        let req: Box<dyn Bootstrap> = Box::new(MockRequest { is_stream: true, batch_size: Some(4) });
        let serialized = serde_json::to_string(&req).unwrap();

        // The type tag "MockRequest" will be added by typetag
        let expected_json = "{\"type\":\"MockRequest\",\"is_stream\":true,\"batch_size\":4}";
        assert_eq!(serialized, expected_json);

        let deserialized: Box<dyn Bootstrap> = serde_json::from_str(&serialized).unwrap();
        assert!(deserialized.is_stream());
        assert_eq!(deserialized.get_batch_size().unwrap(), Some(4));
    }
}
