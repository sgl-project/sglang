use std::sync::Arc;

use crate::config::{PolicyConfig, PolicyError};

use super::{
    cache_aware::CacheAwarePolicy, power_of_two::PowerOfTwoPolicy, random::RandomPolicy,
    round_robin::RoundRobinPolicy, traits::RoutingPolicy,
};

pub struct PolicyFactory;

impl PolicyFactory {
    pub fn create(
        config: &PolicyConfig,
        workers: &[Arc<dyn Worker>],
    ) -> Result<Arc<dyn RoutingPolicy>, PolicyError> {
        match config {
            PolicyConfig::Random => Ok(Arc::new(RandomPolicy::new())),
            PolicyConfig::RoundRobin => Ok(Arc::new(RoundRobinPolicy::new())),
            PolicyConfig::CacheAware { .. } => {
                Ok(Arc::new(CacheAwarePolicy::new(config, workers)?))
            }
            PolicyConfig::PowerOfTwo { .. } => {
                Ok(Arc::new(PowerOfTwoPolicy::new(config, workers)?))
            }
        }
    }
}
