// todo(Yingyi): placeholder, replace by core/worker.rs after task 1 freeze
use std::sync::atomic::AtomicUsize;

use async_trait::async_trait;

#[async_trait]
pub trait Worker: Send + Sync {
    fn is_healthy(&self) -> bool;
    fn load(&self) -> &AtomicUsize;
    fn get_url(&self) -> &str;
}
