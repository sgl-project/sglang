// Data connector module for response storage and conversation storage
//
// Simplified module structure:
// - core.rs: All traits, data types, and errors
// - memory.rs: All in-memory storage implementations
// - noop.rs: All no-op storage implementations
// - oracle.rs: All Oracle ATP storage implementations
// - factory.rs: Storage creation function

mod core;
mod factory;
mod memory;
mod noop;
mod oracle;

// Re-export all core types
pub use core::*;

// Re-export factory function
pub use factory::create_storage;
// Re-export all storage implementations
pub use memory::*;
pub use noop::*;
pub use oracle::*;
