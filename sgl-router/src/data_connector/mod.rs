// Data connector module for response storage and conversation storage
//
// Simplified module structure:
// - core.rs: All traits, data types, and errors
// - memory.rs: All in-memory storage implementations
// - noop.rs: All no-op storage implementations
// - oracle.rs: All Oracle ATP storage implementations
// - factory.rs: Storage creation function

mod common;
mod core;
mod factory;
mod memory;
mod noop;
mod oracle;
mod postgres;

pub use core::{
    Conversation, ConversationId, ConversationItem, ConversationItemId, ConversationItemStorage,
    ConversationStorage, ListParams, NewConversation, NewConversationItem, ResponseId,
    ResponseStorage, SortOrder, StoredResponse,
};

pub use factory::create_storage;
pub use memory::{MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage};
