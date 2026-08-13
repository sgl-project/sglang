//! Rid-keyed handoff for `input_ids`: instead of riding the scheduler ring,
//! the widened ids are parked here at push time and popped by the Python
//! drain (`Server.take_input_ids`), which takes ownership of the vector as a
//! numpy array — the ring carries only the msgpack header.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Same lifecycle contract as `MmResultStore`: park strictly before the ring
/// push, take at the drain, purge for requests that die in between.
#[derive(Clone, Default)]
pub struct InputIdsStore(Arc<Mutex<HashMap<String, Vec<i64>>>>);

impl InputIdsStore {
    pub fn park(&self, rid: String, ids: Vec<i64>) {
        self.0.lock().unwrap().insert(rid, ids);
    }
    pub fn take(&self, rid: &str) -> Option<Vec<i64>> {
        self.0.lock().unwrap().remove(rid)
    }
    pub fn purge(&self, rid: &str) {
        self.0.lock().unwrap().remove(rid);
    }
}
