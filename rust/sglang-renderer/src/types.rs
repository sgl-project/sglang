//! Renderer request primitives.

use serde::{Deserialize, Serialize};

pub type TokenIds = Vec<i32>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}
