//! Renderer request primitives.

use serde::{Deserialize, Serialize};

pub type TokenIds = Vec<i32>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany<T: OneOrManyItem> {
    One(T),
    Many(Vec<T>),
}

pub trait OneOrManyItem: sealed::SealedItem {}

impl<T: sealed::SealedItem> OneOrManyItem for T {}

mod sealed {
    pub trait SealedItem {}

    impl SealedItem for bool {}
    impl SealedItem for i64 {}
    impl SealedItem for String {}
    impl SealedItem for super::TokenIds {}
    impl SealedItem for Option<i64> {}
    impl SealedItem for Option<String> {}
}
