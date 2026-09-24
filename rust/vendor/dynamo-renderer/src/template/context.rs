// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Modified by SGLang: comment-only provenance or spelling metadata; behavior is unchanged.

use super::{ContextMixins, PromptContextMixin};

use chrono::{DateTime, Utc};
use minijinja::value::{Object, Value};
use std::sync::Arc;

impl Object for ContextMixins {
    fn get_value(self: &Arc<Self>, field: &Value) -> Option<Value> {
        match field.as_str()? {
            "datetime" => self.datetime(),
            _ => None,
        }
    }
}

impl ContextMixins {
    pub fn new(allowed_mixins: &[PromptContextMixin]) -> Self {
        ContextMixins {
            context_mixins: allowed_mixins.iter().cloned().collect(),
        }
    }

    /// Implements the `datetime` context mixin.
    /// Different mixins can be implemented here for the same key.
    /// We need to validate that multiple mixins do not conflict with each other.
    fn datetime(&self) -> Option<Value> {
        if self
            .context_mixins
            .contains(&PromptContextMixin::Llama3DateTime)
        {
            let now = chrono::Utc::now();
            Some(Value::from(llama3_datetime(now)))
        } else {
            None
        }
    }
}

fn llama3_datetime(datetime: DateTime<Utc>) -> String {
    datetime.format("%d, %B, %Y").to_string()
}
