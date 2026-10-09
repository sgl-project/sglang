//! PD routing extensions absent from Dynamo's standard OpenAI request types.

use serde::Deserialize;

use crate::message::request::GenerateRequest;
use crate::message::types::{OneOrMany, OneOrManyItem};

#[derive(Default, Deserialize)]
pub(super) struct BootstrapParams {
    bootstrap_host: Option<OneOrMany<String>>,
    bootstrap_port: Option<OneOrMany<Option<i64>>>,
    bootstrap_room: Option<OneOrMany<i64>>,
}

impl BootstrapParams {
    /// Validate all columns before submitting any prompt in the batch.
    pub(super) fn validate(&self, prompt_count: usize) -> Result<(), String> {
        validate_count("bootstrap_host", &self.bootstrap_host, prompt_count)?;
        validate_count("bootstrap_port", &self.bootstrap_port, prompt_count)?;
        validate_count("bootstrap_room", &self.bootstrap_room, prompt_count)
    }

    /// Lists repeat per prompt across `n` choices. Scalar rooms advance in
    /// Python's sample-major order (`sample_index * prompt_count + prompt_index`).
    pub(super) fn apply(
        &self,
        request: &mut GenerateRequest,
        prompt_index: usize,
        expanded_index: usize,
    ) {
        request.bootstrap_host = value_for_prompt(&self.bootstrap_host, prompt_index).cloned();
        request.bootstrap_port = value_for_prompt(&self.bootstrap_port, prompt_index)
            .copied()
            .flatten();
        request.bootstrap_room = match &self.bootstrap_room {
            Some(OneOrMany::One(room)) => Some(room.wrapping_add(expanded_index as i64)),
            value => value_for_prompt(value, prompt_index).copied(),
        };
    }
}

fn validate_count<T: OneOrManyItem>(
    name: &str,
    value: &Option<OneOrMany<T>>,
    prompt_count: usize,
) -> Result<(), String> {
    if let Some(OneOrMany::Many(values)) = value
        && values.len() != prompt_count
    {
        return Err(format!(
            "{name} has {} entries, expected {prompt_count}",
            values.len()
        ));
    }
    Ok(())
}

fn value_for_prompt<T: OneOrManyItem>(
    value: &Option<OneOrMany<T>>,
    prompt_index: usize,
) -> Option<&T> {
    match value {
        Some(OneOrMany::One(value)) => Some(value),
        Some(OneOrMany::Many(values)) => Some(&values[prompt_index]),
        None => None,
    }
}
