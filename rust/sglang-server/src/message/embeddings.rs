//! Positional embedding inputs and Python's owned CPU tensor wire format.

use serde::{Deserialize, Serialize};

use super::types::InputEmbeddings;
use crate::utils::error::Error;

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct PositionalEmbeds {
    pub embeds: InputEmbeddings,
    pub positions: Vec<i64>,
}

impl PositionalEmbeds {
    /// Positions address the final prompt, including media expansion and any
    /// allowed truncation. Check them before the scheduler constructs GPU indices.
    pub fn validate(&self, input_len: usize, hidden_size: u64) -> Result<(), Error> {
        if self.embeds.is_empty()
            || self.embeds.len() != self.positions.len()
            || self.embeds.iter().any(|row| {
                row.len() as u64 != hidden_size || row.iter().any(|value| !value.is_finite())
            })
        {
            return Err(Error::Validation(format!(
                "positional_embed_overrides must have one embedding with {hidden_size} finite values per position"
            )));
        }
        if self
            .positions
            .iter()
            .any(|&position| position < 0 || position as usize >= input_len)
        {
            return Err(Error::Validation(format!(
                "positional_embed_overrides positions must be in [0, {input_len})"
            )));
        }
        Ok(())
    }
}

/// `PositionalEmbeds` is an untagged msgspec array with a torch tensor in slot 0.
/// EXT 2 contains a big-endian metadata length, MessagePack shape/dtype/device,
/// then contiguous float32 bytes. Python's ext_hook copies them into an owned
/// CPU tensor; it never references the scheduler ring after intake returns.
#[derive(Debug)]
pub(super) struct PositionalEmbedsWire<'a>(pub &'a PositionalEmbeds);

impl Serialize for PositionalEmbedsWire<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::Error; // codespell:ignore ser

        let rows = &self.0.embeds;
        let shape = [rows.len(), rows.first().map_or(0, Vec::len)];
        let metadata = rmp_serde::to_vec(&(shape, "float32", "cpu")).map_err(S::Error::custom)?;
        let mut payload = Vec::with_capacity(4 + metadata.len() + shape[0] * shape[1] * 4);
        payload.extend_from_slice(&(metadata.len() as u32).to_be_bytes());
        payload.extend_from_slice(&metadata);
        for value in rows.iter().flatten() {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        (rmpv::Value::Ext(2, payload), &self.0.positions).serialize(serializer)
    }
}
