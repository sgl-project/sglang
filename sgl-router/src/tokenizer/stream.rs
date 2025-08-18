// src/tokenizer/stream.rs

use super::traits;
use crate::metrics::TokenizerMetrics;
use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;

const INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET: usize = 5;

/// DecodeStream will keep the state necessary to produce individual chunks of
/// strings given an input stream of token_ids
pub struct DecodeStream {
    /// The tokenizer used to decode token_ids
    tokenizer: Arc<dyn traits::Tokenizer>,

    skip_special_tokens: bool,

    /// A temporary buffer of the necessary token_ids needed
    /// to produce valid string chunks
    all_token_ids: Vec<u32>,

    prefix_offset: usize,
    read_offset: usize,
}

impl DecodeStream {
    pub fn new(
        tokenizer: Arc<dyn traits::Tokenizer>,
        prompt_token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Self {
        let num_input_tokens = prompt_token_ids.len();
        let prompt_token_ids = prompt_token_ids.to_vec();
        Self {
            tokenizer,
            skip_special_tokens,
            all_token_ids: prompt_token_ids,
            prefix_offset: num_input_tokens
                .saturating_sub(INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET),
            read_offset: num_input_tokens,
        }
    }

    /// Step appends a token_id to the internal state and tries to produce a text chunk.
    /// Returning `None` means the given id is not enough to produce a chunk.
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        let start = Instant::now();

        self.all_token_ids.push(id);

        TokenizerMetrics::record_stream_token();

        let prefix_text = self.tokenizer.decode(
            &self.all_token_ids[self.prefix_offset..self.read_offset],
            self.skip_special_tokens,
        )?;

        let new_text = self.tokenizer.decode(
            &self.all_token_ids[self.prefix_offset..],
            self.skip_special_tokens,
        )?;

        if new_text.len() > prefix_text.len() && !new_text.ends_with("�") {
            let new_text = new_text[prefix_text.len()..].to_string();

            self.prefix_offset = self.read_offset;
            self.read_offset = self.all_token_ids.len();

            TokenizerMetrics::record_stream_step_duration(start.elapsed());

            Ok(Some(new_text))
        } else {
            if new_text.ends_with("�") {
                TokenizerMetrics::record_incomplete_utf8();
            }

            TokenizerMetrics::record_stream_step_duration(start.elapsed());

            Ok(None)
        }
    }

    /// Process multiple tokens at once
    pub fn step_batch(&mut self, token_ids: &[u32]) -> Result<Vec<String>> {
        let mut chunks = Vec::new();

        for &token_id in token_ids {
            if let Some(text) = self.step(token_id)? {
                chunks.push(text);
            }
        }

        Ok(chunks)
    }

    /// Force flush any remaining text
    pub fn flush(&mut self) -> Result<Option<String>> {
        if self.read_offset < self.all_token_ids.len() {
            let remaining = self.tokenizer.decode(
                &self.all_token_ids[self.read_offset..],
                self.skip_special_tokens,
            )?;

            self.read_offset = self.all_token_ids.len();

            if !remaining.is_empty() {
                return Ok(Some(remaining));
            }
        }

        Ok(None)
    }

    /// Get all tokens processed so far
    pub fn tokens(&self) -> &[u32] {
        &self.all_token_ids
    }
}
