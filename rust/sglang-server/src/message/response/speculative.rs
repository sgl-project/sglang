//! Per-request statistics from scheduler counters, including Python's aliases.

use serde::ser::SerializeMap; // codespell:ignore ser
use serde::{Serialize, Serializer};

use super::BatchTokenCounts;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeStats {
    verify_calls: u64,
    correct: u64,
    proposed: u64,
    completion_tokens: u64,
    cap_tokens: Option<u64>,
    block_accept_tokens: Option<u64>,
    correct_histogram: Vec<u64>,
    cap_histogram: Vec<u64>,
}

impl SpeculativeStats {
    pub(super) fn take(batch: &mut BatchTokenCounts, i: usize) -> Option<Self> {
        let verify_calls = batch
            .spec_verify_ct
            .get(i)
            .copied()
            .filter(|&count| count > 0)?;
        let correct = *batch.spec_num_correct_drafts.get(i)?;
        let completion_tokens = *batch.generation_tokens.get(i)?;
        Some(Self {
            verify_calls,
            correct,
            proposed: verify_calls * batch.spec_num_draft_tokens.saturating_sub(1),
            completion_tokens,
            cap_tokens: batch
                .spec_num_cap_tokens
                .get(i)
                .copied()
                .filter(|&count| count > 0),
            block_accept_tokens: batch
                .spec_ragged_verify_cap_accept
                .then(|| batch.spec_num_block_accept_tokens.get(i).copied())
                .flatten(),
            correct_histogram: batch
                .spec_correct_drafts_histogram
                .get_mut(i)
                .map(std::mem::take)
                .unwrap_or_default(),
            cap_histogram: batch
                .spec_cap_lens_histogram
                .get_mut(i)
                .map(std::mem::take)
                .unwrap_or_default(),
        })
    }
}

impl Serialize for SpeculativeStats {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        if self.proposed > 0 {
            map.serialize_entry(
                "spec_accept_rate",
                &(self.correct as f64 / self.proposed as f64),
            )?;
            map.serialize_entry(
                "spec_accept_length",
                &(self.completion_tokens as f64 / self.verify_calls as f64),
            )?;
            map.serialize_entry("spec_num_correct_drafts", &self.correct)?;
            map.serialize_entry("spec_num_proposed_drafts", &self.proposed)?;
            map.serialize_entry("spec_verify_ct", &self.verify_calls)?;
            if let Some(tokens) = self.cap_tokens {
                map.serialize_entry(
                    "spec_cap_length",
                    &(tokens as f64 / self.verify_calls as f64),
                )?;
            }
            if let Some(tokens) = self.block_accept_tokens {
                map.serialize_entry(
                    "spec_block_accept_length",
                    &(tokens as f64 / self.verify_calls as f64),
                )?;
            }
            map.serialize_entry("spec_accepted_drafts", &self.correct)?;
            map.serialize_entry("spec_proposed_drafts", &self.proposed)?;
        }
        if !self.correct_histogram.is_empty() {
            map.serialize_entry("spec_correct_drafts_histogram", &self.correct_histogram)?;
            map.serialize_entry("spec_accept_histogram", &self.correct_histogram)?;
        }
        if !self.cap_histogram.is_empty() {
            map.serialize_entry("spec_cap_lens_histogram", &self.cap_histogram)?;
        }
        map.end()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    #[test]
    fn response_statistics_match_python_for_every_speculative_variant() {
        let fixtures: Vec<Value> = serde_json::from_str(include_str!(
            "../../../testdata/speculative_stats_python.json"
        ))
        .unwrap();
        for fixture in fixtures {
            let mut batch: BatchTokenCounts =
                serde_json::from_value(fixture["columns"].clone()).unwrap();
            let count = batch.generation_tokens.len();
            assert!(batch.valid_batch_size(count), "{}", fixture["name"]);
            for i in 0..count {
                let counts = batch.take_request(i);
                // Visible output can omit stop tokens; acceptance length uses
                // the scheduler's completion count, including its bonus token.
                let actual = serde_json::to_value(counts.metadata(1)).unwrap();
                let spec: serde_json::Map<String, Value> = actual
                    .as_object()
                    .unwrap()
                    .iter()
                    .filter(|(key, _)| key.starts_with("spec_"))
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect();
                assert_eq!(
                    json!(spec),
                    fixture["expected"][i],
                    "{} item {i}",
                    fixture["name"]
                );
            }
        }
    }
}
