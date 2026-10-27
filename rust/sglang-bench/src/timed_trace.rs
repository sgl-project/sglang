//! The `timed-trace` dataset: a JSONL replay trace with one request per line,
//! its arrival timestamp, its lengths, and the prefix-hash ids of its prompt.
//!
//! ```jsonl
//! {"timestamp": 12.4, "input_length": 42, "output_length": 8, "hash_ids": [981, 4412, 7]}
//! ```
//!
//! Prompts carry no text: each hash id expands to a chunk of
//! `--timed-trace-chunk-hash-size` token ids, the last chunk partial so the
//! prompt is exactly `input_length` tokens. A chunk is seeded from its hash and
//! size, so rows that share a hash prefix share a token prefix and the server's
//! prefix cache sees the trace's reuse. Rows keep file order; their timestamps,
//! scaled to seconds by `--timed-trace-sec-multiplier`, are the arrival
//! schedule. The layout and the expansion rule are vLLM's `timed_trace`, so one
//! trace replays through either client with the same token counts.

use std::collections::HashMap;
use std::io::BufRead;
use std::sync::Arc;

use anyhow::{Context, Result};
use rand::prelude::*;
use rand::rngs::StdRng;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::args::Args;
use crate::dataset::{DatasetRow, Prompt, required};

type ChunkCache = HashMap<(u64, usize), Arc<[u32]>>;

/// The first `--num-prompts` rows of `--dataset-path`, in file order.
pub fn load(args: &Args, tokenizer: Option<&Tokenizer>) -> Result<Vec<DatasetRow>> {
    let vocab_size = match args.vocab_size {
        Some(size) => size,
        None => required(tokenizer)?.get_vocab_size(false) as u32,
    };
    anyhow::ensure!(vocab_size > 0, "the tokenizer reports an empty vocabulary");
    let path = &args.dataset_path;
    let file = std::fs::File::open(path).with_context(|| format!("cannot read {path}"))?;

    let mut cache = ChunkCache::new();
    let mut rows = Vec::new();
    for (index, line) in std::io::BufReader::new(file).lines().enumerate() {
        if rows.len() == args.num_prompts {
            break;
        }
        let line = line.with_context(|| format!("cannot read {path}:{}", index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: Value = serde_json::from_str(&line)
            .with_context(|| format!("{path}:{} is not JSON", index + 1))?;
        let row = parse_row(&entry, args, vocab_size, &mut cache)
            .with_context(|| format!("{path}:{}", index + 1))?;
        rows.push(row);
    }
    Ok(rows)
}

fn parse_row(
    entry: &Value,
    args: &Args,
    vocab_size: u32,
    cache: &mut ChunkCache,
) -> Result<DatasetRow> {
    let input_length = field_u64(
        entry,
        &args.timed_trace_label_input_length,
        "--timed-trace-label-input-length",
    )? as usize;
    let output_len = field_u64(
        entry,
        &args.timed_trace_label_output_length,
        "--timed-trace-label-output-length",
    )? as usize;
    let timestamp = field_f64(
        entry,
        &args.timed_trace_label_timestamp,
        "--timed-trace-label-timestamp",
    )? * args.timed_trace_sec_multiplier;
    // The scheduler sleeps on this offset; a non-finite one would panic there.
    anyhow::ensure!(
        timestamp.is_finite(),
        "field '{}' scales to a non-finite offset ({timestamp}); check --timed-trace-sec-multiplier",
        args.timed_trace_label_timestamp
    );
    let hashes = hash_ids(entry, &args.timed_trace_label_hash_ids)?;
    let ids = expand_prompt(
        &hashes,
        input_length,
        args.timed_trace_chunk_hash_size,
        vocab_size,
        cache,
    );
    let prompt_len = ids.len();
    let mut row = DatasetRow::new(Prompt::TokenIds(ids), prompt_len, output_len);
    row.timestamp = Some(timestamp);
    Ok(row)
}

/// Seconds after the first row at which each row is sent. Offsets are taken
/// from the first row, as the Python client's `get_request` does with
/// `--use-trace-timestamps`; a row timestamped before the first fires at once.
pub fn arrival_offsets(rows: &[DatasetRow]) -> Vec<f64> {
    let first = rows.first().and_then(|row| row.timestamp).unwrap_or(0.0);
    rows.iter()
        .map(|row| (row.timestamp.unwrap_or(first) - first).max(0.0))
        .collect()
}

/// `chunk_size` ids per hash until `target_len` is reached, the final chunk
/// partial. The same (hash, size) yields the same chunk, cached across rows.
/// Fewer hashes than the length needs give a shorter prompt, as in vLLM.
fn expand_prompt(
    hash_ids: &[u64],
    target_len: usize,
    chunk_size: usize,
    vocab_size: u32,
    cache: &mut ChunkCache,
) -> Vec<u32> {
    // `target_len` is trace input: reserve what the hashes can produce, not
    // what a corrupt row claims.
    let capacity = target_len.min(hash_ids.len().saturating_mul(chunk_size));
    let mut prompt = Vec::with_capacity(capacity);
    let mut remaining = target_len;
    for &hash in hash_ids {
        if remaining == 0 {
            break;
        }
        let size = remaining.min(chunk_size);
        let chunk = cache.entry((hash, size)).or_insert_with(|| {
            let mut rng = StdRng::seed_from_u64(chunk_seed(hash, size));
            (0..size).map(|_| rng.gen_range(0..vocab_size)).collect()
        });
        prompt.extend_from_slice(chunk);
        remaining -= size;
    }
    prompt
}

/// The golden-ratio multiply is a bijection on u64, so two hashes never
/// collide at one size; the xor with the size can alias across sizes at
/// about 2^-64 odds per pair.
fn chunk_seed(hash_id: u64, size: usize) -> u64 {
    hash_id.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (size as u64)
}

fn field<'v>(entry: &'v Value, label: &str, flag: &str) -> Result<&'v Value> {
    entry.get(label).with_context(|| {
        let available: Vec<&str> = entry
            .as_object()
            .map(|object| object.keys().map(String::as_str).collect())
            .unwrap_or_default();
        format!("field '{label}' not found; the row has {available:?}; set {flag} to the trace's name for it")
    })
}

/// Float-encoded integers (`512.0`) are accepted, as Python's `int()` takes them.
fn field_u64(entry: &Value, label: &str, flag: &str) -> Result<u64> {
    let value = field(entry, label, flag)?;
    value
        .as_u64()
        .or_else(|| {
            value
                .as_f64()
                .filter(|f| f.fract() == 0.0 && (0.0..u64::MAX as f64).contains(f))
                .map(|f| f as u64)
        })
        .with_context(|| format!("field '{label}' must be a non-negative integer, got {value}"))
}

fn field_f64(entry: &Value, label: &str, flag: &str) -> Result<f64> {
    let value = field(entry, label, flag)?;
    value
        .as_f64()
        .with_context(|| format!("field '{label}' must be a number, got {value}"))
}

/// A missing field is an empty prompt, as in vLLM; a present one must be an
/// array of integers. Ids are 64-bit block hashes, so they are kept as `u64`
/// and a negative id keeps its two's-complement bits.
fn hash_ids(entry: &Value, label: &str) -> Result<Vec<u64>> {
    let Some(value) = entry.get(label) else {
        return Ok(Vec::new());
    };
    let array = value
        .as_array()
        .with_context(|| format!("field '{label}' must be an array, got {value}"))?;
    array
        .iter()
        .map(|id| {
            id.as_u64()
                .or_else(|| id.as_i64().map(|i| i as u64))
                .with_context(|| format!("field '{label}' must hold integers, got {id}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use clap::Parser;

    use super::*;

    fn write_trace(name: &str, text: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "sgl-bench-trace-{name}-{}.jsonl",
            std::process::id()
        ));
        std::fs::write(&path, text).unwrap();
        path
    }

    fn args(path: &Path, chunk: usize, multiplier: f64, extra: &[&str]) -> Args {
        let chunk = chunk.to_string();
        let multiplier = multiplier.to_string();
        let mut argv = vec![
            "sglang-bench",
            "--dataset-name",
            "timed-trace",
            "--dataset-path",
            path.to_str().unwrap(),
            "--vocab-size",
            "1000",
            "--timed-trace-chunk-hash-size",
            &chunk,
            "--timed-trace-sec-multiplier",
            &multiplier,
        ];
        argv.extend_from_slice(extra);
        Args::parse_from(argv)
    }

    fn ids(row: &DatasetRow) -> &[u32] {
        match &row.prompt {
            Prompt::TokenIds(ids) => ids,
            Prompt::Text(_) => panic!("timed-trace rows carry ids"),
        }
    }

    /// The prompt is exactly `input_length` long even when that is not a
    /// multiple of the chunk size, and timestamps scale by the multiplier.
    #[test]
    fn rows_have_exact_lengths_and_scaled_offsets() {
        let path = write_trace(
            "basic",
            r#"{"timestamp": 1000, "input_length": 40, "output_length": 7, "hash_ids": [1, 2, 3]}
{"timestamp": 2500, "input_length": 32, "output_length": 9, "hash_ids": [1, 4]}
"#,
        );
        let rows = load(&args(&path, 16, 0.001, &[]), None).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].prompt_len, 40);
        assert_eq!(ids(&rows[0]).len(), 40);
        assert_eq!(rows[0].output_len, 7);
        assert_eq!(rows[0].timestamp, Some(1.0));
        assert_eq!(rows[1].timestamp, Some(2.5));
        assert_eq!(arrival_offsets(&rows), vec![0.0, 1.5]);
        assert!(ids(&rows[0]).iter().all(|&id| id < 1000));
    }

    /// Rows sharing a hash prefix share the token prefix, and differ after it;
    /// this is what lets the server's prefix cache see the trace's reuse.
    #[test]
    fn a_shared_hash_prefix_shares_tokens() {
        let path = write_trace(
            "prefix",
            r#"{"timestamp": 0, "input_length": 32, "output_length": 1, "hash_ids": [7, 8]}
{"timestamp": 1, "input_length": 32, "output_length": 1, "hash_ids": [7, 9]}
"#,
        );
        let rows = load(&args(&path, 16, 1.0, &[]), None).unwrap();
        let (a, b) = (ids(&rows[0]), ids(&rows[1]));
        assert_eq!(a[..16], b[..16]);
        assert_ne!(a[16..], b[16..]);
    }

    /// Two loads expand identically: the seed derives from the trace, not
    /// from the process.
    #[test]
    fn expansion_is_reproducible_across_loads() {
        let path = write_trace(
            "determinism",
            r#"{"timestamp": 0, "input_length": 40, "output_length": 1, "hash_ids": [11, 12, 13]}"#,
        );
        let a = load(&args(&path, 16, 1.0, &[]), None).unwrap();
        let b = load(&args(&path, 16, 1.0, &[]), None).unwrap();
        assert_eq!(ids(&a[0]), ids(&b[0]));
    }

    /// `input_length` is untrusted: with too few hashes the prompt is shorter,
    /// and an absurd value must not reserve memory for tokens no hash yields.
    #[test]
    fn the_hashes_bound_the_prompt_not_input_length() {
        let path = write_trace(
            "short",
            r#"{"timestamp": 0, "input_length": 100, "output_length": 1, "hash_ids": [1]}
{"timestamp": 0, "input_length": 10000000000000, "output_length": 1, "hash_ids": [1]}
"#,
        );
        let rows = load(&args(&path, 16, 1.0, &[]), None).unwrap();
        assert_eq!(rows[0].prompt_len, 16);
        assert_eq!(rows[1].prompt_len, 16);
    }

    #[test]
    fn num_prompts_caps_rows_in_file_order() {
        let path = write_trace(
            "cap",
            r#"{"timestamp": 0, "input_length": 8, "output_length": 1, "hash_ids": [1]}

{"timestamp": 1, "input_length": 8, "output_length": 2, "hash_ids": [2]}
{"timestamp": 2, "input_length": 8, "output_length": 3, "hash_ids": [3]}
"#,
        );
        let rows = load(&args(&path, 16, 1.0, &["--num-prompts", "2"]), None).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].output_len, 2);
    }

    /// A remapped label that is absent names the flag to fix, so a trace with
    /// its own field names is not reported as corrupt.
    #[test]
    fn a_missing_field_names_its_flag() {
        let path = write_trace(
            "missing",
            r#"{"timestamp": 0, "input_length": 8, "output_length": 1, "hash_ids": [1]}"#,
        );
        let error = load(
            &args(
                &path,
                16,
                1.0,
                &["--timed-trace-label-input-length", "input_len"],
            ),
            None,
        )
        .unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("input_len"), "{message}");
        assert!(
            message.contains("--timed-trace-label-input-length"),
            "{message}"
        );
    }

    /// Exporters emit raw 64-bit block hashes above `i64::MAX` and lengths
    /// round-tripped through floats (`16.0`); both are the trace format, not
    /// corruption.
    #[test]
    fn exporter_encodings_are_accepted() {
        let path = write_trace(
            "encodings",
            r#"{"timestamp": 0, "input_length": 16.0, "output_length": 4.0, "hash_ids": [18446744073709551615, -1]}"#,
        );
        let rows = load(&args(&path, 16, 1.0, &[]), None).unwrap();
        assert_eq!(rows[0].prompt_len, 16);
        assert_eq!(rows[0].output_len, 4);
    }

    #[test]
    fn a_non_finite_offset_is_rejected() {
        let path = write_trace(
            "overflow",
            r#"{"timestamp": 1e308, "input_length": 16, "output_length": 1, "hash_ids": [1]}"#,
        );
        let error = load(&args(&path, 16, 1e10, &[]), None).unwrap_err();
        assert!(format!("{error:#}").contains("non-finite offset"));
    }

    /// Arrivals are measured from the first row, and a row timestamped before
    /// it fires at once instead of producing a negative sleep.
    #[test]
    fn offsets_start_at_the_first_row_and_never_go_negative() {
        let path = write_trace(
            "offsets",
            r#"{"timestamp": 5, "input_length": 8, "output_length": 1, "hash_ids": [1]}
{"timestamp": 2, "input_length": 8, "output_length": 1, "hash_ids": [1]}
{"timestamp": 9, "input_length": 8, "output_length": 1, "hash_ids": [1]}
"#,
        );
        let rows = load(&args(&path, 16, 1.0, &[]), None).unwrap();
        assert_eq!(arrival_offsets(&rows), vec![0.0, 0.0, 4.0]);
    }
}
