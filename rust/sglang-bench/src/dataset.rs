//! Prompt generation, ported from `python/sglang/benchmark/datasets/`.
//!
//! The `random`, `random-ids` and `sharegpt` samplers are here; they are the
//! ones a large-batch run uses. Sampling is seeded, so a run reproduces
//! itself, but it does not reproduce the Python script's selection: the two
//! use different random number generators, so the same `--seed` draws
//! different prompts. Totals and distributions match; individual prompts do
//! not.

use std::path::Path;

use anyhow::{Context, Result};
use rand::prelude::*;
use rand::rngs::StdRng;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::args::{Args, DatasetName};

/// What goes on the wire as the prompt.
#[derive(Clone, Debug)]
pub enum Prompt {
    Text(String),
    /// `--tokenize-prompt`: skip the server's tokenizer and send ids.
    TokenIds(Vec<u32>),
}

#[derive(Clone, Debug)]
pub struct DatasetRow {
    pub prompt: Prompt,
    pub prompt_len: usize,
    pub output_len: usize,
    pub text_prompt_len: usize,
    /// Always 0 here: this port has no image datasets. The field exists so the
    /// reported totals keep the Python script's shape.
    pub vision_prompt_len: usize,
}

impl DatasetRow {
    fn new(prompt: Prompt, prompt_len: usize, output_len: usize) -> Self {
        Self {
            prompt,
            prompt_len,
            output_len,
            text_prompt_len: prompt_len,
            vision_prompt_len: 0,
        }
    }
}

/// Whether `--dataset-name` needs the ShareGPT corpus, which the caller
/// resolves (and may download) before sampling.
pub fn needs_corpus(name: DatasetName) -> bool {
    matches!(name, DatasetName::Random | DatasetName::Sharegpt)
}

/// Build the request set for `--dataset-name`. `corpus` is the ShareGPT path
/// when [`needs_corpus`] asked for one.
pub fn load(args: &Args, tokenizer: &Tokenizer, corpus: Option<&Path>) -> Result<Vec<DatasetRow>> {
    let mut rng = StdRng::seed_from_u64(args.seed);
    let rows = match args.dataset_name {
        DatasetName::Random | DatasetName::RandomIds => {
            sample_random(args, tokenizer, corpus, &mut rng)?
        }
        DatasetName::Sharegpt => {
            let corpus = corpus.context("the sharegpt dataset needs a corpus path")?;
            sample_sharegpt(args, tokenizer, corpus, &mut rng)?
        }
    };
    anyhow::ensure!(
        !rows.is_empty(),
        "the dataset produced no requests; check --num-prompts and the dataset filters"
    );
    let input_tokens: usize = rows.iter().map(|row| row.prompt_len).sum();
    let output_tokens: usize = rows.iter().map(|row| row.output_len).sum();
    println!("#Input tokens: {input_tokens}");
    println!("#Output tokens: {output_tokens}");
    Ok(rows)
}

/// Per-request lengths drawn uniformly from `[full * ratio, full]`, as the
/// Python `compute_random_lens` does. A ratio of 0 still has a floor of 1, so
/// no request asks for an empty prompt.
fn random_lens(full_len: usize, range_ratio: f64, count: usize, rng: &mut StdRng) -> Vec<usize> {
    if full_len == 0 {
        return vec![0; count];
    }
    let low = ((full_len as f64 * range_ratio) as usize).max(1);
    (0..count).map(|_| rng.gen_range(low..=full_len)).collect()
}

/// The `random` and `random-ids` datasets.
///
/// `random` repeats or truncates real ShareGPT prompts to the requested
/// length, which keeps the token distribution realistic. `random-ids` builds
/// arithmetic id sequences instead, needing no corpus, at the cost of prompts
/// that no tokenizer would produce.
fn sample_random(
    args: &Args,
    tokenizer: &Tokenizer,
    corpus: Option<&Path>,
    rng: &mut StdRng,
) -> Result<Vec<DatasetRow>> {
    let count = args.num_prompts;
    let mut input_lens = random_lens(args.random_input_len, args.random_range_ratio, count, rng);
    let output_lens = random_lens(args.random_output_len, args.random_range_ratio, count, rng);

    let return_text = !args.tokenize_prompt;
    if return_text {
        // The server adds its own special tokens, so leave room for them or
        // every prompt arrives longer than requested.
        let special = num_special_tokens(tokenizer);
        for len in &mut input_lens {
            *len = len.saturating_sub(special).max(1);
        }
    }

    if args.dataset_name == DatasetName::RandomIds {
        let vocab_size = tokenizer.get_vocab_size(false) as u32;
        anyhow::ensure!(vocab_size > 0, "the tokenizer reports an empty vocabulary");
        let mut rows = Vec::with_capacity(count);
        for (index, (&input_len, &output_len)) in
            input_lens.iter().zip(output_lens.iter()).enumerate()
        {
            let offset: u32 = rng.gen_range(0..vocab_size);
            let ids: Vec<u32> = (0..input_len)
                .map(|position| {
                    let raw = offset as u64 + index as u64 + position as u64;
                    (raw % u64::from(vocab_size)) as u32
                })
                .collect();
            rows.push(DatasetRow::new(
                to_prompt(ids, return_text, tokenizer)?,
                input_len,
                output_len,
            ));
        }
        return Ok(rows);
    }

    let corpus = corpus.context("the random dataset samples prompts from a corpus")?;
    let corpus = load_sharegpt_pairs(corpus, rng)?;
    let mut rows = Vec::with_capacity(count);
    for (prompt, _) in &corpus {
        let index = rows.len();
        if index == count {
            break;
        }
        let token_ids = encode(tokenizer, prompt, true)?;
        if token_ids.is_empty() {
            continue;
        }
        let target = input_lens[index];
        let ids: Vec<u32> = token_ids.iter().copied().cycle().take(target).collect();
        rows.push(DatasetRow::new(
            to_prompt(ids, return_text, tokenizer)?,
            target,
            output_lens[index],
        ));
    }
    anyhow::ensure!(
        rows.len() == count,
        "the corpus held {} usable prompts, fewer than --num-prompts {count}",
        rows.len()
    );
    Ok(rows)
}

/// The `sharegpt` dataset: each conversation's first turn is the prompt and
/// its reply's length is the output length, with the too-short and too-long
/// rows pruned.
fn sample_sharegpt(
    args: &Args,
    tokenizer: &Tokenizer,
    corpus: &Path,
    rng: &mut StdRng,
) -> Result<Vec<DatasetRow>> {
    let corpus = load_sharegpt_pairs(corpus, rng)?;
    let mut rows = Vec::with_capacity(args.num_prompts);
    for (prompt, completion) in &corpus {
        if rows.len() == args.num_prompts {
            break;
        }
        let prompt_len = encode(tokenizer, prompt, true)?.len();
        let output_len = match args.sharegpt_output_len {
            Some(len) => len,
            None => encode(tokenizer, completion, true)?.len(),
        };
        // Prune sequences too short to measure a decode interval on.
        if prompt_len < 2 || output_len < 2 {
            continue;
        }
        if let Some(context_len) = args.sharegpt_context_len
            && prompt_len + output_len > context_len
        {
            continue;
        }
        rows.push(DatasetRow::new(
            Prompt::Text(prompt.clone()),
            prompt_len,
            output_len,
        ));
    }
    Ok(rows)
}

/// A ShareGPT conversation, keeping only what the samplers read.
#[derive(Deserialize)]
struct Conversation {
    #[serde(alias = "conversation", default)]
    conversations: Vec<Turn>,
}

#[derive(Deserialize)]
struct Turn {
    #[serde(default)]
    value: String,
}

/// The first two turns of every conversation that has at least two, shuffled.
fn load_sharegpt_pairs(path: &Path, rng: &mut StdRng) -> Result<Vec<(String, String)>> {
    let bytes = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    let conversations: Vec<Conversation> = serde_json::from_slice(&bytes)
        .with_context(|| format!("{} is not the ShareGPT JSON", path.display()))?;
    let mut pairs: Vec<(String, String)> = conversations
        .into_iter()
        .filter(|conversation| conversation.conversations.len() >= 2)
        .map(|mut conversation| {
            let mut turns = conversation.conversations.drain(..2);
            let prompt = turns.next().expect("two turns").value;
            let completion = turns.next().expect("two turns").value;
            (prompt, completion)
        })
        .collect();
    anyhow::ensure!(
        !pairs.is_empty(),
        "{} held no conversation with two turns",
        path.display()
    );
    pairs.shuffle(rng);
    Ok(pairs)
}

fn to_prompt(ids: Vec<u32>, return_text: bool, tokenizer: &Tokenizer) -> Result<Prompt> {
    if !return_text {
        return Ok(Prompt::TokenIds(ids));
    }
    let text = tokenizer
        .decode(&ids, false)
        .map_err(|error| anyhow::anyhow!("decode failed: {error}"))?;
    Ok(Prompt::Text(text))
}

fn encode(tokenizer: &Tokenizer, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
    let encoding = tokenizer
        .encode(text, add_special_tokens)
        .map_err(|error| anyhow::anyhow!("encode failed: {error}"))?;
    Ok(encoding.get_ids().to_vec())
}

/// How many special tokens the tokenizer adds around a prompt, measured the
/// only way its API allows: encode nothing and count what appears.
fn num_special_tokens(tokenizer: &Tokenizer) -> usize {
    tokenizer
        .encode("", true)
        .map(|encoding| encoding.len())
        .unwrap_or(0)
}

/// Re-tokenize the generated texts to report the server's token count against
/// an independent one. Batched, so the tokenizer's own parallelism applies;
/// the Python script encodes them one at a time, which dominates its
/// post-processing on a large run.
pub fn retokenized_lens(tokenizer: &Tokenizer, texts: &[&str]) -> Result<Vec<usize>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), false)
        .map_err(|error| anyhow::anyhow!("batch encode failed: {error}"))?;
    Ok(encodings.iter().map(tokenizers::Encoding::len).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lengths stay inside `[full * ratio, full]`, and a ratio of 0 keeps the
    /// floor at 1 so no prompt is empty.
    #[test]
    fn random_lens_respect_the_range_ratio() {
        let mut rng = StdRng::seed_from_u64(1);
        let lens = random_lens(100, 0.5, 64, &mut rng);
        assert_eq!(lens.len(), 64);
        assert!(lens.iter().all(|&len| (50..=100).contains(&len)));

        let full = random_lens(10, 1.0, 8, &mut rng);
        assert!(full.iter().all(|&len| len == 10));

        let floored = random_lens(4, 0.0, 32, &mut rng);
        assert!(floored.iter().all(|&len| (1..=4).contains(&len)));

        // An output length of 0 is valid and must not panic on the floor.
        assert_eq!(random_lens(0, 0.5, 3, &mut rng), vec![0, 0, 0]);
    }

    /// The same seed draws the same lengths, so two runs of this binary are
    /// comparable even though they do not match the Python script's draw.
    #[test]
    fn sampling_is_reproducible_under_a_seed() {
        let first = random_lens(64, 0.3, 16, &mut StdRng::seed_from_u64(7));
        let second = random_lens(64, 0.3, 16, &mut StdRng::seed_from_u64(7));
        assert_eq!(first, second);
        let other = random_lens(64, 0.3, 16, &mut StdRng::seed_from_u64(8));
        assert_ne!(first, other);
    }

    /// Both ShareGPT spellings parse, conversations with one turn are dropped,
    /// and only the first two turns are kept.
    #[test]
    fn sharegpt_parsing_accepts_both_key_spellings() {
        let json = serde_json::json!([
            {"conversations": [{"value": "q1"}, {"value": "a1"}, {"value": "ignored"}]},
            {"conversation": [{"value": "q2"}, {"value": "a2"}]},
            {"conversations": [{"value": "orphan"}]},
            {"id": "no conversation key at all"},
        ]);
        let path =
            std::env::temp_dir().join(format!("sgl-bench-sharegpt-{}.json", std::process::id()));
        std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();

        let mut pairs = load_sharegpt_pairs(&path, &mut StdRng::seed_from_u64(3)).unwrap();
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                ("q1".to_owned(), "a1".to_owned()),
                ("q2".to_owned(), "a2".to_owned())
            ]
        );
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_corpus_without_usable_pairs_is_an_error() {
        let path =
            std::env::temp_dir().join(format!("sgl-bench-empty-{}.json", std::process::id()));
        std::fs::write(&path, b"[]").unwrap();
        assert!(load_sharegpt_pairs(&path, &mut StdRng::seed_from_u64(1)).is_err());
        std::fs::remove_file(&path).unwrap();
    }
}
