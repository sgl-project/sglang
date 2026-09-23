//! Concrete sampling parameters and shared preprocessing.
//! Protocol lowering resolves defaults before the Python-compatible
//! `__post_init__` → `normalize` → `verify` pipeline.

use std::collections::BTreeMap;

use crate::{error::RendererError as Error, types::OneOrMany};

use super::regex::RegexPattern;

/// `_SAMPLING_EPS` — temperatures in `[0, eps)` mean greedy decoding.
const SAMPLING_EPS: f64 = 1e-6;
/// `TOP_K_ALL = 1 << 30` — `top_k` sentinel for "consider the whole vocabulary".
const TOP_K_ALL: i64 = 1 << 30;
/// Most stop STRINGS accepted per request. The scheduler scans the decoded text
/// once per stop per decode step, so this is a per-step multiplier: 50k stops
/// measured 20.4 ms/step from a 586 KB body.
const MAX_STOP_COUNT: usize = 32;
/// Longest `stop_regex` accepted. A 1 MB literal pattern takes ~677 ms just to
/// compile, and that cost lands on the scheduler.
const MAX_STOP_REGEX_LEN: usize = 256;
/// Most `stop_regex` patterns accepted per request. Python's `re` cache holds 512
/// (`re._MAXCACHE`), so past that every pattern recompiles on every decode step.
const MAX_STOP_REGEX_COUNT: usize = 32;

/// Concrete sampling parameters normalized and validated during preprocessing.
/// Protocol adapters own serialization into their respective wire formats.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    // --- API parameters (set by callers) ---
    pub max_new_tokens: Option<i64>,
    /// API input alias, copied to `stop_strs` then cleared by `normalize`.
    pub stop: Option<OneOrMany<String>>,
    /// Python `Optional[Set[int]]`. A `null` *element* is a 400 here where Python
    /// filters it out — a typed list can't hold one, and it is malformed input.
    pub stop_token_ids: Option<Vec<i64>>,
    /// API input alias, copied to `stop_regex_strs` then cleared by `normalize`.
    pub stop_regex: Option<OneOrMany<String>>,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: i64,
    pub min_p: f64,
    pub frequency_penalty: f64,
    pub presence_penalty: f64,
    pub repetition_penalty: f64,
    pub min_new_tokens: i64,
    pub n: i64,
    /// `beam_width > 1` requests beam search, which the Rust path rejects below.
    pub beam_width: Option<i64>,
    pub json_schema: Option<String>,
    pub regex: Option<String>,
    pub ebnf: Option<String>,
    pub structural_tag: Option<String>,
    pub ignore_eos: bool,
    pub skip_special_tokens: bool,
    pub spaces_between_special_tokens: bool,
    pub no_stop_trim: bool,
    pub stream_interval: Option<i64>,
    /// Token id (as a string key, matching Python) → bias. Keys are vocab-bounded
    /// by [`verify`](Self::verify).
    pub logit_bias: Option<BTreeMap<String, f64>>,
    pub sampling_seed: Option<i64>,
    /// Opaque JSON object forwarded to a custom logit processor. Python types it
    /// as `Dict[str, JsonScalar | list | dict]`; it is never inspected here.
    pub custom_params: Option<serde_json::Value>,

    // --- Internal fields (populated by the pipeline below, not API-facing) ---
    //
    // These fields are outputs of `normalize`, and a client that
    // could set them would be setting the pipeline's own state. `is_normalized` is
    // the dangerous one — `{"is_normalized": true, "temperature": 0.0}` makes
    // `post_init` early-return, so the greedy mapping never runs and temperature 0
    // reaches the scheduler's `logits.div_()`; `stop` would likewise be dropped
    // without ever reaching `stop_strs`. Only concrete parameters carry them.
    /// From `stop`; a list after `normalize` (Python widens str → [str] there).
    pub stop_strs: Vec<String>,
    /// From `stop_regex`.
    pub stop_regex_strs: Vec<String>,
    pub stop_str_max_len: usize,
    pub stop_regex_max_len: usize,
    /// Set by `normalize`; tells the scheduler its own pass can early-return.
    pub is_normalized: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_new_tokens: Some(128),
            stop: None,
            stop_token_ids: None,
            stop_regex: None,
            temperature: 1.0,
            top_p: 1.0,
            top_k: TOP_K_ALL,
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            min_new_tokens: 0,
            n: 1,
            beam_width: None,
            json_schema: None,
            regex: None,
            ebnf: None,
            structural_tag: None,
            ignore_eos: false,
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
            no_stop_trim: false,
            custom_params: None,
            stream_interval: None,
            logit_bias: None,
            sampling_seed: None,
            stop_strs: Vec::new(),
            stop_regex_strs: Vec::new(),
            stop_str_max_len: 0,
            stop_regex_max_len: 0,
            is_normalized: false,
        }
    }
}

impl SamplingParams {
    /// `__post_init__` → `normalize` → `verify`, the order
    /// `TokenizerManager._create_tokenized_object` runs them in. `Err` is a
    /// request-local 400. `vocab_size` bounds `logit_bias` keys.
    pub fn normalize(&mut self, vocab_size: u64) -> Result<(), Error> {
        self.post_init();
        self.normalize_stops()?;
        self.verify(vocab_size)
    }

    /// Python `__post_init__` (defaults were resolved during protocol lowering):
    /// copy API aliases into internal fields and apply greedy / `top_k` special cases.
    fn post_init(&mut self) {
        // Python's `__post_init__` guard. Without it a second `normalize` reads
        // the aliases `normalize_stops` already cleared and silently wipes
        // `stop_strs`/`stop_regex_strs` to empty — the request would stop
        // matching its stop strings.
        if self.is_normalized {
            return;
        }
        // Moved out, not cloned: `normalize_stops` clears both aliases anyway.
        self.stop_strs = take_one_or_many(self.stop.take());
        self.stop_regex_strs = take_one_or_many(self.stop_regex.take());
        // Python drops null entries and maps an empty set to None.
        if self.stop_token_ids.as_ref().is_some_and(|v| v.is_empty()) {
            self.stop_token_ids = None;
        }
        if (0.0..SAMPLING_EPS).contains(&self.temperature) {
            // Greedy: temperature ~0 → temperature=1.0, top_k=1.
            self.temperature = 1.0;
            self.top_k = 1;
        }
        if self.top_k == -1 {
            self.top_k = TOP_K_ALL; // -1 disables top_k → whole vocabulary
        }
    }

    /// Python `normalize(tokenizer)`: size the stop match windows and clear the
    /// API aliases so they don't ride the wire twice.
    fn normalize_stops(&mut self) -> Result<(), Error> {
        // Match window: UTF-8 byte length is a safe upper bound on the token count.
        self.stop_str_max_len = self.stop_strs.iter().map(|s| s.len()).max().unwrap_or(0);
        // Validate + bound every stop_regex here, before it can reach the
        // scheduler's `re.search` (see `RegexPattern`). A rejected pattern is a
        // 400 for this request; an accepted one carries a bound the scheduler uses
        // to size its match window.
        if self.stop_strs.len() > MAX_STOP_COUNT {
            return Err(bad(format!(
                "at most {MAX_STOP_COUNT} stop strings are allowed, got {}",
                self.stop_strs.len()
            )));
        }
        if self.stop_regex_strs.len() > MAX_STOP_REGEX_COUNT {
            return Err(bad(format!(
                "at most {MAX_STOP_REGEX_COUNT} stop_regex patterns are allowed, got {}",
                self.stop_regex_strs.len()
            )));
        }
        let mut stop_regex_max_len = 0;
        for pattern in &self.stop_regex_strs {
            if pattern.len() > MAX_STOP_REGEX_LEN {
                return Err(bad(format!(
                    "stop_regex is {} bytes, over the {MAX_STOP_REGEX_LEN}-byte limit",
                    pattern.len()
                )));
            }
            let pattern = RegexPattern::try_from(pattern.as_str())
                .map_err(|e| bad(format!("stop_regex {pattern:?} is invalid: {e}")))?;
            stop_regex_max_len = stop_regex_max_len.max(pattern.max_len());
        }
        self.stop_regex_max_len = stop_regex_max_len;

        self.stop = None;
        self.stop_regex = None;
        self.is_normalized = true;
        Ok(())
    }

    /// Python `verify(vocab_size)` — the same ranges, messages and mutual
    /// exclusions, plus the rust-server `n == 1` restriction.
    fn verify(&self, vocab_size: u64) -> Result<(), Error> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(bad(format!(
                "temperature must be a non-negative finite number, got {}",
                self.temperature
            )));
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(bad(format!("top_p must be in (0, 1], got {}", self.top_p)));
        }
        if !(0.0..=1.0).contains(&self.min_p) {
            return Err(bad(format!("min_p must be in [0, 1], got {}", self.min_p)));
        }
        if self.top_k < 1 {
            return Err(bad(format!(
                "top_k must be -1 (disable) or at least 1, got {}",
                self.top_k
            )));
        }
        if !(-2.0..=2.0).contains(&self.frequency_penalty) {
            return Err(bad(format!(
                "frequency_penalty must be in [-2, 2], got {}",
                self.frequency_penalty
            )));
        }
        if !(-2.0..=2.0).contains(&self.presence_penalty) {
            return Err(bad(format!(
                "presence_penalty must be in [-2, 2], got {}",
                self.presence_penalty
            )));
        }
        if !(self.repetition_penalty > 0.0 && self.repetition_penalty <= 2.0) {
            return Err(bad(format!(
                "repetition_penalty must be in (0, 2], got {}",
                self.repetition_penalty
            )));
        }
        if self.min_new_tokens < 0 {
            return Err(bad(format!(
                "min_new_tokens must be non-negative, got {}",
                self.min_new_tokens
            )));
        }
        // `None` = no limit, so the max_new_tokens checks only apply when set.
        if let Some(max_new_tokens) = self.max_new_tokens {
            if max_new_tokens < 0 {
                return Err(bad(format!(
                    "max_new_tokens must be at least 0, got {max_new_tokens}"
                )));
            }
            if self.min_new_tokens > max_new_tokens {
                return Err(bad(format!(
                    "min_new_tokens must be in [0, max_new_tokens({max_new_tokens})], got {}",
                    self.min_new_tokens
                )));
            }
        }
        // A non-numeric bias key raises in the scheduler's `int(key)`, and an
        // out-of-vocabulary one would index past the logits row, so both are
        // rejected here (Python `verify` does the same, in that order). Only the
        // *range* check needs the vocab size (`None` = unknown, skip it); the key
        // format is checked either way, since `int(key)` runs regardless.
        if let Some(logit_bias) = &self.logit_bias {
            for key in logit_bias.keys() {
                let token_id: u64 = key
                    .parse()
                    .map_err(|_| bad(format!("logit_bias keys must be token ids, got {key:?}")))?;
                if token_id >= vocab_size {
                    return Err(bad(format!(
                        "logit_bias must have keys in [0, {}], got {token_id}",
                        vocab_size - 1
                    )));
                }
            }
        }
        // Grammars are mutually exclusive.
        let grammars = [&self.json_schema, &self.regex, &self.ebnf]
            .iter()
            .filter(|g| g.is_some())
            .count();
        if grammars > 1 {
            return Err(bad(
                "Only one of regex, json_schema, or ebnf can be set".into()
            ));
        }
        // Not a Python restriction: the rust from_scheduler maps one rid to one response,
        // so parallel sampling would drop all but the first sample. This is the
        // only place it is rejected — `n` lives in `sampling_params`, where
        // Python reads it, and the `/generate` body has no `n` of its own.
        if self.n != 1 {
            return Err(bad(format!(
                "n must be 1 (parallel sampling is not supported), got {}",
                self.n
            )));
        }
        if let Some(beam_width) = self.beam_width {
            if beam_width < 1 {
                return Err(bad(format!(
                    "beam_width must be at least 1, got {beam_width}."
                )));
            }
            // Also not a Python restriction: beam search returns its candidates
            // in `meta_info.beam_results`, which from_scheduler does not carry.
            if beam_width > 1 {
                return Err(bad(format!(
                    "beam_width must be 1 (beam search is not supported), got {beam_width}"
                )));
            }
        }
        Ok(())
    }
}

fn bad(msg: String) -> Error {
    Error::Validation(msg)
}

/// Widen a `str | [str]` API alias into the list form the internal field holds.
fn take_one_or_many(v: Option<OneOrMany<String>>) -> Vec<String> {
    match v {
        None => Vec::new(),
        Some(OneOrMany::One(s)) => vec![s],
        Some(OneOrMany::Many(v)) => v,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Vocab size for tests that aren't about the vocab bound at all. It is
    /// mandatory now (`ServerArgs::validate_mandatory` rejects a boot without
    /// one), so there is no longer an "unknown vocab" case to pass instead —
    /// this is just a value large enough to stay out of the way.
    const TEST_VOCAB: u64 = 1000;

    /// Bound valid stop regexes and reject malformed patterns before submission.
    #[test]
    fn stop_regex_is_bounded_and_validated() {
        assert_eq!(
            norm(SamplingParams {
                stop_regex: Some(OneOrMany::One("\\d{6}".into())),
                ..Default::default()
            })
            .stop_regex_max_len,
            6
        );
        assert_eq!(
            norm(SamplingParams {
                temperature: 0.7,
                ..Default::default()
            })
            .stop_regex_max_len,
            0
        );
        let _ = norm_err(SamplingParams {
            stop_regex: Some(OneOrMany::One("(".into())),
            ..Default::default()
        });
        let _ = norm_err(SamplingParams {
            stop_regex: Some(OneOrMany::Many(vec!["\\d{6}".into(), "(".into()])),
            ..Default::default()
        });
    }

    fn norm(mut params: SamplingParams) -> SamplingParams {
        params.normalize(TEST_VOCAB).expect("normalizes");
        params
    }

    fn norm_err(mut params: SamplingParams) -> Error {
        params.normalize(TEST_VOCAB).expect_err("must reject")
    }

    #[test]
    fn greedy_sets_temp_one_topk_one() {
        let sp = norm(SamplingParams {
            temperature: 0.0,
            ..Default::default()
        });
        assert_eq!(sp.temperature, 1.0);
        assert_eq!(sp.top_k, 1);
        assert!(sp.is_normalized);
    }

    #[test]
    fn topk_minus_one_becomes_all() {
        assert_eq!(
            norm(SamplingParams {
                temperature: 0.7,
                ..Default::default()
            })
            .top_k,
            TOP_K_ALL
        );
        assert_eq!(
            norm(SamplingParams {
                top_k: -1,
                temperature: 0.7,
                ..Default::default()
            })
            .top_k,
            TOP_K_ALL
        );
    }

    #[test]
    fn stop_list_and_max_len_by_bytes() {
        let sp = norm(SamplingParams {
            stop: Some(OneOrMany::Many(vec!["Question:".into(), "\n\n".into()])),
            ..Default::default()
        });
        assert_eq!(sp.stop_strs.len(), 2);
        assert_eq!(sp.stop_str_max_len, 9); // "Question:" (ASCII)
        // The API alias is cleared, so it never rides the wire twice.
        assert!(sp.stop.is_none());
    }

    /// A multi-byte stop char must use its byte length as the window bound: `𓀀`
    /// is 1 char but 4 UTF-8 bytes (and 3 tokens on Qwen3). Char count (1) would
    /// under-size the tail and miss the stop; byte count (4) ≥ the token span.
    #[test]
    fn stop_str_max_len_uses_bytes_not_chars() {
        let sp = norm(SamplingParams {
            stop: Some(OneOrMany::One("𓀀".into())),
            ..Default::default()
        });
        assert_eq!("𓀀".chars().count(), 1);
        assert_eq!("𓀀".len(), 4);
        assert_eq!(sp.stop_strs, vec!["𓀀".to_string()]); // scalar widened to a list
        assert_eq!(sp.stop_str_max_len, 4);
    }

    #[test]
    fn no_stop_yields_empty_list_zero_len() {
        let sp = norm(SamplingParams {
            temperature: 0.0,
            ..Default::default()
        });
        assert!(sp.stop_strs.is_empty());
        assert_eq!(sp.stop_str_max_len, 0);
    }

    #[test]
    fn unlimited_max_new_tokens_allows_large_minimum() {
        let params = norm(SamplingParams {
            max_new_tokens: None,
            min_new_tokens: 4096,
            ..Default::default()
        });
        assert_eq!(params.max_new_tokens, None);
        assert_eq!(params.min_new_tokens, 4096);
    }

    #[test]
    fn verify_rejects_out_of_range() {
        for (params, want) in [
            (
                SamplingParams {
                    top_p: 2.0,
                    ..Default::default()
                },
                "top_p",
            ),
            (
                SamplingParams {
                    top_k: 0,
                    temperature: 0.7,
                    ..Default::default()
                },
                "top_k",
            ),
            (
                SamplingParams {
                    min_p: 1.5,
                    ..Default::default()
                },
                "min_p",
            ),
            (
                SamplingParams {
                    frequency_penalty: 3.0,
                    ..Default::default()
                },
                "frequency_penalty",
            ),
            (
                SamplingParams {
                    presence_penalty: -3.0,
                    ..Default::default()
                },
                "presence_penalty",
            ),
            (
                SamplingParams {
                    repetition_penalty: 0.0,
                    ..Default::default()
                },
                "repetition_penalty",
            ),
            (
                SamplingParams {
                    max_new_tokens: Some(8),
                    min_new_tokens: 9,
                    ..Default::default()
                },
                "min_new_tokens",
            ),
            (
                SamplingParams {
                    temperature: -0.1,
                    ..Default::default()
                },
                "temperature",
            ),
            (
                SamplingParams {
                    max_new_tokens: Some(-1),
                    ..Default::default()
                },
                "max_new_tokens",
            ),
            (
                SamplingParams {
                    regex: Some("a".into()),
                    ebnf: Some("b".into()),
                    ..Default::default()
                },
                "Only one of",
            ),
            (
                SamplingParams {
                    n: 2,
                    ..Default::default()
                },
                "n must be 1",
            ),
            (
                SamplingParams {
                    beam_width: Some(2),
                    ..Default::default()
                },
                "beam_width must be 1",
            ),
            (
                SamplingParams {
                    beam_width: Some(0),
                    ..Default::default()
                },
                "beam_width must be at least 1",
            ),
        ] {
            let err = norm_err(params).to_string();
            assert!(
                err.contains(want),
                "parameters must be rejected for {want}: {err}"
            );
        }
    }

    /// The inclusive bounds must ACCEPT their endpoints. Only the rejecting side
    /// was covered, and far from the edge (`frequency_penalty: 3.0`), so flipping
    /// any `..=` to `..` — or `>= 1` to `> 1` — would 400 legitimate requests
    /// without failing a single test.
    #[test]
    fn verify_accepts_inclusive_boundaries() {
        for params in [
            SamplingParams {
                top_p: 1.0,
                temperature: 0.7,
                ..Default::default()
            },
            SamplingParams {
                min_p: 0.0,
                temperature: 0.7,
                ..Default::default()
            },
            SamplingParams {
                min_p: 1.0,
                temperature: 0.7,
                ..Default::default()
            },
            SamplingParams {
                top_k: 1,
                temperature: 0.7,
                ..Default::default()
            },
            SamplingParams {
                frequency_penalty: 2.0,
                ..Default::default()
            },
            SamplingParams {
                frequency_penalty: -2.0,
                ..Default::default()
            },
            SamplingParams {
                presence_penalty: 2.0,
                ..Default::default()
            },
            SamplingParams {
                presence_penalty: -2.0,
                ..Default::default()
            },
            SamplingParams {
                repetition_penalty: 2.0,
                ..Default::default()
            },
            SamplingParams {
                max_new_tokens: Some(0),
                ..Default::default()
            },
            SamplingParams {
                min_new_tokens: 0,
                ..Default::default()
            },
            // min == max is in range: `[0, max_new_tokens]` is inclusive.
            SamplingParams {
                max_new_tokens: Some(8),
                min_new_tokens: 8,
                ..Default::default()
            },
            // Greedy: temperature 0 is the documented sentinel, not an under-run.
            SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
            SamplingParams {
                n: 1,
                ..Default::default()
            },
        ] {
            let mut sp = params;
            sp.normalize(TEST_VOCAB)
                .unwrap_or_else(|e| panic!("{sp:?} is in range but was rejected: {e}"));
        }
    }

    /// And the first value past each endpoint is still rejected — the pair of
    /// tests brackets the boundary instead of testing one side of it.
    #[test]
    fn verify_rejects_just_past_the_boundaries() {
        for params in [
            SamplingParams {
                top_p: 0.0,
                temperature: 0.7,
                ..Default::default()
            }, // exclusive lower bound
            SamplingParams {
                repetition_penalty: 0.0,
                ..Default::default()
            }, // exclusive lower bound
            SamplingParams {
                top_k: 0,
                temperature: 0.7,
                ..Default::default()
            },
            SamplingParams {
                min_p: 1.0000001,
                temperature: 0.7,
                ..Default::default()
            },
            SamplingParams {
                frequency_penalty: 2.0000001,
                ..Default::default()
            },
            SamplingParams {
                presence_penalty: -2.0000001,
                ..Default::default()
            },
            SamplingParams {
                repetition_penalty: 2.0000001,
                ..Default::default()
            },
            SamplingParams {
                max_new_tokens: Some(-1),
                ..Default::default()
            },
            SamplingParams {
                min_new_tokens: -1,
                ..Default::default()
            },
            SamplingParams {
                max_new_tokens: Some(8),
                min_new_tokens: 9,
                ..Default::default()
            },
        ] {
            let mut sp = params;
            assert!(
                sp.normalize(TEST_VOCAB).is_err(),
                "{sp:?} is out of range but was accepted"
            );
        }
    }

    /// `normalize` must be idempotent: `post_init` reads the API aliases, which
    /// `normalize_stops` clears, so without Python's `if self.is_normalized:
    /// return` guard a second call wipes `stop_strs` and drops the stop bound to
    /// zero — silently, leaving a request that never stops.
    #[test]
    fn normalize_is_idempotent() {
        let mut once = norm(SamplingParams {
            stop: Some(OneOrMany::Many(vec!["END".into(), "STOP".into()])),
            stop_regex: Some(OneOrMany::One("\\d{3}".into())),
            ..Default::default()
        });
        let twice = {
            let mut p = once.clone();
            p.normalize(TEST_VOCAB).expect("second normalize");
            p
        };
        assert_eq!(once, twice, "a second normalize must change nothing");
        assert_eq!(twice.stop_strs, vec!["END".to_string(), "STOP".to_string()]);
        assert_eq!(twice.stop_str_max_len, 4);
        assert_eq!(twice.stop_regex_max_len, 3);

        // Greedy handling must not re-fire either: temperature is 1.0 after the
        // first pass, which is not in the greedy window.
        once.normalize(TEST_VOCAB).unwrap();
        assert_eq!(once.top_k, twice.top_k);
    }

    /// `logit_bias` keys index the logits row, so an out-of-vocab id is a 400
    /// (Python `verify`'s vocab bound). The bound is exclusive, and it always
    /// applies — `vocab_size` is mandatory, so there is no "unknown vocab" path
    /// that skips this.
    #[test]
    fn logit_bias_keys_are_vocab_bounded() {
        let mut sp = SamplingParams {
            logit_bias: Some(BTreeMap::from([("1000".into(), 1.0)])),
            ..Default::default()
        };
        assert!(sp.clone().normalize(1000).is_err());
        assert!(sp.normalize(1001).is_ok());

        let mut sp = SamplingParams {
            logit_bias: Some(BTreeMap::from([("999".into(), -1.0)])),
            ..Default::default()
        };
        assert!(sp.normalize(1000).is_ok());
    }

    /// The key *format* check is separate from the vocab bound: the scheduler
    /// does `logit_bias[i, int(key)]`, so a key that is not a parseable
    /// non-negative integer has to be a 400 in its own right — a range check
    /// alone would let `"abc"` or `"1.5"` through to that indexing.
    #[test]
    fn logit_bias_keys_must_be_parseable_token_ids() {
        for params in [
            SamplingParams {
                logit_bias: Some(BTreeMap::from([("abc".into(), 1.0)])),
                ..Default::default()
            },
            SamplingParams {
                logit_bias: Some(BTreeMap::from([("-1".into(), 1.0)])),
                ..Default::default()
            },
            SamplingParams {
                logit_bias: Some(BTreeMap::from([("1.5".into(), 1.0)])),
                ..Default::default()
            },
            SamplingParams {
                logit_bias: Some(BTreeMap::from([("".into(), 1.0)])),
                ..Default::default()
            },
        ] {
            // Every key here is well inside TEST_VOCAB's range (or unparsable),
            // so only the format check can be what rejects it.
            let _ = norm_err(params);
        }
        assert!(
            norm(SamplingParams {
                logit_bias: Some(BTreeMap::from([("7".into(), 1.0)])),
                ..Default::default()
            })
            .logit_bias
            .is_some()
        );
    }

    /// Both `stop_regex` caps, neither of which had a test: deleting either `if`
    /// left the suite green. The count cap bounds per-step recompilation (Python's
    /// `re` cache is 512 entries); the length cap bounds compile time (a 1 MB
    /// literal pattern measured ~677 ms).
    #[test]
    fn stop_regex_count_and_length_are_capped() {
        let over: Vec<String> = (0..MAX_STOP_REGEX_COUNT + 1)
            .map(|i| format!("a{i}"))
            .collect();
        let params = SamplingParams {
            stop_regex: Some(OneOrMany::Many(over)),
            ..Default::default()
        };
        assert!(norm_err(params).to_string().contains("at most"));

        let at_cap: Vec<String> = (0..MAX_STOP_REGEX_COUNT).map(|i| format!("a{i}")).collect();
        let mut params = SamplingParams {
            stop_regex: Some(OneOrMany::Many(at_cap)),
            ..Default::default()
        };
        assert!(
            params.normalize(TEST_VOCAB).is_ok(),
            "the cap itself must be accepted"
        );

        let long = "a".repeat(MAX_STOP_REGEX_LEN + 1);
        let params = SamplingParams {
            stop_regex: Some(OneOrMany::One(long)),
            ..Default::default()
        };
        let err = norm_err(params).to_string();
        assert!(err.contains("over the"), "{err}");
    }

    /// The commoner field had no limit at all: the scheduler scans the decoded text
    /// once per stop per decode step.
    #[test]
    fn stop_string_count_is_capped() {
        let stops: Vec<String> = (0..MAX_STOP_COUNT + 1).map(|i| i.to_string()).collect();
        let params = SamplingParams {
            stop: Some(OneOrMany::Many(stops)),
            ..Default::default()
        };
        let err = norm_err(params).to_string();
        assert!(err.contains("at most"), "{err}");
    }
}
