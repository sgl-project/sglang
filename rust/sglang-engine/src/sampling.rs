//! CPU reference sampling primitives for model-executor implementations.
//!
//! The production sampler will use accelerator kernels, but must preserve the
//! same token-selection rules. This reference defines first-index tie behavior
//! for greedy sampling and gives executor/parity tests a small deterministic
//! oracle.

/// Greedy selection error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GreedySampleError {
    #[error("cannot sample from an empty logits row")]
    EmptyLogits,
    #[error("logits contain NaN at vocabulary index {0}")]
    NaNLogit(usize),
    #[error("logits do not contain a finite sampling distribution")]
    NoFiniteDistribution,
}

/// Sample one token using temperature, top-k, and top-p filtering.
///
/// The seeded CPU implementation is a reference for model backends. It uses
/// descending-logit order with token index as the tie-breaker, applies top-k
/// before nucleus filtering, and retains the token that crosses the top-p
/// cumulative probability threshold.
pub fn sample_from_logits(
    logits: &[f32],
    temperature: f32,
    top_k: u32,
    top_p: f32,
    seed: u64,
) -> Result<usize, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::Greedy(GreedySampleError::EmptyLogits));
    }
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(SamplingError::InvalidParameter("temperature"));
    }
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        return Err(SamplingError::InvalidParameter("top_p"));
    }
    if temperature == 0.0 {
        return greedy_sample(logits).map_err(SamplingError::Greedy);
    }

    let mut ranked = Vec::with_capacity(logits.len());
    for (index, &logit) in logits.iter().enumerate() {
        if logit.is_nan() {
            return Err(SamplingError::Greedy(GreedySampleError::NaNLogit(index)));
        }
        if logit.is_finite() {
            ranked.push((index, f64::from(logit) / f64::from(temperature)));
        }
    }
    if ranked.is_empty() {
        return Err(SamplingError::NoFiniteDistribution);
    }
    ranked.sort_by(|(left_index, left), (right_index, right)| {
        right
            .total_cmp(left)
            .then_with(|| left_index.cmp(right_index))
    });
    if top_k > 0 {
        ranked.truncate((top_k as usize).min(ranked.len()));
    }

    let max_logit = ranked[0].1;
    let mut probs: Vec<_> = ranked
        .into_iter()
        .map(|(index, logit)| (index, (logit - max_logit).exp()))
        .collect();
    let sum: f64 = probs.iter().map(|(_, probability)| probability).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(SamplingError::NoFiniteDistribution);
    }
    for (_, probability) in &mut probs {
        *probability /= sum;
    }

    if top_p < 1.0 {
        let mut cumulative = 0.0;
        let mut keep = 0;
        for (_, probability) in &probs {
            cumulative += probability;
            keep += 1;
            if cumulative >= f64::from(top_p) {
                break;
            }
        }
        probs.truncate(keep.max(1));
        let retained_sum: f64 = probs.iter().map(|(_, probability)| probability).sum();
        for (_, probability) in &mut probs {
            *probability /= retained_sum;
        }
    }

    let draw = uniform01(seed) * probs.iter().map(|(_, p)| p).sum::<f64>();
    let mut cumulative = 0.0;
    for (index, probability) in &probs {
        cumulative += probability;
        if draw < cumulative {
            return Ok(*index);
        }
    }
    Ok(probs.last().expect("non-empty distribution").0)
}

fn uniform01(seed: u64) -> f64 {
    let mut state = if seed == 0 {
        0x9e37_79b9_7f4a_7c15
    } else {
        seed
    };
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    (state >> 11) as f64 / ((1_u64 << 53) as f64)
}

/// Invalid sampling parameters or a logits row without a usable distribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SamplingError {
    #[error(transparent)]
    Greedy(GreedySampleError),
    #[error("invalid sampling parameter: {0}")]
    InvalidParameter(&'static str),
    #[error("logits do not contain a finite sampling distribution")]
    NoFiniteDistribution,
}

/// Select the maximum-logit token, preserving the first token on ties.
///
/// Negative infinity is accepted because model execution commonly masks
/// vocabulary entries with `-inf`; if all entries are masked, index zero
/// matches first-index argmax behavior.
pub fn greedy_sample(logits: &[f32]) -> Result<usize, GreedySampleError> {
    let (&first, rest) = logits.split_first().ok_or(GreedySampleError::EmptyLogits)?;
    if first.is_nan() {
        return Err(GreedySampleError::NaNLogit(0));
    }
    let mut best_index = 0;
    let mut best_logit = first;
    for (offset, &logit) in rest.iter().enumerate() {
        if logit.is_nan() {
            return Err(GreedySampleError::NaNLogit(offset + 1));
        }
        if logit > best_logit {
            best_logit = logit;
            best_index = offset + 1;
        }
    }
    Ok(best_index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_selects_maximum_and_first_index_on_ties() {
        assert_eq!(greedy_sample(&[0.2, 4.0, 4.0, -1.0]), Ok(1));
    }

    #[test]
    fn greedy_accepts_masked_negative_infinity() {
        assert_eq!(greedy_sample(&[f32::NEG_INFINITY, -2.0]), Ok(1));
        assert_eq!(
            greedy_sample(&[f32::NEG_INFINITY, f32::NEG_INFINITY]),
            Ok(0)
        );
    }

    #[test]
    fn greedy_rejects_empty_or_nan_logits() {
        assert_eq!(greedy_sample(&[]), Err(GreedySampleError::EmptyLogits));
        assert_eq!(
            greedy_sample(&[0.0, f32::NAN]),
            Err(GreedySampleError::NaNLogit(1))
        );
    }

    #[test]
    fn top_k_and_top_p_keep_the_highest_token() {
        let logits = [2.0, 1.0, -4.0];
        assert_eq!(sample_from_logits(&logits, 1.0, 1, 1.0, 7), Ok(0));
        assert_eq!(sample_from_logits(&logits, 1.0, 0, 0.01, 7), Ok(0));
    }

    #[test]
    fn seeded_sampling_is_reproducible_and_temperature_zero_is_greedy() {
        let logits = [0.0, 1.0, 2.0];
        assert_eq!(
            sample_from_logits(&logits, 1.0, 0, 1.0, 123),
            sample_from_logits(&logits, 1.0, 0, 1.0, 123)
        );
        assert_eq!(sample_from_logits(&logits, 0.0, 0, 1.0, 123), Ok(2));
    }

    #[test]
    fn sampling_rejects_invalid_parameters_and_all_masked_logits() {
        assert_eq!(
            sample_from_logits(&[1.0], 1.0, 0, 0.0, 1),
            Err(SamplingError::InvalidParameter("top_p"))
        );
        assert_eq!(
            sample_from_logits(&[f32::NEG_INFINITY], 1.0, 0, 1.0, 1),
            Err(SamplingError::NoFiniteDistribution)
        );
    }
}
