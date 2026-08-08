use anyhow::{anyhow, Result};
use std::str::FromStr;

/// Converts String to Bucket Object
impl FromStr for BucketRange {
    type Err = anyhow::Error;

    fn from_str(spec: &str) -> Result<Self> {
        let (name, range_part) = spec
            .split_once(':')
            .ok_or_else(|| anyhow!("missing ':' in the bucket spec: {spec:?}"))?;

        let (min, max) = range_part
            .split_once('-')
            .ok_or_else(|| anyhow!("missing '-' in the bucket spec: {spec:?}"))?;

        let min_tokens: u32 = min
            .parse()
            .map_err(|_| anyhow!("invalid min token count {min:?} in {spec:?}"))?;
        let max_tokens: Option<u32> = if max.is_empty() {
            None
        } else {
            Some(
                max.parse()
                    .map_err(|_| anyhow!("invalid max token count {max:?} in {spec:?}"))?,
            )
        };
        Ok(BucketRange {
            name: name.to_string(),
            min_tokens,
            max_tokens,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BucketRange {
    pub name: String,
    pub min_tokens: u32,
    pub max_tokens: Option<u32>,
}

impl BucketRange {
    pub fn contains(&self, token_count: u32) -> bool {
        if token_count < self.min_tokens {
            return false;
        }

        match self.max_tokens {
            Some(max_tokens) => token_count <= max_tokens,
            None => true,
        }
    }
}

/// Check a bucket list for cross-bucket invariants that a single
/// spec parse cannot see: duplicate names, overlapping ranges, and
/// min > max.
pub fn validate_bucket_ranges(buckets: &[BucketRange]) -> Result<()> {
    let mut seen = std::collections::HashSet::new();
    for bucket in buckets {
        if !seen.insert(&bucket.name) {
            return Err(anyhow!("duplicate bucket name {:?}", bucket.name));
        }
        if let Some(max) = bucket.max_tokens {
            if bucket.min_tokens > max {
                return Err(anyhow!(
                    "bucket {:?} has min_tokens {} > max_tokens {}",
                    bucket.name,
                    bucket.min_tokens,
                    max
                ));
            }
        }
    }

    // Sort by lower bound so overlaps can be detected by comparing
    // neighbours: an open-ended range covers everything above its min.
    let mut sorted: Vec<&BucketRange> = buckets.iter().collect();
    sorted.sort_by_key(|b| b.min_tokens);
    for pair in sorted.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        let overlaps = match a.max_tokens {
            None => true,
            Some(a_max) => a_max >= b.min_tokens,
        };
        if overlaps {
            return Err(anyhow!(
                "bucket ranges overlap: {:?} ({}-{:?}) and {:?} ({}-{:?})",
                a.name,
                a.min_tokens,
                a.max_tokens,
                b.name,
                b.min_tokens,
                b.max_tokens
            ));
        }
    }
    Ok(())
}

pub fn select_bucket(token_count: u32, buckets: &[BucketRange]) -> Option<&BucketRange> {
    buckets.iter().find(|b| b.contains(token_count))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_range_contains_token_count() {
        let bucket = BucketRange {
            name: "short".to_string(),
            min_tokens: 0,
            max_tokens: Some(2048),
        };

        assert!(bucket.contains(1000));
        assert!(!bucket.contains(4096));
    }
}
