//! Gauge with gt/le bucket labels for Grafana heatmap visualization.
//!
//! Unlike Prometheus Histogram which uses cumulative buckets, this uses
//! non-cumulative buckets (gt < value <= le) suitable for heatmap display.
//!
//! Note: Keep in sync with Python implementation in
//! python/sglang/srt/utils/gauge_histogram.py

use std::sync::LazyLock;

use metrics::gauge;

pub struct BucketLabels {
    upper_bounds: &'static [u64],
    le_labels: Vec<&'static str>,
    gt_labels: Vec<&'static str>,
}

impl BucketLabels {
    pub fn new(upper_bounds: &'static [u64]) -> Self {
        let leak_str = |n: u64| Box::leak(n.to_string().into_boxed_str()) as &'static str;

        let mut le_labels: Vec<&'static str> = upper_bounds.iter().map(|&b| leak_str(b)).collect();
        le_labels.push("+Inf");

        let mut gt_labels: Vec<&'static str> = vec!["0"];
        gt_labels.extend(upper_bounds.iter().map(|&b| leak_str(b)));

        Self {
            upper_bounds,
            le_labels,
            gt_labels,
        }
    }

    pub fn upper_bounds(&self) -> &[u64] {
        self.upper_bounds
    }

    pub fn len(&self) -> usize {
        self.le_labels.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&'static str, &'static str)> + '_ {
        std::iter::zip(&self.gt_labels, &self.le_labels).map(|(&gt, &le)| (gt, le))
    }

    pub fn find_bucket_index(&self, value: u64) -> usize {
        self.upper_bounds
            .iter()
            .position(|&bound| value <= bound)
            .unwrap_or(self.len() - 1)
    }

    /// Compute bucket counts from observations.
    pub fn compute_bucket_counts(&self, observations: &[u64]) -> Vec<usize> {
        let mut counts = vec![0usize; self.len()];
        for &value in observations {
            let idx = self.find_bucket_index(value);
            counts[idx] += 1;
        }
        counts
    }
}

pub struct GaugeHistogram {
    name: &'static str,
    buckets: &'static BucketLabels,
}

impl GaugeHistogram {
    pub const fn new(name: &'static str, buckets: &'static BucketLabels) -> Self {
        Self { name, buckets }
    }

    /// Set bucket counts directly.
    pub fn set_raw(&self, values: &[usize]) {
        for ((gt, le), &count) in self.buckets.iter().zip(values.iter()) {
            gauge!(self.name, "gt" => gt, "le" => le).set(count as f64);
        }
    }

    /// Compute bucket counts from observations and set them.
    pub fn set_by_current_observations(&self, observations: &[u64]) {
        let counts = self.buckets.compute_bucket_counts(observations);
        self.set_raw(&counts);
    }

    pub fn buckets(&self) -> &BucketLabels {
        self.buckets
    }
}

pub const INFLIGHT_AGE_BUCKET_BOUNDS: &[u64] =
    &[30, 60, 180, 300, 600, 1200, 3600, 7200, 14400, 28800, 86400];
pub static INFLIGHT_AGE_BUCKETS: LazyLock<BucketLabels> =
    LazyLock::new(|| BucketLabels::new(INFLIGHT_AGE_BUCKET_BOUNDS));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_labels() {
        let buckets = BucketLabels::new(&[10, 30, 60]);

        assert_eq!(buckets.le_labels, vec!["10", "30", "60", "+Inf"]);
        assert_eq!(buckets.gt_labels, vec!["0", "10", "30", "60"]);
        assert_eq!(buckets.len(), 4);
    }

    #[test]
    fn test_find_bucket_index() {
        let buckets = BucketLabels::new(&[10, 30, 60]);

        assert_eq!(buckets.find_bucket_index(5), 0);
        assert_eq!(buckets.find_bucket_index(10), 0);
        assert_eq!(buckets.find_bucket_index(11), 1);
        assert_eq!(buckets.find_bucket_index(30), 1);
        assert_eq!(buckets.find_bucket_index(31), 2);
        assert_eq!(buckets.find_bucket_index(60), 2);
        assert_eq!(buckets.find_bucket_index(61), 3);
        assert_eq!(buckets.find_bucket_index(1000), 3);
    }

    #[test]
    fn test_iter() {
        let buckets = BucketLabels::new(&[10, 30]);
        let pairs: Vec<_> = buckets.iter().collect();
        assert_eq!(pairs, vec![("0", "10"), ("10", "30"), ("30", "+Inf")]);
    }

    #[test]
    fn test_compute_bucket_counts_empty() {
        let buckets = BucketLabels::new(&[10, 30, 60]);
        let counts = buckets.compute_bucket_counts(&[]);
        assert_eq!(counts, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_compute_bucket_counts_distribution() {
        let buckets = BucketLabels::new(&[10, 30, 60]);
        // Values: 5 (<=10), 10 (<=10), 15 (<=30), 40 (<=60), 100 (+Inf)
        let counts = buckets.compute_bucket_counts(&[5, 10, 15, 40, 100]);
        assert_eq!(counts, vec![2, 1, 1, 1]);
    }

    #[test]
    fn test_compute_bucket_counts_all_in_one_bucket() {
        let buckets = BucketLabels::new(&[10, 30, 60]);
        let counts = buckets.compute_bucket_counts(&[1, 2, 3, 4, 5]);
        assert_eq!(counts, vec![5, 0, 0, 0]);
    }

    #[test]
    fn test_compute_bucket_counts_all_overflow() {
        let buckets = BucketLabels::new(&[10, 30, 60]);
        let counts = buckets.compute_bucket_counts(&[100, 200, 300]);
        assert_eq!(counts, vec![0, 0, 0, 3]);
    }

    #[test]
    fn test_inflight_age_buckets() {
        let buckets = &*INFLIGHT_AGE_BUCKETS;

        assert_eq!(buckets.le_labels.first(), Some(&"30"));
        assert_eq!(buckets.le_labels.last(), Some(&"+Inf"));
        assert_eq!(buckets.gt_labels.first(), Some(&"0"));
        assert_eq!(buckets.gt_labels.last(), Some(&"86400"));
        assert_eq!(buckets.le_labels.len(), buckets.gt_labels.len());
    }
}
