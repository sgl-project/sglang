//! Gauge with gt/le bucket labels for Grafana heatmap visualization.
//!
//! Unlike Prometheus Histogram which uses cumulative buckets, this uses
//! non-cumulative buckets (gt < value <= le) suitable for heatmap display.
//!
//! Note: Keep in sync with Python implementation in
//! python/sglang/srt/metrics/collector.py (GaugeHistogram class)

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
}

pub struct GaugeHistogram {
    name: &'static str,
    buckets: &'static BucketLabels,
}

impl GaugeHistogram {
    pub const fn new(name: &'static str, buckets: &'static BucketLabels) -> Self {
        Self { name, buckets }
    }

    pub fn set(&self, values: &[usize]) {
        for ((gt, le), &count) in self.buckets.iter().zip(values.iter()) {
            gauge!(self.name, "gt" => gt, "le" => le).set(count as f64);
        }
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
    fn test_inflight_age_buckets() {
        let buckets = &*INFLIGHT_AGE_BUCKETS;

        assert_eq!(buckets.le_labels.first(), Some(&"30"));
        assert_eq!(buckets.le_labels.last(), Some(&"+Inf"));
        assert_eq!(buckets.gt_labels.first(), Some(&"0"));
        assert_eq!(buckets.gt_labels.last(), Some(&"86400"));
        assert_eq!(buckets.le_labels.len(), buckets.gt_labels.len());
    }
}
