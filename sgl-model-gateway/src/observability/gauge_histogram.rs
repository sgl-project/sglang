//! Gauge with gt/le bucket labels for Grafana heatmap visualization.
//!
//! Unlike Prometheus Histogram which uses cumulative buckets, this uses
//! non-cumulative buckets (gt < value <= le) suitable for heatmap display.
//!
//! Note: Keep in sync with Python implementation in
//! python/sglang/srt/utils/gauge_histogram.py

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

    pub fn len(&self) -> usize {
        self.le_labels.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&'static str, &'static str)> + '_ {
        std::iter::zip(&self.gt_labels, &self.le_labels).map(|(&gt, &le)| (gt, le))
    }

    /// Compute bucket counts from observations.
    pub fn compute_bucket_counts(&self, observations: &[u64]) -> Vec<usize> {
        let mut counts = vec![0usize; self.len()];
        for &value in observations {
            // Equivalent to Python's bisect.bisect_left. O(log n).
            let idx = self.upper_bounds.partition_point(|&bound| bound < value);
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

#[cfg(test)]
mod tests {
    use super::*;

    // --- Bucket Labels Tests ---

    #[test]
    fn test_bucket_labels_basic() {
        let buckets = BucketLabels::new(&[10, 30, 60]);
        let pairs: Vec<_> = buckets.iter().collect();
        assert_eq!(
            pairs,
            vec![("0", "10"), ("10", "30"), ("30", "60"), ("60", "+Inf")]
        );
    }

    #[test]
    fn test_bucket_labels_single_bound() {
        let buckets = BucketLabels::new(&[100]);
        let pairs: Vec<_> = buckets.iter().collect();
        assert_eq!(pairs, vec![("0", "100"), ("100", "+Inf")]);
    }

    #[test]
    fn test_bucket_labels_many_bounds() {
        let buckets = BucketLabels::new(&[1, 2, 5, 10]);
        let pairs: Vec<_> = buckets.iter().collect();
        assert_eq!(
            pairs,
            vec![
                ("0", "1"),
                ("1", "2"),
                ("2", "5"),
                ("5", "10"),
                ("10", "+Inf")
            ]
        );
    }

    #[test]
    fn test_bucket_labels_len() {
        let buckets = BucketLabels::new(&[10, 30, 60]);
        assert_eq!(buckets.len(), 4);
    }

    // --- Bucket Counts Tests ---

    #[test]
    fn test_compute_bucket_counts_empty() {
        let buckets = BucketLabels::new(&[10, 30, 60]);
        assert_eq!(buckets.compute_bucket_counts(&[]), vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_compute_bucket_counts_single_value_first_bucket() {
        let buckets = BucketLabels::new(&[10, 30, 60]);
        assert_eq!(buckets.compute_bucket_counts(&[5]), vec![1, 0, 0, 0]);
    }

    #[test]
    fn test_compute_bucket_counts_single_value_last_bucket() {
        let buckets = BucketLabels::new(&[10, 30, 60]);
        assert_eq!(buckets.compute_bucket_counts(&[100]), vec![0, 0, 0, 1]);
    }

    #[test]
    fn test_compute_bucket_counts_exact_boundary_values() {
        // Values at exact boundaries: 10 -> (0,10], 30 -> (10,30], 60 -> (30,60]
        let buckets = BucketLabels::new(&[10, 30, 60]);
        assert_eq!(
            buckets.compute_bucket_counts(&[10, 30, 60]),
            vec![1, 1, 1, 0]
        );
    }

    #[test]
    fn test_compute_bucket_counts_just_above_boundary() {
        // 11 -> (10,30], 31 -> (30,60], 61 -> (60,+Inf]
        let buckets = BucketLabels::new(&[10, 30, 60]);
        assert_eq!(
            buckets.compute_bucket_counts(&[11, 31, 61]),
            vec![0, 1, 1, 1]
        );
    }

    #[test]
    fn test_compute_bucket_counts_multiple_values_same_bucket() {
        let buckets = BucketLabels::new(&[10, 30, 60]);
        assert_eq!(
            buckets.compute_bucket_counts(&[1, 2, 3, 4, 5]),
            vec![5, 0, 0, 0]
        );
    }

    #[test]
    fn test_compute_bucket_counts_all_overflow() {
        let buckets = BucketLabels::new(&[10, 30, 60]);
        assert_eq!(
            buckets.compute_bucket_counts(&[100, 200, 300]),
            vec![0, 0, 0, 3]
        );
    }

    #[test]
    fn test_compute_bucket_counts_distribution() {
        // 5 (<=10), 10 (<=10), 15 (<=30), 40 (<=60), 100 (+Inf)
        let buckets = BucketLabels::new(&[10, 30, 60]);
        assert_eq!(
            buckets.compute_bucket_counts(&[5, 10, 15, 40, 100]),
            vec![2, 1, 1, 1]
        );
    }
}
