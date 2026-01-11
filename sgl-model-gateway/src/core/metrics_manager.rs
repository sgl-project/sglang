use anyhow::ensure;
use openmetrics_parser::{MetricFamily, MetricsExposition, MetricNumber, PrometheusType, PrometheusValue};
use std::collections::HashMap;
use tracing::warn;

#[derive(Debug)]
pub struct MetricPack {
    pub labels: Vec<(String, String)>,
    pub metrics_text: String,
}

type PrometheusExposition = MetricsExposition<PrometheusType, PrometheusValue>;
type PrometheusFamily = MetricFamily<PrometheusType, PrometheusValue>;

/// Aggregate Prometheus metrics scraped from multiple sources into a unified one
pub fn aggregate_metrics(metric_packs: Vec<MetricPack>) -> anyhow::Result<String> {
    let mut expositions = vec![];
    for metric_pack in metric_packs {
        let metrics_text = &metric_pack.metrics_text;
        // openmetrics_parser doesn't handle colons in metric names; replace with underscores
        let metrics_text = metrics_text.replace(":", "_");

        let exposition = match openmetrics_parser::prometheus::parse_prometheus(&metrics_text) {
            Ok(x) => x,
            Err(err) => {
                warn!(
                    "aggregate_metrics error when parsing text: pack={:?} err={:?}",
                    metric_pack, err
                );
                continue;
            }
        };
        let exposition = transform_metrics(exposition, &metric_pack.labels);
        expositions.push(exposition);
    }

    let text = try_reduce(expositions.into_iter(), merge_exposition)?
        .map(|x| format!("{x}"))
        .unwrap_or_default();
    Ok(text)
}

fn transform_metrics(
    mut exposition: PrometheusExposition,
    extra_labels: &[(String, String)],
) -> PrometheusExposition {
    for family in exposition.families.values_mut() {
        *family = family.with_labels(extra_labels.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    }
    exposition
}

fn merge_exposition(
    a: PrometheusExposition,
    b: PrometheusExposition,
) -> anyhow::Result<PrometheusExposition> {
    let mut ans = a;
    for (name, family_b) in b.families.into_iter() {
        let family_merged = if let Some(family_a) = ans.families.remove(&name) {
            merge_family(family_a, family_b)?
        } else {
            family_b
        };
        ans.families.insert(name, family_merged);
    }
    Ok(ans)
}

fn merge_family(a: PrometheusFamily, b: PrometheusFamily) -> anyhow::Result<PrometheusFamily> {
    ensure!(
        a.get_label_names() == b.get_label_names(),
        "Label names should agree a={:?} b={:?}",
        a.get_label_names(),
        b.get_label_names()
    );
    a.with_samples(b.into_iter_samples())
        .map_err(|e| anyhow::anyhow!("failed to merge samples: {e:?}"))
}

fn try_reduce<I, T, E, F>(iterable: I, f: F) -> Result<Option<T>, E>
where
    I: IntoIterator<Item = T>,
    F: FnMut(T, T) -> Result<T, E>,
{
    let mut it = iterable.into_iter();
    let first = match it.next() {
        None => return Ok(None),
        Some(x) => x,
    };

    Ok(Some(it.try_fold(first, f)?))
}

pub fn extract_gauge_metrics(text: String, target_metric_family: &str) -> HashMap<isize, isize> {
    let metrics_text = text.replace(":", "_");
    let exposition = match openmetrics_parser::prometheus::parse_prometheus(&metrics_text) {
        Ok(x) => x,
        Err(err) => {
            warn!(
                "parse_load_response error when parsing text: pack={:?} err={:?}",
                metrics_text, err
            );
            return HashMap::new();
        }
    };
    extract_metrics(
        &exposition,
        target_metric_family,
        &PrometheusValue::Gauge(MetricNumber::Float(0.0)))
}

pub fn extract_metrics(
    exposition: &PrometheusExposition,
    target_metric_family: &str,
    target_value_type: &PrometheusValue
) -> HashMap<isize, isize> {
    let mut result = HashMap::new();
    let Some(target_families) = exposition.families.get(target_metric_family) else {
        warn!("{} don't exist!", target_metric_family);
        return result;
    };

    for sample in target_families.iter_samples() {
        let label_set = match sample.get_labelset() {
            Ok(l_set) => l_set,
            Err(e) => {
                warn!("The metric is missing the dp_rank label{}, skipping.", e);
                continue;
            }
        };

        let dp_rank_str = match label_set.get_label_value("dp_rank") {
            Some(val) => val,
            None => {
                warn!("Don't find dp_rank");
                "0"
            }
        };

        let dp_rank = match dp_rank_str.parse::<isize>() {
            Ok(rank_num) => rank_num,
            Err(e) => {
                warn!("Failed to parse dp_rank value {} as number: {}", dp_rank_str, e);
                0
            }
        };

        let metric_value = match (&target_value_type, &sample.value) {
            (PrometheusValue::Gauge(_), PrometheusValue::Gauge(val)) => {
                val.as_f64()
            }
            (target_type, actual_type) => {
                warn!("Unadapted PrometheusValue. Expected:{:?}, Actual:{:?}.", target_type, actual_type);
                continue;
            }
        };

        let value = metric_value as isize;
        result.insert(dp_rank, value);
    }
    result
}
