use std::collections::HashMap;

pub(crate) struct MetricPack {
    pub(crate) labels: HashMap<String, String>,
    pub(crate) metrics_text: String,
}

pub(crate) fn aggregate_metrics(metric_packs: Vec<MetricPack>) -> anyhow::Result<String> {
    for metric_pack in metric_packs {
        let metrics = openmetrics_parser::prometheus::parse_prometheus(&metric_pack.metrics_text)?;
        todo!()
    }
    todo!()
}
