pub(crate) struct EngineMetricsOutput {
    pub(crate) worker_base_url: String,
    pub(crate) body_text: String,
}

pub(crate) fn compute_engine_metrics(engine_responses: Vec<EngineMetricsOutput>) -> anyhow::Result<String> {
    for engine_response in engine_responses {
        openmetrics_parser::prometheus::parse_prometheus(&engine_response.body_text)?;
    }
    todo!()
}
