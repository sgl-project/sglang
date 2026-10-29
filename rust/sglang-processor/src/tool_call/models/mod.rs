//! Parser names SGLang ports from Python, where Dynamo's parsers split output
//! differently. Every other name goes to Dynamo.

mod deepseek_v4;
mod kimi_k3;
mod qwen3_coder;

use serde_json::Value;

use super::ToolDetector;

/// The detector a `--tool-call-parser` name selects, over the request's tools.
pub(crate) fn tool_detector(tool_parser: &str, tools: &[Value]) -> Option<Box<dyn ToolDetector>> {
    let tool_names = || {
        tools
            .iter()
            .filter_map(|tool| tool["function"]["name"].as_str().map(str::to_owned))
            .collect()
    };
    Some(match tool_parser {
        "deepseekv4" => Box::new(deepseek_v4::DsmlDetector::new(
            deepseek_v4::DSML_TAGS,
            tool_names(),
        )),
        "deepseekv41" => Box::new(deepseek_v4::DsmlDetector::new(
            deepseek_v4::DSML_TAGS_V41,
            tool_names(),
        )),
        // `KimiK3Detector` keeps calls to tools the request does not list.
        "kimi_k3" => Box::new(kimi_k3::KimiK3Detector::default()),
        // Typed argument values follow the tool schemas.
        "qwen3_coder" => Box::new(qwen3_coder::Qwen3CoderDetector::new(tools.to_vec())),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_call::ToolCallItem;
    use serde_json::{Value, json};

    /// `tests/fixtures/tool_parity/*.json`, from `tests/scripts/generate_tool_parity.py`.
    #[test]
    fn tool_fixtures_match_sglang() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tool_parity");
        for entry in std::fs::read_dir(dir).unwrap() {
            let fixture: Value =
                serde_json::from_str(&std::fs::read_to_string(entry.unwrap().path()).unwrap())
                    .unwrap();
            let parser = fixture["parser"].as_str().unwrap();
            let tools = fixture["tools"].as_array().unwrap();
            let detector = || tool_detector(parser, tools).unwrap();
            // An argument delta has no name: `None` in Python, empty here.
            let calls = |calls: Vec<ToolCallItem>| -> Value {
                calls
                    .into_iter()
                    .map(|c| {
                        json!([
                            c.tool_index,
                            (!c.name.is_empty()).then_some(c.name),
                            c.parameters
                        ])
                    })
                    .collect()
            };
            for case in fixture["cases"].as_array().unwrap() {
                let chunks: Vec<&str> = case["chunks"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| c.as_str().unwrap())
                    .collect();
                let mut streaming = detector();
                let steps: Vec<Value> = chunks
                    .iter()
                    .enumerate()
                    .map(|(i, chunk)| {
                        let (normal, found) = streaming.parse_stream(chunk, i + 1 == chunks.len());
                        json!([normal, calls(found)])
                    })
                    .collect();
                let (normal, found) = detector().parse_non_stream(&chunks.concat());
                let got = json!({"steps": steps, "unary": [normal, calls(found)]});
                let want = json!({"steps": case["steps"], "unary": case["unary"]});
                assert_eq!(got, want, "{parser} {:?}", chunks);
            }
        }
    }
}
