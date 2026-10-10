//! Parser names SGLang ports from Python, where Dynamo's parsers split output
//! differently. Every other name goes to Dynamo.

mod deepseek_v4;

use super::ToolDetector;

/// The detector a `--tool-call-parser` name selects, over the request's tool names.
pub(crate) fn tool_detector(
    tool_parser: &str,
    tool_names: Vec<String>,
) -> Option<Box<dyn ToolDetector>> {
    let detector = match tool_parser {
        "deepseekv4" => deepseek_v4::DsmlDetector::new(deepseek_v4::DSML_TAGS, tool_names),
        "deepseekv41" => deepseek_v4::DsmlDetector::new(deepseek_v4::DSML_TAGS_V41, tool_names),
        _ => return None,
    };
    Some(Box::new(detector))
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
            let names: Vec<String> = fixture["tools"]
                .as_array()
                .unwrap()
                .iter()
                .map(|tool| tool["function"]["name"].as_str().unwrap().to_owned())
                .collect();
            let detector = || tool_detector(parser, names.clone()).unwrap();
            let calls = |calls: Vec<ToolCallItem>| -> Value {
                calls
                    .into_iter()
                    .map(|c| json!([c.tool_index, c.name, c.parameters]))
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
