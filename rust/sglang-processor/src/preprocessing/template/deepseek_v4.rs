//! Adapt SGLang's DeepSeek V4 effort profiles to Dynamo's native formatter.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeepSeekV4Profile {
    Preview,
    Official,
}

pub(super) fn dynamo_reasoning_effort(
    profile: DeepSeekV4Profile,
    effort: Option<&str>,
) -> &'static str {
    match (profile, effort) {
        (DeepSeekV4Profile::Preview, Some("max")) | (DeepSeekV4Profile::Official, Some("high")) => {
            "high"
        }
        (DeepSeekV4Profile::Official, Some("max")) => "max",
        // Dynamo's low effort preserves thinking without adding a prefix.
        _ => "low",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use dynamo_protocols::types::CreateChatCompletionRequest;
    use dynamo_renderer::PromptFormatter;
    use dynamo_renderer::deepseek::v4::DeepSeekV4Formatter;

    use super::super::{ChatFormatter, TemplateArgsRequest};
    use super::DeepSeekV4Profile;

    #[test]
    fn deepseek_v4_profiles_map_effort_without_coercing_unsupported_tiers() {
        fn render(
            profile: DeepSeekV4Profile,
            effort: Option<&str>,
            thinking: Option<bool>,
            environment_effort: Option<&str>,
        ) -> String {
            let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "model": "test",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Hello"}
                ]
            }))
            .unwrap();
            let mut args = HashMap::new();
            if let Some(effort) = effort {
                args.insert("reasoning_effort".into(), serde_json::json!(effort));
            }
            if let Some(thinking) = thinking {
                args.insert("thinking".into(), serde_json::json!(thinking));
            }
            ChatFormatter::DeepSeekV4 {
                formatter: PromptFormatter::OAI(Arc::new(DeepSeekV4Formatter::new_chat())),
                profile,
                environment_effort: environment_effort.map(str::to_owned),
            }
            .render(&TemplateArgsRequest {
                request: &request,
                args,
            })
            .unwrap()
        }

        let baseline = "<｜begin▁of▁sentence｜>Be concise.<｜User｜>Hello<｜Assistant｜><think>";
        for (profile, high_prefix, max_prefix) in [
            (DeepSeekV4Profile::Preview, None, "Absolute maximum"),
            (
                DeepSeekV4Profile::Official,
                Some("Absolute maximum"),
                "Beyond maximum",
            ),
        ] {
            for (effort, prefix) in [
                (None, None),
                (Some("low"), None),
                (Some("high"), high_prefix),
                (Some("max"), Some(max_prefix)),
                (Some("xhigh"), None),
            ] {
                let prompt = render(profile, effort, Some(true), None);
                assert_eq!(
                    prompt.matches("Reasoning Effort:").count(),
                    usize::from(prefix.is_some()),
                    "{profile:?}, {effort:?}: {prompt}"
                );
                if let Some(prefix) = prefix {
                    assert!(prompt.starts_with(&format!(
                        "<｜begin▁of▁sentence｜>Reasoning Effort: {prefix}"
                    )));
                    assert_eq!(
                        prompt.split_once("\n\n").unwrap().1,
                        baseline.strip_prefix("<｜begin▁of▁sentence｜>").unwrap()
                    );
                } else {
                    assert_eq!(prompt, baseline);
                }
            }
            let disabled = baseline.replace("<think>", "</think>");
            assert_eq!(render(profile, None, None, None), disabled);
            assert_eq!(render(profile, Some("max"), Some(false), None), disabled);
            assert_eq!(
                render(profile, None, Some(true), Some("max")),
                render(profile, Some("max"), Some(true), None)
            );
            assert_eq!(
                render(profile, Some("low"), Some(true), Some("max")),
                baseline
            );
        }
    }
}
