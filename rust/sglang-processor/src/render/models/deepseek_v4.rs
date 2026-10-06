//! Adapt SGLang's DeepSeek V4 effort profiles to Dynamo's native formatter.

use crate::model_files::resolve_model_file;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4Profile {
    Preview,
    Official,
}

pub(crate) fn dynamo_reasoning_effort(
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

pub(crate) fn resolve_dsv4_profile(
    profile: Option<&str>,
    model_source: &str,
    revision: Option<&str>,
) -> Result<DeepSeekV4Profile, String> {
    if let Some(profile) = profile {
        return match profile {
            "preview" => Ok(DeepSeekV4Profile::Preview),
            "official" => Ok(DeepSeekV4Profile::Official),
            _ => Err(format!(
                "invalid dsv4_reasoning_effort_profile: {profile:?}; expected \"preview\" or \"official\""
            )),
        };
    }
    let Some(encoder) = resolve_model_file(model_source, revision, "encoding/encoding_dsv4.py")
    else {
        return Ok(DeepSeekV4Profile::Preview);
    };
    let Ok(metadata) = std::fs::metadata(&encoder) else {
        return Ok(DeepSeekV4Profile::Preview);
    };
    if metadata.len() > 1 << 20 {
        return Ok(DeepSeekV4Profile::Preview);
    }
    let Ok(source) = std::fs::read_to_string(encoder) else {
        return Ok(DeepSeekV4Profile::Preview);
    };
    let default = top_level_python_assignment(&source, "DEFAULT_REASONING_EFFORT")
        .and_then(python_string_literal);
    let prompt_keys = top_level_python_assignment(&source, "REASONING_EFFORT_PROMPTS")
        .and_then(python_dict_keys)
        .unwrap_or_default();
    if default.as_deref() == Some("low")
        && ["low", "high", "max"]
            .iter()
            .all(|key| prompt_keys.iter().any(|candidate| candidate == key))
    {
        Ok(DeepSeekV4Profile::Official)
    } else {
        Ok(DeepSeekV4Profile::Preview)
    }
}

fn top_level_python_assignment<'a>(source: &'a str, name: &str) -> Option<&'a str> {
    let mut offset = 0;
    for line in source.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if !trimmed.starts_with(char::is_whitespace)
            && let Some((target, _)) = trimmed.split_once('=')
            && target
                .split(':')
                .next()
                .is_some_and(|target| target.trim() == name)
        {
            let equals = line.find('=')?;
            return Some(&source[offset + equals + 1..]);
        }
        offset += line.len();
    }
    None
}

fn python_string_literal(source: &str) -> Option<String> {
    let source = source.trim_start();
    let quote = source.chars().next()?;
    if !matches!(quote, '\'' | '"') {
        return None;
    }
    let mut escaped = false;
    let mut value = String::new();
    for character in source[quote.len_utf8()..].chars() {
        if escaped {
            value.push(character);
            escaped = false;
        } else if character == '\\' {
            escaped = true;
        } else if character == quote {
            return Some(value);
        } else {
            value.push(character);
        }
    }
    None
}

fn python_dict_keys(source: &str) -> Option<Vec<String>> {
    let source = source.trim_start();
    if !source.starts_with('{') {
        return None;
    }
    let mut keys = Vec::new();
    let mut depth = 0usize;
    let mut index = 0usize;
    let bytes = source.as_bytes();
    while index < bytes.len() {
        match bytes[index] {
            b'{' | b'[' | b'(' => {
                depth += 1;
                index += 1;
            }
            b'}' | b']' | b')' => {
                depth = depth.checked_sub(1)?;
                index += 1;
                if depth == 0 {
                    return Some(keys);
                }
            }
            quote @ (b'\'' | b'"') => {
                let start = index + 1;
                index = start;
                let mut escaped = false;
                while index < bytes.len() {
                    if escaped {
                        escaped = false;
                    } else if bytes[index] == b'\\' {
                        escaped = true;
                    } else if bytes[index] == quote {
                        break;
                    }
                    index += 1;
                }
                if index == bytes.len() {
                    return None;
                }
                let value = std::str::from_utf8(&bytes[start..index]).ok()?;
                index += 1;
                if depth == 1 {
                    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
                        index += 1;
                    }
                    if bytes.get(index) == Some(&b':') {
                        keys.push(value.to_owned());
                    }
                }
            }
            b'#' => {
                while index < bytes.len() && bytes[index] != b'\n' {
                    index += 1;
                }
            }
            _ => index += 1,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use dynamo_protocols::types::CreateChatCompletionRequest;
    use dynamo_renderer::PromptFormatter;
    use dynamo_renderer::deepseek::v4::DeepSeekV4Formatter;

    use super::{DeepSeekV4Profile, resolve_dsv4_profile};
    use crate::render::{ChatFormatter, TemplateArgsRequest};

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

    #[test]
    fn deepseek_v4_profile_resolution_uses_override_then_checkpoint_source() {
        let directory = std::env::temp_dir().join(format!(
            "sglang-processor-deepseek-v4-profile-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(directory.join("encoding")).unwrap();
        std::fs::write(
            directory.join("encoding/encoding_dsv4.py"),
            "DEFAULT_REASONING_EFFORT: str = 'low'\n\
REASONING_EFFORT_PROMPTS = {'low': '', 'high': 'absolute', 'max': 'beyond'}",
        )
        .unwrap();
        let source = directory.to_string_lossy();
        assert_eq!(
            resolve_dsv4_profile(None, &source, None).unwrap(),
            DeepSeekV4Profile::Official
        );
        assert_eq!(
            resolve_dsv4_profile(Some("preview"), &source, None).unwrap(),
            DeepSeekV4Profile::Preview
        );
        assert!(resolve_dsv4_profile(Some("future"), &source, None).is_err());

        std::fs::write(
            directory.join("encoding/encoding_dsv4.py"),
            r#"DEFAULT_REASONING_EFFORT = "high"
REASONING_EFFORT_PROMPTS = {"low": "", "high": "absolute", "max": "Beyond maximum"}"#,
        )
        .unwrap();
        assert_eq!(
            resolve_dsv4_profile(None, &source, None).unwrap(),
            DeepSeekV4Profile::Preview
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
