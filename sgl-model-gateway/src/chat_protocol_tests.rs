#[cfg(test)]
mod tests {
    use openai_protocol::{chat::ChatCompletionRequest, validated::Normalizable};
    use validator::Validate;
    use serde_json::json;

    fn request(top_logprobs: u32) -> ChatCompletionRequest {
        serde_json::from_value(json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "logprobs": true,
            "top_logprobs": top_logprobs
        }))
        .unwrap()
    }

    #[test]
    fn accepts_extended_head_without_modifying_request() {
        for count in [0, 16, 20, 21, 128] {
            let mut body = request(count);
            body.normalize();
            assert!(body.validate().is_ok(), "{count}");
            assert_eq!(
                serde_json::to_value(&body).unwrap()["top_logprobs"],
                count
            );
        }
    }

    #[test]
    fn rejects_head_above_backend_limit() {
        for count in [129, u32::MAX] {
            let errors = request(count).validate().unwrap_err();
            assert!(errors.field_errors().contains_key("top_logprobs"));
        }
    }

    #[test]
    fn preserves_other_protocol_validation() {
        let mut body = request(128);
        body.temperature = Some(3.0);
        let errors = body.validate().unwrap_err();
        assert!(errors.field_errors().contains_key("temperature"));
    }

    #[test]
    fn optional_head_and_invalid_json_types() {
        let value = json!({"model":"test","messages":[{"role":"user","content":"hello"}]});
        let body: ChatCompletionRequest = serde_json::from_value(value.clone()).unwrap();
        assert!(body.validate().is_ok());
        for bad in [json!(-1), json!(1.5), json!("128")] {
            let mut value = value.clone();
            value["top_logprobs"] = bad;
            assert!(serde_json::from_value::<ChatCompletionRequest>(value).is_err());
        }
    }
    #[test]
    fn preserves_session_extensions() {
        let mut value = serde_json::to_value(request(128)).unwrap();
        value["input_ids"] = json!([1, 2, 3]);
        value["return_meta_info"] = json!(true);
        let mut body: ChatCompletionRequest = serde_json::from_value(value).unwrap();
        body.normalize();
        assert!(body.validate().is_ok());
        let outgoing = serde_json::to_value(body).unwrap();
        assert_eq!(outgoing["input_ids"], json!([1, 2, 3]));
        assert_eq!(outgoing["return_meta_info"], json!(true));
        assert_eq!(outgoing["top_logprobs"], json!(128));
    }

    #[test]
    fn requires_logprobs_for_extended_head() {
        let mut body = request(128);
        body.logprobs = false;
        assert!(body.validate().is_err());
    }

}
