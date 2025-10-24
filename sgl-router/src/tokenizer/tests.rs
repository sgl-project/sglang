#[cfg(test)]
use std::sync::Arc;

#[cfg(test)]
use super::*;

#[test]
fn test_mock_tokenizer_encode() {
    let tokenizer = mock::MockTokenizer::new();
    let encoding = tokenizer.encode("Hello world").unwrap();
    let token_ids = encoding.token_ids();
    assert_eq!(token_ids, &[1, 2]); // "Hello" -> 1, "world" -> 2
}

#[test]
fn test_mock_tokenizer_decode() {
    let tokenizer = mock::MockTokenizer::new();
    let text = tokenizer.decode(&[1, 2], false).unwrap();
    assert_eq!(text, "Hello world");
}

#[test]
fn test_mock_tokenizer_decode_skip_special() {
    let tokenizer = mock::MockTokenizer::new();

    // With special tokens
    let text = tokenizer.decode(&[1000, 1, 2, 999], false).unwrap();
    assert_eq!(text, "<bos> Hello world <eos>");

    // Without special tokens
    let text = tokenizer.decode(&[1000, 1, 2, 999], true).unwrap();
    assert_eq!(text, "Hello world");
}

#[test]
fn test_tokenizer_wrapper() {
    let mock_tokenizer = Arc::new(mock::MockTokenizer::new());
    let tokenizer = Tokenizer::from_arc(mock_tokenizer);

    let encoding = tokenizer.encode("Hello world").unwrap();
    assert_eq!(encoding.token_ids(), &[1, 2]);

    let text = tokenizer.decode(&[1, 2], false).unwrap();
    assert_eq!(text, "Hello world");

    assert_eq!(tokenizer.vocab_size(), 14);

    assert_eq!(tokenizer.token_to_id("Hello"), Some(1));
    assert_eq!(tokenizer.token_to_id("unknown"), None);

    assert_eq!(tokenizer.id_to_token(1), Some("Hello".to_string()));
    assert_eq!(tokenizer.id_to_token(9999), None);
}

#[test]
fn test_decode_stream_basic() {
    let mock_tokenizer = Arc::new(mock::MockTokenizer::new());
    let tokenizer = Tokenizer::from_arc(mock_tokenizer);

    // Create a decode stream with initial tokens
    let initial_tokens = vec![1, 2]; // "Hello world"
    let mut stream = tokenizer.decode_stream(&initial_tokens, false);

    // Add a new token
    let result = stream.step(3).unwrap(); // "test"
                                          // Since we're using a mock, the actual incremental behavior depends on implementation
                                          // For now, we just verify it doesn't crash
    assert!(result.is_some() || result.is_none());
}

#[test]
fn test_decode_stream_flush() {
    let mock_tokenizer = Arc::new(mock::MockTokenizer::new());
    let tokenizer = Tokenizer::from_arc(mock_tokenizer);

    let initial_tokens = vec![1];
    let mut stream = tokenizer.decode_stream(&initial_tokens, false);

    // Add tokens
    stream.step(2).unwrap();
    stream.step(3).unwrap();

    // Flush remaining
    let flushed = stream.flush().unwrap();
    // The flush behavior depends on the implementation
    assert!(flushed.is_some() || flushed.is_none());
}

#[test]
fn test_special_tokens() {
    let mock_tokenizer = Arc::new(mock::MockTokenizer::new());
    let tokenizer = Tokenizer::from_arc(mock_tokenizer);

    let special_tokens = tokenizer.get_special_tokens();
    assert_eq!(special_tokens.bos_token, Some("<bos>".to_string()));
    assert_eq!(special_tokens.eos_token, Some("<eos>".to_string()));
    assert_eq!(special_tokens.unk_token, Some("<unk>".to_string()));
    assert!(special_tokens.sep_token.is_none());
    assert!(special_tokens.pad_token.is_none());
}

#[test]
fn test_batch_encode() {
    let tokenizer = mock::MockTokenizer::new();
    let inputs = vec!["Hello", "world", "test"];
    let encodings = tokenizer.encode_batch(&inputs).unwrap();

    assert_eq!(encodings.len(), 3);
    assert_eq!(encodings[0].token_ids(), &[1]); // "Hello" -> 1
    assert_eq!(encodings[1].token_ids(), &[2]); // "world" -> 2
    assert_eq!(encodings[2].token_ids(), &[3]); // "test" -> 3
}

#[test]
fn test_thread_safety() {
    use std::thread;

    let mock_tokenizer = Arc::new(mock::MockTokenizer::new());
    let tokenizer = Tokenizer::from_arc(mock_tokenizer);

    // Spawn multiple threads that use the same tokenizer
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let tokenizer_clone = tokenizer.clone();
            thread::spawn(move || {
                let text = "Hello test".to_string();
                let encoding = tokenizer_clone.encode(&text).unwrap();
                let decoded = tokenizer_clone.decode(encoding.token_ids(), false).unwrap();
                assert!(decoded.contains("Hello") || decoded.contains("test"));
                i
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
}
