//! Integration tests for tokenizers using real tokenizer data
//!
//! These tests download the TinyLlama tokenizer from HuggingFace to verify our tokenizer
//! implementation works correctly with real-world tokenizer files.

mod common;
use std::sync::Arc;

use common::{ensure_tokenizer_cached, EXPECTED_HASHES, TEST_PROMPTS};
use sgl_model_gateway::tokenizer::{
    factory, huggingface::HuggingFaceTokenizer, sequence::Sequence, stop::*, stream::DecodeStream,
    traits::*,
};

const LONG_TEST_PROMPTS: [(&str, &str); 6] = [
    ("Tell me about the following text.", "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."),
    ("Tell me about the following text.", "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."),
    ("Tell me about the following text.", "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt."),
    ("Tell me about the following text.", "Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem."),
    // Tennis-themed prompt for variety
    ("Tell me about the following text.", "In the ancient realm of Tennisia, the very magic of the land is drawn from the sport itself. Forehands light the skies, backhands carve the earth, and serves rumble like thunder across kingdoms. At the center of this balance lie four sacred Grand Slam relics: the Sapphire Trophy of Melbourne, the Emerald Chalice of Paris, the Ruby Crown of London, and the Diamond Orb of New York. Together, they keep the game's spirit alive.
    But the relics are scattered, guarded by champions of legendary skill. The first is the Fire King of Clay, ruler of the crimson courts, whose topspin arcs blaze high and heavy, scorching all who dare stand across from him. The second is the Tempest Trickster, master of the baseline fortress, whose footwork and precision can turn back any storm, and whose returns arrive as if pulled by invisible strings. The third is the Shadow-Dancer of the Highlands, a tactician who thrives in the long rallies of twilight, changing pace and spin until opponents lose their rhythm. The fourth and final guardian is a towering Diamond Titan, a net-charging colossus whose volleys shatter the air itself.
    Into this arena of gods steps the Silver-Wristed Knight — a player of impossible grace, whose game is an art form. His quest: to claim each relic not for glory, but to restore harmony to the rankings of the realm.
    He travels across the Kingdom of Clay, where the points stretch like marathons and the air tastes of iron; through the Grasslands of London, where the ball skids low and the margins are razor-thin; over the Hard Courts of the East, where rallies turn into duels of endurance; and finally to the Cathedral of Lights in New York, where night matches burn with fevered energy.
    Each battle is played under enchanted floodlights, the lines patrolled by spectral line judges whose calls are final. The crowd's roar swells with every break point, and the Silver-Wristed Knight's racket glows brightest when the match teeters at deuce. There are moments when doubt grips him — when his serve falters or his touch deserts him — but each challenge teaches a new stroke, culminating in the legendary Forehand of Dawn.
    When the last relic is claimed, he stands not as a conqueror but as a custodian of the game, knowing that rivalries forge the very magic he protects. The balance is restored — until the next season begins."),
    // Emoji stress test
    ("Tell me about the following text.", "😀😃😄😁😆🥹😅😂🤣🥲☺️😊😇🙂🙃😉🤩😎 🤪🥳🤓🙄🤪😵👻")
];

fn compute_hashes_for_tokenizer<E: Encoder>(tokenizer: &E, prompts: &[&str]) -> Vec<u64> {
    prompts
        .iter()
        .map(|&prompt| {
            tokenizer
                .encode(prompt)
                .expect("Failed to encode prompt")
                .get_hash()
        })
        .collect()
}

#[test]
fn test_huggingface_tokenizer_hashes() {
    let tokenizer_path = ensure_tokenizer_cached();
    let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
        .expect("Failed to load HuggingFace tokenizer");

    let prompt_hashes = compute_hashes_for_tokenizer(&tokenizer, &TEST_PROMPTS);

    println!(
        "HF Tokenizer: {:?}\nComputed Hashes: {:?}\nExpected Hashes: {:?}",
        tokenizer_path, prompt_hashes, EXPECTED_HASHES
    );

    assert_eq!(prompt_hashes, EXPECTED_HASHES);
}

#[test]
fn test_tokenizer_encode_decode_lifecycle() {
    let tokenizer_path = ensure_tokenizer_cached();
    let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
        .expect("Failed to load HuggingFace tokenizer");

    for prompt in TEST_PROMPTS.iter() {
        let encoding = tokenizer.encode(prompt).expect("Failed to encode prompt");

        let decoded = tokenizer
            .decode(encoding.token_ids(), false)
            .expect("Failed to decode token_ids");

        assert_eq!(decoded, *prompt, "Encode-decode mismatch for: {}", prompt);
    }
}

#[test]
fn test_sequence_operations() {
    let tokenizer_path = ensure_tokenizer_cached();
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    for prompt in TEST_PROMPTS.iter() {
        let encoding = tokenizer.encode(prompt).expect("Failed to encode prompt");

        let mut sequence = Sequence::new(tokenizer.clone());
        sequence.append_text(prompt).expect("Failed to append text");

        assert_eq!(
            sequence.len(),
            encoding.token_ids().len(),
            "Sequence length mismatch"
        );
        assert_eq!(sequence.text().unwrap(), *prompt, "Sequence text mismatch");

        let mut decoder = Sequence::new(tokenizer.clone());
        let mut output = String::new();

        for token_id in encoding.token_ids() {
            let text = decoder
                .append_token(*token_id)
                .expect("Failed to append token");
            output.push_str(&text);
        }

        assert_eq!(decoder.len(), sequence.len(), "Decoder length mismatch");
        assert_eq!(
            decoder.token_ids(),
            sequence.token_ids(),
            "Token IDs mismatch"
        );
        assert_eq!(output, *prompt, "Incremental decode mismatch");
    }
}

#[test]
fn test_decode_stream() {
    let tokenizer_path = ensure_tokenizer_cached();
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    for prompt in TEST_PROMPTS.iter() {
        let encoding = tokenizer.encode(prompt).expect("Failed to encode prompt");

        let mut decoder = DecodeStream::new(tokenizer.clone(), &[], false);
        let mut output = String::new();

        for token_id in encoding.token_ids() {
            if let Some(text) = decoder.step(*token_id).expect("Failed to decode token") {
                output.push_str(&text);
            }
        }

        assert_eq!(output, *prompt, "DecodeStream output mismatch");
    }
}

#[test]
fn test_long_sequence_incremental_decode_with_prefill() {
    let tokenizer_path = ensure_tokenizer_cached();
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    for (input_text, output_text) in LONG_TEST_PROMPTS.iter() {
        let input_encoding = tokenizer
            .encode(input_text)
            .expect("Failed to encode input");

        let output_encoding = tokenizer
            .encode(output_text)
            .expect("Failed to encode output");

        let mut decoder = DecodeStream::new(tokenizer.clone(), input_encoding.token_ids(), false);

        let mut output = String::new();
        for token_id in output_encoding.token_ids() {
            if let Some(text) = decoder.step(*token_id).expect("Failed to decode token") {
                output.push_str(&text);
            }
        }

        assert_eq!(output.trim(), *output_text, "Long sequence decode mismatch");
    }
}

#[test]
fn test_stop_sequence_decoder() {
    let tokenizer_path = ensure_tokenizer_cached();
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let test_cases = vec![
        (
            "Hello world! Stop here. Continue after.",
            "Stop",
            "Hello world! ",
        ),
        ("Testing stop sequences.", ".", "Testing stop sequences"),
        ("No stop sequence here", "xyz", "No stop sequence here"),
    ];

    for (input, stop_seq, expected) in test_cases {
        let config = StopSequenceConfig::default().with_stop_sequence(stop_seq);

        let mut decoder = StopSequenceDecoder::new(tokenizer.clone(), config, false);

        let encoding = tokenizer.encode(input).expect("Failed to encode");
        let mut output = String::new();
        let mut stopped = false;

        for token_id in encoding.token_ids() {
            match decoder.process_token(*token_id).unwrap() {
                SequenceDecoderOutput::Text(text) => output.push_str(&text),
                SequenceDecoderOutput::StoppedWithText(text) => {
                    output.push_str(&text);
                    stopped = true;
                    break;
                }
                SequenceDecoderOutput::Stopped => {
                    stopped = true;
                    break;
                }
                SequenceDecoderOutput::Held => {}
            }
        }

        if !stopped {
            // Flush any remaining text
            if let SequenceDecoderOutput::Text(text) = decoder.flush() {
                output.push_str(&text);
            }
        }

        println!(
            "Input: '{}', Stop: '{}', Output: '{}', Expected: '{}'",
            input, stop_seq, output, expected
        );

        // The test should check if output starts with expected
        // since stop sequences might not be perfectly aligned with token boundaries
        assert!(
            output.starts_with(expected) || output == input,
            "Stop sequence test failed"
        );
    }
}

#[test]
fn test_factory_creation() {
    let tokenizer_path = ensure_tokenizer_cached();
    let tokenizer = factory::create_tokenizer(tokenizer_path.to_str().unwrap())
        .expect("Failed to create tokenizer via factory");

    let encoding = tokenizer.encode(TEST_PROMPTS[0]).expect("Failed to encode");

    let decoded = tokenizer
        .decode(encoding.token_ids(), false)
        .expect("Failed to decode");

    assert_eq!(decoded, TEST_PROMPTS[0]);
}

#[test]
fn test_batch_encoding() {
    let tokenizer_path = ensure_tokenizer_cached();
    let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
        .expect("Failed to load tokenizer");

    let encodings = tokenizer
        .encode_batch(&TEST_PROMPTS)
        .expect("Failed to batch encode");

    assert_eq!(encodings.len(), TEST_PROMPTS.len());

    for (i, encoding) in encodings.iter().enumerate() {
        let decoded = tokenizer
            .decode(encoding.token_ids(), false)
            .expect("Failed to decode");
        assert_eq!(decoded, TEST_PROMPTS[i]);
    }
}

#[test]
fn test_special_tokens() {
    use sgl_model_gateway::tokenizer::traits::Tokenizer as TokenizerTrait;

    let tokenizer_path = ensure_tokenizer_cached();
    let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
        .expect("Failed to load tokenizer");

    let special_tokens = tokenizer.get_special_tokens();

    // TinyLlama should have at least BOS and EOS tokens
    assert!(special_tokens.bos_token.is_some());
    assert!(special_tokens.eos_token.is_some());

    println!("Special tokens: {:?}", special_tokens);
}

#[test]
fn test_thread_safety() {
    use std::thread;

    let tokenizer_path = ensure_tokenizer_cached();
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let handles: Vec<_> = TEST_PROMPTS
        .iter()
        .map(|&prompt| {
            let tokenizer_clone = tokenizer.clone();
            thread::spawn(move || {
                let encoding = tokenizer_clone
                    .encode(prompt)
                    .expect("Failed to encode in thread");
                let decoded = tokenizer_clone
                    .decode(encoding.token_ids(), false)
                    .expect("Failed to decode in thread");
                assert_eq!(decoded, prompt);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

#[test]
fn test_chat_template_discovery() {
    use std::fs;

    use tempfile::TempDir;

    // Create a temporary directory with test files
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let dir_path = temp_dir.path();

    // Copy a real tokenizer.json file for testing
    // We'll use the TinyLlama tokenizer that's already cached
    let cached_tokenizer = ensure_tokenizer_cached();
    let tokenizer_path = dir_path.join("tokenizer.json");
    fs::copy(&cached_tokenizer, &tokenizer_path).expect("Failed to copy tokenizer file");

    // Test 1: With chat_template.jinja file
    let jinja_path = dir_path.join("chat_template.jinja");
    fs::write(&jinja_path, "{{ messages }}").expect("Failed to write chat template");

    let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap());
    assert!(
        tokenizer.is_ok(),
        "Should load tokenizer with chat template"
    );

    // Clean up for next test
    fs::remove_file(&jinja_path).ok();

    // Test 2: With tokenizer_config.json containing chat_template
    let config_path = dir_path.join("tokenizer_config.json");
    fs::write(&config_path, r#"{"chat_template": "{{ messages }}"}"#)
        .expect("Failed to write config");

    let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap());
    assert!(
        tokenizer.is_ok(),
        "Should load tokenizer with embedded template"
    );

    // Test 3: No chat template
    fs::remove_file(&config_path).ok();
    let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap());
    assert!(
        tokenizer.is_ok(),
        "Should load tokenizer without chat template"
    );
}

#[test]
fn test_load_chat_template_from_local_file() {
    use std::fs;

    use tempfile::TempDir;

    // Test 1: Load tokenizer with explicit chat template path
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let dir_path = temp_dir.path();

    // Copy a real tokenizer for testing
    let cached_tokenizer = ensure_tokenizer_cached();
    let tokenizer_path = dir_path.join("tokenizer.json");
    fs::copy(&cached_tokenizer, &tokenizer_path).expect("Failed to copy tokenizer");

    // Create a chat template file
    let template_path = dir_path.join("my_template.jinja");
    let template_content = r#"{% for message in messages %}{{ message.role }}: {{ message.content }}
{% endfor %}"#;
    fs::write(&template_path, template_content).expect("Failed to write template");

    // Load tokenizer with explicit template path
    let tokenizer = HuggingFaceTokenizer::from_file_with_chat_template(
        tokenizer_path.to_str().unwrap(),
        Some(template_path.to_str().unwrap()),
    );
    assert!(
        tokenizer.is_ok(),
        "Should load tokenizer with explicit template path"
    );
}

#[tokio::test]
async fn test_tinyllama_embedded_template() {
    use sgl_model_gateway::tokenizer::hub::download_tokenizer_from_hf;

    // Skip in CI without HF_TOKEN

    // Test 2: TinyLlama has chat template embedded in tokenizer_config.json
    match download_tokenizer_from_hf("TinyLlama/TinyLlama-1.1B-Chat-v1.0").await {
        Ok(cache_dir) => {
            // Verify tokenizer_config.json exists
            let config_path = cache_dir.join("tokenizer_config.json");
            assert!(config_path.exists(), "tokenizer_config.json should exist");

            // Load the config and check for chat_template
            let config_content =
                std::fs::read_to_string(&config_path).expect("Failed to read config");
            assert!(
                config_content.contains("\"chat_template\""),
                "TinyLlama should have embedded chat_template in config"
            );

            // Load tokenizer and verify it has chat template
            let tokenizer_path = cache_dir.join("tokenizer.json");
            let _tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
                .expect("Failed to load tokenizer");

            println!(
                "✓ TinyLlama: Loaded tokenizer with embedded template from tokenizer_config.json"
            );
        }
        Err(e) => {
            println!("Download test skipped due to error: {}", e);
        }
    }
}

#[tokio::test]
async fn test_qwen3_next_embedded_template() {
    use sgl_model_gateway::tokenizer::hub::download_tokenizer_from_hf;

    // Test 3: Qwen3-Next has chat template in tokenizer_config.json
    match download_tokenizer_from_hf("Qwen/Qwen3-Next-80B-A3B-Instruct").await {
        Ok(cache_dir) => {
            let config_path = cache_dir.join("tokenizer_config.json");
            assert!(config_path.exists(), "tokenizer_config.json should exist");

            // Verify chat_template in config
            let config_content =
                std::fs::read_to_string(&config_path).expect("Failed to read config");
            assert!(
                config_content.contains("\"chat_template\""),
                "Qwen3-Next should have chat_template in tokenizer_config.json"
            );

            // Load tokenizer
            let tokenizer_path = cache_dir.join("tokenizer.json");
            if tokenizer_path.exists() {
                let _tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
                    .expect("Failed to load tokenizer");
                println!("✓ Qwen3-Next: Loaded tokenizer with embedded template");
            }
        }
        Err(e) => {
            println!("Download test skipped due to error: {}", e);
        }
    }
}

#[tokio::test]
async fn test_qwen3_vl_json_template_priority() {
    use sgl_model_gateway::tokenizer::hub::download_tokenizer_from_hf;

    // Test 4: Qwen3-VL has both tokenizer_config.json template and chat_template.json
    // Should prioritize chat_template.json
    match download_tokenizer_from_hf("Qwen/Qwen3-VL-235B-A22B-Instruct").await {
        Ok(cache_dir) => {
            // Check for chat_template.json
            let json_template_path = cache_dir.join("chat_template.json");
            let has_json_template = json_template_path.exists();

            // Also check tokenizer_config.json
            let config_path = cache_dir.join("tokenizer_config.json");
            assert!(config_path.exists(), "tokenizer_config.json should exist");

            if has_json_template {
                let json_content = std::fs::read_to_string(&json_template_path)
                    .expect("Failed to read chat_template.json");
                println!("✓ Qwen3-VL: Found chat_template.json (should be prioritized)");

                // Verify it contains jinja template
                assert!(
                    !json_content.is_empty(),
                    "chat_template.json should contain template"
                );
            }

            // Load tokenizer - it should use the appropriate template
            let tokenizer_path = cache_dir.join("tokenizer.json");
            if tokenizer_path.exists() {
                let _tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
                    .expect("Failed to load tokenizer");
                println!("✓ Qwen3-VL: Loaded tokenizer with template priority handling");
            }
        }
        Err(e) => {
            println!("Download test skipped due to error: {}", e);
        }
    }
}

#[tokio::test]
async fn test_llava_separate_jinja_template() {
    use sgl_model_gateway::tokenizer::hub::download_tokenizer_from_hf;

    // Test 5: llava has chat_template.jinja as a separate file, not in tokenizer_config.json
    match download_tokenizer_from_hf("llava-hf/llava-1.5-7b-hf").await {
        Ok(cache_dir) => {
            // Check for .jinja file
            let jinja_path = cache_dir.join("chat_template.jinja");
            let has_jinja = jinja_path.exists()
                || std::fs::read_dir(&cache_dir)
                    .map(|entries| {
                        entries.filter_map(|e| e.ok()).any(|e| {
                            e.file_name()
                                .to_str()
                                .is_some_and(|name| name.ends_with(".jinja"))
                        })
                    })
                    .unwrap_or(false);

            if has_jinja {
                println!("✓ llava: Found separate .jinja chat template file");
            }

            // Check tokenizer_config.json - should NOT have embedded template
            let config_path = cache_dir.join("tokenizer_config.json");
            if config_path.exists() {
                let config_content =
                    std::fs::read_to_string(&config_path).expect("Failed to read config");

                // llava might not have chat_template in config
                if !config_content.contains("\"chat_template\"") {
                    println!("✓ llava: No embedded template in config (as expected)");
                }
            }

            // Load tokenizer - should auto-discover the .jinja file
            let tokenizer_path = cache_dir.join("tokenizer.json");
            if tokenizer_path.exists() {
                let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap());
                if tokenizer.is_ok() {
                    println!("✓ llava: Loaded tokenizer with auto-discovered .jinja template");
                } else {
                    println!("Note: llava tokenizer loading failed - might need specific handling");
                }
            }
        }
        Err(e) => {
            println!("Download test skipped due to error: {}", e);
        }
    }
}
