#![cfg(feature = "mm")]

use sglang_processor::MmProcessor;
use sglang_processor::dynamo_multimodal::{
    models::qwen3_asr::Qwen3AsrSpec,
    processor::{PositionOutput, TensorData},
    registry::ProcessorSpec,
};

fn wav(sample_rate: u32, samples: u32) -> Vec<u8> {
    let bytes = 2 * samples;
    let mut wav = b"RIFF".to_vec();
    wav.extend((36 + bytes).to_le_bytes());
    wav.extend(b"WAVEfmt ");
    wav.extend(16u32.to_le_bytes());
    wav.extend(1u16.to_le_bytes()); // PCM
    wav.extend(1u16.to_le_bytes()); // mono
    wav.extend(sample_rate.to_le_bytes());
    wav.extend((2 * sample_rate).to_le_bytes());
    wav.extend(2u16.to_le_bytes()); // block alignment
    wav.extend(16u16.to_le_bytes());
    wav.extend(b"data");
    wav.extend(bytes.to_le_bytes());
    for index in 0..samples {
        wav.extend(((index % 251) as i16 * 16 - 2000).to_le_bytes());
    }
    wav
}

#[test]
fn wav_uses_the_shared_family_and_expands_the_prompt() {
    let processor = MmProcessor::from_spec(ProcessorSpec::Qwen3Asr(Qwen3AsrSpec {
        audio_token_id: 2,
        audio_start_id: 1,
        audio_end_id: 3,
    }))
    .unwrap();
    let prompt = [7, 1, 2, 3, 9];
    let audio = wav(16000, 16_001);
    let output = processor.process_audio(&prompt, &audio).unwrap();
    assert_eq!(output.offsets, [(2, 15)]);
    assert_eq!(
        output.input_ids,
        [&prompt[..2], &[2; 14], &prompt[3..]].concat()
    );
    let PositionOutput::MRope { positions, delta } = output.positions else {
        panic!("expected M-RoPE positions");
    };
    assert_eq!(delta, 0);
    for row in positions.chunks_exact(output.input_ids.len()) {
        assert!(row.iter().copied().eq(0..output.input_ids.len() as i64));
    }
    let [item] = &output.items[..] else {
        panic!("expected one audio item")
    };
    assert_eq!(item.feature.shape, [1, 128, 3000]);
    let [(name, mask)] = &item.aux[..] else {
        panic!("expected one audio mask")
    };
    assert_eq!(name, "feature_attention_mask");
    assert_eq!(mask.shape, [1, 3000]);
    let TensorData::I64(mask) = &mask.data else {
        panic!("expected an I64 mask")
    };
    assert!(mask[..101].iter().all(|&value| value == 1));
    assert!(mask[101..].iter().all(|&value| value == 0));

    assert!(processor.process_audio(&prompt, b"bad WAV").is_err());
    assert!(processor.process_audio(&prompt, &wav(8000, 1000)).is_err());
    assert!(processor.process_audio(&[1, 3], &audio).is_err());
}
