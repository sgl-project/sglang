use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array4;
use sgl_model_gateway::multimodal::vision::image_processor::PreprocessedImages;

fn bench_vision_flat_access(c: &mut Criterion) {
    // 1. Setup: 4 images, 3 channels, 336x336 resolution (Typical LLaVA Batch)
    let shape = (4, 3, 336, 336);
    let pixel_values = Array4::<f32>::zeros(shape);
    let images = PreprocessedImages::new(pixel_values, vec![576; 4], vec![(336, 336); 4]);

    // 2. The Benchmark Loop
    c.bench_function("vision_pixel_values_flat", |b| {
        b.iter(|| {
            // Access the flat view
            let data = images.pixel_values_flat();

            // black_box ensures the compiler doesn't skip the work
            black_box(data);
        })
    });
}

criterion_group!(benches, bench_vision_flat_access);
criterion_main!(benches);
