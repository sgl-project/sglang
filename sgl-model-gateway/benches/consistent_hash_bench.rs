use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use smg::mesh::consistent_hash::ConsistentHashRing;

fn setup_ring(node_count: usize) -> ConsistentHashRing {
    let mut ring = ConsistentHashRing::new();
    for i in 0..node_count {
        ring.add_node(&format!("node-{}", i));
    }
    ring
}

fn bench_consistent_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("ConsistentHashRing");

    for size in [10, 100, 500].iter() {
        let ring = setup_ring(*size);
        let key = "test-request-key-for-rate-limiting";
        let node_name = "node-5";

        group.bench_with_input(BenchmarkId::new("get_owners", size), size, |b, _| {
            b.iter(|| {
                black_box(ring.get_owners(key));
            });
        });
        group.bench_with_input(BenchmarkId::new("is_owner", size), size, |b, _| {
            b.iter(|| {
                black_box(ring.is_owner(key, node_name));
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_consistent_hash);
criterion_main!(benches);
