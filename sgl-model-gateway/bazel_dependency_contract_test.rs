#[test]
fn generated_aliases_keep_tonic_versions_distinct() {
    let current = tonic::Status::ok("tonic 0.14");
    let legacy = tonic_v12::Status::ok("tonic 0.12");

    assert_eq!(current.code(), tonic::Code::Ok);
    assert_eq!(legacy.code(), tonic_v12::Code::Ok);
}
