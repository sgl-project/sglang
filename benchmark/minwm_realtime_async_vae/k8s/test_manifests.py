from pathlib import Path

from validate_manifests import find, load_documents, requirement_values, validate


def test_gpu_nodepools_are_spot_only_and_bounded():
    documents = load_documents()
    validate(documents)


def test_l40s_alternate_is_spot_only_and_never_in_base_topology():
    documents = load_documents(("l40s-vae.yaml",))
    nodepool = find(documents, "NodePool", "minwm-async-vae-l40s")
    assert requirement_values(nodepool, "karpenter.sh/capacity-type") == ["spot"]
    assert all(
        value.startswith("g6e.")
        for value in requirement_values(nodepool, "node.kubernetes.io/instance-type")
    )
    kustomization = (Path(__file__).parent / "kustomization.yaml").read_text()
    assert "l40s-vae.yaml" not in kustomization


def test_vae_deployment_can_land_on_either_l4_or_l40s_pool():
    base_documents = load_documents()
    l40s_documents = load_documents(("l40s-vae.yaml",))
    deployment = find(base_documents, "Deployment", "minwm-async-vae")
    selector = deployment["spec"]["template"]["spec"]["nodeSelector"]
    assert selector["seedleap.ai/vae-worker"] == "true"
    assert "karpenter.sh/nodepool" not in selector

    for documents, name in (
        (base_documents, "minwm-async-vae-l4"),
        (l40s_documents, "minwm-async-vae-l40s"),
    ):
        nodepool = find(documents, "NodePool", name)
        labels = nodepool["spec"]["template"]["metadata"]["labels"]
        assert labels["seedleap.ai/vae-worker"] == "true"


def test_wan22_5b_uses_the_matching_taehv_checkpoint():
    for filename in ("h100-denoiser.yaml", "l4-vae.yaml"):
        manifest = (Path(__file__).parent / filename).read_text()
        assert "taew2_2.pth" in manifest
        assert "taew2_1.pth" not in manifest


def test_webui_enables_i2v_and_t2v_in_production_manifest():
    manifest = (Path(__file__).parent / "h100-denoiser.yaml").read_text()
    assert '"generationModes":["i2v","t2v"]' in manifest
    assert '"t2vDefaultNumFrames":121' in manifest
