# SPDX-License-Identifier: Apache-2.0


def is_module_list_entry(name: str, container_name: str) -> bool:
    # Match only direct block entries, not their inner submodules.
    parts = name.split(".")
    return len(parts) >= 2 and parts[-2] == container_name and parts[-1].isdigit()


def is_module_list_entry_in(name: str, container_names: tuple[str, ...]) -> bool:
    parts = name.split(".")
    return len(parts) >= 2 and parts[-2] in container_names and parts[-1].isdigit()


def is_layer(name: str, module: object) -> bool:
    return is_module_list_entry(name, "layers")


def is_block(name: str, module: object) -> bool:
    return is_module_list_entry(name, "blocks")


def is_t5_block(name: str, module: object) -> bool:
    return is_module_list_entry(name, "block")


def is_transformer_block(name: str, module: object) -> bool:
    return is_module_list_entry(name, "transformer_blocks")


def is_double_block(name: str, module: object) -> bool:
    return is_module_list_entry(name, "double_blocks")


def is_single_block(name: str, module: object) -> bool:
    return is_module_list_entry(name, "single_blocks")


def is_refiner_block(name: str, module: object) -> bool:
    return is_module_list_entry(name, "refiner_blocks")


def is_blocks_or_double_blocks(name: str, module: object) -> bool:
    return is_module_list_entry_in(name, ("blocks", "double_blocks"))


def is_blocks_or_transformer_blocks(name: str, module: object) -> bool:
    return is_module_list_entry_in(name, ("blocks", "transformer_blocks"))


def is_zimage_layer(name: str, module: object) -> bool:
    last_part = name.split(".")[-1]
    # Preserve Z-Image's finer historical FSDP granularity for perf.
    return last_part.isdigit() and (
        "layers" in name or "noise_refiner" in name or "context_refiner" in name
    )


def is_embed_tokens(name: str, module: object) -> bool:
    return name.endswith("embed_tokens")


def is_embeddings(name: str, module: object) -> bool:
    return name.endswith("embeddings")


def is_final_norm(name: str, module: object) -> bool:
    return name.endswith("norm")


def is_shared(name: str, module: object) -> bool:
    return name.endswith("shared")


def is_final_layer_norm(name: str, module: object) -> bool:
    return name.endswith("final_layer_norm")


def is_txt_in(name: str, module: object) -> bool:
    return name.split(".")[-1] == "txt_in"
