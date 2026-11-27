GDN_BEAM_REPLAY_WINDOW = 3


def records_per_state_slot(value_dim: int, key_dim: int, element_size: int) -> int:
    return (value_dim * key_dim * element_size) // (4 * (value_dim + key_dim + 1))
