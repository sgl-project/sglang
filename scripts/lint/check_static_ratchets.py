#!/usr/bin/env python3

from check_decode_bookkeeping_ownership import (
    check_bookkeeping_sites_match_owner_allowlist,
    check_spec_v2_draft_workers_do_no_scheduler_bookkeeping,
)
from check_global_config_read_ratchet import (
    check_configured_size_call_sites,
    check_global_config_read_ratchet,
    check_no_renamed_accessor_imports,
)
from check_legacy_global_ratchet import check_legacy_global_ratchet
from check_module_state_ratchet import check_module_state_ratchet
from check_parallel_adoption_ratchet import check_parallel_adoption_ratchet
from check_server_args_mutation_ratchet import check_server_args_mutation_ratchet


def main():
    checks = (
        check_bookkeeping_sites_match_owner_allowlist,
        check_spec_v2_draft_workers_do_no_scheduler_bookkeeping,
        check_global_config_read_ratchet,
        check_configured_size_call_sites,
        check_no_renamed_accessor_imports,
        check_legacy_global_ratchet,
        check_module_state_ratchet,
        check_parallel_adoption_ratchet,
        check_server_args_mutation_ratchet,
    )
    for check in checks:
        check()


if __name__ == "__main__":
    main()
