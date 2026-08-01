# Stack review vehicle — DO NOT MERGE

This PR aggregates the full diff of a stacked series so CI can run once over
the whole stack and reviewers can see the combined change. Merge the member
PRs in stack order instead; if this PR is ever used as a squash vehicle,
drop the commit that adds this file first.

Members (merge in order):

1. config: route parallel config-leaf reads through get_parallel()
2. test: recover the config-namespace-migration deferrals
3. runtime_context: per-role namespace enforcement behind SGLANG_ROLE_NAMESPACES
4. docs: rewrite the runtime-context skill for the namespace-bag config model
