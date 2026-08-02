# Stack review vehicle — DO NOT MERGE

This PR aggregates the full diff of a stacked series so CI can run once over
the whole stack and reviewers can see the combined change. Merge the member
PRs in stack order instead; if this PR is ever used as a squash vehicle,
drop the commit that adds this file first.

Members (merge in order):

1. config: retire three ServerArgs.override sites that no reader needed
2. spec: give the v2 draft workers their own ServerArgs copy
3. config: keep runtime hicache and weight-version updates off ServerArgs
4. observability: publish the generated forward-pass-metrics endpoint to the bags
5. config: retire the last process-global config field reads
