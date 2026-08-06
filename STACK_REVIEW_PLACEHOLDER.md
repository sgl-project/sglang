# Stack review / CI vehicle — not for merge

This PR carries the full `cheng/gc-final-closeout` series so it can be
reviewed and CI-tested as one diff. It is **not** meant to be merged: the
member PRs merge one by one (rebasing each on the previous merge), and this
PR is closed at the end.

Member PRs (one commit each, based on the previous member for a clean
per-commit diff):

1. `config: retire ServerArgs.derive; per-runner values are constructor arguments`
2. `config: delete the dead get_server_args() bindings across the repo`
3. `moe: the shared-experts-fusion decision is a per-runner moe flag`

This file exists only to give the vehicle a diff distinct from member 3 and
is deleted when the vehicle closes.
