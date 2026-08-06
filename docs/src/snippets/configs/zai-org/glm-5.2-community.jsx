// GLM-5.2 community-contributed deployment configs.
//
// These are NOT matrix cells. They render in the "Community Configs" section at the
// bottom of the page via _community.jsx — see that file's header for the full schema.
//
// Empty for now: this file exists so the section is wired up and shows its
// "submit yours" empty state. Contributions land here rather than in glm-5.2.jsx
// `cells` whenever the hardware is something SGLang does not run in CI, so the verified
// matrix stays a team-reproduced artifact.
//
// DO NOT HAND-WRITE ENTRIES. Generate one block per contributing PR:
//
//     node docs/scripts/gen_community_entry.mjs --pr <number>
//
// The generator imports the PR's config and benchmarks files as objects at both the PR
// head and its base, structurally diffs them to find the added cells, and emits
// flags / env / modelName / dockerImage / hardware / sglangVersion verbatim. Paste the
// block it prints into the array below. The only field a human fills in is `source.org`,
// which GitHub does not know.
//
// Benchmark numbers, editorial notes, and status badges are deliberately not part of the
// schema — measurements and discussion stay in the linked PR, where the workload they
// were taken under is visible. `check_cookbook_configs.mjs` rejects those fields by name.

export const community = [];
