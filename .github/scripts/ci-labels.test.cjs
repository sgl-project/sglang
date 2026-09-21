const assert = require("node:assert/strict");
const { test } = require("node:test");

const { resolveCiLabels } = require("./ci-labels.cjs");

// `snapshot` is what the event payload froze; `live` is what the API returns.
// They differ whenever a label was added or removed after the run started.
function stub({ snapshot = null, live = [], sha = "head" } = {}) {
  const calls = { pullsGet: 0, byCommit: 0 };
  const wrap = (names) => names.map((name) => ({ name }));
  const github = {
    rest: {
      pulls: {
        get: async (args) => {
          calls.pullsGet++;
          assert.equal(args.pull_number, 42);
          return { data: { labels: wrap(live) } };
        },
      },
      repos: {
        listPullRequestsAssociatedWithCommit: async (args) => {
          calls.byCommit++;
          assert.equal(args.commit_sha, sha);
          return { data: live === null ? [] : [{ labels: wrap(live) }] };
        },
      },
    },
  };
  const context = {
    repo: { owner: "owner", repo: "repo" },
    sha,
    payload: snapshot
      ? { pull_request: { number: 42, head: { sha }, labels: wrap(snapshot) } }
      : {},
  };
  return { github, context, calls };
}

async function axes(opts) {
  const { github, context, calls } = stub(opts);
  const { labels, ...resolved } = await resolveCiLabels(github, context);
  return { resolved, calls };
}

test("no labels leaves every axis off", async () => {
  const { resolved } = await axes({ snapshot: [] });
  assert.deepEqual(resolved, {
    bypassFailFast: false,
    parallelStages: false,
    maxConcurrency: false,
    highestPriority: false,
  });
});

test("each label switches on its own axis only", async () => {
  for (const [label, axis] of [
    ["bypass-fail-fast", "bypassFailFast"],
    ["parallel-stages", "parallelStages"],
    ["max-concurrency", "maxConcurrency"],
  ]) {
    const { resolved } = await axes({ snapshot: [], live: [label] });
    const on = Object.entries(resolved)
      .filter(([, v]) => v)
      .map(([k]) => k);
    assert.deepEqual(on, [axis], `${label} should switch on ${axis} alone`);
  }
});

test("highest-priority is the union of the three", async () => {
  const { resolved } = await axes({ snapshot: [], live: ["highest-priority"] });
  assert.deepEqual(resolved, {
    bypassFailFast: true,
    parallelStages: true,
    maxConcurrency: true,
    highestPriority: true,
  });
});

test("a label added after the event is picked up on rerun", async () => {
  const { resolved, calls } = await axes({
    snapshot: [],
    live: ["parallel-stages"],
  });
  assert.equal(resolved.parallelStages, true);
  assert.equal(calls.pullsGet, 1);
});

test("a label removed after the event stops applying", async () => {
  const { resolved } = await axes({ snapshot: ["max-concurrency"], live: [] });
  assert.equal(resolved.maxConcurrency, false);
});

test("a non-PR event resolves through the commit's associated PR", async () => {
  const { resolved, calls } = await axes({
    snapshot: null,
    live: ["highest-priority"],
  });
  assert.equal(resolved.highestPriority, true);
  assert.equal(calls.pullsGet, 0);
  assert.equal(calls.byCommit, 1);
});

test("a commit with no associated PR leaves every axis off", async () => {
  const { github, context } = stub({ snapshot: null });
  github.rest.repos.listPullRequestsAssociatedWithCommit = async () => ({
    data: [],
  });
  const { labels, ...resolved } = await resolveCiLabels(github, context);
  assert.deepEqual(labels, []);
  assert.equal(Object.values(resolved).some(Boolean), false);
});

test("a failed lookup leaves every axis off rather than throwing", async () => {
  const github = {
    rest: {
      pulls: { get: async () => { throw new Error("502"); } },
      repos: { listPullRequestsAssociatedWithCommit: async () => { throw new Error("502"); } },
    },
  };
  const context = {
    repo: { owner: "owner", repo: "repo" },
    sha: "head",
    payload: { pull_request: { number: 42, head: { sha: "head" }, labels: [] } },
  };
  const warn = console.warn;
  console.warn = () => {};
  try {
    const { labels, ...resolved } = await resolveCiLabels(github, context);
    assert.deepEqual(labels, []);
    assert.equal(Object.values(resolved).some(Boolean), false);
  } finally {
    console.warn = warn;
  }
});

test("legacy label names no longer switch anything on", async () => {
  const { resolved } = await axes({
    snapshot: [],
    live: ["bypass-fastfail", "high priority", "high-priority"],
  });
  assert.equal(Object.values(resolved).some(Boolean), false);
});
