import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("../../src/snippets/_muse_spark_swa_ratio.jsx", import.meta.url), "utf8");
// Mintlify imports the named export without preceding module-level declarations.
const exportSource = source.slice(source.indexOf("export const museSparkSwaRatio"));
const { museSparkSwaRatio } = await import("data:text/javascript," + encodeURIComponent(exportSource));

test("short contexts need equal SWA and full-attention token capacities", () => {
  for (const length of [1, 1024, 2048]) {
    assert.equal(museSparkSwaRatio.calculate(length), 1);
  }
});

test("long contexts balance request capacity, without a layer-count multiplier", () => {
  assert.equal(museSparkSwaRatio.calculate("100000"), 0.02048);
  assert.equal(museSparkSwaRatio.calculate(131072), 0.015625);
  assert.equal(museSparkSwaRatio.calculate(1048576), 0.001953125);
  for (const concurrency of [1, 8, 64]) {
    const fullCapacity = concurrency * 100000;
    const swaCapacity = fullCapacity * museSparkSwaRatio.calculate(100000);
    assert.equal(swaCapacity / 2048, concurrency);
  }
});

test("the window boundary is continuous and the ratio decreases beyond it", () => {
  assert.equal(museSparkSwaRatio.calculate(2049), 2048 / 2049);
  assert.ok(museSparkSwaRatio.calculate(100001) < museSparkSwaRatio.calculate(100000));
});

test("zero and invalid inputs never produce a launch value", () => {
  for (const input of [0, "0", "", " ", -1, 1.5, 1048577, "abc", NaN, Infinity, null, undefined]) {
    assert.equal(museSparkSwaRatio.calculate(input), null, `input: ${String(input)}`);
  }
});
