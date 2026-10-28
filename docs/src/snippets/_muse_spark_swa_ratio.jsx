// Mintlify evaluates the named export in isolation, so keep its constants inside it.
export const museSparkSwaRatio = (() => {
  const windowSize = 2048;
  const maxContextLength = 1048576;

  return {
    windowSize,
    maxContextLength,
    calculate: (input) => {
      if (typeof input !== "number" && typeof input !== "string") return null;
      if (typeof input === "string" && input.trim() === "") return null;
      const length = Number(input);
      if (!Number.isInteger(length) || length <= 0 || length > maxContextLength) return null;
      // The flag compares token capacities, independently of each pool's layer count.
      return Math.min(1, windowSize / length);
    },
  };
})();
