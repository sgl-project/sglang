export const useUrlStatePersistence = (
  values,
  setValues,
  { prefix = '', includeKeys = null } = {}
) => {
  const scopedKey = (key) => (prefix ? `${prefix}-${key}` : key);
  const allowedKeys = includeKeys ? new Set(includeKeys) : null;
  const shouldPersistKey = (key) => !allowedKeys || allowedKeys.has(key);
  const [isHydrated, setIsHydrated] = useState(false);

  // READ from URL — runs once after mount
  useEffect(() => {
    if (typeof window === 'undefined' || isHydrated) return;

    const params = new URLSearchParams(window.location.search); // fixed
    const parsedValues = {};
    let hasChanges = false;

    Object.keys(values).forEach((key) => {
      if (!shouldPersistKey(key)) return;
      const queryKey = scopedKey(key);
      if (!params.has(queryKey)) return;
      const rawValue = params.get(queryKey) || '';
      parsedValues[key] = Array.isArray(values[key])
        ? rawValue ? rawValue.split(',').filter(Boolean) : []
        : rawValue;
      hasChanges = true;
    });

    if (hasChanges) setValues((prev) => ({ ...prev, ...parsedValues }));
    setIsHydrated(true);
  }, [isHydrated]); // fixed — no `values` here, avoids re-run loop

  // WRITE to URL — runs after hydration whenever values change
  useEffect(() => {
    if (typeof window === 'undefined' || !isHydrated) return;

    const params = new URLSearchParams(window.location.search); // fixed

    Object.entries(values).forEach(([key, value]) => {
      if (!shouldPersistKey(key)) return;
      const queryKey = scopedKey(key);
      if (Array.isArray(value)) {
        value.length > 0 ? params.set(queryKey, value.join(',')) : params.delete(queryKey);
        return;
      }
      const normalized = value == null ? '' : String(value);
      normalized.length > 0 ? params.set(queryKey, normalized) : params.delete(queryKey);
    });

    const nextSearch = params.toString();
    const nextUrl = `${window.location.pathname}${nextSearch ? `?${nextSearch}` : ''}${window.location.hash}`;
    window.history.replaceState(window.history.state, '', nextUrl);
  }, [isHydrated, values]);
};