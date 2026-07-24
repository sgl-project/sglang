export const ReleaseLookup = () => {
  const INDEX_URL = '/release_lookup/release_index.json';
  const SHORT_HASH_LEN = 8;
  const REPO_URL = 'https://github.com/sgl-project/sglang';

  const [index, setIndex] = useState(null);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [ready, setReady] = useState(false);
  const [status, setStatus] = useState({ text: 'Initializing…', error: false });
  const [result, setResult] = useState(null);

  // --- Index parsing (ported verbatim from the Sphinx tool) ---
  function formatDate(iso) {
    if (!iso) return 'Unknown';
    try {
      return new Date(iso).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
    } catch (e) {
      return iso;
    }
  }

  function getTagInfo(ref) {
    const tag = index.tags[ref];
    return { name: tag[0], date: tag[1], type: tag[2] === 1 ? 'gateway' : 'main' };
  }

  function parseTagRef(ref) {
    if (typeof ref === 'string' && /^[mg]\d+$/.test(ref))
      return { type: ref[0], idx: parseInt(ref.slice(1)) };
    return null;
  }

  function prefixSearch(prefix) {
    const keys = index.sortedCommitKeys;
    if (!keys) return null;
    let lo = 0, hi = keys.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (keys[mid] < prefix) lo = mid + 1; else hi = mid;
    }
    if (lo < keys.length && keys[lo].indexOf(prefix) === 0) return keys[lo];
    return null;
  }

  useEffect(() => {
    let cancelled = false;
    setStatus({ text: 'Downloading index…', error: false });
    fetch(INDEX_URL)
      .then((r) => {
        if (!r.ok) throw new Error('Index not found. It is generated on each release.');
        return r.json();
      })
      .then((data) => {
        if (cancelled) return;
        setIndex({
          tags: data.t,
          prs: data.p,
          commits: data.c,
          sortedCommitKeys: Object.keys(data.c).sort(),
        });
        const tagCount = data.t.length;
        const prCount = Object.keys(data.p).length;
        setStatus({ text: 'Ready. Indexed ' + tagCount + ' releases and ' + prCount + ' PRs.', error: false });
        setReady(true);
      })
      .catch((e) => {
        if (cancelled) return;
        setStatus({ text: 'Error: ' + e.message, error: true });
        setReady(false);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  function search() {
    if (!index) return;
    const raw = query.trim();
    if (!raw) return;

    let queryType = 'unknown', key = raw;
    const urlMatch = raw.match(/\/pull\/(\d+)/);
    if (urlMatch) { key = urlMatch[1]; queryType = 'pr'; }
    else if (/^#?\d+$/.test(raw)) { key = raw.replace('#', ''); queryType = 'pr'; }
    else if (/^[0-9a-fA-F]{7,40}$/.test(raw)) { key = raw.toLowerCase(); queryType = 'commit'; }

    let tagData = null;
    if (queryType === 'pr') {
      tagData = index.prs[key];
    } else if (queryType === 'commit') {
      const sk = key.slice(0, SHORT_HASH_LEN);
      tagData = index.commits[sk];
      if (!tagData) { const mk = prefixSearch(sk); if (mk) tagData = index.commits[mk]; }
    }

    const tagRefs = [];
    if (tagData) {
      if (typeof tagData === 'string') {
        const p = parseTagRef(tagData);
        if (p) tagRefs.push(p.idx);
      } else if (typeof tagData === 'object') {
        if ('m' in tagData) tagRefs.push(tagData.m);
        if ('g' in tagData) tagRefs.push(tagData.g);
      }
    }

    if (tagRefs.length === 0) {
      const label = queryType === 'pr' ? 'PR #' + key : 'Commit ' + key.substring(0, 7);
      setResult({ found: false, queryType, label });
      return;
    }

    const releases = tagRefs.map((ref) => {
      const info = getTagInfo(ref);
      return {
        name: info.name,
        date: formatDate(info.date),
        type: info.type,
        url: REPO_URL + '/releases/tag/' + encodeURIComponent(info.name),
      };
    });
    setResult({ found: true, releases });
  }

  function onKeyPress(e) {
    if (e.key === 'Enter') search();
  }

  return (
    <div className="release-lookup-container">
      <style dangerouslySetInnerHTML={{ __html: `
        .release-lookup-container {
          background-color: #ffffff;
          padding: 2rem;
          border-radius: 12px;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
          max-width: 600px;
          margin: 1.5rem 0;
        }
        .release-lookup-container .rl-input-group { display: flex; gap: 10px; margin-bottom: 1.2rem; }
        .release-lookup-container input[type="text"] {
          flex: 1; padding: 10px 14px; border: 2px solid #e2e8f0; border-radius: 8px;
          font-size: 0.95rem; outline: none; transition: border-color 0.2s; color: #1e293b;
        }
        .release-lookup-container input[type="text"]:focus {
          border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }
        .release-lookup-container input[type="text"]::placeholder { color: #94a3b8; }
        .release-lookup-container .rl-btn {
          padding: 10px 20px; background-color: #3b82f6; color: white; border: none;
          border-radius: 8px; font-size: 0.95rem; font-weight: 600; cursor: pointer;
          transition: background-color 0.2s;
        }
        .release-lookup-container .rl-btn:hover { background-color: #2563eb; }
        .release-lookup-container .rl-btn:disabled { background-color: #cbd5e1; cursor: not-allowed; }
        .release-lookup-container .rl-result-content { padding: 1rem; border-radius: 8px; margin-bottom: 0.75rem; }
        .release-lookup-container .rl-success { background-color: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; }
        .release-lookup-container .rl-error { background-color: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }
        .release-lookup-container .rl-row { display: flex; justify-content: space-between; margin-bottom: 0.4rem; align-items: baseline; }
        .release-lookup-container .rl-row:last-child { margin-bottom: 0; }
        .release-lookup-container .rl-label { font-weight: 600; margin-right: 1rem; min-width: 70px; }
        .release-lookup-container .rl-tag-link { color: #3b82f6; text-decoration: none; font-weight: bold; font-size: 1.05rem; }
        .release-lookup-container .rl-tag-link:hover { text-decoration: underline; }
        .release-lookup-container .rl-badge {
          display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;
          font-weight: 600; text-transform: uppercase;
        }
        .release-lookup-container .rl-badge-main { background-color: #dbeafe; color: #1e40af; }
        .release-lookup-container .rl-badge-gateway { background-color: #f3e8ff; color: #6b21a8; }
        .release-lookup-container .rl-status { margin-top: 0.8rem; font-size: 0.85rem; color: #64748b; min-height: 18px; }
        .release-lookup-container .rl-loader {
          display: inline-block; width: 16px; height: 16px; border: 3px solid rgba(59, 130, 246, 0.2);
          border-radius: 50%; border-top-color: #3b82f6; animation: rl-spin 1s linear infinite;
          margin-right: 6px; vertical-align: text-bottom;
        }
        @keyframes rl-spin { to { transform: rotate(360deg); } }
      ` }} />

      <div className="rl-input-group">
        <input
          type="text"
          value={query}
          placeholder="PR # (e.g. 1425), PR URL, or commit hash"
          autoComplete="off"
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={onKeyPress}
        />
        <button className="rl-btn" disabled={!ready} onClick={search}>Search</button>
      </div>

      {loading && (
        <div style={{ color: '#64748b', marginBottom: '0.8rem' }}>
          <span className="rl-loader"></span> Loading index…
        </div>
      )}

      {result && !result.found && (
        <div className="rl-result-content rl-error">
          <div className="rl-row"><span className="rl-label">Status</span><span>Not Found</span></div>
          <div style={{ marginTop: '6px' }}>
            The {result.queryType} <strong>{result.label}</strong> has not been included in any release yet, or is not in the index.
          </div>
        </div>
      )}

      {result && result.found && result.releases.map((rel, i) => (
        <div className="rl-result-content rl-success" key={i}>
          <div className="rl-row">
            <span className="rl-label">Release</span>
            <a href={rel.url} target="_blank" rel="noreferrer" className="rl-tag-link">{rel.name}</a>
          </div>
          <div className="rl-row"><span className="rl-label">Date</span><span>{rel.date}</span></div>
          <div className="rl-row">
            <span className="rl-label">Module</span>
            <span className={'rl-badge ' + (rel.type === 'gateway' ? 'rl-badge-gateway' : 'rl-badge-main')}>{rel.type}</span>
          </div>
        </div>
      ))}

      <div className={'rl-status' + (status.error ? ' rl-status-error' : '')} style={status.error ? { color: '#991b1b' } : undefined}>
        {status.text}
      </div>
    </div>
  );
};
