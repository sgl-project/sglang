const GROUPS = {
  serve: {
    step: "02",
    eyebrow: "Serve-time overlays",
    title: "Tune the loaded pipeline",
    description: "Startup flags that change kernels, precision, or execution policy without creating another base topology.",
  },
  request: {
    step: "03",
    eyebrow: "Request-time features",
    title: "Choose per generation",
    description: "Sampling fields that can change from one request to the next while the same server stays resident.",
  },
};

const QualityBadge = ({ quality }) => (
  <span className="sgd-feature-badge" data-tone={quality.tone}>
    {quality.label}
  </span>
);

const Feature = ({ feature, copied, onCopy }) => (
  <details className="sgd-feature-item">
    <summary>
      <span className="sgd-feature-marker" aria-hidden="true">
        {feature.marker}
      </span>
      <span className="sgd-feature-summary-copy">
        <strong>{feature.title}</strong>
        <span>{feature.summary}</span>
      </span>
      <span className="sgd-feature-summary-meta">
        <QualityBadge quality={feature.quality} />
        <span className="sgd-feature-scope">{feature.scope}</span>
      </span>
      <span className="sgd-feature-chevron" aria-hidden="true">↘</span>
    </summary>

    <div className="sgd-feature-body">
      <dl className="sgd-feature-contract">
        <div>
          <dt>Default</dt>
          <dd>{feature.defaultValue}</dd>
        </div>
        <div>
          <dt>Quality contract</dt>
          <dd>{feature.contract}</dd>
        </div>
        <div>
          <dt>Compatibility</dt>
          <dd>{feature.compatibility}</dd>
        </div>
      </dl>

      <div className="sgd-feature-recipes">
        {feature.recipes.map((recipe) => {
          const copyId = `${feature.id}:${recipe.label}`;
          return (
            <div className="sgd-feature-recipe" key={copyId}>
              <div className="sgd-feature-recipe-heading">
                <strong>{recipe.label}</strong>
                {recipe.quality && <QualityBadge quality={recipe.quality} />}
              </div>
              {recipe.description && <p>{recipe.description}</p>}
              {recipe.code && (
                <div className="sgd-feature-code">
                  <pre><code>{recipe.code}</code></pre>
                  <button
                    type="button"
                    onClick={() => onCopy(copyId, recipe.code)}
                    aria-label={`Copy ${recipe.label} overlay`}
                  >
                    {copied === copyId ? "Copied" : "Copy overlay"}
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  </details>
);

const FeatureGroup = ({ type, features, copied, onCopy }) => {
  const group = GROUPS[type];
  return (
    <section className="sgd-feature-group" data-group={type}>
      <header>
        <span>{group.eyebrow}</span>
        <strong>{group.title}</strong>
        <p>{group.description}</p>
      </header>
      <div className="sgd-feature-list">
        {features.map((feature) => (
          <Feature
            key={feature.id}
            feature={feature}
            copied={copied}
            onCopy={onCopy}
          />
        ))}
      </div>
    </section>
  );
};

export const DiffusionFeatureGuide = ({ serveFeatures = [], requestFeatures = [] }) => {
  const [copied, setCopied] = useState("");

  const copy = (id, code) => {
    navigator.clipboard.writeText(code);
    setCopied(id);
    window.setTimeout(() => setCopied(""), 1400);
  };

  return (
    <section className="not-prose sgd-feature-guide" aria-label="Feature configuration guide">
      <div className="sgd-feature-flow" aria-label="Configuration order">
        <div className="sgd-feature-step" data-step="base">
          <span>01</span>
          <strong>Base recipe</strong>
          <small>topology · weights · mode</small>
        </div>
        <span className="sgd-feature-connector" aria-hidden="true">→</span>
        {Object.entries(GROUPS).map(([type, group], index) => (
          <div key={type} className="sgd-feature-flow-fragment">
            <div className="sgd-feature-step" data-step={type}>
              <span>{group.step}</span>
              <strong>{group.eyebrow}</strong>
              <small>{type === "serve" ? "startup flags" : "sampling fields"}</small>
            </div>
            {index === 0 && <span className="sgd-feature-connector" aria-hidden="true">→</span>}
          </div>
        ))}
      </div>

      <div className="sgd-feature-groups">
        <FeatureGroup type="serve" features={serveFeatures} copied={copied} onCopy={copy} />
        <FeatureGroup type="request" features={requestFeatures} copied={copied} onCopy={copy} />
      </div>
    </section>
  );
};
