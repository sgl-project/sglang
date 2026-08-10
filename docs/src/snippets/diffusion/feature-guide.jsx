export const DiffusionFeatureGuide = ({ serveFeatures = [], requestFeatures = [] }) => {
  const groups = {
    serve: {
      eyebrow: "Serve-time · restart required",
      title: "Server options",
      description: "Add a startup flag, then restart the server.",
    },
    request: {
      eyebrow: "Request-time · no restart",
      title: "Generation options",
      description: "Add a field to one request without reloading the model.",
    },
  };

  const qualityBadge = (quality) => (
    <span className="sgd-feature-badge" data-tone={quality.tone}>
      {quality.label}
    </span>
  );

  const copy = async (event, code) => {
    const button = event.currentTarget;
    await navigator.clipboard.writeText(code);
    button.textContent = "Copied";
    window.setTimeout(() => {
      button.textContent = "Copy setting";
    }, 1400);
  };

  const featureView = (feature) => (
    <details className="sgd-feature-item" key={feature.id}>
      <summary>
        <span className="sgd-feature-summary-copy">
          <strong>{feature.title}</strong>
          <span>{feature.summary}</span>
        </span>
        <span className="sgd-feature-summary-meta">
          {qualityBadge(feature.quality)}
          <span className="sgd-feature-scope">{feature.scope}</span>
        </span>
        <span className="sgd-feature-action" aria-hidden="true">
          Details <span>↘</span>
        </span>
      </summary>

      <div className="sgd-feature-body">
        <dl className="sgd-feature-contract">
          <div>
            <dt>Default setting</dt>
            <dd>{feature.defaultValue}</dd>
          </div>
          <div>
            <dt>Quality</dt>
            <dd>{feature.contract}</dd>
          </div>
          <div>
            <dt>Works with</dt>
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
                  {recipe.quality && qualityBadge(recipe.quality)}
                </div>
                {recipe.description && <p>{recipe.description}</p>}
                {recipe.code && (
                  <div className="sgd-feature-code">
                    <pre><code>{recipe.code}</code></pre>
                    <button
                      type="button"
                      onClick={(event) => copy(event, recipe.code)}
                      aria-label={`Copy ${recipe.label} setting`}
                    >
                      Copy setting
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

  const featureGroup = (type, features) => {
    const group = groups[type];
    return (
      <section className="sgd-feature-group" data-group={type}>
        <header>
          <span>{group.eyebrow}</span>
          <strong>{group.title}</strong>
          <p>{group.description}</p>
        </header>
        <div className="sgd-feature-list">
          {features.map(featureView)}
        </div>
      </section>
    );
  };

  return (
    <section className="not-prose sgd-feature-guide" aria-label="Feature configuration guide">
      <div className="sgd-feature-groups">
        {featureGroup("serve", serveFeatures)}
        {featureGroup("request", requestFeatures)}
      </div>
    </section>
  );
};
