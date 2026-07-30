export const DiffusionModelTags = ({ tags = [], pullUp = false }) => {
  const normalizedTags = Array.isArray(tags) ? tags : [tags];
  const className = `not-prose sgd-model-tags${pullUp ? ' sgd-model-tags--pull-up' : ''}`;

  return (
    <div className={className}>
      {normalizedTags.map((tag) => (
        <span key={tag} className="sgd-chip">
          {tag}
        </span>
      ))}
    </div>
  );
};
