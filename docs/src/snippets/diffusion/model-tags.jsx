export const DiffusionModelTags = ({ tags = [] }) => {
  const normalizedTags = Array.isArray(tags) ? tags : [tags];

  return (
    <div className="not-prose sgd-model-tags">
      {normalizedTags.map((tag) => (
        <span key={tag} className="sgd-chip">
          {tag}
        </span>
      ))}
    </div>
  );
};
