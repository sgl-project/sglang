export const DiffusionModelTags = ({ tags = [] }) => {
  const normalizedTags = Array.isArray(tags) ? tags : [tags];

  return (
    <div
      className="not-prose"
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '6px',
        margin: '8px 0 30px',
      }}
    >
      {normalizedTags.map((tag) => (
        <span
          key={tag}
          style={{
            display: 'inline-block',
            padding: '4px 9px',
            borderRadius: '999px',
            background: '#ecfeff',
            color: '#155e75',
            fontSize: '12px',
            fontWeight: 750,
          }}
        >
          {tag}
        </span>
      ))}
    </div>
  );
};
