interface TopKItem {
  position: number;
  score: number;
}

interface TopKListProps {
  items: TopKItem[];
}

export function TopKList({ items }: TopKListProps) {
  return (
    <div className="topk-list">
      {items.map((item, rank) => (
        <div key={item.position} className="topk-row">
          <div className="topk-left">
            <div>
              <strong>#{rank + 1}</strong> pos {item.position}
            </div>
            <div className="topk-hint">score {(item.score * 100).toFixed(1)}%</div>
          </div>
          <div className="topk-right">{(item.score * 100).toFixed(1)}%</div>
        </div>
      ))}
    </div>
  );
}
