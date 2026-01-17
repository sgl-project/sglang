interface KPICardProps {
  label: string;
  value: string;
  hint?: string;
  progress?: number;
}

export function KPICard({ label, value, hint, progress }: KPICardProps) {
  return (
    <div className="kpi">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      {hint && <div className="kpi-hint">{hint}</div>}
      {progress !== undefined && (
        <div className="kpi-bar">
          <span style={{ width: `${progress * 100}%` }} />
        </div>
      )}
    </div>
  );
}
