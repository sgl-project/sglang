import { ManifoldZone } from '../../api/types';

interface ZoneBadgeComparisonProps {
  leftZone: ManifoldZone;
  rightZone: ManifoldZone;
  leftConfidence: number;
  rightConfidence: number;
  zoneChanged: boolean;
}

function getZoneColor(zone: ManifoldZone): string {
  switch (zone) {
    case 'syntax_floor':
      return '#55d6a6';
    case 'semantic_bridge':
      return '#7aa2ff';
    case 'long_range':
      return '#c678dd';
    case 'structure_ripple':
      return '#ffcc66';
    case 'diffuse':
      return '#ff7b72';
    default:
      return '#888';
  }
}

function getZoneDescription(zone: ManifoldZone): string {
  switch (zone) {
    case 'syntax_floor':
      return 'Focus on immediate syntax/grammar';
    case 'semantic_bridge':
      return 'Paragraph-level semantic retrieval';
    case 'long_range':
      return 'Document-level dependencies';
    case 'structure_ripple':
      return 'Periodic/structural patterns';
    case 'diffuse':
      return 'Exploratory/uncertain attention';
    default:
      return 'Unknown zone';
  }
}

function formatZoneName(zone: ManifoldZone): string {
  return zone.replace('_', ' ');
}

export function ZoneBadgeComparison({
  leftZone,
  rightZone,
  leftConfidence,
  rightConfidence,
  zoneChanged,
}: ZoneBadgeComparisonProps) {
  return (
    <div className="zone-comparison">
      <div className="zone-side left">
        <span className="zone-session-label">Session A</span>
        <div
          className="zone-badge"
          style={{
            borderColor: getZoneColor(leftZone),
            backgroundColor: `${getZoneColor(leftZone)}20`,
          }}
        >
          <span
            className="zone-name"
            style={{ color: getZoneColor(leftZone) }}
          >
            {formatZoneName(leftZone)}
          </span>
        </div>
        <div className="zone-confidence">
          <span className="confidence-label">Confidence</span>
          <div className="confidence-bar">
            <div
              className="confidence-fill"
              style={{
                width: `${leftConfidence * 100}%`,
                backgroundColor: getZoneColor(leftZone),
              }}
            />
          </div>
          <span className="confidence-value">
            {(leftConfidence * 100).toFixed(0)}%
          </span>
        </div>
        <span className="zone-description">{getZoneDescription(leftZone)}</span>
      </div>

      <div className="zone-arrow-container">
        {zoneChanged ? (
          <div className="zone-arrow changed">
            <span className="arrow-symbol">&#x2192;</span>
            <span className="zone-changed-label">Zone Changed</span>
          </div>
        ) : (
          <div className="zone-arrow same">
            <span className="arrow-symbol">=</span>
            <span className="zone-same-label">Same Zone</span>
          </div>
        )}
      </div>

      <div className="zone-side right">
        <span className="zone-session-label">Session B</span>
        <div
          className="zone-badge"
          style={{
            borderColor: getZoneColor(rightZone),
            backgroundColor: `${getZoneColor(rightZone)}20`,
          }}
        >
          <span
            className="zone-name"
            style={{ color: getZoneColor(rightZone) }}
          >
            {formatZoneName(rightZone)}
          </span>
        </div>
        <div className="zone-confidence">
          <span className="confidence-label">Confidence</span>
          <div className="confidence-bar">
            <div
              className="confidence-fill"
              style={{
                width: `${rightConfidence * 100}%`,
                backgroundColor: getZoneColor(rightZone),
              }}
            />
          </div>
          <span className="confidence-value">
            {(rightConfidence * 100).toFixed(0)}%
          </span>
        </div>
        <span className="zone-description">{getZoneDescription(rightZone)}</span>
      </div>
    </div>
  );
}
