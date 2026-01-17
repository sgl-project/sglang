import { useManifoldStore } from '../../stores/useManifoldStore';
import { ClusterDefinition, ManifoldZone } from '../../api/types';

// Zone colors for badges
const ZONE_BADGE_COLORS: Record<ManifoldZone, string> = {
  syntax_floor: '#55d6a6',
  semantic_bridge: '#7aa2ff',
  long_range: '#ffcc66',
  structure_ripple: '#ff6b6b',
  diffuse: '#9ca3af',
  unknown: '#6b7280',
};

function formatZoneName(zone: ManifoldZone): string {
  return zone
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

interface ClusterCardProps {
  cluster: ClusterDefinition;
  isSelected: boolean;
  onClick: () => void;
}

function ClusterCard({ cluster, isSelected, onClick }: ClusterCardProps) {
  const zoneColor = ZONE_BADGE_COLORS[cluster.dominant_zone];

  return (
    <div
      className={`cluster-card ${isSelected ? 'selected' : ''}`}
      onClick={onClick}
      style={{ borderLeftColor: zoneColor }}
    >
      <div className="cluster-header">
        <span className="cluster-name">{cluster.name}</span>
        <span className="cluster-count">{cluster.point_count} pts</span>
      </div>

      <div className="cluster-zone">
        <span
          className="zone-badge"
          style={{ backgroundColor: zoneColor }}
        >
          {formatZoneName(cluster.dominant_zone)}
        </span>
      </div>

      <div className="cluster-metrics">
        <div className="metric-mini">
          <span className="metric-label">Consensus</span>
          <span className="metric-value">{cluster.avg_consensus.toFixed(2)}</span>
        </div>
        <div className="metric-mini">
          <span className="metric-label">Hubness</span>
          <span className="metric-value">{cluster.avg_hubness.toFixed(2)}</span>
        </div>
        <div className="metric-mini">
          <span className="metric-label">Entropy</span>
          <span className="metric-value">{cluster.avg_entropy.toFixed(2)}</span>
        </div>
      </div>

      {cluster.top_experts && cluster.top_experts.length > 0 && (
        <div className="cluster-experts">
          <span className="expert-label">Top Experts:</span>
          <span className="expert-ids">
            {cluster.top_experts.slice(0, 3).join(', ')}
          </span>
        </div>
      )}
    </div>
  );
}

export function ClusterList() {
  const clusters = useManifoldStore((state) => state.getClusters());
  const selectedPointId = useManifoldStore((state) => state.selectedPointId);
  const sessionPoints = useManifoldStore((state) => state.sessionPoints);
  const selectPoint = useManifoldStore((state) => state.selectPoint);

  // Find which cluster the selected point belongs to
  const selectedClusterId = selectedPointId
    ? sessionPoints.find((p) => p.session_id === selectedPointId)?.cluster_id
    : null;

  const handleClusterClick = (cluster: ClusterDefinition) => {
    // Select first point in this cluster
    const clusterPoint = sessionPoints.find((p) => p.cluster_id === cluster.cluster_id);
    if (clusterPoint) {
      selectPoint(clusterPoint.session_id);
    }
  };

  if (clusters.length === 0) {
    return (
      <div className="cluster-list empty">
        <div className="empty-state">
          <span className="empty-icon">~</span>
          <span className="empty-text">No clusters detected</span>
          <span className="empty-hint">
            Start a conversation to populate the manifold
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="cluster-list">
      <div className="cluster-list-header">
        <span className="list-title">Clusters</span>
        <span className="list-count">{clusters.length} found</span>
      </div>

      <div className="cluster-cards">
        {clusters.map((cluster) => (
          <ClusterCard
            key={cluster.cluster_id}
            cluster={cluster}
            isSelected={selectedClusterId === cluster.cluster_id}
            onClick={() => handleClusterClick(cluster)}
          />
        ))}
      </div>
    </div>
  );
}
