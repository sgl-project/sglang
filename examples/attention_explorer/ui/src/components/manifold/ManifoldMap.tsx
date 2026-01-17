import { useRef, useEffect, useCallback, useState } from 'react';
import { useManifoldStore, ManifoldScope } from '../../stores/useManifoldStore';
import { ManifoldPoint, ManifoldZone } from '../../api/types';

// Zone colors for rendering
const ZONE_COLORS: Record<ManifoldZone, string> = {
  syntax_floor: 'rgba(85, 214, 166, 0.8)',
  semantic_bridge: 'rgba(122, 162, 255, 0.8)',
  long_range: 'rgba(255, 204, 102, 0.8)',
  structure_ripple: 'rgba(255, 107, 107, 0.8)',
  diffuse: 'rgba(156, 163, 175, 0.6)',
  unknown: 'rgba(156, 163, 175, 0.4)',
};

// Scope display names
const SCOPE_LABELS: Record<ManifoldScope, string> = {
  current: 'This Session',
  recent: 'Recent (50)',
  saved: 'Saved Only',
  all: 'All Sessions',
};

interface ManifoldMapProps {
  onPointClick?: (sessionId: string) => void;
  height?: number;
}

export function ManifoldMap({ onPointClick, height = 280 }: ManifoldMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 400, height });

  // Store state
  const scope = useManifoldStore((state) => state.scope);
  const setScope = useManifoldStore((state) => state.setScope);
  const getFilteredPoints = useManifoldStore((state) => state.getFilteredPoints);
  const getClusters = useManifoldStore((state) => state.getClusters);
  const getCurrentSessionPoint = useManifoldStore((state) => state.getCurrentSessionPoint);
  const selectedPointId = useManifoldStore((state) => state.selectedPointId);
  const hoveredPointId = useManifoldStore((state) => state.hoveredPointId);
  const selectPoint = useManifoldStore((state) => state.selectPoint);
  const hoverPoint = useManifoldStore((state) => state.hoverPoint);

  // Get computed values
  const points = getFilteredPoints();
  const clusters = getClusters();
  const currentPoint = getCurrentSessionPoint();

  // Handle canvas resize
  useEffect(() => {
    if (!containerRef.current) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setCanvasSize({
          width: entry.contentRect.width,
          height: height,
        });
      }
    });

    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [height]);

  // Find point at coordinates
  const findPointAt = useCallback(
    (x: number, y: number): ManifoldPoint | null => {
      const { width, height } = canvasSize;
      const hitRadius = 8;

      // Check current session point first
      if (currentPoint) {
        const px = currentPoint.coords[0] * width;
        const py = currentPoint.coords[1] * height;
        const dist = Math.sqrt(Math.pow(x - px, 2) + Math.pow(y - py, 2));
        if (dist <= hitRadius) return currentPoint;
      }

      // Check other points
      for (const point of points) {
        const px = point.coords[0] * width;
        const py = point.coords[1] * height;
        const dist = Math.sqrt(Math.pow(x - px, 2) + Math.pow(y - py, 2));
        if (dist <= hitRadius) return point;
      }

      return null;
    },
    [points, currentPoint, canvasSize]
  );

  // Mouse handlers
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      const point = findPointAt(x, y);
      hoverPoint(point?.session_id || null);

      canvas.style.cursor = point ? 'pointer' : 'default';
    },
    [findPointAt, hoverPoint]
  );

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      const point = findPointAt(x, y);
      if (point) {
        selectPoint(point.session_id);
        onPointClick?.(point.session_id);
      }
    },
    [findPointAt, selectPoint, onPointClick]
  );

  const handleMouseLeave = useCallback(() => {
    hoverPoint(null);
  }, [hoverPoint]);

  // Draw the canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle device pixel ratio
    const dpr = window.devicePixelRatio || 1;
    const { width: w, height: h } = canvasSize;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Draw cluster backgrounds (subtle ellipses)
    for (const cluster of clusters) {
      const cx = cluster.centroid[0] * w;
      const cy = cluster.centroid[1] * h;
      const radius = Math.max(cluster.radius * w, 30);

      ctx.fillStyle = ZONE_COLORS[cluster.dominant_zone].replace('0.8', '0.15');
      ctx.beginPath();
      ctx.ellipse(cx, cy, radius * 1.2, radius, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw cluster labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.font = '11px system-ui, sans-serif';
    for (const cluster of clusters) {
      const cx = cluster.centroid[0] * w;
      const cy = cluster.centroid[1] * h;
      ctx.fillText(cluster.name, cx - 20, cy - 20);
    }

    // Draw session points
    for (const point of points) {
      const px = point.coords[0] * w;
      const py = point.coords[1] * h;
      const isSelected = point.session_id === selectedPointId;
      const isHovered = point.session_id === hoveredPointId;

      // Point size based on state
      let radius = 3;
      if (isSelected) radius = 6;
      else if (isHovered) radius = 5;

      // Draw point
      ctx.fillStyle = ZONE_COLORS[point.manifold_zone] || ZONE_COLORS.unknown;
      ctx.beginPath();
      ctx.arc(px, py, radius, 0, Math.PI * 2);
      ctx.fill();

      // Selection ring
      if (isSelected) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(px, py, radius + 3, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Hover ring
      if (isHovered && !isSelected) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(px, py, radius + 2, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    // Draw current session point ("you are here")
    if (currentPoint) {
      const px = currentPoint.coords[0] * w;
      const py = currentPoint.coords[1] * h;
      const isCurrentSelected = currentPoint.session_id === selectedPointId;

      // Pulsing outer ring
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px, py, isCurrentSelected ? 10 : 8, 0, Math.PI * 2);
      ctx.stroke();

      // Inner filled circle
      ctx.fillStyle = ZONE_COLORS[currentPoint.manifold_zone] || 'rgba(255, 255, 255, 0.8)';
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fill();

      // Label
      ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
      ctx.font = '12px system-ui, sans-serif';
      ctx.fillText('you', px + 12, py + 4);
    }

    // Draw empty state message if no points
    if (points.length === 0 && !currentPoint) {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.font = '14px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No session data yet', w / 2, h / 2 - 10);
      ctx.font = '12px system-ui, sans-serif';
      ctx.fillText('Start a conversation to see it here', w / 2, h / 2 + 10);
      ctx.textAlign = 'left';
    }
  }, [points, clusters, currentPoint, selectedPointId, hoveredPointId, canvasSize]);

  return (
    <div className="manifold-map-container" ref={containerRef}>
      {/* Scope selector */}
      <div className="scope-selector">
        {(Object.keys(SCOPE_LABELS) as ManifoldScope[]).map((s) => (
          <button
            key={s}
            className={`scope-btn ${scope === s ? 'active' : ''}`}
            onClick={() => setScope(s)}
          >
            {SCOPE_LABELS[s]}
          </button>
        ))}
      </div>

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        className="manifold-canvas"
        style={{ width: '100%', height: `${height}px` }}
        onMouseMove={handleMouseMove}
        onClick={handleClick}
        onMouseLeave={handleMouseLeave}
      />

      {/* Stats bar */}
      <div className="manifold-stats">
        <span className="stat">
          {points.length} point{points.length !== 1 ? 's' : ''}
        </span>
        <span className="stat">
          {clusters.length} cluster{clusters.length !== 1 ? 's' : ''}
        </span>
        {hoveredPointId && (
          <span className="stat highlight">
            Hovering: {hoveredPointId.slice(0, 12)}...
          </span>
        )}
      </div>
    </div>
  );
}
