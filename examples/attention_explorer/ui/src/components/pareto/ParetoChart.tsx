/**
 * ParetoChart - Interactive scatter plot for quality vs compression trade-off
 *
 * Features:
 * - SVG-based scatter plot with zoom/pan
 * - Pareto frontier highlight
 * - Hover tooltips
 * - Color coding by method/quality/tiling
 * - Click to select config
 */

import { useMemo, useCallback, useRef, useState } from 'react';
import { useParetoStore, useFilteredPoints, ParetoPoint } from '../../stores/useParetoStore';
import { QualityTier, QuantizationMethod, TilingMode } from '../../api/comparisonSchema';

// ============================================================================
// CONSTANTS
// ============================================================================

const CHART_PADDING = { top: 40, right: 40, bottom: 60, left: 70 };
const POINT_RADIUS = 8;
const POINT_RADIUS_HOVER = 12;

const METHOD_COLORS: Record<QuantizationMethod, string> = {
  none: '#666666',
  sinq: '#3b82f6',      // blue
  asinq: '#8b5cf6',     // purple
  awq: '#10b981',       // green
  gptq: '#f59e0b',      // amber
  squeezellm: '#ef4444', // red
  fp8: '#06b6d4',       // cyan
  marlin: '#ec4899',    // pink
};

const QUALITY_COLORS: Record<QualityTier, string> = {
  excellent: '#22c55e',
  good: '#84cc16',
  acceptable: '#eab308',
  degraded: '#f97316',
  failed: '#ef4444',
};

const TILING_COLORS: Record<TilingMode, string> = {
  '1D': '#3b82f6',
  '2D': '#8b5cf6',
};

// ============================================================================
// HELPERS
// ============================================================================

function getAxisLabel(axis: string): string {
  switch (axis) {
    case 'compressionRatio': return 'Compression Ratio';
    case 'memoryMb': return 'Memory (MB)';
    case 'nbits': return 'Bits';
    case 'meanJaccard': return 'Jaccard Similarity';
    case 'weightedJaccard': return 'Weighted Jaccard';
    case 'spearman': return 'Spearman Correlation';
    case 'massRetained': return 'Mass Retained';
    default: return axis;
  }
}

function getPointValue(point: ParetoPoint, axis: string): number {
  switch (axis) {
    case 'compressionRatio': return point.compressionRatio;
    case 'memoryMb': return point.memoryMb;
    case 'nbits': return point.nbits;
    case 'meanJaccard': return point.meanJaccard;
    case 'weightedJaccard': return point.weightedJaccard;
    case 'spearman': return point.spearman;
    case 'massRetained': return point.massRetained;
    default: return 0;
  }
}

// ============================================================================
// COMPONENT
// ============================================================================

interface ParetoChartProps {
  width: number;
  height: number;
}

export function ParetoChart({ width, height }: ParetoChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; point: ParetoPoint } | null>(null);

  const points = useFilteredPoints();
  const xAxis = useParetoStore((state) => state.xAxis);
  const yAxis = useParetoStore((state) => state.yAxis);
  const colorBy = useParetoStore((state) => state.colorBy);
  const selectedPointId = useParetoStore((state) => state.selectedPointId);
  const hoveredPointId = useParetoStore((state) => state.hoveredPointId);
  const selectPoint = useParetoStore((state) => state.selectPoint);
  const hoverPoint = useParetoStore((state) => state.hoverPoint);

  // Compute scales
  const { xScale, yScale, xMin, xMax, yMin, yMax } = useMemo(() => {
    if (points.length === 0) {
      return {
        xScale: (v: number) => v,
        yScale: (v: number) => v,
        xMin: 0, xMax: 10, yMin: 0, yMax: 1,
      };
    }

    const xValues = points.map((p) => getPointValue(p, xAxis));
    const yValues = points.map((p) => getPointValue(p, yAxis));

    const xMin = Math.min(...xValues) * 0.95;
    const xMax = Math.max(...xValues) * 1.05;
    const yMin = Math.min(...yValues) * 0.95;
    const yMax = Math.max(...yValues) * 1.05;

    const chartWidth = width - CHART_PADDING.left - CHART_PADDING.right;
    const chartHeight = height - CHART_PADDING.top - CHART_PADDING.bottom;

    const xScale = (v: number) => CHART_PADDING.left + ((v - xMin) / (xMax - xMin)) * chartWidth;
    const yScale = (v: number) => height - CHART_PADDING.bottom - ((v - yMin) / (yMax - yMin)) * chartHeight;

    return { xScale, yScale, xMin, xMax, yMin, yMax };
  }, [points, xAxis, yAxis, width, height]);

  // Get point color
  const getPointColor = useCallback((point: ParetoPoint): string => {
    switch (colorBy) {
      case 'method':
        return METHOD_COLORS[point.method] || '#666';
      case 'qualityTier':
        return QUALITY_COLORS[point.qualityTier] || '#666';
      case 'tilingMode':
        return TILING_COLORS[point.tilingMode] || '#666';
      default:
        return '#666';
    }
  }, [colorBy]);

  // Generate Pareto frontier line
  const frontierPath = useMemo(() => {
    const frontierPoints = points
      .filter((p) => p.isOnFrontier)
      .sort((a, b) => getPointValue(a, xAxis) - getPointValue(b, xAxis));

    if (frontierPoints.length < 2) return '';

    const pathPoints = frontierPoints.map((p) => {
      const x = xScale(getPointValue(p, xAxis));
      const y = yScale(getPointValue(p, yAxis));
      return `${x},${y}`;
    });

    return `M ${pathPoints.join(' L ')}`;
  }, [points, xAxis, yAxis, xScale, yScale]);

  // Generate grid lines
  const gridLines = useMemo(() => {
    const xTicks = 5;
    const yTicks = 5;
    const lines: JSX.Element[] = [];

    for (let i = 0; i <= xTicks; i++) {
      const v = xMin + (i / xTicks) * (xMax - xMin);
      const x = xScale(v);
      lines.push(
        <line
          key={`x-${i}`}
          x1={x}
          y1={CHART_PADDING.top}
          x2={x}
          y2={height - CHART_PADDING.bottom}
          stroke="rgba(255,255,255,0.1)"
          strokeDasharray="2,4"
        />
      );
      lines.push(
        <text
          key={`x-label-${i}`}
          x={x}
          y={height - CHART_PADDING.bottom + 20}
          fill="rgba(255,255,255,0.5)"
          fontSize="11"
          textAnchor="middle"
        >
          {v.toFixed(1)}
        </text>
      );
    }

    for (let i = 0; i <= yTicks; i++) {
      const v = yMin + (i / yTicks) * (yMax - yMin);
      const y = yScale(v);
      lines.push(
        <line
          key={`y-${i}`}
          x1={CHART_PADDING.left}
          y1={y}
          x2={width - CHART_PADDING.right}
          y2={y}
          stroke="rgba(255,255,255,0.1)"
          strokeDasharray="2,4"
        />
      );
      lines.push(
        <text
          key={`y-label-${i}`}
          x={CHART_PADDING.left - 10}
          y={y + 4}
          fill="rgba(255,255,255,0.5)"
          fontSize="11"
          textAnchor="end"
        >
          {v.toFixed(2)}
        </text>
      );
    }

    return lines;
  }, [xMin, xMax, yMin, yMax, xScale, yScale, width, height, xAxis]);

  // Quality threshold lines
  const thresholdLines = useMemo(() => {
    if (yAxis !== 'meanJaccard' && yAxis !== 'weightedJaccard') return null;

    const thresholds = [
      { value: 0.8, label: 'Excellent', color: '#22c55e' },
      { value: 0.6, label: 'Good', color: '#84cc16' },
      { value: 0.4, label: 'Acceptable', color: '#eab308' },
    ];

    return thresholds.map((t) => {
      const y = yScale(t.value);
      if (y < CHART_PADDING.top || y > height - CHART_PADDING.bottom) return null;

      return (
        <g key={t.label}>
          <line
            x1={CHART_PADDING.left}
            y1={y}
            x2={width - CHART_PADDING.right}
            y2={y}
            stroke={t.color}
            strokeOpacity={0.3}
            strokeDasharray="4,4"
          />
          <text
            x={width - CHART_PADDING.right + 5}
            y={y + 4}
            fill={t.color}
            fontSize="10"
            opacity={0.7}
          >
            {t.label}
          </text>
        </g>
      );
    });
  }, [yAxis, yScale, width, height]);

  // Event handlers
  const handlePointClick = useCallback((point: ParetoPoint) => {
    selectPoint(selectedPointId === point.id ? null : point.id);
  }, [selectPoint, selectedPointId]);

  const handlePointHover = useCallback((point: ParetoPoint | null, event?: React.MouseEvent) => {
    hoverPoint(point?.id || null);
    if (point && event) {
      setTooltip({
        x: event.clientX,
        y: event.clientY,
        point,
      });
    } else {
      setTooltip(null);
    }
  }, [hoverPoint]);

  return (
    <div className="pareto-chart-container">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="pareto-chart"
      >
        {/* Background */}
        <rect
          x={CHART_PADDING.left}
          y={CHART_PADDING.top}
          width={width - CHART_PADDING.left - CHART_PADDING.right}
          height={height - CHART_PADDING.top - CHART_PADDING.bottom}
          fill="rgba(0,0,0,0.2)"
          rx={4}
        />

        {/* Grid */}
        {gridLines}

        {/* Quality thresholds */}
        {thresholdLines}

        {/* Pareto frontier */}
        {frontierPath && (
          <path
            d={frontierPath}
            fill="none"
            stroke="rgba(122, 162, 255, 0.6)"
            strokeWidth={2}
            strokeDasharray="6,3"
          />
        )}

        {/* Points */}
        {points.map((point) => {
          const x = xScale(getPointValue(point, xAxis));
          const y = yScale(getPointValue(point, yAxis));
          const isSelected = selectedPointId === point.id;
          const isHovered = hoveredPointId === point.id;
          const radius = isHovered ? POINT_RADIUS_HOVER : POINT_RADIUS;

          return (
            <g key={point.id}>
              {/* Selection ring */}
              {isSelected && (
                <circle
                  cx={x}
                  cy={y}
                  r={radius + 4}
                  fill="none"
                  stroke="#fff"
                  strokeWidth={2}
                />
              )}

              {/* Frontier indicator */}
              {point.isOnFrontier && (
                <circle
                  cx={x}
                  cy={y}
                  r={radius + 2}
                  fill="none"
                  stroke="rgba(122, 162, 255, 0.8)"
                  strokeWidth={2}
                />
              )}

              {/* Main point */}
              <circle
                cx={x}
                cy={y}
                r={radius}
                fill={getPointColor(point)}
                stroke={isSelected ? '#fff' : 'rgba(0,0,0,0.3)'}
                strokeWidth={isSelected ? 2 : 1}
                style={{
                  cursor: 'pointer',
                  transition: 'r 0.15s ease',
                }}
                onClick={() => handlePointClick(point)}
                onMouseEnter={(e) => handlePointHover(point, e)}
                onMouseLeave={() => handlePointHover(null)}
              />

              {/* Label for frontier points */}
              {point.isOnFrontier && (
                <text
                  x={x}
                  y={y - radius - 6}
                  fill="rgba(255,255,255,0.7)"
                  fontSize="10"
                  textAnchor="middle"
                >
                  {point.nbits}b
                </text>
              )}
            </g>
          );
        })}

        {/* Axis labels */}
        <text
          x={width / 2}
          y={height - 10}
          fill="rgba(255,255,255,0.7)"
          fontSize="12"
          textAnchor="middle"
        >
          {getAxisLabel(xAxis)}
        </text>
        <text
          x={15}
          y={height / 2}
          fill="rgba(255,255,255,0.7)"
          fontSize="12"
          textAnchor="middle"
          transform={`rotate(-90, 15, ${height / 2})`}
        >
          {getAxisLabel(yAxis)}
        </text>

        {/* Title */}
        <text
          x={width / 2}
          y={20}
          fill="rgba(255,255,255,0.9)"
          fontSize="14"
          fontWeight="600"
          textAnchor="middle"
        >
          Quantization Pareto Frontier
        </text>
      </svg>

      {/* Tooltip */}
      {tooltip && (
        <div
          className="pareto-tooltip"
          style={{
            position: 'fixed',
            left: tooltip.x + 15,
            top: tooltip.y - 10,
            zIndex: 1000,
          }}
        >
          <div className="tooltip-header">{tooltip.point.configName}</div>
          <div className="tooltip-row">
            <span>Jaccard:</span>
            <span className={`tier-${tooltip.point.qualityTier}`}>
              {(tooltip.point.meanJaccard * 100).toFixed(1)}%
            </span>
          </div>
          <div className="tooltip-row">
            <span>Compression:</span>
            <span>{tooltip.point.compressionRatio.toFixed(2)}x</span>
          </div>
          <div className="tooltip-row">
            <span>Memory:</span>
            <span>{tooltip.point.memoryMb.toFixed(0)} MB</span>
          </div>
          <div className="tooltip-row">
            <span>Quality:</span>
            <span className={`tier-${tooltip.point.qualityTier}`}>
              {tooltip.point.qualityTier}
            </span>
          </div>
          {tooltip.point.isOnFrontier && (
            <div className="tooltip-badge">On Pareto Frontier</div>
          )}
        </div>
      )}
    </div>
  );
}
