import { useMemo } from 'react';
import { SegmentType } from '../../api/types';

interface SegmentInfo {
  type: SegmentType;
  startIndex: number;
  endIndex: number;
  tokenCount: number;
}

interface SegmentTimelineProps {
  content: string;
  tokens: string[];
  onSegmentClick?: (segment: SegmentInfo) => void;
  selectedSegment?: SegmentType | null;
  compact?: boolean;
}

/**
 * Parse content to detect think/output segments
 * Returns segment boundaries based on <think> tags
 */
function parseSegments(content: string, tokens: string[]): SegmentInfo[] {
  const segments: SegmentInfo[] = [];

  // Find <think> tags
  const thinkMatch = content.match(/<think>([\s\S]*?)<\/think>/);

  if (!thinkMatch) {
    // No think section - single output segment
    return [{
      type: 'assistant_final',
      startIndex: 0,
      endIndex: tokens.length,
      tokenCount: tokens.length,
    }];
  }

  const thinkStart = content.indexOf('<think>');
  const thinkEnd = content.indexOf('</think>') + '</think>'.length;

  // Map character positions to token indices
  let charCount = 0;
  let thinkStartToken = 0;
  let thinkEndToken = tokens.length;

  for (let i = 0; i < tokens.length; i++) {
    const tokenStart = charCount;
    const tokenEnd = charCount + tokens[i].length;

    // Token is before think section
    if (tokenEnd <= thinkStart && i + 1 > thinkStartToken) {
      // This token is still before think
    }
    // Token contains or is after <think>
    else if (tokenStart < thinkStart && tokenEnd > thinkStart) {
      thinkStartToken = i;
    }
    else if (tokenStart >= thinkStart && thinkStartToken === 0) {
      thinkStartToken = i;
    }

    // Token contains or is after </think>
    if (tokenStart >= thinkEnd && thinkEndToken === tokens.length) {
      thinkEndToken = i;
    }

    charCount += tokens[i].length;
  }

  // Pre-think segment (if any content before <think>)
  if (thinkStartToken > 0) {
    segments.push({
      type: 'assistant_final',
      startIndex: 0,
      endIndex: thinkStartToken,
      tokenCount: thinkStartToken,
    });
  }

  // Think segment
  if (thinkEndToken > thinkStartToken) {
    segments.push({
      type: 'assistant_think',
      startIndex: thinkStartToken,
      endIndex: thinkEndToken,
      tokenCount: thinkEndToken - thinkStartToken,
    });
  }

  // Post-think segment (final output)
  if (thinkEndToken < tokens.length) {
    segments.push({
      type: 'assistant_final',
      startIndex: thinkEndToken,
      endIndex: tokens.length,
      tokenCount: tokens.length - thinkEndToken,
    });
  }

  return segments;
}

function getSegmentLabel(type: SegmentType): string {
  switch (type) {
    case 'assistant_think': return 'Think';
    case 'assistant_final': return 'Output';
    case 'user': return 'User';
    case 'system': return 'System';
    default: return type;
  }
}

function getSegmentIcon(type: SegmentType): string {
  switch (type) {
    case 'assistant_think': return '💭';
    case 'assistant_final': return '💬';
    case 'user': return '👤';
    case 'system': return '⚙️';
    default: return '📝';
  }
}

export function SegmentTimeline({
  content,
  tokens,
  onSegmentClick,
  selectedSegment,
  compact = false,
}: SegmentTimelineProps) {
  const segments = useMemo(() => parseSegments(content, tokens), [content, tokens]);
  const totalTokens = tokens.length;

  if (segments.length <= 1 && segments[0]?.type === 'assistant_final') {
    // No segmentation needed for simple output
    return null;
  }

  return (
    <div className={`segment-timeline ${compact ? 'compact' : ''}`}>
      <div className="segment-bar">
        {segments.map((seg, i) => {
          const width = (seg.tokenCount / totalTokens) * 100;
          const isSelected = selectedSegment === seg.type;

          return (
            <div
              key={i}
              className={`segment-block ${seg.type} ${isSelected ? 'selected' : ''}`}
              style={{ width: `${Math.max(width, 2)}%` }}
              onClick={() => onSegmentClick?.(seg)}
              title={`${getSegmentLabel(seg.type)}: ${seg.tokenCount} tokens (${width.toFixed(0)}%)`}
            >
              {!compact && width > 15 && (
                <span className="segment-label">
                  <span className="segment-icon">{getSegmentIcon(seg.type)}</span>
                  {getSegmentLabel(seg.type)}
                </span>
              )}
            </div>
          );
        })}
      </div>

      {!compact && (
        <div className="segment-legend">
          {segments.map((seg, i) => (
            <div
              key={i}
              className={`segment-legend-item ${seg.type} ${selectedSegment === seg.type ? 'selected' : ''}`}
              onClick={() => onSegmentClick?.(seg)}
            >
              <span className="segment-legend-icon">{getSegmentIcon(seg.type)}</span>
              <span className="segment-legend-label">{getSegmentLabel(seg.type)}</span>
              <span className="segment-legend-count">{seg.tokenCount}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Export the parsing function for use elsewhere
export { parseSegments };
export type { SegmentInfo };
