import { useState, useMemo } from 'react';
import { AttentionEntry, SegmentType } from '../../api/types';
import { Token } from './Token';
import { parseSegments, SegmentInfo } from './SegmentTimeline';

interface SegmentedTokenLineProps {
  content: string;
  tokens: string[];
  attention?: AttentionEntry[];
  type: 'input' | 'output';
  defaultCollapsed?: boolean;
}

interface TokenSegment {
  info: SegmentInfo;
  tokens: string[];
  startIndex: number;
  attention?: AttentionEntry[];
}

function getSegmentLabel(type: SegmentType): string {
  switch (type) {
    case 'assistant_think': return 'Reasoning';
    case 'assistant_final': return 'Output';
    default: return type;
  }
}

export function SegmentedTokenLine({
  content,
  tokens,
  attention,
  type,
  defaultCollapsed = true,
}: SegmentedTokenLineProps) {
  const segments = useMemo(() => parseSegments(content, tokens), [content, tokens]);

  // Track collapsed state for each segment
  const [collapsedSegments, setCollapsedSegments] = useState<Set<number>>(() => {
    // Default: collapse think segments
    const collapsed = new Set<number>();
    if (defaultCollapsed) {
      segments.forEach((seg, i) => {
        if (seg.type === 'assistant_think') {
          collapsed.add(i);
        }
      });
    }
    return collapsed;
  });

  // Build token segments with their data
  const tokenSegments: TokenSegment[] = useMemo(() => {
    return segments.map((seg) => ({
      info: seg,
      tokens: tokens.slice(seg.startIndex, seg.endIndex),
      startIndex: seg.startIndex,
      attention: attention?.slice(seg.startIndex, seg.endIndex),
    }));
  }, [segments, tokens, attention]);

  const toggleSegment = (index: number) => {
    setCollapsedSegments((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  // If no think segments, render normally
  const hasThinkSegment = segments.some((s) => s.type === 'assistant_think');
  if (!hasThinkSegment) {
    return (
      <div className="token-line">
        {tokens.map((text, index) => (
          <Token
            key={index}
            text={text}
            index={index}
            type={type}
            attention={attention?.[index]}
          />
        ))}
      </div>
    );
  }

  return (
    <div className="segmented-token-line">
      {tokenSegments.map((segment, segIndex) => {
        const isCollapsed = collapsedSegments.has(segIndex);
        const isThink = segment.info.type === 'assistant_think';

        return (
          <div
            key={segIndex}
            className={`token-segment ${segment.info.type} ${isCollapsed ? 'collapsed' : 'expanded'}`}
          >
            {/* Segment header */}
            <div
              className="segment-header"
              onClick={() => isThink && toggleSegment(segIndex)}
              role={isThink ? 'button' : undefined}
              tabIndex={isThink ? 0 : undefined}
            >
              <span className="segment-indicator">
                {isThink ? (isCollapsed ? '▶' : '▼') : ''}
              </span>
              <span className="segment-type-badge">
                {isThink ? '💭' : '💬'} {getSegmentLabel(segment.info.type)}
              </span>
              <span className="segment-token-count">
                {segment.tokens.length} tokens
              </span>
              {isThink && isCollapsed && (
                <span className="segment-preview">
                  {segment.tokens.slice(0, 5).join('').slice(0, 50)}...
                </span>
              )}
            </div>

            {/* Token content */}
            {!isCollapsed && (
              <div className={`segment-tokens ${isThink ? 'think-tokens' : ''}`}>
                {segment.tokens.map((text, localIndex) => {
                  const globalIndex = segment.startIndex + localIndex;
                  return (
                    <Token
                      key={globalIndex}
                      text={text}
                      index={globalIndex}
                      type={type}
                      attention={attention?.[globalIndex]}
                    />
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
