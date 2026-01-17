import { useState, useCallback, useEffect, useMemo, useRef, CSSProperties, ReactElement } from 'react';
import { List, ListImperativeAPI } from 'react-window';
import { AttentionEntry } from '../../api/types';
import { Token } from './Token';
import { useUIStore } from '../../stores/useUIStore';

interface VirtualTokenListProps {
  tokens: string[];
  attention?: AttentionEntry[];
  type: 'input' | 'output';
  height?: number;
}

// Props passed to each row via rowProps
interface RowData {
  tokens: string[];
  attention?: AttentionEntry[];
  type: 'input' | 'output';
  tokensPerRow: number;
}

// Constants for virtualization
const ROW_HEIGHT = 28; // Height of each token row
const MIN_HEIGHT = 150;
const MAX_HEIGHT = 400;

// Row component for react-window v2
function TokenRow({
  index,
  style,
  tokens,
  attention,
  type,
  tokensPerRow,
}: {
  ariaAttributes: { 'aria-posinset': number; 'aria-setsize': number; role: 'listitem' };
  index: number;
  style: CSSProperties;
} & RowData): ReactElement {
  const rowStart = index * tokensPerRow;
  const rowEnd = Math.min(rowStart + tokensPerRow, tokens.length);
  const rowTokens = tokens.slice(rowStart, rowEnd);

  return (
    <div style={style} className="virtual-token-row">
      {rowTokens.map((text: string, localIndex: number) => {
        const globalIndex = rowStart + localIndex;
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
  );
}

export function VirtualTokenList({
  tokens,
  attention,
  type,
  height: propHeight,
}: VirtualTokenListProps) {
  const listRef = useRef<ListImperativeAPI>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [jumpValue, setJumpValue] = useState('');
  const [containerWidth, setContainerWidth] = useState(600);

  const selectedTokenIndex = useUIStore((state) => state.selectedTokenIndex);
  const drawerTokenIndex = useUIStore((state) => state.drawerTokenIndex);

  // Measure container width for tokens-per-row calculation
  useEffect(() => {
    if (!containerRef.current) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width);
      }
    });

    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Calculate tokens per row based on container width
  // Assume average token width of ~40px
  const tokensPerRow = useMemo(() => {
    const avgTokenWidth = 40;
    return Math.max(5, Math.floor(containerWidth / avgTokenWidth));
  }, [containerWidth]);

  // Calculate number of rows
  const rowCount = Math.ceil(tokens.length / tokensPerRow);

  // Calculate list height
  const listHeight = useMemo(() => {
    if (propHeight) return propHeight;
    const calculatedHeight = rowCount * ROW_HEIGHT;
    return Math.min(Math.max(calculatedHeight, MIN_HEIGHT), MAX_HEIGHT);
  }, [rowCount, propHeight]);

  // Row props for react-window v2
  const rowProps = useMemo<RowData>(
    () => ({
      tokens,
      attention,
      type,
      tokensPerRow,
    }),
    [tokens, attention, type, tokensPerRow]
  );

  // Scroll to selected token
  useEffect(() => {
    const targetIndex = selectedTokenIndex ?? drawerTokenIndex;
    if (targetIndex !== null && listRef.current) {
      const rowIndex = Math.floor(targetIndex / tokensPerRow);
      listRef.current.scrollToRow({ index: rowIndex, align: 'center' });
    }
  }, [selectedTokenIndex, drawerTokenIndex, tokensPerRow, listRef]);

  // Handle jump-to-token
  const handleJump = useCallback(() => {
    const index = parseInt(jumpValue, 10);
    if (!isNaN(index) && index >= 0 && index < tokens.length && listRef.current) {
      const rowIndex = Math.floor(index / tokensPerRow);
      listRef.current.scrollToRow({ index: rowIndex, align: 'center' });
      // Also trigger token selection
      useUIStore.getState().pinDrawer(index);
    }
  }, [jumpValue, tokens.length, tokensPerRow, listRef]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter') {
        handleJump();
      }
    },
    [handleJump]
  );

  return (
    <div className="virtual-token-list" ref={containerRef}>
      {/* Jump-to-token search */}
      <div className="virtual-token-header">
        <div className="virtual-token-stats">
          <span className="badge">{tokens.length.toLocaleString()} tokens</span>
          <span className="badge">{rowCount} rows</span>
          <span className="hint-text">virtualized</span>
        </div>
        <div className="jump-to-token">
          <input
            type="number"
            placeholder="Jump to #"
            value={jumpValue}
            onChange={(e) => setJumpValue(e.target.value)}
            onKeyDown={handleKeyDown}
            min={0}
            max={tokens.length - 1}
            className="jump-input"
          />
          <button onClick={handleJump} className="jump-btn" title="Jump to token">
            →
          </button>
        </div>
      </div>

      {/* Virtualized token list */}
      <List<RowData>
        listRef={listRef}
        rowCount={rowCount}
        rowHeight={ROW_HEIGHT}
        rowComponent={TokenRow}
        rowProps={rowProps}
        className="virtual-token-scroll"
        style={{ height: listHeight, width: '100%' }}
      />

      {/* Scroll indicator */}
      <div className="virtual-token-footer">
        <span className="hint-text">
          {selectedTokenIndex !== null
            ? `Selected: Token #${selectedTokenIndex}`
            : 'Click a token to select'}
        </span>
      </div>
    </div>
  );
}

// Threshold for when to use virtualization
export const VIRTUALIZATION_THRESHOLD = 500;
