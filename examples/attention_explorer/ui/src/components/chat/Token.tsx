import { useCallback } from 'react';
import { useUIStore } from '../../stores/useUIStore';
import { AttentionEntry, isRawMode } from '../../api/types';

interface TokenProps {
  text: string;
  index: number;
  type: 'input' | 'output';
  attention?: AttentionEntry;
}

export function Token({ text, index, type, attention }: TokenProps) {
  const selectedTokenIndex = useUIStore((state) => state.selectedTokenIndex);
  const hoveredTokenIndex = useUIStore((state) => state.hoveredTokenIndex);
  const selectToken = useUIStore((state) => state.selectToken);
  const hoverToken = useUIStore((state) => state.hoverToken);

  const isSelected = type === 'output' && selectedTokenIndex === index;
  const isHovered = type === 'output' && hoveredTokenIndex === index;

  // Check topk_mass for low-mass indicator
  const isLowMass = attention && isRawMode(attention) && (attention.topk_mass ?? 1) < 0.55;

  const handleClick = useCallback(() => {
    if (type === 'output') {
      if (isSelected) {
        selectToken(null);
        return;
      }
      // Simply select the token - the InsightPanel will read attention from stored messages
      selectToken(index);
    }
  }, [type, index, isSelected, selectToken]);

  const handleMouseEnter = useCallback(() => {
    if (type === 'output') {
      hoverToken(index);
    }
  }, [type, index, hoverToken]);

  const handleMouseLeave = useCallback(() => {
    hoverToken(null);
  }, [hoverToken]);

  const classNames = [
    'tok',
    isSelected && 'selected',
    isHovered && 'hovered',
    isLowMass && 'lowmass',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <span
      className={classNames}
      onClick={handleClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      data-index={index}
      data-type={type}
    >
      {text}
    </span>
  );
}
