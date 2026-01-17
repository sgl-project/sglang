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
  const drawerState = useUIStore((state) => state.drawerState);
  const drawerTokenIndex = useUIStore((state) => state.drawerTokenIndex);
  const openDrawerHover = useUIStore((state) => state.openDrawerHover);
  const closeDrawerHover = useUIStore((state) => state.closeDrawerHover);
  const pinDrawer = useUIStore((state) => state.pinDrawer);

  const isSelected = type === 'output' && selectedTokenIndex === index;
  const isHovered = type === 'output' && hoveredTokenIndex === index;
  const isPinned = type === 'output' && drawerState === 'pinned' && drawerTokenIndex === index;

  // Check topk_mass for low-mass indicator
  const isLowMass = attention && isRawMode(attention) && (attention.topk_mass ?? 1) < 0.55;

  const handleClick = useCallback(() => {
    if (type === 'output') {
      // Pin/unpin drawer on click
      pinDrawer(index);
    }
  }, [type, index, pinDrawer]);

  const handleMouseEnter = useCallback(() => {
    if (type === 'output') {
      // Open drawer on hover (unless pinned to another token)
      openDrawerHover(index);
    }
  }, [type, index, openDrawerHover]);

  const handleMouseLeave = useCallback(() => {
    if (type === 'output') {
      // Start close delay (drawer will close after delay unless re-hovered)
      closeDrawerHover();
    }
  }, [type, closeDrawerHover]);

  const classNames = [
    'tok',
    isSelected && 'selected',
    isHovered && 'hovered',
    isPinned && 'pinned',
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
