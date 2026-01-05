import { AttentionEntry } from '../../api/types';
import { Token } from './Token';

interface TokenLineProps {
  tokens: string[];
  attention?: AttentionEntry[];
  type: 'input' | 'output';
}

export function TokenLine({ tokens, attention, type }: TokenLineProps) {
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
