// SGLang OpenAI-compatible client with attention capture

import {
  AttentionCaptureParams,
  ChatCompletionChunkWithAttention,
  ChatCompletionWithAttention,
  AttentionEntry,
  MoERoutingEntry,
  Program,
  extractFingerprint,
} from './types';

// ============================================================================
// CONFIGURATION
// ============================================================================

export interface ClientConfig {
  baseUrl: string;
  model: string;
  program: Program;
  onToken?: (token: string, index: number) => void;
  onAttention?: (entry: AttentionEntry, index: number) => void;
  onMoE?: (entry: MoERoutingEntry, index: number) => void;
  onFinish?: (fullResponse: ChatCompletionWithAttention) => void;
  onError?: (error: Error) => void;
}

const PROGRAM_CONFIGS: Record<Program, AttentionCaptureParams> = {
  prod: {
    return_attention_tokens: true,
    top_k_attention: 10,
  },
  debug: {
    return_attention_tokens: true,
    top_k_attention: 10,
    attention_capture_layer_ids: [7, 15, 23, 31],
    return_moe_routing: true,
    moe_routing_top_k: 2,
  },
  discovery: {
    return_attention_tokens: true,
    top_k_attention: 10,
    attention_sketch_mode: true,
    return_moe_routing: true,
  },
};

// ============================================================================
// STREAMING CLIENT
// ============================================================================

export class AttentionStreamClient {
  private config: ClientConfig;
  private abortController: AbortController | null = null;

  constructor(config: ClientConfig) {
    this.config = config;
  }

  setProgram(program: Program) {
    this.config.program = program;
  }

  abort() {
    this.abortController?.abort();
    this.abortController = null;
  }

  async stream(messages: Array<{ role: string; content: string }>): Promise<void> {
    this.abort();
    this.abortController = new AbortController();

    const captureParams = PROGRAM_CONFIGS[this.config.program];

    const response = await fetch(`${this.config.baseUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.config.model,
        messages,
        stream: true,
        ...captureParams,
      }),
      signal: this.abortController.signal,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API error: ${response.status} - ${error}`);
    }

    if (!response.body) {
      throw new Error('No response body');
    }

    await this.processStream(response.body);
  }

  private async processStream(body: ReadableStream<Uint8Array>): Promise<void> {
    const reader = body.getReader();
    const decoder = new TextDecoder();

    const buffer = new StreamBuffer();
    let tokenIndex = 0;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        const events = parseSSEEvents(text);

        for (const event of events) {
          if (event === '[DONE]') {
            this.config.onFinish?.(buffer.toResponse());
            return;
          }

          try {
            const chunk: ChatCompletionChunkWithAttention = JSON.parse(event);
            const delta = chunk.choices[0]?.delta;

            if (delta?.content) {
              const tokens = tokenizeIncremental(delta.content, buffer.pendingText);
              buffer.pendingText = tokens.pending;

              for (const token of tokens.complete) {
                buffer.tokens.push(token);
                this.config.onToken?.(token, tokenIndex);
                tokenIndex++;
              }
            }

            if (chunk.attention_token) {
              buffer.attention.push(chunk.attention_token);
              this.config.onAttention?.(chunk.attention_token, buffer.attention.length - 1);
            }

            if (chunk.moe_routing_step) {
              buffer.moe.push(chunk.moe_routing_step);
              this.config.onMoE?.(chunk.moe_routing_step, buffer.moe.length - 1);
            }

            if (chunk.choices[0]?.finish_reason) {
              buffer.finishReason = chunk.choices[0].finish_reason;
            }
          } catch (e) {
            console.warn('Failed to parse SSE event:', event, e);
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
}

// ============================================================================
// STREAM BUFFER
// ============================================================================

class StreamBuffer {
  tokens: string[] = [];
  attention: AttentionEntry[] = [];
  moe: MoERoutingEntry[] = [];
  pendingText: string = '';
  finishReason: string | null = null;

  toResponse(): ChatCompletionWithAttention {
    return {
      id: `chat-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: 'default',
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: this.tokens.join(''),
          },
          finish_reason: this.finishReason || 'stop',
        },
      ],
      attention_tokens: this.attention,
      moe_routing: this.moe,
    };
  }
}

// ============================================================================
// SSE PARSING
// ============================================================================

function parseSSEEvents(text: string): string[] {
  const events: string[] = [];
  const lines = text.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = line.slice(6).trim();
      if (data) {
        events.push(data);
      }
    }
  }

  return events;
}

// ============================================================================
// TOKEN ALIGNMENT
// ============================================================================

interface TokenizeResult {
  complete: string[];
  pending: string;
}

function tokenizeIncremental(newText: string, pending: string): TokenizeResult {
  const combined = pending + newText;
  const tokens: string[] = [];
  let current = '';

  for (let i = 0; i < combined.length; i++) {
    const char = combined[i];

    if (char === ' ' || char === '\n' || char === '\t') {
      if (current) {
        tokens.push(current);
        current = '';
      }
      tokens.push(char);
    } else {
      current += char;
    }
  }

  const endsWithBoundary = /[\s]$/.test(newText);

  if (endsWithBoundary || current === '') {
    if (current) tokens.push(current);
    return { complete: tokens, pending: '' };
  } else {
    return { complete: tokens, pending: current };
  }
}

// ============================================================================
// TOKEN-ATTENTION ALIGNER
// ============================================================================

export class TokenAttentionAligner {
  private tokens: string[] = [];
  private attention: Map<number, AttentionEntry> = new Map();
  private prefillLength: number = 0;

  setPrefillLength(length: number) {
    this.prefillLength = length;
  }

  addToken(token: string): number {
    const index = this.tokens.length;
    this.tokens.push(token);
    return index;
  }

  addAttention(entry: AttentionEntry): void {
    const step =
      'step' in entry ? entry.step : 'decode_step' in entry ? entry.decode_step : this.attention.size;

    const tokenIndex = this.prefillLength + step;
    this.attention.set(tokenIndex, entry);
  }

  getAttention(tokenIndex: number): AttentionEntry | null {
    return this.attention.get(tokenIndex) || null;
  }

  getAligned(): Array<{ token: string; index: number; attention: AttentionEntry | null }> {
    return this.tokens.map((token, index) => ({
      token,
      index,
      attention: this.attention.get(index) || null,
    }));
  }

  getAlignmentStats(): { tokens: number; attention: number; aligned: number; missing: number } {
    const decodeTokens = this.tokens.length - this.prefillLength;
    const aligned = Array.from(this.attention.keys()).filter((i) => i >= this.prefillLength).length;

    return {
      tokens: this.tokens.length,
      attention: this.attention.size,
      aligned,
      missing: Math.max(0, decodeTokens - aligned),
    };
  }
}

// ============================================================================
// NON-STREAMING CLIENT
// ============================================================================

export async function fetchWithAttention(
  baseUrl: string,
  messages: Array<{ role: string; content: string }>,
  params: AttentionCaptureParams = {}
): Promise<ChatCompletionWithAttention> {
  const response = await fetch(`${baseUrl}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'default',
      messages,
      stream: false,
      ...params,
    }),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}
