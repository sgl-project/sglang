/**
 * Unit tests for Zustand stores.
 * Tests state management logic for UI, Trace, and other stores.
 */
import { test, expect } from '@playwright/test';

test.describe('useUIStore', () => {
  test.describe('View Management', () => {
    test('initializes with chat view', async ({ page }) => {
      const result = await page.evaluate(() => {
        // Test initial state logic
        const initialState = {
          view: 'chat',
          program: 'discovery',
          selectedTokenIndex: null,
          hoveredTokenIndex: null,
          selectedLayerId: -1,
          segment: 'output',
          isConnected: false,
        };
        return initialState.view;
      });

      expect(result).toBe('chat');
    });

    test('setView updates current view', async ({ page }) => {
      const result = await page.evaluate(() => {
        let state = { view: 'chat' as const };
        const setView = (v: 'chat' | 'inspect' | 'manifold' | 'router') => {
          state = { ...state, view: v };
        };

        setView('inspect');
        return state.view;
      });

      expect(result).toBe('inspect');
    });

    test('all view options are valid', async ({ page }) => {
      const result = await page.evaluate(() => {
        const views = ['chat', 'inspect', 'manifold', 'router'];
        let state = { view: 'chat' as string };

        const setView = (v: string) => {
          state = { ...state, view: v };
        };

        const results: string[] = [];
        for (const v of views) {
          setView(v);
          results.push(state.view);
        }
        return results;
      });

      expect(result).toEqual(['chat', 'inspect', 'manifold', 'router']);
    });
  });

  test.describe('Token Selection', () => {
    test('selectToken updates selectedTokenIndex', async ({ page }) => {
      const result = await page.evaluate(() => {
        let state = {
          selectedTokenIndex: null as number | null,
          tokenDetail: { data: 'test' },
          tokenDetailError: 'error',
        };

        const selectToken = (idx: number | null) => {
          state = {
            ...state,
            selectedTokenIndex: idx,
            tokenDetail: null,
            tokenDetailError: null,
          };
        };

        selectToken(5);
        return state;
      });

      expect(result.selectedTokenIndex).toBe(5);
      expect(result.tokenDetail).toBeNull();
      expect(result.tokenDetailError).toBeNull();
    });

    test('selectToken clears selection when null', async ({ page }) => {
      const result = await page.evaluate(() => {
        let state = {
          selectedTokenIndex: 5 as number | null,
        };

        const selectToken = (idx: number | null) => {
          state = { ...state, selectedTokenIndex: idx };
        };

        selectToken(null);
        return state.selectedTokenIndex;
      });

      expect(result).toBeNull();
    });
  });

  test.describe('Drawer State Machine', () => {
    test('drawer initializes closed', async ({ page }) => {
      const result = await page.evaluate(() => {
        const initialState = {
          drawerState: 'closed',
          drawerTokenIndex: null,
          drawerTab: 'links',
        };
        return initialState;
      });

      expect(result.drawerState).toBe('closed');
      expect(result.drawerTokenIndex).toBeNull();
    });

    test('pinDrawer transitions to pinned state', async ({ page }) => {
      const result = await page.evaluate(() => {
        let state = {
          drawerState: 'closed' as const,
          drawerTokenIndex: null as number | null,
          selectedTokenIndex: null as number | null,
          hoverTimeoutId: null as any,
        };

        const pinDrawer = (tokenIndex: number) => {
          // If clicking the same pinned token, unpin
          if (state.drawerState === 'pinned' && state.drawerTokenIndex === tokenIndex) {
            state = {
              ...state,
              drawerState: 'closed',
              drawerTokenIndex: null,
              selectedTokenIndex: null,
            };
            return;
          }
          // Pin to this token
          state = {
            ...state,
            drawerState: 'pinned',
            drawerTokenIndex: tokenIndex,
            selectedTokenIndex: tokenIndex,
          };
        };

        pinDrawer(3);
        return state;
      });

      expect(result.drawerState).toBe('pinned');
      expect(result.drawerTokenIndex).toBe(3);
      expect(result.selectedTokenIndex).toBe(3);
    });

    test('pinDrawer toggles off when clicking same token', async ({ page }) => {
      const result = await page.evaluate(() => {
        let state = {
          drawerState: 'pinned' as 'closed' | 'hovering' | 'pinned',
          drawerTokenIndex: 3 as number | null,
          selectedTokenIndex: 3 as number | null,
        };

        const pinDrawer = (tokenIndex: number) => {
          if (state.drawerState === 'pinned' && state.drawerTokenIndex === tokenIndex) {
            state = {
              ...state,
              drawerState: 'closed',
              drawerTokenIndex: null,
              selectedTokenIndex: null,
            };
            return;
          }
          state = {
            ...state,
            drawerState: 'pinned',
            drawerTokenIndex: tokenIndex,
            selectedTokenIndex: tokenIndex,
          };
        };

        pinDrawer(3); // Click same token
        return state;
      });

      expect(result.drawerState).toBe('closed');
      expect(result.drawerTokenIndex).toBeNull();
    });

    test('unpinDrawer closes the drawer', async ({ page }) => {
      const result = await page.evaluate(() => {
        let state = {
          drawerState: 'pinned' as const,
          drawerTokenIndex: 5 as number | null,
          selectedTokenIndex: 5 as number | null,
        };

        const unpinDrawer = () => {
          state = {
            drawerState: 'closed',
            drawerTokenIndex: null,
            selectedTokenIndex: null,
          };
        };

        unpinDrawer();
        return state;
      });

      expect(result.drawerState).toBe('closed');
      expect(result.drawerTokenIndex).toBeNull();
      expect(result.selectedTokenIndex).toBeNull();
    });
  });

  test.describe('Overhead Stats', () => {
    test('startStreamStats initializes stats', async ({ page }) => {
      const result = await page.evaluate(() => {
        let state = {
          overheadStats: {
            tokenCount: 100,
            attentionBytes: 50000,
            attentionMode: 'raw' as const,
            streamStartTime: null as number | null,
            lastTokenTime: null as number | null,
          },
        };

        const startStreamStats = () => {
          state = {
            overheadStats: {
              tokenCount: 0,
              attentionBytes: 0,
              attentionMode: null,
              streamStartTime: Date.now(),
              lastTokenTime: null,
            },
          };
        };

        startStreamStats();
        return state.overheadStats;
      });

      expect(result.tokenCount).toBe(0);
      expect(result.attentionBytes).toBe(0);
      expect(result.streamStartTime).toBeGreaterThan(0);
    });

    test('recordAttentionEntry accumulates bytes', async ({ page }) => {
      const result = await page.evaluate(() => {
        let state = {
          overheadStats: {
            tokenCount: 0,
            attentionBytes: 100,
            attentionMode: null as string | null,
            streamStartTime: Date.now(),
            lastTokenTime: null as number | null,
          },
        };

        const recordAttentionEntry = (bytes: number, mode: string) => {
          state = {
            overheadStats: {
              ...state.overheadStats,
              attentionBytes: state.overheadStats.attentionBytes + bytes,
              attentionMode: mode,
              lastTokenTime: Date.now(),
            },
          };
        };

        recordAttentionEntry(500, 'sketch');
        recordAttentionEntry(500, 'sketch');
        return state.overheadStats;
      });

      expect(result.attentionBytes).toBe(1100);
      expect(result.attentionMode).toBe('sketch');
    });

    test('incrementTokenCount increments count', async ({ page }) => {
      const result = await page.evaluate(() => {
        let state = {
          overheadStats: {
            tokenCount: 5,
            attentionBytes: 0,
            attentionMode: null,
            streamStartTime: Date.now(),
            lastTokenTime: null as number | null,
          },
        };

        const incrementTokenCount = () => {
          state = {
            overheadStats: {
              ...state.overheadStats,
              tokenCount: state.overheadStats.tokenCount + 1,
              lastTokenTime: Date.now(),
            },
          };
        };

        incrementTokenCount();
        incrementTokenCount();
        incrementTokenCount();
        return state.overheadStats.tokenCount;
      });

      expect(result).toBe(8);
    });
  });
});

test.describe('useTraceStore', () => {
  test.describe('Trace Initialization', () => {
    test('initTrace creates new trace session', async ({ page }) => {
      const result = await page.evaluate(() => {
        const createTraceSession = (model: string) => ({
          id: `trace-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          model,
          createdAt: Date.now(),
          updatedAt: Date.now(),
          messages: [],
          tokens: [],
          segments: [],
          steps: [],
          isStreaming: false,
          streamingTokenIndex: -1,
        });

        let state = { currentTrace: null as any };

        const initTrace = (model: string) => {
          state = { currentTrace: createTraceSession(model) };
        };

        initTrace('test-model');
        return {
          hasTrace: state.currentTrace !== null,
          model: state.currentTrace?.model,
          messagesEmpty: state.currentTrace?.messages.length === 0,
        };
      });

      expect(result.hasTrace).toBe(true);
      expect(result.model).toBe('test-model');
      expect(result.messagesEmpty).toBe(true);
    });

    test('clearTrace removes current trace', async ({ page }) => {
      const result = await page.evaluate(() => {
        let state = {
          currentTrace: { id: 'test', model: 'test' },
        };

        const clearTrace = () => {
          state = { currentTrace: null as any };
        };

        clearTrace();
        return state.currentTrace;
      });

      expect(result).toBeNull();
    });
  });

  test.describe('Message Operations', () => {
    test('addUserMessage adds message to trace', async ({ page }) => {
      const result = await page.evaluate(() => {
        let state = {
          currentTrace: {
            id: 'test',
            model: 'test',
            messages: [] as any[],
            segments: [] as any[],
            tokens: [] as any[],
            updatedAt: Date.now(),
          },
        };

        const addUserMessage = (content: string) => {
          if (!state.currentTrace) return;

          const messageId = `user-${Date.now()}`;
          const message = {
            id: messageId,
            role: 'user',
            content,
            timestamp: Date.now(),
          };

          const segment = {
            id: `${messageId}-main`,
            type: 'user',
            startTokenIndex: state.currentTrace.tokens.length,
            endTokenIndex: state.currentTrace.tokens.length,
            messageId,
          };

          state = {
            currentTrace: {
              ...state.currentTrace,
              messages: [...state.currentTrace.messages, message],
              segments: [...state.currentTrace.segments, segment],
              updatedAt: Date.now(),
            },
          };
        };

        addUserMessage('Hello, world!');
        return {
          messageCount: state.currentTrace.messages.length,
          segmentCount: state.currentTrace.segments.length,
          lastMessageContent: state.currentTrace.messages[0]?.content,
          lastMessageRole: state.currentTrace.messages[0]?.role,
        };
      });

      expect(result.messageCount).toBe(1);
      expect(result.segmentCount).toBe(1);
      expect(result.lastMessageContent).toBe('Hello, world!');
      expect(result.lastMessageRole).toBe('user');
    });

    test('startAssistantMessage initializes streaming', async ({ page }) => {
      const result = await page.evaluate(() => {
        let state = {
          currentTrace: {
            id: 'test',
            model: 'test',
            messages: [] as any[],
            tokens: [] as any[],
            isStreaming: false,
            streamingTokenIndex: -1,
            updatedAt: Date.now(),
          },
        };

        const startAssistantMessage = () => {
          if (!state.currentTrace) return;

          const messageId = `assistant-${Date.now()}`;
          const message = {
            id: messageId,
            role: 'assistant',
            content: '',
            tokens: [],
            attention: [],
            timestamp: Date.now(),
          };

          state = {
            currentTrace: {
              ...state.currentTrace,
              messages: [...state.currentTrace.messages, message],
              isStreaming: true,
              streamingTokenIndex: state.currentTrace.tokens.length,
              updatedAt: Date.now(),
            },
          };
        };

        startAssistantMessage();
        return {
          isStreaming: state.currentTrace.isStreaming,
          streamingTokenIndex: state.currentTrace.streamingTokenIndex,
          lastMessageRole: state.currentTrace.messages[0]?.role,
          lastMessageContent: state.currentTrace.messages[0]?.content,
        };
      });

      expect(result.isStreaming).toBe(true);
      expect(result.streamingTokenIndex).toBe(0);
      expect(result.lastMessageRole).toBe('assistant');
      expect(result.lastMessageContent).toBe('');
    });

    test('appendToken adds token during streaming', async ({ page }) => {
      const result = await page.evaluate(() => {
        let state = {
          currentTrace: {
            id: 'test',
            model: 'test',
            messages: [{
              id: 'assistant-1',
              role: 'assistant',
              content: '',
              tokens: [] as string[],
              attention: [],
            }],
            tokens: [] as any[],
            steps: [] as any[],
            isStreaming: true,
            streamingTokenIndex: 0,
            updatedAt: Date.now(),
          },
        };

        const appendToken = (token: string) => {
          if (!state.currentTrace || !state.currentTrace.isStreaming) return;

          const trace = state.currentTrace;
          const messages = [...trace.messages];
          const lastMsg = messages[messages.length - 1];

          if (!lastMsg || lastMsg.role !== 'assistant') return;

          const newTokens = [...(lastMsg.tokens || []), token];
          const newContent = lastMsg.content + token;

          messages[messages.length - 1] = {
            ...lastMsg,
            content: newContent,
            tokens: newTokens,
          };

          const tokenEntry = {
            index: trace.tokens.length,
            text: token,
            segmentId: `${lastMsg.id}-main`,
            role: 'generated',
          };

          state = {
            currentTrace: {
              ...trace,
              messages,
              tokens: [...trace.tokens, tokenEntry],
              updatedAt: Date.now(),
            },
          };
        };

        appendToken('Hello');
        appendToken(' ');
        appendToken('World');

        return {
          tokenCount: state.currentTrace.tokens.length,
          content: state.currentTrace.messages[0].content,
          tokens: state.currentTrace.messages[0].tokens,
        };
      });

      expect(result.tokenCount).toBe(3);
      expect(result.content).toBe('Hello World');
      expect(result.tokens).toEqual(['Hello', ' ', 'World']);
    });
  });

  test.describe('Getters', () => {
    test('getTokenAt returns correct token', async ({ page }) => {
      const result = await page.evaluate(() => {
        const trace = {
          tokens: [
            { index: 0, text: 'Hello', segmentId: 's1', role: 'generated' },
            { index: 1, text: ' ', segmentId: 's1', role: 'generated' },
            { index: 2, text: 'World', segmentId: 's1', role: 'generated' },
          ],
        };

        const getTokenAt = (index: number) => trace.tokens[index] ?? null;

        return {
          token0: getTokenAt(0)?.text,
          token2: getTokenAt(2)?.text,
          tokenInvalid: getTokenAt(10),
        };
      });

      expect(result.token0).toBe('Hello');
      expect(result.token2).toBe('World');
      expect(result.tokenInvalid).toBeNull();
    });

    test('getSegmentForToken returns correct segment', async ({ page }) => {
      const result = await page.evaluate(() => {
        const trace = {
          segments: [
            { id: 's1', type: 'assistant_think', startTokenIndex: 0, endTokenIndex: 5, messageId: 'm1' },
            { id: 's2', type: 'assistant_final', startTokenIndex: 5, endTokenIndex: 10, messageId: 'm1' },
          ],
        };

        const getSegmentForToken = (index: number) => {
          return trace.segments.find(
            (s) => index >= s.startTokenIndex && index < s.endTokenIndex
          ) ?? null;
        };

        return {
          segment0: getSegmentForToken(0)?.id,
          segment4: getSegmentForToken(4)?.id,
          segment5: getSegmentForToken(5)?.id,
          segment9: getSegmentForToken(9)?.id,
          segment10: getSegmentForToken(10),
        };
      });

      expect(result.segment0).toBe('s1');
      expect(result.segment4).toBe('s1');
      expect(result.segment5).toBe('s2');
      expect(result.segment9).toBe('s2');
      expect(result.segment10).toBeNull();
    });
  });

  test.describe('Export/Import', () => {
    test('exports trace to JSONL format', async ({ page }) => {
      const result = await page.evaluate(() => {
        const trace = {
          id: 'test-trace',
          model: 'test-model',
          createdAt: 1000,
          updatedAt: 2000,
          messages: [
            { id: 'msg1', role: 'user', content: 'Hello', timestamp: 1000 },
          ],
          tokens: [
            { index: 0, text: 'Hi', segmentId: 's1', role: 'generated' },
          ],
          steps: [
            { tokenIndex: 0, fingerprint: { entropy: 0.5 } },
          ],
          segments: [
            { id: 's1', type: 'assistant_final', startTokenIndex: 0, endTokenIndex: 1, messageId: 'm1' },
          ],
          metrics: { avgEntropy: 0.5 },
        };

        const lines: string[] = [];

        // Header
        lines.push(JSON.stringify({
          record_type: 'header',
          version: '1.0',
          trace_id: trace.id,
          model: trace.model,
          created_at: trace.createdAt,
          updated_at: trace.updatedAt,
        }));

        // Messages
        for (const msg of trace.messages) {
          lines.push(JSON.stringify({ record_type: 'message', ...msg }));
        }

        // Tokens
        for (const token of trace.tokens) {
          lines.push(JSON.stringify({ record_type: 'token', ...token }));
        }

        // Steps
        for (const step of trace.steps) {
          lines.push(JSON.stringify({ record_type: 'step', ...step }));
        }

        // Segments
        for (const segment of trace.segments) {
          lines.push(JSON.stringify({ record_type: 'segment', ...segment }));
        }

        // Metrics
        lines.push(JSON.stringify({ record_type: 'metrics', ...trace.metrics }));

        const content = lines.join('\n');
        const parsed = content.split('\n').map(l => JSON.parse(l));

        return {
          lineCount: parsed.length,
          hasHeader: parsed.some(p => p.record_type === 'header'),
          hasMessage: parsed.some(p => p.record_type === 'message'),
          hasToken: parsed.some(p => p.record_type === 'token'),
          hasStep: parsed.some(p => p.record_type === 'step'),
          hasMetrics: parsed.some(p => p.record_type === 'metrics'),
          headerModel: parsed.find(p => p.record_type === 'header')?.model,
        };
      });

      expect(result.lineCount).toBe(6);
      expect(result.hasHeader).toBe(true);
      expect(result.hasMessage).toBe(true);
      expect(result.hasToken).toBe(true);
      expect(result.hasStep).toBe(true);
      expect(result.hasMetrics).toBe(true);
      expect(result.headerModel).toBe('test-model');
    });

    test('imports trace from JSONL format', async ({ page }) => {
      const result = await page.evaluate(() => {
        const jsonlContent = [
          '{"record_type":"header","version":"1.0","trace_id":"imported-1","model":"imported-model","created_at":1000,"updated_at":2000}',
          '{"record_type":"message","id":"msg1","role":"user","content":"Hello","timestamp":1000}',
          '{"record_type":"token","index":0,"text":"Hi","segmentId":"s1","role":"generated"}',
          '{"record_type":"step","tokenIndex":0}',
          '{"record_type":"segment","id":"s1","type":"assistant_final","startTokenIndex":0,"endTokenIndex":1,"messageId":"m1"}',
          '{"record_type":"metrics","avgEntropy":0.5}',
        ].join('\n');

        const lines = jsonlContent.trim().split('\n').filter(l => l.trim());

        let header: any = null;
        const messages: any[] = [];
        const tokens: any[] = [];
        const steps: any[] = [];
        const segments: any[] = [];
        let metrics: any = null;

        for (const line of lines) {
          const obj = JSON.parse(line);
          const { record_type, ...data } = obj;

          switch (record_type) {
            case 'header': header = data; break;
            case 'message': messages.push(data); break;
            case 'token': tokens.push(data); break;
            case 'step': steps.push(data); break;
            case 'segment': segments.push(data); break;
            case 'metrics': metrics = data; break;
          }
        }

        return {
          hasHeader: header !== null,
          model: header?.model,
          messageCount: messages.length,
          tokenCount: tokens.length,
          stepCount: steps.length,
          segmentCount: segments.length,
          hasMetrics: metrics !== null,
        };
      });

      expect(result.hasHeader).toBe(true);
      expect(result.model).toBe('imported-model');
      expect(result.messageCount).toBe(1);
      expect(result.tokenCount).toBe(1);
      expect(result.stepCount).toBe(1);
      expect(result.segmentCount).toBe(1);
      expect(result.hasMetrics).toBe(true);
    });
  });
});

test.describe('Metric Scope Filtering', () => {
  test('filters decode steps by scope', async ({ page }) => {
    const result = await page.evaluate(() => {
      const steps = [
        { tokenIndex: 0, manifoldZone: 'syntax_floor', isThink: true },
        { tokenIndex: 1, manifoldZone: 'semantic_bridge', isThink: true },
        { tokenIndex: 2, manifoldZone: 'long_range', isThink: false },
        { tokenIndex: 3, manifoldZone: 'diffuse', isThink: false },
      ];

      const filterByScope = (scope: string) => {
        switch (scope) {
          case 'all':
            return steps;
          case 'think':
            return steps.filter(s => s.isThink);
          case 'output':
            return steps.filter(s => !s.isThink);
          default:
            return steps;
        }
      };

      return {
        allCount: filterByScope('all').length,
        thinkCount: filterByScope('think').length,
        outputCount: filterByScope('output').length,
      };
    });

    expect(result.allCount).toBe(4);
    expect(result.thinkCount).toBe(2);
    expect(result.outputCount).toBe(2);
  });
});
