#!/usr/bin/env python3
"""
fm_langchain.py — LangChain + LangGraph Orchestration Agent (Port 7801)
Pipelines: map-reduce summarise, RAG, research, code review, strategy.
Calls OpenRouter agent (7791) for LLM — no LangChain install required.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os, json, time, signal, sqlite3, logging, urllib.request
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _l in _vault.read_text().splitlines():
        if "=" in _l and not _l.startswith("#"):
            _k, _, _v = _l.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

PORT       = int(os.getenv("LANGCHAIN_PORT", "7801"))
ROOT       = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB         = ROOT / "database" / "sovereign.db"
LOG        = ROOT / "logs" / "langchain.log"
OR_URL     = os.getenv("OPENROUTER_URL", "http://127.0.0.1:7791")
LS_URL     = os.getenv("LANGSMITH_URL",  "http://127.0.0.1:7803")

for p in (ROOT, LOG.parent, DB.parent): p.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [LANGCHAIN] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()])
log = logging.getLogger("langchain")

def _db_init():
    c = sqlite3.connect(DB, timeout=10); c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS langchain_runs (
        id INTEGER PRIMARY KEY, pipeline TEXT, input_len INT, output_len INT,
        steps INT, latency_ms REAL, status TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.commit(); c.close()

def _db_log(pipeline, input_len, output_len, steps, latency_ms, status):
    try:
        c = sqlite3.connect(DB, timeout=5)
        c.execute("INSERT INTO langchain_runs (pipeline,input_len,output_len,steps,latency_ms,status) VALUES (?,?,?,?,?,?)",
                  (pipeline, input_len, output_len, steps, latency_ms, status))
        c.commit(); c.close()
    except Exception as e: log.warning("db: %s", e)

# ── LLM call via OpenRouter ───────────────────────────────────────────────────

def _llm(prompt: str, system: str = "", task: str = "draft", tier: str = "balanced") -> str:
    payload = json.dumps({"task":task,"tier":tier,"prompt":prompt,
                          "system":system,"max_tokens":1024,"cache":True}).encode()
    req = urllib.request.Request(f"{OR_URL}/route", data=payload,
        headers={"Content-Type":"application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read()).get("content","")
    except Exception as e:
        return f"[LLM unavailable: {e}]"

# ── StateGraph implementation (LangGraph-compatible) ─────────────────────────

class StateGraph:
    def __init__(self, state_schema: type = dict):
        self.nodes:  dict[str, callable] = {}
        self.edges:  dict[str, str]      = {}
        self.cond:   dict[str, tuple]    = {}  # node → (fn, {result: next_node})
        self.entry:  str                 = ""

    def add_node(self, name: str, fn: callable):
        self.nodes[name] = fn

    def add_edge(self, src: str, dst: str):
        self.edges[src] = dst

    def add_conditional_edges(self, src: str, condition_fn: callable, mapping: dict):
        self.cond[src] = (condition_fn, mapping)

    def set_entry_point(self, name: str):
        self.entry = name

    def invoke(self, initial_state: dict) -> dict:
        state    = dict(initial_state)
        current  = self.entry
        visited  = []
        max_steps = 20
        while current and current != "__end__" and len(visited) < max_steps:
            visited.append(current)
            fn    = self.nodes.get(current)
            if fn: state = fn(state)
            if current in self.cond:
                cond_fn, mapping = self.cond[current]
                result  = cond_fn(state)
                current = mapping.get(result, "__end__")
            else:
                current = self.edges.get(current, "__end__")
        state["__visited__"] = visited
        return state

# ── built-in pipelines ────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 80) -> list[str]:
    words = text.split(); chunks = []; i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size])); i += chunk_size - overlap
    return chunks or [text]

def chain_summarize(text: str) -> dict:
    """Map-reduce summarisation."""
    chunks    = _chunk_text(text)
    summaries = [_llm(f"Summarise this passage in 3 sentences:\n{c}", task="summarize", tier="fast") for c in chunks]
    combined  = "\n".join(summaries)
    final     = _llm(f"Combine these summaries into one coherent summary:\n{combined}", task="summarize")
    return {"pipeline":"summarize","chunks":len(chunks),"summary":final}

def chain_rag(query: str, context_docs: list) -> dict:
    """RAG: inject context, generate answer."""
    context = "\n\n".join(f"[Doc {i+1}]: {d}" for i, d in enumerate(context_docs[:5]))
    answer  = _llm(f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based only on the context.",
                   system="You are a precise assistant. Use only the provided context.", task="extract")
    return {"pipeline":"rag","query":query,"docs_used":len(context_docs[:5]),"answer":answer}

def chain_research(topic: str) -> dict:
    """Multi-step research synthesis."""
    queries  = _llm(f"Generate 5 specific research questions about: {topic}. Return as numbered list.", task="draft", tier="fast")
    synthesis = _llm(f"You are a researcher. Synthesise what is known about: {topic}\n\nKey questions to address:\n{queries}",
                     system="Provide structured, factual synthesis with key insights.", task="analyse")
    return {"pipeline":"research","topic":topic,"queries":queries,"synthesis":synthesis}

def chain_code_review(code: str, language: str = "python") -> dict:
    """Security + quality review."""
    security = _llm(f"Review this {language} code for security vulnerabilities:\n```{language}\n{code}\n```",
                    system="You are a security-focused code reviewer. List vulnerabilities and fixes.", task="analyse")
    quality  = _llm(f"Review this {language} code for quality, style, and performance:\n```{language}\n{code}\n```",
                    task="analyse", tier="balanced")
    return {"pipeline":"code_review","language":language,"security":security,"quality":quality}

def chain_strategy(goal: str, constraints: str = "") -> dict:
    """Multi-role strategic planning via StateGraph."""
    g = StateGraph()

    def analyst(state):
        state["analysis"] = _llm(f"Goal: {goal}\nConstraints: {constraints}\nAnalyse the opportunity and risks.", task="analyse")
        return state

    def planner(state):
        state["plan"] = _llm(f"Goal: {goal}\nAnalysis: {state.get('analysis','')}\nCreate a detailed 5-step action plan.", task="strategy", tier="premium")
        return state

    def critic(state):
        state["critique"] = _llm(f"Critique this plan and suggest improvements:\n{state.get('plan','')}", task="analyse")
        return state

    g.add_node("analyst", analyst); g.add_node("planner", planner); g.add_node("critic", critic)
    g.add_edge("analyst","planner"); g.add_edge("planner","critic")
    g.set_entry_point("analyst")
    result = g.invoke({"goal": goal})
    return {"pipeline":"strategy","goal":goal,"analysis":result.get("analysis"),
            "plan":result.get("plan"),"critique":result.get("critique"),"steps":result.get("__visited__",[])}

PIPELINES = {
    "summarize":    {"fn":chain_summarize,  "args":["text"],              "description":"Map-reduce text summarisation"},
    "rag":          {"fn":chain_rag,        "args":["query","context_docs"],"description":"Retrieval-augmented generation"},
    "research":     {"fn":chain_research,   "args":["topic"],             "description":"Multi-step research synthesis"},
    "code_review":  {"fn":chain_code_review,"args":["code","language"],   "description":"Security + quality code review"},
    "strategy":     {"fn":chain_strategy,   "args":["goal","constraints"],"description":"Multi-agent strategic planning"},
}

class LCHandler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _r(self, code, body):
        p = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(p))); self.end_headers(); self.wfile.write(p)

    def do_GET(self):
        ep = self.path.split("?")[0]
        if ep == "/health":
            try:
                with urllib.request.urlopen(f"{OR_URL}/health", timeout=2) as r: or_ok = True
            except: or_ok = False
            self._r(200, {"status":"ok","pipelines":list(PIPELINES),"openrouter_reachable":or_ok})
        elif ep == "/pipelines":
            self._r(200, {k:{"args":v["args"],"description":v["description"]} for k,v in PIPELINES.items()})
        elif ep == "/runs":
            try:
                c = sqlite3.connect(DB, timeout=5)
                rows = c.execute("SELECT * FROM langchain_runs ORDER BY ts DESC LIMIT 20").fetchall()
                c.close(); self._r(200, {"runs": rows})
            except Exception as e: self._r(500,{"error":str(e)})
        else:
            self._r(404, {"error":"not_found"})

    def do_POST(self):
        try:
            n  = int(self.headers.get("Content-Length",0)); d = json.loads(self.rfile.read(n))
            t0 = time.time(); ep = self.path.split("?")[0]

            if ep == "/run":
                name    = d.get("pipeline","")
                p_cfg   = PIPELINES.get(name)
                if not p_cfg: self._r(404,{"error":"unknown_pipeline","available":list(PIPELINES)}); return
                kwargs  = {k: d.get(k, d.get("input","")) for k in p_cfg["args"]}
                # fix list arg
                if "context_docs" in kwargs and isinstance(kwargs["context_docs"], str):
                    kwargs["context_docs"] = [kwargs["context_docs"]]
                result  = p_cfg["fn"](**kwargs)
                lat     = (time.time()-t0)*1000
                _db_log(name, len(str(kwargs)), len(str(result)), 1, lat, "ok")
                log.info("pipeline=%s latency=%.0fms", name, lat)
                self._r(200, result)

            elif ep == "/chain":
                steps  = d.get("steps", [])  # [{prompt, role, system}]
                inputs = d.get("inputs","")
                state  = inputs; outputs = []
                for s in steps:
                    prompt = s.get("prompt","").format(input=state)
                    state  = _llm(prompt, system=s.get("system",""), task=s.get("task","draft"))
                    outputs.append({"role":s.get("role",""),"output":state})
                lat = (time.time()-t0)*1000
                _db_log("chain", len(str(inputs)), len(str(outputs)), len(steps), lat, "ok")
                self._r(200, {"steps":len(steps),"outputs":outputs,"final":state})

            elif ep == "/graph":
                nodes  = d.get("nodes",{})
                edges  = d.get("edges",{})
                entry  = d.get("entry","")
                state  = d.get("initial_state",{})
                g      = StateGraph()
                for name, prompt_tpl in nodes.items():
                    def make_node(tpl):
                        def node_fn(s): s[name+"_out"] = _llm(tpl.format(**s)); return s
                        return node_fn
                    g.add_node(name, make_node(prompt_tpl))
                for src, dst in edges.items(): g.add_edge(src, dst)
                g.set_entry_point(entry)
                result = g.invoke(state)
                lat    = (time.time()-t0)*1000
                _db_log("graph", len(str(state)), len(str(result)), len(result.get("__visited__",[])), lat, "ok")
                self._r(200, result)

            else:
                self._r(404,{"error":"unknown_path"})
        except json.JSONDecodeError: self._r(400,{"error":"invalid_json"})
        except Exception as e: log.error("handler: %s",e); self._r(500,{"error":str(e)})

_running = True
def _shutdown(*_): global _running; _running = False
signal.signal(signal.SIGTERM, _shutdown); signal.signal(signal.SIGINT, _shutdown)

def main():
    _db_init(); server = HTTPServer(("0.0.0.0", PORT), LCHandler)
    log.info("LangChain/LangGraph agent on port %d | pipelines=%d", PORT, len(PIPELINES))
    try:
        while _running: server.handle_request()
    finally: server.server_close()

if __name__ == "__main__": main()
