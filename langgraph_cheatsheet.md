# LangGraph Cheat Sheet

Quick-reference for everything in the *From Zero to Professional Agentic AI Developer* guide. Jump to a section, copy the pattern, adapt.

---

## 1. Mental Model

- **Agent** = observe → decide action → act (usually tool call) → observe result → loop until done.
- **Workflow** = developer hard-codes the sequence.
- **Graph, not chain** — plain chains (A → B → C) can't express loops, branches, retries, or "pause for human." LangGraph models the agent as a state machine: nodes (circles) + edges (arrows) + shared state (the notebook everyone reads/writes).
- **ReAct loop** — `agent (LLM decides) → tool call? → tools (run) → back to agent → ... → done → END`. Backbone of almost every agent you'll build.

### The four primitives (know cold)

| Primitive | What it is | Analogy |
|---|---|---|
| **State** | Typed schema (dict-like) shared by all nodes | The shared notebook |
| **Node** | `(state) -> partial_state_update` function | One step of work |
| **Edge** | Rule for "what runs next" (fixed or conditional) | The arrows |
| **Graph** | Nodes + edges compiled into a runnable | The whole machine |

### Ecosystem

- **LangChain** — component library (model wrappers, tool decorators, prompts) + high-level `create_agent`.
- **LangGraph** — low-level graph runtime: durable execution, persistence, streaming, HITL, multi-agent.
- **LangSmith** — observability/tracing (optional, invaluable once non-trivial).
- **Deep Agents** — higher-level package on LangGraph for planning + subagents + filesystem tools.
- You don't need LangChain to use LangGraph, but in practice you'll use `langchain_core` (tools, message types) + a chat model wrapper.

---

## 2. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U langgraph langchain langchain-openai langgraph-checkpoint-sqlite langgraph-supervisor
export OPENAI_API_KEY="sk-..."
# or export ANTHROPIC_API_KEY="..." with langchain-anthropic
```

Smoke test:
```python
from langchain.chat_models import init_chat_model
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
print(llm.invoke("Say 'LangGraph is up and running' and nothing else.").content)
```

`init_chat_model` is the model-agnostic entry point — swap the string, graph code never changes.

Optional tracing:
```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="ls-..."
```

---

## 3. Core Syntax Deep Dive

### 3.1 Defining State

```python
from typing import TypedDict
class State(TypedDict):
    question: str
    answer: str
```

- Default reducer = **overwrite** ("last write wins"). Node returns only the keys it's changing.
- **Custom reducer** — accumulate instead of overwrite:
```python
from typing import Annotated
import operator
class State(TypedDict):
    question: str
    logs: Annotated[list[str], operator.add]   # concatenates lists
```
Or write your own:
```python
def keep_max(existing: int, new: int) -> int:
    return max(existing, new)
class State(TypedDict):
    best_score: Annotated[int, keep_max]
```

- **MessagesState** — built-in chat schema, already has `messages: Annotated[list[AnyMessage], add_messages]`. `add_messages` appends new messages, overwrites in place if same `id` (streaming), and auto-converts plain dicts (`{"role": "user", "content": "hi"}`) into message objects.
```python
from langgraph.graph import MessagesState
class State(MessagesState):
    user_id: str   # add your own fields on top
```

### 3.2 Nodes

- A node = `(state) -> dict`. Optionally also takes `config` and a `Runtime` object.
- **Rules:** never mutate state in place and return nothing — always return the partial update. A node can return `{}`. Function name becomes default node name in `add_node`, but explicit string names are clearer.

```python
def research(state: State) -> dict:
    q = state["question"]
    return {"answer": f"researched: {q}"}
```

### 3.3 Building the graph — StateGraph

```python
from langgraph.graph import StateGraph, START, END
builder = StateGraph(State)
builder.add_node("research", research)
builder.add_node("summarize", summarize)
builder.add_edge(START, "research")
builder.add_edge("research", "summarize")
builder.add_edge("summarize", END)
graph = builder.compile()   # REQUIRED before running — validates reachability, catches typos
```

Run modes:
```python
graph.invoke(inputs)                 # blocks, returns final state
graph.stream(inputs)                 # yields incrementally (Part 7)
await graph.ainvoke(inputs)          # async
await graph.astream(inputs)          # async streaming
```

### 3.4 Conditional edges — branching

```python
def route(state: State) -> str:
    if len(state["answer"]) < 10:
        return "retry"
    return "done"

builder.add_conditional_edges(
    "research", route,
    {"retry": "research", "done": "summarize"},   # optional mapping dict
)
```
This retry-loop-back-to-self pattern is how you build the ReAct agent↔tools loop with zero `while` loops in your own code — the graph *is* the loop.

### 3.5 Command — routing + state update in one object

```python
from typing import Literal
from langgraph.types import Command

def research(state: State) -> Command[Literal["summarize", "research"]]:
    answer = do_research(state["question"])
    if len(answer) < 10:
        return Command(update={"answer": answer}, goto="research")
    return Command(update={"answer": answer}, goto="summarize")
```
When a node returns `Command`, you still `add_node` it normally but **don't add outgoing edges** — routing happens at runtime. Heavily used in multi-agent handoffs (Part 8) and HITL (Part 6).

### 3.6 Parallel fan-out with Send

```python
from langgraph.types import Send

def plan(state: State) -> list[Send]:
    return [Send("research_one", {"topic": t}) for t in state["subtopics"]]

builder.add_conditional_edges("plan", plan)
```
Each `Send` runs the target node once per item, in parallel, isolated input; outputs merge back via whatever reducer you defined (why `operator.add`-style reducers matter for parallel branches).

### 3.7 Full runnable example (self-critiquing loop)

```python
from typing import Annotated, TypedDict
import operator
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

class State(TypedDict):
    topic: str
    draft: str
    critique: str
    attempts: Annotated[int, operator.add]

def write(state: State) -> dict:
    prompt = f"Write one punchy sentence about: {state['topic']}"
    if state.get("critique"):
        prompt += f"\nAddress this critique: {state['critique']}"
    return {"draft": llm.invoke(prompt).content, "attempts": 1}

def critique(state: State) -> dict:
    prompt = f"Critique this sentence in 5 words or fewer, or say 'GOOD':\n{state['draft']}"
    return {"critique": llm.invoke(prompt).content}

def should_continue(state: State) -> str:
    if "GOOD" in state["critique"] or state["attempts"] >= 3:
        return END
    return "write"

builder = StateGraph(State)
builder.add_node("write", write)
builder.add_node("critique", critique)
builder.add_edge(START, "write")
builder.add_edge("write", "critique")
builder.add_conditional_edges("critique", should_continue, {"write": "write", END: END})
graph = builder.compile()

final = graph.invoke({"topic": "LangGraph", "attempts": 0})
print(final["draft"], f"took {final['attempts']} attempt(s)")
```

**Drill 3:** Build a 2-node graph `State{n: int, history: Annotated[list[int], operator.add]}` where node `double` doubles `n` and appends it to `history`, looping until `n > 100`, then ends. Print `history`.

---

## 4. Tools & the ReAct Loop

### 4.1 Defining a tool

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"It's sunny in {city}."
```
Docstring isn't decoration — the LLM reads it to decide when/how to call the tool. Type hints define the JSON schema.

### 4.2 Binding tools to a model

```python
tools = [get_weather, convert_currency]
llm_with_tools = llm.bind_tools(tools)
response = llm_with_tools.invoke("What's the weather in Tokyo?")
print(response.tool_calls)
# [{'name': 'get_weather', 'args': {'city': 'Tokyo'}, 'id': '...'}]
```
`bind_tools` doesn't execute anything — just tells the model these functions exist.

### 4.3 ToolNode

```python
from langgraph.prebuilt import ToolNode
tool_node = ToolNode(tools)
```
Expects state with a `messages` key where the last message is an `AIMessage` with `tool_calls`. Runs every requested tool (parallel if several), returns `{"messages": [ToolMessage(...), ...]}`.

### 4.4 Hand-written ReAct loop (memorize this shape)

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
tools = [add]
llm_with_tools = llm.bind_tools(tools)

def call_model(state: MessagesState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END

builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")
graph = builder.compile()

result = graph.invoke({"messages": [{"role": "user", "content": "what's 17 plus 25?"}]})
print(result["messages"][-1].content)
```

### 4.5 The shortcut: create_agent

```python
from langchain.agents import create_agent
agent = create_agent(
    model="gpt-4o-mini",
    tools=[add, get_weather],
    system_prompt="You are a precise, concise assistant. Use tools when needed.",
)
result = agent.invoke({"messages": [{"role": "user", "content": "Weather in Tokyo, then add 2+2"}]})
```
Note: `langgraph.prebuilt.create_react_agent` is the older name for the same job; `create_agent` (in `langchain`) is now recommended. Both return a compiled graph — `.invoke`, `.stream`, checkpointers, interrupts all apply unchanged. **Use `create_agent` for standard loops; drop to raw `StateGraph` for custom branching, multi-agent, or non-standard state.**

**Drill 4:** Write `@tool square(n: int) -> int`. Build a hand-written ReAct graph (not `create_agent`) answering "what is 9 squared, then add 10 to it," requiring two sequential tool calls.

---

## 5. Persistence & Memory

Without persistence, every `.invoke()` starts blank and a crash mid-run loses everything. A **checkpointer** snapshots full state after every node.

### 5.1 Checkpointers

```python
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()          # dev/testing only, lost on process exit
graph = builder.compile(checkpointer=checkpointer)

from langgraph.checkpoint.sqlite import SqliteSaver
# with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer: ...

from langgraph.checkpoint.postgres import PostgresSaver
# checkpointer = PostgresSaver.from_conn_string("postgresql://...")
```

### 5.2 Threads — unit of persisted conversation

```python
config = {"configurable": {"thread_id": "user-42-session-1"}}
graph.invoke({"messages": [{"role": "user", "content": "My name is Sam."}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, config)
# -> "Your name is Sam." — messages accumulated under this thread_id
```
Change `thread_id` → completely fresh, isolated conversation. Typical key: `thread_id = f"{user_id}:{conversation_id}"`.

> ⚠️ **Gotcha:** re-invoking with the *same* `thread_id` loads and appends to the entire prior conversation — it doesn't start fresh. If you want a clean run, use a new `thread_id`.

### 5.3 Inspecting/editing state

```python
snapshot = graph.get_state(config)
print(snapshot.values)   # current state dict
print(snapshot.next)     # which node runs next (empty tuple if finished)

graph.update_state(config, {"messages": [{"role": "system", "content": "Be brief."}]})
```

### 5.4 Time travel

```python
history = list(graph.get_state_history(config))   # newest first
older_checkpoint = history[3]
graph.invoke(None, {"configurable": {
    "thread_id": "user-42-session-1",
    "checkpoint_id": older_checkpoint.config["configurable"]["checkpoint_id"],
}})
```
Basis for "undo," debugging tools, A/B branching from a fixed reasoning point.

### 5.5 Long-term memory — Store

Checkpointers persist *one thread's* history. A `Store` persists facts **across every future thread** (e.g., "user prefers metric units"), independent of `thread_id`.

```python
from langgraph.store.memory import InMemoryStore
store = InMemoryStore()
graph = builder.compile(checkpointer=checkpointer, store=store)

from langgraph.store.base import BaseStore
def remember_preference(state: MessagesState, config: dict, *, store: BaseStore) -> dict:
    user_id = config["configurable"]["user_id"]
    namespace = ("preferences", user_id)
    store.put(namespace, "units", {"value": "metric"})
    item = store.get(namespace, "units")
    return {}
```
Some Store implementations support semantic search (`store.search(namespace, query="...")`) — embedding-based recall without exact keys.

> **Rule of thumb:** checkpointer = memory *within* a conversation (short-term). Store = memory *across* conversations (long-term, per-user).

**Drill 5:** Take Part 4's ReAct agent, compile with `InMemorySaver`, have a two-turn conversation on the same `thread_id` where turn 2 references only what was said in turn 1.

---

## 6. Human-in-the-Loop (HITL)

Real agents need to pause before irreversible/high-stakes actions and wait for a human. `interrupt()` is a first-class, checkpoint-backed primitive.

> **Requirement:** a checkpointer is **mandatory** for `interrupt()` — pausing means "serialize state and stop," and there must be somewhere to serialize to.

### 6.1 interrupt() — pausing a node

```python
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

def approval_node(state: State) -> dict:
    decision = interrupt({
        "question": "Approve sending this email?",
        "preview": state["draft_email"],
    })
    return {"approved": decision == "approve"}
```
When execution hits `interrupt(...)`:
1. Raises an internal exception that unwinds cleanly (**don't** wrap in try/except — you'll swallow the pause).
2. Serializes the entire state at that point.
3. Returns control to the caller, with your payload surfaced under `__interrupt__`.

```python
config = {"configurable": {"thread_id": "email-123"}}
result = graph.invoke({"draft_email": "Hi team, ..."}, config)
print(result["__interrupt__"])
# [Interrupt(value={'question': ..., 'preview': ...}, ...)]
```

### 6.2 Resuming — Command(resume=...)

```python
result = graph.invoke(Command(resume="approve"), config)
```
`Command(resume=...)` is passed as graph input in place of a normal state dict. Whatever you pass becomes `interrupt()`'s return value inside the node — execution continues from there.

> ⚠️ **Critical gotcha:** on resume, LangGraph **re-runs the interrupted node from the top**, not from the exact `interrupt()` line. Any code *before* `interrupt()` in that node re-executes too. Keep everything before `interrupt()` side-effect-free. If a node has multiple `interrupt()` calls, keep them in a fixed, deterministic order — never inside a conditional that might skip differently across runs (resume matches interrupts positionally against the replay).

### 6.3 Pattern: approve / reject / edit a tool call

```python
from typing import Literal
from langgraph.types import Command, interrupt

def review_tool_call(state: MessagesState) -> Command[Literal["tools", "agent"]]:
    last = state["messages"][-1]
    decision = interrupt({
        "question": "Approve this tool call?",
        "tool_calls": last.tool_calls,
    })
    if decision["type"] == "approve":
        return Command(goto="tools")
    if decision["type"] == "edit":
        last.tool_calls[0]["args"] = decision["args"]
        return Command(update={"messages": [last]}, goto="tools")
    # reject
    return Command(
        update={"messages": [{"role": "tool", "tool_call_id": last.tool_calls[0]["id"],
                               "content": "Rejected by human."}]},
        goto="agent",
    )

builder.add_node("review", review_tool_call)
builder.add_edge("agent", "review")   # instead of agent -> tools directly
```
Resume with structured data matching what the node expects:
```python
graph.invoke(Command(resume={"type": "approve"}), config)
graph.invoke(Command(resume={"type": "edit", "args": {"city": "Osaka"}}), config)
graph.invoke(Command(resume={"type": "reject"}), config)
```

### 6.4 Simpler alternative: interrupt_before

```python
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["tools"])
```
Coarse-grained "always pause before this node." No payload, no per-run conditional logic — fastest way to add a blanket approval gate, but less flexible than explicit `interrupt()`.

**Drill 6:** Add a review node to your Part 4 ReAct graph that pauses before every tool call and only proceeds on `Command(resume="approve")`; test both approve and reject paths in the same thread.

---

## 7. Streaming

`.invoke()` blocks until the whole graph finishes. `.stream()` yields incrementally.

| `stream_mode` | Yields |
|---|---|
| `"values"` | Full state snapshot after each node |
| `"updates"` | Only the partial dict each node returned |
| `"messages"` | Token-by-token LLM output (for chat UIs) |
| `"custom"` | Arbitrary values emitted via `get_stream_writer()` |

```python
for chunk in graph.stream({"messages": [...]}, config, stream_mode="values"):
    print(chunk)

for mode, chunk in graph.stream(inputs, config, stream_mode=["updates", "messages"]):
    if mode == "messages":
        token, metadata = chunk
        print(token.content, end="", flush=True)
    else:
        print("\n[state update]", chunk)
```

Custom progress events:
```python
from langgraph.config import get_stream_writer

def crawl(state: State) -> dict:
    writer = get_stream_writer()
    for i, url in enumerate(state["urls"]):
        writer({"progress": f"{i+1}/{len(state['urls'])}"})
        fetch(url)
    return {"done": True}
```
Consume with `stream_mode="custom"`.

Async twin: `await graph.astream(inputs, config, stream_mode="values")` — needed inside FastAPI/async servers.

**Drill 7:** Take the Part 5 memory-enabled agent and stream with `stream_mode="messages"`, printing tokens as they arrive.

---

## 8. Multi-Agent Systems

One agent with 20 tools + a giant prompt degrades fast (wrong tool picks, lost context, undebuggable). Fix: split into specialists, each with a narrow toolset/prompt, coordinated by explicit structure.

### 8.1 The handoff primitive

Everything (supervisor, network, hierarchical) is built from: **a node returns `Command(goto=<other_agent>, update=<state>)`.**

```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

def make_handoff_tool(agent_name: str):
    @tool(f"transfer_to_{agent_name}", description=f"Hand off the task to {agent_name}.")
    def handoff(state: Annotated[dict, InjectedState],
                tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        tool_message = {
            "role": "tool", "content": f"Transferred to {agent_name}",
            "name": f"transfer_to_{agent_name}", "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,
            update={"messages": [tool_message]},
            graph=Command.PARENT,   # jump out of this agent's subgraph into the parent graph
        )
    return handoff
```
`graph=Command.PARENT` matters whenever each agent is itself a compiled subgraph (the normal case with `create_agent`) — tells LangGraph "route in the outer graph, not inside my own local graph."

### 8.2 Pattern: supervisor

Supervisor LLM never does the work — only decides which specialist acts next.

```python
from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

research_agent = create_agent(llm, tools=[web_search], name="research_agent",
    system_prompt="You research facts. Do not write code. Report findings only.")
coder_agent = create_agent(llm, tools=[run_python], name="coder_agent",
    system_prompt="You write and run Python. Do not do open-ended research.")

workflow = create_supervisor(
    [research_agent, coder_agent], model=llm,
    prompt=("You supervise a research_agent and a coder_agent. "
            "Route research questions to research_agent, coding tasks to coder_agent. "
            "Assign one agent at a time. Do not do the work yourself."),
)
app = workflow.compile(checkpointer=InMemorySaver())
result = app.invoke({"messages": [{"role": "user", "content": "Find Python's creation year, then print it doubled."}]})
```
`create_supervisor` auto-generates handoff tools, wires each worker as a node, and routes back to the supervisor after each worker finishes (`add_handoff_back_messages=True` default).

### 8.3 Hand-rolled supervisor with StateGraph

For custom logic (non-LLM router, custom shared state fields) the prebuilt helper doesn't expose:

```python
from typing import Literal
from langgraph.graph import StateGraph, MessagesState, START

class TeamState(MessagesState):
    next_agent: str

def supervisor_node(state: TeamState) -> Command[Literal["research_agent", "coder_agent", "__end__"]]:
    decision = llm.invoke([
        {"role": "system", "content": "Reply with exactly one word: research_agent, coder_agent, or done."},
        *state["messages"],
    ]).content.strip()
    if decision == "done":
        return Command(goto="__end__")
    return Command(goto=decision)

builder = StateGraph(TeamState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("research_agent", research_agent)
builder.add_node("coder_agent", coder_agent)
builder.add_edge(START, "supervisor")
builder.add_edge("research_agent", "supervisor")
builder.add_edge("coder_agent", "supervisor")
graph = builder.compile()
```

### 8.4 Pattern: network (peer-to-peer, no central supervisor)

```python
addition_expert = create_agent(llm, tools=[add, make_handoff_tool("multiplication_expert")],
    name="addition_expert", system_prompt="You only do addition. Hand off multiplication questions.")
multiplication_expert = create_agent(llm, tools=[multiply, make_handoff_tool("addition_expert")],
    name="multiplication_expert", system_prompt="You only do multiplication. Hand off addition questions.")

builder = StateGraph(MessagesState)
builder.add_node("addition_expert", addition_expert)
builder.add_node("multiplication_expert", multiplication_expert)
builder.add_edge(START, "addition_expert")
graph = builder.compile()
```

### When to use which

- **Supervisor** — anything you need to reason about, debug, or guardrail (most production systems).
- **Network** — agents are true peers, no natural coordinator.
- **Hierarchical** (supervisor-of-supervisors) — once one supervisor's tool list gets unwieldy; nest another `create_supervisor` team as one "agent" under a top-level supervisor.

**Drill 8:** Build a 3-agent supervisor team: `research_agent` (web_search), `writer_agent` (no tools, drafts from provided facts), `critic_agent` (no tools, returns "GOOD" or specific feedback). Route: research → writer → critic → (writer again if feedback, else done).

---

## 9. Subgraphs, Structured Output & Error Handling

### 9.1 Subgraphs

Any compiled graph is a valid node in another graph (how `create_agent`'s output plugs into `add_node` in Part 8).

```python
sub_builder = StateGraph(State)
sub_builder.add_node("step_a", step_a)
sub_builder.add_node("step_b", step_b)
sub_builder.add_edge(START, "step_a")
sub_builder.add_edge("step_a", "step_b")
sub_builder.add_edge("step_b", END)
subgraph = sub_builder.compile()

parent_builder = StateGraph(State)
parent_builder.add_node("preprocess", preprocess)
parent_builder.add_node("sub_workflow", subgraph)   # compiled graph AS a node
parent_builder.add_edge(START, "preprocess")
parent_builder.add_edge("preprocess", "sub_workflow")
parent_builder.add_edge("sub_workflow", END)
```
If schemas differ, LangGraph maps overlapping keys automatically; for fully independent schemas, wrap the subgraph call in your own translating node function.

### 9.2 Structured output

```python
from pydantic import BaseModel, Field

class RouteDecision(BaseModel):
    next_agent: str = Field(description="one of: research_agent, coder_agent, done")
    reason: str

router_llm = llm.with_structured_output(RouteDecision)

def supervisor_node(state) -> dict:
    decision = router_llm.invoke(state["messages"])
    return {"next_agent": decision.next_agent}
```
More reliable than asking the model to "reply in JSON" and hand-parsing.

### 9.3 Error handling and retries

```python
def call_flaky_api(state: State) -> dict:
    try:
        return {"result": flaky_api_call(state["query"])}
    except Exception as e:
        return {"error": str(e)}

def route_on_error(state: State) -> str:
    return "handle_error" if state.get("error") else "continue"

builder.add_conditional_edges("call_flaky_api", route_on_error,
    {"handle_error": "handle_error", "continue": "continue"})
```
LLM/tool call retries:
```python
llm = init_chat_model("gpt-4o-mini", model_provider="openai").with_retry(stop_after_attempt=3)
```
Whole-graph loop safety net:
```python
graph.invoke(inputs, {"configurable": {"thread_id": "t1"}, "recursion_limit": 25})
```
Raises `GraphRecursionError` if exceeded — critical guardrail for any loop whose exit condition might be wrong.

---

## 10. Projects — What Each One Locks In

| Project | Goal | Skills |
|---|---|---|
| **1 — Research Assistant Agent** | Single tool-using agent: web search + math, remembers across turns, streams token-by-token | `StateGraph`, `MessagesState`, tools, `ToolNode`, conditional edges, `InMemorySaver`, `thread_id`, `stream_mode="messages"` |
| **2 — HITL Email Approval Agent** | Draft email → human approve/edit/reject before "send" | `interrupt()`, `Command(resume=...)`, `Command(goto=...)`, checkpointer requirement, mid-run state editing |
| **3 — Multi-Agent Content Team** | Supervisor + research/writer/critic producing a fact-checked paragraph, critic can bounce back to writer | `create_agent`, `create_supervisor`, handoff routing, checkpointing a multi-agent graph |
| **4 — Capstone: Autonomous Ops Agent** | IT-ops assistant: diagnose (read-only tools) → propose fix → human approval before destructive action → remembers team policy across sessions → streams live | Multi-agent supervisor + tools + checkpointer/Store + `interrupt`/`Command` + streaming + structured output — all combined |

### Capstone architecture (Part 14)

```
user ───────► supervisor
              ├──► diagnostics_agent (read-only tools, no approval needed)
              └──► action_agent (has restart_service tool)
                        └──► human_approval (interrupt before any action tool runs)
```
- **Policy memory** via `Store`: `store.get(("policy", team), tool_name)` — if a team previously chose "approve_always," skip asking again. Scoped per `team_id`, not global.
- Verify: first incident pauses (`snapshot.next` non-empty); `"approve_always"` writes to store; second incident **same team** doesn't pause; new team **does** pause again (proves per-team scoping).

**Production checklist (where to take Project 4 next):**
- Swap `InMemorySaver`/`InMemoryStore` → Postgres-backed versions.
- Swap stub tools for real infra APIs (PagerDuty, Kubernetes, cloud SDK) — same graph shape.
- Put `human_approval`'s `interrupt()` payload in front of a real UI (Slack button, dashboard) calling back with `Command(resume=...)`.
- Add LangSmith tracing (`LANGSMITH_TRACING=true`) for auditability.
- Add a `recursion_limit` so a misbehaving loop can't run away.
- Consider `interrupt_before=["human_approval"]` for a coarse first version, then move to fine-grained `interrupt()` payloads as needed.

---

## 11. Syntax Cheat Sheet — Imports & One-Liners

### Imports you'll reach for constantly
```python
from typing import TypedDict, Annotated, Literal
import operator
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import interrupt, Command, Send
from langgraph.prebuilt import ToolNode, InjectedState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.config import get_stream_writer
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool, InjectedToolCallId
from langgraph_supervisor import create_supervisor
```

### State
```python
class State(TypedDict):
    field: str                                  # overwrite reducer (default)
    log: Annotated[list[str], operator.add]      # append reducer

class State(MessagesState):                      # built-in messages + add_messages reducer
    extra_field: str
```

### Graph construction
```python
builder = StateGraph(State)
builder.add_node("name", fn)
builder.add_edge(START, "name")
builder.add_edge("a", "b")
builder.add_conditional_edges("a", router_fn, {"x": "b", "y": END})
graph = builder.compile()                                            # REQUIRED before running
graph = builder.compile(checkpointer=cp, store=st, interrupt_before=["node"])

def node(state) -> Command[Literal["b", "c"]]:
    return Command(update={"k": v}, goto="b")                        # route + update in one

def planner(state) -> list[Send]:
    return [Send("worker", {"item": x}) for x in state["items"]]     # parallel fan-out
```

### Running
```python
graph.invoke(inputs, config)
graph.stream(inputs, config, stream_mode="values")                   # full state each step
graph.stream(inputs, config, stream_mode="updates")                  # partial diffs each step
graph.stream(inputs, config, stream_mode="messages")                 # token-by-token
graph.stream(inputs, config, stream_mode=["updates", "messages"])    # multiplex
await graph.ainvoke(...); await graph.astream(...)
```

### Tools
```python
@tool
def my_tool(arg: str) -> str:
    """Docstring = what the LLM reads to decide when/how to call this."""
    return "..."

llm_with_tools = llm.bind_tools([my_tool])
tool_node = ToolNode([my_tool])
agent = create_agent(model="gpt-4o-mini", tools=[my_tool], system_prompt="...")
```

### Persistence
```python
config = {"configurable": {"thread_id": "abc"}}
graph.get_state(config).values             # current state
graph.get_state(config).next               # next node(s) to run
graph.update_state(config, {"k": v})        # patch state mid-thread
list(graph.get_state_history(config))       # time travel
```

### Human-in-the-loop
```python
def node(state):
    answer = interrupt({"question": "...", "payload": state["x"]})
    return {"y": answer}

graph.invoke(inputs, config)                                # runs until interrupt
graph.get_state(config).values["__interrupt__"]              # or check returned dict's __interrupt__
graph.invoke(Command(resume="approve"), config)               # resumes with this value
```

### Multi-agent
```python
create_supervisor([agent_a, agent_b], model=llm, prompt="...").compile(checkpointer=cp)

def handoff(state: Annotated[dict, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    return Command(goto="other_agent", update={...}, graph=Command.PARENT)
```

### Structured output & guardrails
```python
router_llm = llm.with_structured_output(MyPydanticModel)
llm = llm.with_retry(stop_after_attempt=3)
graph.invoke(inputs, {"configurable": {...}, "recursion_limit": 25})
```

---

## 12. Important Extra Notes (from hands-on Q&A)

- **Thread persistence surprise:** re-invoking with the same `thread_id` doesn't start clean — it loads the *entire* prior conversation for that thread and appends your new message. If you see unexpected old context in output, check whether you're reusing a `thread_id` across unrelated test runs. Use a fresh `thread_id` for a clean slate.
- **Prompt scope matters in multi-agent systems:** a restriction in one agent's `system_prompt` (e.g. writer_agent's "use only provided research") does **not** apply to other agents (supervisor, research_agent) unless stated in *their* prompts too. Don't assume a rule you set on one agent constrains the whole team.
- **Kernel/env mismatches (Jupyter):** `ModuleNotFoundError` after installing in terminal but not in notebook usually means the notebook kernel points to a different Python than the terminal's venv. Check with `import sys; print(sys.executable)`, fix via kernel picker, or install directly from the notebook: `!{sys.executable} -m pip install <package>`. Also try restarting the kernel — installs after kernel launch aren't picked up until restart.
- **`interrupt()` re-execution gotcha (worth repeating):** on resume, the *entire node function* reruns from the top — not just the code after `interrupt()`. Anything with side effects (DB writes, API calls, print statements) placed before the `interrupt()` call will fire twice across the pause + resume. Keep that region pure.
- **`Command` has two distinct jobs** that look similar but serve different purposes: (1) `Command(goto=..., update=...)` for intra-graph routing decisions (HITL branching, supervisor routing), and (2) `Command(resume=...)` purely as the *input* you pass to `.invoke()` to unblock a paused `interrupt()`. Don't conflate the two — `resume` is never combined with `goto`/`update` in the same object.
- **`graph=Command.PARENT`** is only needed when the target of a `Command(goto=...)` lives in a *different* compiled subgraph than the node returning the `Command` — e.g., handoffs between `create_agent`-built agents. Omit it for routing within a single flat `StateGraph`.
