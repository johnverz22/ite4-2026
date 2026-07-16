# Activity 16: From ReAct Loop to LangGraph

## Porting Your Chatbot to a State Graph Architecture

---

## What You Already Have (Weeks 1–5 Recap)

| Week | Component | What Your Chatbot Does |
|------|-----------|------------------------|
| W1 | Identity + Working Memory | System prompt defines persona; `WorkingMemory.history` tracks conversation |
| W2 | Structured Perception | Pydantic `AgentAction` validates each model decision |
| W3 | Long-Term Memory (Qdrant) | `search_documents` retrieves chunks from a vector store |
| W4 | Self-Evaluation | Mock `evaluate_answer` (or real RAG Triad judges) scores quality |
| W5 | ReAct Loop | `run_react_loop` cycles through reason → act → observe in a `for` loop |

### The Core Difference

```
ReAct Loop (Week 5)                    LangGraph (Week 6)
─────────────────                      ─────────────────
for i in range(max_iter):              graph = StateGraph(AgentState)
    action = llm(history)              graph.add_node("generate", ...)
    if action.tool_name:               graph.add_node("evaluate", ...)
        result = tools[action]         graph.add_conditional_edges(
        history.append(result)             "evaluate", should_retry, {...}
    else:                              graph.compile()
        return answer                  app.invoke(state)
```

**Why LangGraph matters:**
- **Explicit state type** — every possible field is declared upfront in `AgentState`
- **Graph visualization** — you can see the flow: generate → evaluate → (rewrite or end)
- **Conditional edges** — routing logic lives in one function, not scattered in `if/else` branches
- **Built-in safety** — `recursion_limit` catches infinite loops at the framework level

---

## Step 1: Define the LangGraph State

Replace your loose `WorkingMemory` and ad-hoc variables with a typed state:

```python
from typing import TypedDict


class AgentState(TypedDict):
    question: str
    original_question: str
    retrieved_chunk: str
    answer: str
    context_relevance: float
    groundedness: float
    answer_relevance: float
    iteration: int
    log: list
```

Every field has one job:
| Field | Set By | Purpose |
|-------|--------|---------|
| `question` | `user` / `rewrite` | The current (possibly rewritten) query |
| `original_question` | `user` | Saved for reporting — never changes |
| `retrieved_chunk` | `generate` | What Qdrant returned |
| `answer` | `generate` | The LLM's answer |
| `context_relevance` | `evaluate` | Score: is the chunk relevant to the question? |
| `groundedness` | `evaluate` | Score: is the answer supported by the chunk? |
| `answer_relevance` | `evaluate` | Score: does the answer address the question? |
| `iteration` | `generate` | Counter for the safety cap |
| `log` | `generate` | Diagnostic list of every turn |

**CHECKPOINT 1:** Write the `AgentState` TypedDict exactly as shown. Add one field of your own: `route_decision: str` to record whether each iteration ended with "accept" or "rewrite".

---

## Step 2: Port Your Tools into Standalone Functions

From your `react_agent_demo.py`, keep these as-is:

```python
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
qdrant = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "course_memory"


def search_documents(query: str) -> str:
    """Qdrant retrieval — returns a clean string."""
    if not query.strip():
        return "No query provided."
    try:
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        query_vector = response.embeddings[0].values
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3,
        )
        if not results:
            return "No relevant information found."
        return results[0].payload.get("text_segment", "")
    except Exception as e:
        return f"Search error: {e}"


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


def clarify(question: str) -> str:
    """Fallback — ask the user for more information."""
    return f"[Clarify] {question}"
```

These functions do **not** change. LangGraph calls them the same way your ReAct loop did.

**CHECKPOINT 2:** Run each function in isolation. Confirm `search_documents("What is ReAct?")` returns a real chunk from Qdrant.

---

## Step 3: Build the Generate Node

This replaces the `run_react_loop` function's core logic:

```python
def generate_node(state: AgentState) -> AgentState:
    """Retrieve context and produce an answer."""
    question = state["question"]

    chunk = search_documents(question)

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.

Question: {question}

Context: {chunk}

Answer:"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    answer = response.text

    state["retrieved_chunk"] = chunk
    state["answer"] = answer
    state["iteration"] = state.get("iteration", 0) + 1

    state["log"] = state.get("log", [])
    state["log"].append({
        "iteration": state["iteration"],
        "question": question,
        "chunk_preview": chunk[:100] if chunk else "",
        "answer_preview": answer[:100] if answer else "",
    })

    print(f"  [Generate] Iteration {state['iteration']}")
    return state
```

**Key difference from your ReAct loop:** The generate node does **not** decide which tool to call. It always retrieves and generates. Routing decisions move to the conditional edge.

**CHECKPOINT 3:** Run `generate_node` manually with a test state:
```python
test = {"question": "What is chunking?", "iteration": 0, "log": []}
result = generate_node(test)
print(result["answer"][:200])
```

---

## Step 4: Build the Evaluate Node

Replace your mock `evaluate_answer` with a real RAG Triad judge. Each judge asks the LLM to score one metric:

```python
from pydantic import BaseModel


class JudgeVerdict(BaseModel):
    score: float
    reason: str


def judge_metric(criterion: str, reference: str, target: str) -> JudgeVerdict:
    """Score a single RAG Triad metric using an LLM judge."""
    prompt = f"""You are an evaluation agent. Score on a scale of 0.0 to 1.0.

Criterion: {criterion}

Reference:
{reference}

Text to evaluate:
{target}

Return JSON: {{"score": <float>, "reason": "<one sentence>"}}"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JudgeVerdict,
        ),
    )
    return JudgeVerdict.model_validate_json(response.text)
```

The evaluate node calls all three judges:

```python
def evaluate_node(state: AgentState) -> AgentState:
    """Score the output using RAG Triad judges."""
    cr = judge_metric(
        "Is the retrieved chunk relevant to answering the question?",
        state["question"],
        state["retrieved_chunk"],
    )
    gr = judge_metric(
        "Is the answer factually supported by the retrieved chunk?",
        state["retrieved_chunk"],
        state["answer"],
    )
    ar = judge_metric(
        "Does the answer correctly address the user's question?",
        state["question"],
        state["answer"],
    )

    state["context_relevance"] = cr.score
    state["groundedness"] = gr.score
    state["answer_relevance"] = ar.score

    # Append scores to the latest log entry
    if state["log"]:
        state["log"][-1].update({
            "context_relevance": cr.score,
            "groundedness": gr.score,
            "answer_relevance": ar.score,
            "cr_reason": cr.reason,
            "gr_reason": gr.reason,
            "ar_reason": ar.reason,
        })

    print(f"  [Evaluate] CR: {cr.score:.2f} | GR: {gr.score:.2f} | AR: {ar.score:.2f}")
    return state
```

**CHECKPOINT 4:** Run `evaluate_node` with a state that has `question`, `retrieved_chunk`, and `answer` populated. Confirm all three scores are between 0.0 and 1.0.

---

## Step 5: Build the Conditional Edge

This is the brain of the graph — it reads the scores and decides what to do next:

```python
THRESHOLD = 0.7
MAX_ITERATIONS = 3


def should_retry(state: AgentState) -> str:
    """Route to 'end' if scores are good, otherwise 'rewrite'."""
    if state["iteration"] >= MAX_ITERATIONS:
        print(f"  [Decision] Max iterations ({MAX_ITERATIONS}) reached.")
        state["route_decision"] = "accept_maxed"
        return "end"

    min_score = min(
        state["context_relevance"],
        state["groundedness"],
        state["answer_relevance"],
    )

    if min_score >= THRESHOLD:
        print(f"  [Decision] All scores >= {THRESHOLD}. Accepting.")
        state["route_decision"] = "accept"
        return "end"

    print(f"  [Decision] Min score {min_score:.2f} < {THRESHOLD}. Rewriting...")
    state["route_decision"] = "rewrite"
    return "rewrite"
```

**CHECKPOINT 5:** Test the function with different score combinations:

| `context_relevance` | `groundedness` | `answer_relevance` | `iteration` | Expected Return |
|---------------------|----------------|---------------------|--------------|-----------------|
| 0.9 | 0.8 | 0.95 | 1 | `"end"` |
| 0.9 | 0.3 | 0.95 | 1 | `"rewrite"` |
| 0.5 | 0.4 | 0.6 | 3 | `"end"` (maxed) |

---

## Step 6: Build the Rewrite Node

When scores are too low, the rewrite node uses the LLM to produce a better query:

```python
def rewrite_node(state: AgentState) -> AgentState:
    """Rewrite the query to improve retrieval on the next cycle."""
    original = state["question"]
    chunk = state["retrieved_chunk"]

    rewrite_prompt = f"""The following query did not retrieve good context.
Rewrite it to be more specific and likely to find relevant information
in a vector database.

Original query: {original}
Retrieved chunk (not helpful): {chunk[:200]}

Return ONLY the rewritten query, nothing else."""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=rewrite_prompt,
    )
    state["question"] = response.text.strip()

    print(f"  [Rewrite] '{original[:50]}...' -> '{state['question'][:50]}...'")
    return state
```

**CHECKPOINT 6:** Run `rewrite_node` on a state where the chunk was a poor match. Compare the original and rewritten queries. Does the rewritten version add specificity?

---

## Step 7: Assemble the Graph

```python
from langgraph.graph import StateGraph, END


def build_graph() -> StateGraph:
    """Assemble the self-correcting RAG graph."""
    graph = StateGraph(AgentState)

    graph.add_node("generate", generate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("rewrite", rewrite_node)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges("evaluate", should_retry, {
        "end": END,
        "rewrite": "rewrite",
    })
    graph.add_edge("rewrite", "generate")

    return graph.compile()
```

The graph looks like this:

```
         ┌──────────┐
         │ generate │
         └────┬─────┘
              │
              ▼
         ┌──────────┐
         │ evaluate │
         └────┬─────┘
              │
         ┌────┴────┐
         │         │
    [accept]   [rewrite]
         │         │
         ▼         ▼
        END    ┌──────────┐
               │ rewrite  │
               └────┬─────┘
                    │
                    └──► back to generate
```

**CHECKPOINT 7:** Run `build_graph()` and confirm no errors. Print the graph:
```python
app = build_graph()
print(app.get_graph())
```

---

## Step 8: Run the Agent

```python
def run_agent(question: str) -> dict:
    """Run the self-correcting agent on a single question."""
    app = build_graph()

    initial_state = {
        "question": question,
        "original_question": question,
        "retrieved_chunk": "",
        "answer": "",
        "context_relevance": 0.0,
        "groundedness": 0.0,
        "answer_relevance": 0.0,
        "iteration": 0,
        "log": [],
    }

    result = app.invoke(initial_state, config={"recursion_limit": 10})
    return result


if __name__ == "__main__":
    test_questions = [
        "What is ReAct?",
        "What is the travel budget?",
        "What did we learn about chunking?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        print(f"{'='*60}")

        result = run_agent(q)

        print(f"\nFinal Answer: {result['answer'][:200]}")
        print(f"Iterations used: {result['iteration']}")
        print(f"Final Scores — CR: {result['context_relevance']:.2f} | "
              f"GR: {result['groundedness']:.2f} | AR: {result['answer_relevance']:.2f}")
        print(f"Route decision: {result.get('route_decision', 'N/A')}")

        print(f"\n  Log:")
        for entry in result.get("log", []):
            print(f"    #{entry['iteration']}: \"{entry['question'][:40]}...\" "
                  f"CR={entry.get('context_relevance', '?'):.2f} "
                  f"GR={entry.get('groundedness', '?'):.2f} "
                  f"AR={entry.get('answer_relevance', '?'):.2f}")
```

**CHECKPOINT 8:** Run all three test questions. For each:
1. Confirm the graph compiles and runs without errors.
2. Confirm the log shows `generate -> evaluate -> (end or rewrite)`.
3. If self-correction triggers, confirm the rewritten query is semantically different from the original.
4. Confirm `recursion_limit` prevents runaway execution.

---

## Step 9: Test the Self-Correction Path

Deliberately ask a vague question to trigger the rewrite cycle:

```python
tricky_questions = [
    "Tell me about that thing from class",
    "What did the course say?",
    "Do the stuff",
]
```

For each, verify:
1. The evaluate node returns low scores (< 0.7).
2. `should_retry` routes to `"rewrite"`.
3. The rewrite node produces a more specific query.
4. The second `generate` call with the rewritten query returns better scores — OR the graph gracefully stops at `MAX_ITERATIONS`.

**CHECKPOINT 9:** Run the tricky questions. Print the full log for each. In your reflection, explain why the original query failed and how the rewrite helped (or why it did not).

---

## Step 10: Compare ReAct Loop vs. LangGraph

Run the same question through both architectures:

```python
def react_loop_baseline(question: str) -> dict:
    """Your Week 5 ReAct loop — returns answer and transcript."""
    memory = WorkingMemory()
    memory.add_message("system", SYSTEM_PROMPT)
    memory = run_react_loop(question, memory)

    # Extract final answer from memory
    for msg in reversed(memory.history):
        if msg["role"] == "assistant" and "tool_input" not in msg.get("content", ""):
            return {"answer": msg["content"], "loop_type": "react"}
    return {"answer": "[No answer]", "loop_type": "react"}
```

Compare on these dimensions:

| Dimension | ReAct Loop (Week 5) | LangGraph (Week 6) |
|-----------|--------------------|--------------------|
| **State** | `WorkingMemory.history` (list of dicts) | Typed `AgentState` with typed fields |
| **Routing** | `if/elif/else` in `mock_llm` | `should_retry` conditional edge |
| **Evaluation** | Mock `evaluate_answer` (pass/fail) | 3 RAG Triad judges (0.0–1.0 scores) |
| **Self-correction** | Manual retry (if you coded it) | Automatic via node + conditional edge |
| **Safety** | `max_iterations` in `for` loop | `recursion_limit` + `MAX_ITERATIONS` |
| **Traceability** | Print statements | Structured `log` with scores per iteration |

**CHECKPOINT 10:** Run 5 questions through both architectures. Create a comparison table like the one above with your actual results (answers, scores, iterations).

---

## Graded Activity — LangGraph Port

### File Structure

```
activity16/
├── activity16_langgraph_agent.py    # Full LangGraph implementation
├── activity16_comparison.md         # ReAct vs. LangGraph analysis
└── activity16_reflection.md         # Reflection questions
```

### Requirement A: LangGraph Agent Implementation (50%)

Port your Week 5 chatbot to a LangGraph state graph with all of the following:

| Component | Points | Must Include |
|-----------|--------|--------------|
| `AgentState` TypedDict | 5 | `question`, `original_question`, `retrieved_chunk`, `answer`, 3 RAG Triad scores, `iteration`, `log` |
| `generate_node` | 10 | Calls `search_documents` (real Qdrant), generates answer via LLM, increments `iteration`, appends to `log` |
| `evaluate_node` | 10 | Calls all 3 RAG Triad judges, stores scores in state, appends to latest log entry with reasons |
| `should_retry` conditional edge | 10 | Checks `MAX_ITERATIONS` first, then checks all 3 scores against `THRESHOLD`, returns `"end"` or `"rewrite"` |
| `rewrite_node` | 5 | Uses LLM to produce a semantically different query, stores result in `state["question"]` |
| Graph assembly + execution | 10 | Compiles without errors, runs with `recursion_limit`, logs all iterations |

### Requirement B: Self-Correction Demonstration (20%)

Run your agent on **at least 6 questions**:

- 3 "happy path" questions that your Qdrant collection can answer well (expect 1 iteration each)
- 3 "tricky" vague questions that should trigger self-correction (expect 2+ iterations)

**Must demonstrate:**
1. At least one query completes in 1 iteration (all scores >= threshold).
2. At least one query triggers the rewrite node and improves on retry.
3. At least one query reaches `MAX_ITERATIONS` and gracefully stops.

Include the console output in your submission showing all 6 runs with full logs.

### Requirement C: Comparison Report (15%)

Write `activity16_comparison.md` with:

1. **Architecture Diagram** — Draw (in ASCII or text) the Week 5 ReAct loop flow and the Week 6 LangGraph flow. Label the key differences.

2. **Comparison Table** — Run 5 identical questions through both architectures and compare:

| Question | ReAct Answer | LangGraph Answer | ReAct Iters | LangGraph Iters | Which was better? |
|----------|-------------|-----------------|-------------|-----------------|-------------------|

3. **Failure Analysis** — For any question where the two architectures gave different answers, explain why. Was it the evaluation? The rewrite? The tool routing?

### Requirement D: Reflection (15%)

Write `activity16_reflection.md` answering:

1. What was the hardest part of converting your ReAct loop to LangGraph? Why?

2. Your `AgentState` includes an extra field `route_decision` you added in Step 1. How did you use it in debugging? Would you add any other fields?

3. Compare the safety mechanisms:
   - ReAct loop: `for i in range(max_iterations)` with `break`
   - LangGraph: `MAX_ITERATIONS` in `should_retry` + `recursion_limit` in `app.invoke()`
   
   Why are both mechanisms necessary in LangGraph? What happens if you set `recursion_limit` to 3?

4. The evaluate node makes 3 LLM calls per iteration. If `MAX_ITERATIONS = 3`, that is up to 9 judge calls per question. Is this cost justified? What would you change to reduce cost while preserving quality?

5. In your Week 5 ReAct loop, the `mock_llm` function routed between `search_documents`, `calculate`, and `clarify`. In this LangGraph port, there is only one tool (`search_documents`). How would you add `calculate` and `clarify` back into the graph? Would they be new nodes or something else?

---

## Grading Rubric

| Criteria | Excellent (100%) | Satisfactory (70%) | Needs Work (40%) |
|----------|-----------------|--------------------|-------------------|
| **A: LangGraph Implementation** | All 6 components working, graph compiles, runs 3+ queries, logs are complete | 5 of 6 components, graph compiles but minor bugs | < 4 components or graph does not compile |
| **B: Self-Correction** | 6 queries run, at least 1 triggers rewrite with improvement, 1 reaches max gracefully | 6 queries run, self-correction triggers but improvement unclear | < 6 queries or no self-correction |
| **C: Comparison Report** | Architecture diagram, 5-query comparison with analysis, failure explanation | Comparison table present but thin analysis | Missing diagram or no meaningful comparison |
| **D: Reflection** | All 5 questions answered with specific references to your code and results | Answers present but generic | Missing or superficial answers |

---

## Appendix: Common Migration Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Forgetting to increment `iteration` | Infinite loop — `should_retry` never sees `MAX_ITERATIONS` | Add `state["iteration"] += 1` at the top of `generate_node` |
| `AgentState` missing a field | KeyError when node tries to read/write the field | Declare all fields in the TypedDict, including `route_decision` and `log` |
| `search_documents` returns `None` | Crash in generate node when formatting prompt | Ensure the function always returns a string (use `return ""` as fallback) |
| Not saving `original_question` | After rewrite, the original intent is lost | Set `original_question` in the initial state and never modify it |
| `recursion_limit` too low | Graph terminates mid-cycle with no answer | Set to at least `(MAX_ITERATIONS * 2) + 2` (minimum 8–10) |
| Judge prompt returns invalid JSON | Crash in `model_validate_json` | Add `response_mime_type="application/json"` to the judge config |
| No `log` field in initial state | KeyError when `generate_node` calls `state.get("log", [])` | Add `"log": []` to the initial state dict |
