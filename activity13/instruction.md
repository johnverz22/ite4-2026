# Activity 13 Project Guideline: ReAct Loop & Tool Calling

## Integrating ReAct and Tool Selection into Your Chatbot


---

## What You Already Have (Weeks 1–4 Recap)

| Week | Component | What Your Chatbot Does |
|------|-----------|------------------------|
| W1 | Identity + Working Memory | System prompt defines persona; chat history tracks conversation |
| W2 | Structured Perception | Pydantic schemas validate all LLM inputs/outputs |
| W3 | Long-Term Memory (Qdrant) | Documents chunked, embedded, stored in Qdrant |
| W4 | Self-Evaluation (RAG Triad) | LLM-as-a-Judge scores context relevance, groundedness, answer relevance |

### What Your Current Pipeline Looks Like

```
User Query
    │
    ▼
[Embed Query] ──► [Qdrant Search] ──► [Retrieved Chunk]
    │                                       │
    │                                       ▼
    └──────────────────► [LLM Generate] ◄──┘
                              │
                              ▼
                         [RAG Triad Judge]
```

**The problem:** Retrieval is hardcoded. Every query embeds, searches Qdrant, and generates — regardless of whether the question needs external knowledge. The agent has no choice about what to do next.

### What You're Building This Week

```
User Query
    │
    ▼
┌─ REASON: "Do I need a tool?"
│     │
│     ┌──────────┬──────────┬───────────┐
│     ▼          ▼          ▼           ▼
│ search_    calculate  clarify    answer
│ documents            (fallback)  directly
│     │          │          │
│     ▼          ▼          ▼
│  [result]   [result]   [result]
│     │          │          │
│     └─────┬────┘          │
│           ▼               │
│     ┌─ OBSERVE ◄──────────┘
│     │     │
│     │     ▼
│     └─ REASON again → loop or answer
│
▼
[RAG Triad Judge] → PASS or self-correct
```

---

## Step-by-Step Build Process

### Step 1: Define Your Tool Palette

Create three Python functions. For this guide, use a **mock knowledge base** (dictionary) for `search_documents` so you can focus on routing mechanics. The real Qdrant integration comes in the separate Memory Bridging activity.

**`search_documents` — Retrieval from a mock knowledge base**

```python
def search_documents(query: str) -> str:
    """Search the knowledge base for factual information."""
    kb = {
        "ReAct": "ReAct stands for Reasoning + Acting. It interleaves thought and tool calls.",
        "Qdrant": "Qdrant is a vector database for long-term agent memory.",
        "chunking": "Semantic chunking splits documents at natural boundaries.",
        "budget": "The travel budget is $2000 for flights and $500 for accommodations.",
        "RAG Triad": "The RAG Triad evaluates context relevance, groundedness, and answer relevance.",
    }
    for key, doc in kb.items():
        if key.lower() in query.lower():
            return f"[Found] {doc}"
    return "[Not found] No relevant information in the knowledge base."
```

**`calculate` — Arithmetic**

```python
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"
```

**`clarify` — Fallback for ambiguous queries**

```python
def clarify(question: str) -> str:
    """Ask for clarification when the request is ambiguous."""
    return f"[Clarify] {question}"
```

**CHECKPOINT 1:** Run each function manually. Confirm `search_documents("What is ReAct?")` returns a relevant result, `calculate("45 * 12")` returns `540`, and `clarify("What do you mean?")` returns the clarification template.

---

### Step 2: Register Tools with the Google GenAI SDK

Each function needs a `FunctionDeclaration` that tells the LLM:
1. **Name** — what to call in the code
2. **Description** — when to use this tool (this drives routing decisions)
3. **Parameters** — what arguments the tool expects

```python
from google.genai import types

TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description=(
                "Search the knowledge base for factual information about "
                "course topics, budgets, or stored knowledge. Use this when "
                "the question asks about specific content or stored facts."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={"query": types.Schema(type="STRING")},
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="calculate",
            description=(
                "Perform a mathematical calculation given an arithmetic "
                "expression. Use for any numeric computation."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={"expression": types.Schema(type="STRING")},
                required=["expression"],
            ),
        ),
        types.FunctionDeclaration(
            name="clarify",
            description=(
                "Ask the user a clarifying question when their request is "
                "ambiguous or too vague to handle. Use as a fallback when "
                "you cannot determine what the user needs."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={"question": types.Schema(type="STRING")},
                required=["question"],
            ),
        ),
    ]
)

# Dispatch registry — maps tool name → Python function
AVAILABLE_FUNCTIONS = {
    "search_documents": search_documents,
    "calculate": calculate,
    "clarify": clarify,
}
```

**Key insight:** The `description` field is the **only signal** the LLM gets to decide which tool to call. Weak descriptions (e.g., "searches stuff") cause misrouting. Strong descriptions (e.g., "Search the knowledge base for factual information about course topics") trigger correctly on factual questions.

**CHECKPOINT 2:** Review each description. Could a human reading only these descriptions pick the right tool for each query? If descriptions overlap (same keywords), rewrite them to be distinct.

---

### Step 3: Build the ReAct Loop

The ReAct loop is the engine: it sends the conversation to the LLM, checks if the LLM wants to call a tool, executes the tool, appends the result to memory, and repeats until the LLM produces a final answer.

```python
def react_loop(question: str, max_iterations: int = 5, system_prompt: str = "") -> list[dict]:
    """
    Run a ReAct loop and return a labeled transcript.
    Each entry: {"phase": "USER"|"ACTION"|"OBSERVE"|"ANSWER"|"SYSTEM", "content": str}
    """
    transcript = [{"phase": "USER", "content": question}]
    history = [types.Content(role="user", parts=[types.Part(text=question)])]

    for turn in range(max_iterations):
        # --- REASON: Send conversation to the LLM ---
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=[TOOLS],
            ),
        )

        part = response.candidates[0].content.parts[0]

        # --- ACT: Did the LLM request a tool call? ---
        if part.function_call:
            fc = part.function_call
            tool_name = fc.name
            tool_args = dict(fc.args)

            transcript.append({
                "phase": "ACTION",
                "tool": tool_name,
                "content": f"{tool_name}({tool_args})",
            })

            # Execute the tool
            func = AVAILABLE_FUNCTIONS.get(tool_name)
            result = func(**tool_args) if func else f"Unknown tool: {tool_name}"

            # --- OBSERVE: Append the result ---
            transcript.append({"phase": "OBSERVE", "content": result})

            # Append both the model's action and the tool result to working memory
            history.append(response.candidates[0].content)
            history.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=tool_name,
                                response={"result": result},
                            )
                        )
                    ],
                )
            )
        else:
            # --- ANSWER: No tool call → final answer ---
            transcript.append({"phase": "ANSWER", "content": part.text})
            return transcript

    transcript.append({"phase": "SYSTEM", "content": f"Max iterations ({max_iterations}) reached."})
    return transcript
```

**CHECKPOINT 3:** Run `react_loop("What is ReAct?")`. Confirm the transcript shows:
```
[USER]    What is ReAct?
[ACTION]  search_documents({'query': 'What is ReAct?'})
[OBSERVE] [Found] ReAct stands for Reasoning + Acting...
[ANSWER]  ReAct is a pattern that interleaves reason and act...
```

---

### Step 4: Trace and Label Transcripts

Add a print function to visualize the cognitive cycle:

```python
def print_transcript(transcript: list[dict]):
    """Pretty-print a labeled transcript."""
    print(f"\n{'=' * 60}")
    for entry in transcript:
        phase = entry["phase"]
        content = entry["content"]
        print(f"  [{phase:7}] {content}")
    print(f"{'=' * 60}")


def demo_query(question: str):
    """Run one query through the ReAct loop and print the transcript."""
    print(f"\n>>> QUERY: {question}")
    transcript = react_loop(question, system_prompt="You are a helpful assistant with tools.")
    print_transcript(transcript)
```

**CHECKPOINT 4:** Run these four test queries and confirm the tool routing is correct:

| Query | Expected First Tool | Why? |
|-------|-------------------|------|
| `"What is the travel budget?"` | `search_documents` | Factual recall |
| `"Calculate 15% of 2000"` | `calculate` | Math computation |
| `"Help me with that thing"` | `clarify` | Ambiguous intent |
| `"Hello!"` | *(no tool, direct answer)* | Greeting, no tool needed |

If the LLM picks the wrong tool, improve the tool's `description` field and re-test. This iteration — test → observe misroute → improve description → re-test — is the core skill of tool selection engineering.

---

### Step 5: Test Routing Accuracy

Create a test suite to measure how often the agent picks the correct tool:

```python
ROUTING_TESTS = [
    ("What is ReAct?", "search_documents"),
    ("What is the travel budget?", "search_documents"),
    ("What did we learn about chunking?", "search_documents"),
    ("Calculate 45 * 12", "calculate"),
    ("What is 15% of 3000?", "calculate"),
    ("What is 2 to the power of 10?", "calculate"),
    ("Help me with that thing", "clarify"),
    ("Do the stuff I asked", "clarify"),
    ("Hello!", None),           # No tool → answer directly
    ("Thank you!", None),       # No tool → answer directly
]


def test_routing_accuracy():
    """Run all routing tests and report accuracy."""
    correct = 0

    for query, expected in ROUTING_TESTS:
        transcript = react_loop(query)
        # Find the first tool called (if any)
        actual_tool = None
        for entry in transcript:
            if entry.get("phase") == "ACTION":
                actual_tool = entry.get("tool")
                break

        match = actual_tool == expected
        status = "✓" if match else "✗"
        correct += 1 if match else 0
        print(f"  {status} expected={str(expected):20s} got={str(actual_tool):20s} | {query[:50]}")

    total = len(ROUTING_TESTS)
    print(f"\nAccuracy: {correct}/{total} ({correct / total * 100:.0f}%)")
```

**CHECKPOINT 5:** Aim for ≥ 80% accuracy. For each misroute:
- Read the transcript: what tool did it call instead?
- Is the description too similar to another tool's?
- Would adding "Use this when..." to the description fix it?

---

### Step 6: Integrate RAG Triad Evaluation (From Week 4)

After the ReAct loop produces an answer, wrap it with your existing RAG Triad evaluation:

```python
def run_with_evaluation(question: str) -> dict:
    """ReAct loop + RAG Triad quality gate."""
    transcript = react_loop(question)

    # Extract final answer and retrieved chunk from transcript
    answer = None
    chunk = None
    for entry in transcript:
        if entry["phase"] == "ANSWER":
            answer = entry["content"]
        if entry["phase"] == "OBSERVE":
            chunk = entry["content"]

    # Run Week 4's RAG Triad evaluation (import from your week4 code)
    triad = score_rag_triad(question, chunk or "", answer or "")

    result = {
        "question": question,
        "answer": answer,
        "transcript": transcript,
        "context_relevance": triad.context_relevance.score,
        "groundedness": triad.groundedness.score,
        "answer_relevance": triad.answer_relevance.score,
        "passed": triad.passed(),
    }

    if not triad.passed() and triad.weakest_leg()[0] != "context_relevance":
        # Self-correct (groundedness or relevance failure)
        corrected = self_correct_answer(question, chunk or "", answer or "", triad)
        result["corrected_answer"] = corrected
        result["was_corrected"] = True

    return result
```

**CHECKPOINT 6:** Run a query your mock KB can answer (e.g., "What is ReAct?"). Confirm the Triad scores are logged and the quality gate decision (PASS/FAIL) is printed.

---

## Week 5 Project — Graded Activity

### File Structure

```
activity13/
├── project_react_loop.py        # ReAct loop + 3 tools + transcript logging
├── routing_test_suite.py        # 10-query test + accuracy report
├── evaluation_integration.py    # RAG Triad wrapper around the ReAct loop
└── routing_report_week5.md      # Analysis + reflection
```

### Requirement A: ReAct Loop with 3 Tools (40%)

Implement a working ReAct loop with `search_documents` (mock KB), `calculate`, and `clarify`.

**Must demonstrate:**
1. Labeled transcripts (USER, ACTION, OBSERVE, ANSWER).
2. Correct routing for all 5 query types:
   - Factual → `search_documents`
   - Math → `calculate`
   - Ambiguous → `clarify`
   - Greeting → direct answer (no tool)
   - Multi-step → two tools in sequence (e.g., "What's the budget and 15% of it?")

### Requirement B: Routing Test Suite (25%)

Create `routing_test_suite.py` as shown in Step 5. Run on **at least 10 queries** and report accuracy.

**Must demonstrate:**
1. ≥ 80% routing accuracy.
2. For any failures, explain in your report which tool description caused the misroute and how you would fix it.

### Requirement C: RAG Triad Integration (20%)

Wrap your ReAct loop with the Week 4 RAG Triad evaluation.

**Must demonstrate:**
1. Every query outputs context_relevance, groundedness, answer_relevance scores.
2. Self-correction triggers when scores fall below threshold.
3. Transcript and Triad scores are logged together.

### Requirement D: Reflection Report (15%)

Write `routing_report_week5.md` with these sections:

**Section 1 — Routing Results Table**

| Query | Expected Tool | Actual Tool | Correct? |
|-------|---------------|-------------|----------|
| What is ReAct? | search_documents | search_documents | ✓ |
| ... | ... | ... | ... |

**Section 2 — Failure Analysis** (if any misroutes)
For each failure: which tool was called instead? Which description keywords likely confused the model?

**Section 3 — Reflection**
1. Which query type was hardest for your agent to route correctly? Why?
2. How did your tool descriptions change between your first attempt and your final version?
3. If you added a fourth tool to your palette (e.g., `get_current_date`, `format_summary`), what would its description look like?

---

## Grading Rubric

| Criteria | Excellent (100%) | Satisfactory (70%) | Needs Work (40%) |
|----------|-----------------|--------------------|-------------------|
| **A: ReAct Loop** | Full loop, 3 tools, labeled transcripts, multi-step queries work | Loop works, missing labels or one tool | No loop or 1 tool |
| **B: Routing Tests** | 10+ queries, ≥ 80% accuracy, failure analysis | 10 queries, < 80% accuracy | < 10 queries |
| **C: RAG Triad** | Scores logged, self-correction triggers | Scores logged, no correction | Not integrated |
| **D: Reflection** | Specific failure analysis, concrete description improvements | Basic answers | Superficial |

---

## Appendix: Tool Description Anti-Patterns

| Weak Description | Strong Description |
|-----------------|--------------------|
| "Searches the database" | "Search the knowledge base for factual information about course topics or stored data. Use when the question asks about specific content not in the conversation." |
| "Does math" | "Evaluate a mathematical expression (e.g., '45 * 12', '15% of 200', '2^10'). Use for any numeric calculation." |
| "Talks to user" | "Ask the user a clarifying question when their request is too ambiguous to handle. Use as a fallback when no other tool fits." |
| "Does stuff with data" | Avoid entirely — too vague for any routing decision. |

---
