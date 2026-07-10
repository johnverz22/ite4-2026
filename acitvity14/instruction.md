# Activity14: Bridging Memory and Action

## Wrapping Qdrant Retrieval as a Tool & Improving Vector Search

**Prerequisite:** Complete `activity13.md` (ReAct Loop & Tool Calling) first. Make a copy and apply the following guidelinews

---

## Step 1: Replace Mock KB with Real Qdrant

Your existing project already has a Qdrant retrieval function. Instead of the mock dictionary, wire it directly into your tool palette:

```python
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
qdrant = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "course_memory"  # Use your existing collection
```

### Replace the mock function

**Before (from main guide):**
```python
def search_documents(query: str) -> str:
    kb = {
        "ReAct": "ReAct stands for Reasoning + Acting...",
        ...
    }
    for key, doc in kb.items():
        if key.lower() in query.lower():
            return f"[Found] {doc}"
    return "[Not found] No relevant information in the knowledge base."
```

**After (real Qdrant):**
```python
def search_documents(query: str) -> str:
    """
    Search the persistent Qdrant knowledge base for factual information.
    Embeds the query, searches the vector DB, and returns the top chunk.
    """
    try:
        # 1. Embed the query using gemini-embedding-2
        response = gemini_client.models.embed_content(
            model="gemini-embedding-2",
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        query_vector = response.embeddings[0].values

        # 2. Search Qdrant
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3,
        )

        if not results:
            return "No relevant information found in the knowledge base."

        # 3. Return the top chunk's text
        return results[0].payload.get("text_segment", "No text in payload.")

    except Exception as e:
        return f"Qdrant search error: {e}"
```

**CHECKPOINT 1:** Run `react_loop("What is ReAct?")` with the new `search_documents`. Confirm it returns real content from your Qdrant collection — not the mock dictionary result.

---

## Step 2: Handle Edge Cases Gracefully

The tool must always return a clean string — never a raw Qdrant object or a stack trace.

```python
def search_documents(query: str) -> str:
    """Qdrant retrieval wrapped as a tool — always returns a clean string."""
    if not query.strip():
        return "No query provided."

    try:
        response = gemini_client.models.embed_content(
            model="gemini-embedding-2",
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        query_vector = response.embeddings[0].values
    except Exception as e:
        return f"Embedding failed: {e}"

    try:
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3,
        )
    except Exception as e:
        return f"Qdrant connection error: {e}"

    if not results:
        return "No relevant information found in the knowledge base."

    text = results[0].payload.get("text_segment", "")
    return text if text else "Found a result but no text payload."
```

**Why it matters:** If the tool returns `None` or raises an unhandled exception, the ReAct loop breaks. If it returns a `PointStruct` object instead of a string, the LLM cannot read it in the OBSERVE step.

---

## Step 3: Improve Vector Search Quality

Your chatbot uses `gemini-embedding-2` (3072-dim dense vectors). Dense embeddings have two known weaknesses:

| Problem | What Happens | Why It Hurts |
|---------|--------------|--------------|
| **Embedding Dilution** | Long document vectors average specific details into a generic "mean." | Query about "budget limits for flights" retrieves a generic "travel" doc instead of the exact sentence. |
| **Asymmetric Retrieval** | Short query (10–20 tokens) vs. long document (200+ tokens). Gemini produces different embedding distributions for different lengths. | Short query vectors land in a different region of the 3072-dim space than document vectors. Cosine similarity returns low scores even for semantically related content. |

### Improvement 1: Query Expansion (Counters Asymmetry)

Before embedding, expand short queries into document-style passages:

```python
def expand_query(short_query: str) -> str:
    """Expand a short user query into a document-style passage to reduce
    asymmetric embedding mismatch between short queries and long documents."""
    prompt = f"""Rewrite the following short query as a detailed, paragraph-style
statement suitable for similarity search. Include key terms and context.

Original query: {short_query}

Expanded passage:"""

    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1),
    )
    return response.text.strip()


# Updated search_documents with query expansion:
def search_documents(query: str) -> str:
    if not query.strip():
        return "No query provided."

    try:
        # Step 1: Expand the query
        expanded = expand_query(query)

        # Step 2: Embed the EXPANDED text
        response = gemini_client.models.embed_content(
            model="gemini-embedding-2",
            contents=expanded,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        query_vector = response.embeddings[0].values

        # Step 3: Search Qdrant
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
```

**When to use:** Always for single-sentence queries (< 20 tokens). Skip for multi-sentence or detailed queries (they are already long enough).

### Improvement 2: Hybrid Search — Dense + Keyword (Counters Dilution)

Dense vectors lose exact keyword matches. Combine dense similarity with keyword overlap to preserve specific terms:

```python
def hybrid_search(query: str, dense_vector: list[float], alpha: float = 0.7) -> str:
    """Combine dense vector similarity with keyword scoring.
    alpha=0.7 means 70% weight on dense similarity, 30% on keyword overlap."""
    # Retrieve extra candidates for re-ranking
    dense_results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=dense_vector,
        limit=10,
    )

    if not dense_results:
        return "No relevant information found."

    query_terms = set(query.lower().split())
    scored = []

    for point in dense_results:
        dense_score = point.score
        text = point.payload.get("text_segment", "").lower()

        # Keyword overlap score
        doc_terms = set(text.split())
        overlap = len(query_terms & doc_terms)
        if len(query_terms) > 0:
            tf_sum = sum(text.count(term) for term in query_terms)
            keyword_score = (overlap / len(query_terms)) * min(1.0, tf_sum / 10.0)
        else:
            keyword_score = 0.0

        # Weighted hybrid score
        hybrid = alpha * dense_score + (1 - alpha) * keyword_score

        scored.append((hybrid, text))

    # Sort by hybrid score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


# Updated search_documents with hybrid search:
def search_documents(query: str) -> str:
    if not query.strip():
        return "No query provided."
    try:
        response = gemini_client.models.embed_content(
            model="gemini-embedding-2",
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        query_vector = response.embeddings[0].values
        return hybrid_search(query, query_vector, alpha=0.7)
    except Exception as e:
        return f"Search error: {e}"
```

**Tuning `alpha`:**
- **0.5–0.6:** Keyword-heavy — good for code docs or queries with specific proper nouns
- **0.7–0.8:** Balanced — default for general factual questions
- **0.9–1.0:** Semantic-heavy — good for conceptual or abstract questions

### Improvement 3: LLM Re-Ranking (Counters Both)

Retrieve top 10, then use an LLM call to rank by actual relevance:

```python
import json


def rerank_with_llm(query: str, candidates: list[str], top_k: int = 3) -> list[str]:
    """Use the LLM to re-rank retrieved chunks by relevance to the query."""
    prompt = f"""Given the user query below, rank the following document
chunks from most relevant (score=1.0) to least relevant (score=0.0).

Query: {query}

Chunks:
"""
    for i, chunk in enumerate(candidates):
        prompt += f"\n[{i}] {chunk[:200]}..."

    prompt += "\n\nReturn ONLY a JSON array of scores in the same order: [score_0, score_1, ...]"

    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )

    scores = json.loads(response.text)
    paired = list(zip(candidates, scores))
    paired.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in paired[:top_k]]


# Updated search_documents with re-ranking:
def search_documents(query: str) -> str:
    if not query.strip():
        return "No query provided."
    try:
        response = gemini_client.models.embed_content(
            model="gemini-embedding-2",
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        query_vector = response.embeddings[0].values

        # Retrieve top 10 candidates
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=10,
        )
        candidates = [
            r.payload.get("text_segment", "")
            for r in results
            if r.payload.get("text_segment")
        ]
        if not candidates:
            return "No relevant information found."

        # Re-rank with LLM
        top_results = rerank_with_llm(query, candidates, top_k=1)
        return top_results[0]

    except Exception as e:
        return f"Search error: {e}"
```


---

## Deciding Which Improvement to Use

| Scenario | Recommended Fix |
|----------|----------------|
| Short user queries (< 20 tokens) retrieve wrong chunks | **Query Expansion** (Improvement 1) |
| Long documents where specific terms keep getting lost | **Hybrid Search** (Improvement 2) + **Semantic Chunking** (Improvement 4) |
| Top-1 result is often wrong but top-5 contains the answer | **LLM Re-ranking** (Improvement 3) |
| Mixed problems | (4) re-chunk → (1) expand query → (2) hybrid search → (3) LLM re-rank |

---

## Graded Activity — Memory Bridging Extension

### File Structure

```
activity14/
├── bridging_search_tool.py       # Qdrant-wrapped search_documents + improvements
├── search_comparison.py          # Before/after comparison script
└── search_quality_report.md      # Analysis of improvement impact
```

### Requirement A: Qdrant Tool Integration (30%)

Replace the mock KB in your ReAct loop with real Qdrant retrieval.

**Must demonstrate:**
1. The `search_documents` tool connects to your existing Qdrant collection.
2. It embeds the query using `gemini-embedding-2` before searching.
3. It returns a clean string (not a raw Qdrant object).
4. It handles errors gracefully (empty results, connection failures, embedding errors).

### Requirement B: Search Improvement Implementation (40%)

Implement **at least two** of the four improvements:

| Improvement | Points | Complexity |
|-------------|--------|------------|
| Query Expansion (Improvement 1) | 18 | Medium |
| Hybrid Search (Improvement 2) | 22 | High |
| LLM Re-ranking (Improvement 3) | 22 | High |

### Requirement C: Before/After Comparison (20%)

Create `search_comparison.py` that runs the same 10 queries against:

1. **Baseline:** Your original Qdrant search (before any improvements)
2. **Improved:** Your search with improvements applied
3. **Oracle:** The correct chunk you expect to retrieve (ground truth)

```python
TEST_QUERIES = [
    {
        "query": "What distance metric does Qdrant use?",
        "expected_chunk": "Cosine distance",
        "topics": ["qdrant", "setup"],
    },
    {
        "query": "What is the travel budget for flights?",
        "expected_chunk": "$2000",
        "topics": ["budget", "travel"],
    },
    # ... add 8 more covering your stored corpus
]


def compare_search(queries):
    """Run baseline vs. improved search and report results."""
    for item in queries:
        q = item["query"]

        # Baseline: original search (or mock KB as proxy)
        baseline = original_search_documents(q)

        # Improved: with your chosen improvements
        improved = search_documents(q)

        # Score: does the result contain the expected text?
        expected = item["expected_chunk"].lower()
        baseline_hit = expected in baseline.lower() if baseline else False
        improved_hit = expected in improved.lower() if improved else False

        print(f"  Q: {q}")
        print(f"    Baseline: {baseline_hit} | Improved: {improved_hit}")
        print(f"    Baseline chunk: {baseline[:80]}...")
        print(f"    Improved chunk: {improved[:80]}...")
```

### Requirement D: Quality Report (10%)

Write `search_quality_report.md` with:

1. **Improvements Chosen:** Which two (or more) did you implement and why?
2. **Results Table:**

| Query | Expected Contains? | Baseline Hit | Improved Hit | Improvement? |
|-------|-------------------|--------------|--------------|--------------|
| What distance metric... | "Cosine distance" | ✗ | ✓ | +1 |
| ... | ... | ... | ... | ... |

3. **Analysis:** For queries the baseline missed but improved found, what was the root cause? (Dilution? Asymmetry?)
4. **Trade-offs:** What is the cost of your improvements? (Extra LLM calls? Latency? API costs?)


## Appendix: Wiring It All Together — Updated Tool Registration

When your `search_documents` uses real Qdrant, update the tool description to match:

```python
types.FunctionDeclaration(
    name="search_documents",
    description=(
        "Search the persistent Qdrant knowledge base for factual information "
        "about course topics, user preferences, or stored documents. Use this "
        "when the question requires knowledge not already in the conversation."
    ),
    parameters=...
)
```

The `AVAILABLE_FUNCTIONS` dict stays the same — it already points to `search_documents`:

```python
AVAILABLE_FUNCTIONS = {
    "search_documents": search_documents,  # Now calls real Qdrant
    "calculate": calculate,
    "clarify": clarify,
}
```

Your ReAct loop from the main guide needs **zero changes** — it calls `AVAILABLE_FUNCTIONS[tool_name]` regardless of whether `search_documents` uses a mock KB or real Qdrant. This is the power of the tool dispatch pattern.
