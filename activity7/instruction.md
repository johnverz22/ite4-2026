# Activity 7: Qdrant Memory and Chunking Strategy

---

## Preliminary Step: Environment Setup (Windows + WSL + Docker)

Before writing any Python code, you must set up your local infrastructure. We will run Qdrant inside a Docker container managed via Windows Subsystem for Linux (WSL).

### 1. Install and Configure WSL 2

1. Open PowerShell as Administrator and run:
```powershell
wsl --install

```


2. Restart your computer if prompted.
3. Ensure your WSL is set to version 2 by running:
```powershell
wsl --set-default-version 2

```

### 2. Spin Up Qdrant

Open your terminal (Command Prompt, PowerShell, or WSL terminal) and execute the following command to download and start a local instance of Qdrant:

```bash
docker run -d -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant

```

> **What do these flags mean?**
> * `-d`: Runs the container in "detached" mode (in the background).
> * `-p 6333:6333`: Exposes the HTTP API port so your Python script can communicate with it.
> * `-v`: Creates a persistent volume (`qdrant_storage`) so your database collections aren't wiped when the container stops.
> 
> 

Verify it is running by opening your browser and visiting: `http://localhost:6333/dashboard`

---

## Step 1: Dependencies and Environment Setup

Create a file named `activity_07.py`. Copy the initial setup below, which handles imports, environment variables, and our test document.

```python
import os
import re
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

COLLECTION_NAME = "activity7_memory"
VECTOR_SIZE = 8  # Match the size of our custom vocabulary matrix

SOURCE_TEXT = """
Qdrant is a vector database designed for similarity search and retrieval.

Chunking is the process of dividing a document into smaller pieces before embedding.
If a chunk is too large, it may contain too much unrelated information.
If a chunk is too small, it may lose the surrounding context needed for the answer.

Overlap helps preserve meaning when a sentence or idea crosses a boundary.
Metadata such as source, section, and strategy makes debugging easier.
"""

```

---

## Step 2: Implement Strategy 1 – Fixed-Size Chunking

Fixed-size chunking uses a sliding window based on character counts. It is simple to implement but can cut through sentences mid-word if not carefully managed.

### Code Challenge:

Complete the logic inside the `while` loop below to properly slice the text into a chunk and increment the `start` position while taking the `overlap` into account.

```python
def fixed_size_chunk(text: str, chunk_size: int = 140, overlap: int = 30) -> list[str]:
    """Split text into overlapping character windows."""
    chunks: list[str] = []
    start = 0

    while start < len(text):
        # 1. Determine the end boundary (don't exceed the text length)
        end = min(start + chunk_size, len(text))
        
        # 2. Extract and clean the current chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # ------ YOUR CODE HERE ------
        # Task: Update 'start' so that the next iteration begins back by the 'overlap' value.
        start = end - overlap
        # ----------------------------

        # Safety checks to prevent infinite loops on small text
        if start <= 0 and end == len(text):
            break
        if start < 0:
            start = 0

    return chunks

```

### Checkpoint 1: Visualizing Boundaries

> If a document is exactly 200 characters long, `chunk_size` is 100, and `overlap` is 20:
> * Where does the first chunk start and end? `[0, ___]`
> * Where does the second chunk start and end? `[___, ___]`


---

## Step 3: Implement Strategy 2 – Structural Paragraph Chunking

Paragraph chunking respects document formatting by breaking text only where blank lines occur, keeping related sentences together.

### Code Challenge:

Fill in the regular expression pattern inside `re.split()` that identifies a blank line (one or more newline characters, potentially with empty spacing between them).

```python
def paragraph_chunk(text: str) -> list[str]:
    """Split text on blank line boundaries."""
    # ------ YOUR CODE HERE ------
    # Task: Replace None with a regex pattern that catches one or more newlines separating paragraphs.
    regex_pattern = r"\n\s*\n" 
    # ----------------------------
    
    paragraphs = [p.strip() for p in re.split(regex_pattern, text) if p.strip()]
    return paragraphs

```

---

## Step 3: The Embedding Generator

To avoid complex API setups, we are using a **deterministic keyword sketch vector**. It checks for the frequency of 8 specific vocabulary words.

### Code Challenge:

Complete the normalization logic. Normalization scales the vector down so that a longer chunk containing a word multiple times doesn't automatically "win" purely based on raw size.

```python
def embed_text(text: str) -> list[float]:
    """Create a tiny deterministic 8-dimension vector representation."""
    vocab = ["qdrant", "chunking", "embedding", "overlap", "metadata", "retrieval", "context", "vector"]
    lowered = text.lower()
    vector = [float(lowered.count(word)) for word in vocab]

    # Calculate the vector magnitude (Euclidean norm)
    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0:
        return [0.0 for _ in vector]

    # ------ YOUR CODE HERE ------
    # Task: Return a list where every element 'v' in 'vector' is divided by its 'norm'.
    return [v / norm for v in vector]
    # ----------------------------

```

---

## Step 4: Storing Points with Metadata in Qdrant

When storing data in a vector database, we don't just store vectors; we attach a **payload** (metadata) that tracks its source origin and extraction history.

### Code Challenge:

Fill out the dictionary properties inside `PointStruct` to attach the raw text string, the strategy name (`fixed_size` or `paragraph`), and its numerical sequence index.

```python
def store_chunks(client, collection_name: str, chunks: list[str], strategy: str) -> None:
    """Insert each chunk into Qdrant with metadata payloads."""
    points = []

    for index, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=f"{strategy}-{index}",
                vector=embed_text(chunk),
                # ------ YOUR CODE HERE ------
                payload={
                    "text": chunk,
                    "strategy": strategy,
                    "chunk_index": index,
                    "source": "sample_doc",
                },
                # ----------------------------
            )
        )

    client.upsert(collection_name=collection_name, points=points)

```

---

## Step 5: Querying Vector Spaces

```python
def retrieve_best_match(client, collection_name: str, query_vector: list[float]):
    """Query the collection and return the single top-scoring result."""
    result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    return result.points[0] if result.points else None

```

---

## Step 6: Orchestration & Analysis

Now construct your execution orchestrator. Copy this block to complete your script:

```python
def main():
    # Initialize connection to local storage engine running on Docker
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

    # Reset collection space
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    # Process strategies
    fixed_chunks = fixed_size_chunk(SOURCE_TEXT)
    paragraph_chunks = paragraph_chunk(SOURCE_TEXT)

    # Print out splits for review
    print("--- Fixed-size Chunks ---")
    for i, chunk in enumerate(fixed_chunks): print(f" {i}: {chunk}")
    print("\n--- Paragraph Chunks ---")
    for i, chunk in enumerate(paragraph_chunks): print(f" {i}: {chunk}")

    # Write to DB
    store_chunks(client, COLLECTION_NAME, fixed_chunks, "fixed_size")
    store_chunks(client, COLLECTION_NAME, paragraph_chunks, "paragraph")

    # Target Query
    query_text = "Why does overlap help when chunking a document?"
    query_vector = embed_text(query_text)

    match = retrieve_best_match(client, COLLECTION_NAME, query_vector)
    
    print(f"\nQuery: {query_text}")
    if match:
        payload = match.payload
        print(f"\nBest Match Strategy Found: [ {payload.get('strategy').upper()} ]")
        print(f"Chunk Index Location: {payload.get('chunk_index')}")
        print(f"Text Returned:\n\"{payload.get('text')}\"")

if __name__ == "__main__":
    main()

```

---

## Final Critical Reflection Checkpoints

To complete this activity, execute your script and answer these 4 questions in your workspace submission markdown:

1. **Which chunking strategy returned the most relevant text for your query?** Look closely at the exact string fragment returned—did it capture the *entire sentence* context or was it cut off?
2. **What happened to the text structure in Fixed-Size Chunk #2 vs. Paragraph Chunk #2?** Identify how boundaries changed word availability.
3. **Hypothetical Application:** Imagine you are building a production AI system for a company's internal HR manual handbook. Why might relying *exclusively* on Fixed-Size character chunking create bad answers for employees?
4. **The Metadata Payload:** Why do we spend computing effort storing things like `chunk_index` and `strategy` inside the database alongside raw vectors? Why can't we just store the text string alone?
