## Scenario 4: Video Transcript Semantic Indexing & Stateful RAG Chat Agent

### Step 1: Core Setup and Client Initialization

First, make sure you have the required packages installed:

```bash
pip install google-genai qdrant-client llama-index-core llama-index-embeddings-google

```

Next, configure your Qdrant collection to match Gemini's native **3,072-dimensional** vector architecture.

```python
import uuid
import os
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Initialize System Clients
gemini_client = genai.Client()
qdrant_client = QdrantClient(url="http://localhost:6333")

COLLECTION_TRANSCRIPTS = "video_transcripts"

# Configure collection with optimized metric sizing
if not qdrant_client.collection_exists(COLLECTION_TRANSCRIPTS):
    qdrant_client.create_collection(
        collection_name=COLLECTION_TRANSCRIPTS,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
    )
    print(f"Collection '{COLLECTION_TRANSCRIPTS}' initialized.")

```

---

### Step 2: Processing Video Transcripts via LlamaIndex

This function implements the gold standard for audio/video transcripts: it sets a **`buffer_size=3`** to group spoken fragments together into coherent conversational windows before evaluating where a topic shifts.

```python
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.google import GeminiEmbedding

def process_and_index_transcript(video_title: str, raw_transcript: str):
    """
    Groups conversational video transcript fragments using a sliding semantic window
    and indexes them into Qdrant.
    """
    # 1. Initialize LlamaIndex's Google Embedding wrapper
    embed_model = GeminiEmbedding(model_name="models/gemini-embedding-2")
    
    # 2. Configure the Semantic Splitter specifically for messy spoken text
    splitter = SemanticSplitterNodeParser(
        buffer_size=3,                         # Groups sentences to smooth over spoken pauses
        breakpoint_percentile_threshold=95,   # Cuts when a clear new topic begins
        embed_model=embed_model
    )
    
    # 3. Parse the transcript text into semantic nodes
    doc = Document(text=raw_transcript)
    nodes = splitter.get_nodes_from_documents([doc])
    
    print(f"\nProcessing '{video_title}': Split transcript into {len(nodes)} semantic topics.")
    
    points = []
    for node in nodes:
        # Use the latest GenAI SDK to generate the optimized document vector
        response = gemini_client.models.embed_content(
            model="gemini-embedding-2",
            contents=node.text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        vector = response.embeddings[0].values
        
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text_segment": node.text,
                    "video_title": video_title
                }
            )
        )
        
    # Bulk upload points into Qdrant
    if points:
        qdrant_client.upsert(collection_name=COLLECTION_TRANSCRIPTS, points=points)
        print(f"Successfully indexed all segments for '{video_title}'.")

```

---

### Step 3: Vector Search & Multi-Turn Conversational Chat

This setup creates a live, stateful chat stream using `gemini-client.chats.create()`. It intercepts every user prompt, pulls relevant video timestamps/segments from Qdrant, and updates Gemini's system instructions on the fly.

```python
def search_transcripts(query: str, limit: int = 2) -> str:
    """Queries Qdrant using a specialized retrieval query vector."""
    response = gemini_client.models.embed_content(
        model="gemini-embedding-2",
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    query_vector = response.embeddings[0].values
    
    results = qdrant_client.search(
        collection_name=COLLECTION_TRANSCRIPTS,
        query_vector=query_vector,
        limit=limit,
        score_threshold=0.70
    )
    
    if not results:
        return "No relevant video context found matching this query."
        
    context_blocks = []
    for hit in results:
        context_blocks.append(
            f"From Video [{hit.payload['video_title']}]:\n\"{hit.payload['text_segment']}\""
        )
        
    return "\n\n".join(context_blocks)


def start_transcript_chat_agent():
    """Launches a live conversational loop over your indexed videos."""
    # Create the stateful continuous chat session
    chat_session = gemini_client.chats.create(model="gemini-2.5-flash")
    
    print("\n=======================================================")
    print("Video Transcript AI Agent Online. Type 'quit' to exit.")
    print("=======================================================\n")
    
    while True:
        user_query = input("You: ").strip()
        if not user_query:
            continue
        if user_query.lower() in ["exit", "quit"]:
            break
            
        # 1. Pull relevant background context from video transcripts
        video_context = search_transcripts(user_query, limit=2)
        
        # 2. Inject context into instructions while maintaining conversation state
        dynamic_instruction = f"""
        You are an advanced AI assistant specializing in analyzing video content.
        Answer the user's questions utilizing the verified video transcript segments provided below.
        Always cite the source video title when referencing facts.
        
        [RETRIEVED VIDEO TRANSCRIPT SECTIONS]
        {video_context}
        """
        
        # 3. Transmit the user message through the continuous session
        response = chat_session.send_message(
            user_query,
            config=types.GenerateContentConfig(
                system_instruction=dynamic_instruction,
                temperature=0.3 # Low temperature for accurate synthesis
            )
        )
        
        print(f"\nAgent: {response.text}\n")

```

---

### Step 4: Run the Complete Pipeline

Here is a sample execution. We simulate a messy raw transcript (with spoken fragments and run-on thoughts), index it, and launch the chat agent interface.

```python
if __name__ == "__main__":
    # Simulate a raw video lecture transcript
    sample_transcript = (
        "okay hello everyone today we are talking about building solar panels so basically "
        "the key ingredient you need is high grade silicon because silicon is what captures the photons "
        "right and then another thing people forget is the framing you must use anodized aluminum "
        "for the outer frame structural reasons mostly because if it rains steel is going to rust "
        "and ruin the roof array installation and that is a major problem we want to avoid."
    )
    
    # 1. Ingest and parse semantically
    process_and_index_transcript(
        video_title="Solar Engineering Lecture 101", 
        raw_transcript=sample_transcript
    )
    
    # 2. Start the conversation loop
    # Try asking: "What metal should I use for the frame and why?"
    start_transcript_chat_agent()

```
