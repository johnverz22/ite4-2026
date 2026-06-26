
## State-Sustained Conversational Chat with Long-Term Memory

### Core Workflow

1. **Initialize a Chat Session:** Start a conversational chat stream with a baseline system behavior template.
2. **Context Injection:** On every incoming message, query Qdrant using the query vector to fetch relevant long-term memories.
3. **Dynamic Configuration:** Apply the extracted long-term memories into the chat runtime configuration before sending the user's message down the stream.

---

### Implementation Code

```python
import uuid
import time
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue

# Initialize SDK Clients
gemini_client = genai.Client()
qdrant_client = QdrantClient(url="http://localhost:6333") 

COLLECTION_NAME = "agent_long_term_memory"

# Ensure Qdrant is initialized with Gemini-compatible vector configurations
if not qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
    )
    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="user_id",
        field_schema="keyword"
    )

def get_gemini_embedding(text: str) -> list[float]:
    """Generates a dense vector using the up-to-date GenAI SDK."""
    response = gemini_client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )
    return response.embeddings[0].values

def save_memory(user_id: str, memory_text: str):
    """Stores factual background metadata explicitly to Qdrant memory."""
    vector = get_gemini_embedding(memory_text)
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"user_id": user_id, "memory": memory_text}
            )
        ]
    )

def recall_relevant_memories(user_id: str, query: str, limit: int = 2) -> list[str]:
    """Searches Qdrant for memories matching the current conversation turn context."""
    query_vector = get_gemini_embedding(query)
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        ),
        limit=limit,
        score_threshold=0.70
    )
    return [hit.payload["memory"] for hit in search_results]

```

### Running the Live Multi-Turn Memory Conversation

This execution demonstrates how the model updates its long-term understanding without losing track of the current short-term conversation thread.

```python
# Unique session tracker
USER_SESSION_ID = "user_dev_99"

# Seed the database with some historical facts from prior weeks
save_memory(USER_SESSION_ID, "User mentioned his favorite programming language is Python.")
save_memory(USER_SESSION_ID, "User has a dog named Barnaby who is a golden retriever.")

# 1. Initialize Gemini's stateful chat object
chat = gemini_client.chats.create(model="gemini-2.5-flash")

# --- CONVERSATION TURN 1 ---
user_msg_1 = "Hey! Can you recommend a quick script template to build a web scraper? Keep my tech preferences in mind."

# Fetch relevant memory context for Turn 1
memories_turn_1 = recall_relevant_memories(USER_SESSION_ID, user_msg_1)
print(f"[Memory System Log] Found relevant memories: {memories_turn_1}")

# Construct dynamic system instruction containing memory state
sys_instruction_1 = f"""
You are an advanced personal AI companion. 
Long-term facts known about this user:
{chr(10).join(f'- {m}' for m in memories_turn_1)}
Tailor recommendations to match their profile seamlessly without explicitly confirming you fetched database records.
"""

# Send message through the stateful chat session with the step's specific runtime configuration
response_1 = chat.send_message(
    user_msg_1,
    config=types.GenerateContentConfig(system_instruction=sys_instruction_1)
)
print(f"\nUser: {user_msg_1}")
print(f"Gemini: {response_1.text}\n")


# --- CONVERSATION TURN 2 (Testing standard short-term chat continuity) ---
user_msg_2 = "That looks perfect! What kind of toy should I get my dog that wouldn't shred easily under a strong bite?"

# Fetch relevant memory context for Turn 2 (Will look for dog-related facts)
memories_turn_2 = recall_relevant_memories(USER_SESSION_ID, user_msg_2)
print(f"[Memory System Log] Found relevant memories: {memories_turn_2}")

sys_instruction_2 = f"""
You are an advanced personal AI companion. 
Long-term facts known about this user:
{chr(10).join(f'- {m}' for m in memories_turn_2)}
Tailor recommendations to match their profile seamlessly without explicitly confirming you fetched database records.
"""

# Send second message through the continuous chat stream
response_2 = chat.send_message(
    user_msg_2,
    config=types.GenerateContentConfig(system_instruction=sys_instruction_2)
)
print(f"User: {user_msg_2}")
print(f"Gemini: {response_2.text}\n")

```

---

### How Gemini Handles This Under the Hood

1. **In Turn 1:** Qdrant returns *"User mentioned his favorite programming language is Python"*. Gemini immediately responds with a boilerplate structured in Python (`BeautifulSoup` or `requests`).
2. **In Turn 2:** When you ask about your dog, the vector search drops the Python context entirely and pulls *"User has a dog named Barnaby who is a golden retriever"*. Gemini customizes its recommendation precisely for a large-breed Retriever chewing profile (heavy duty KONGs, durable rubber), keeping the conversation completely natural.
