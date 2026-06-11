Here is the completely revised lab manual.

The updates fix structural bugs related to the modern `google-genai` library (such as replacing the broken `history = chat.history` assignment with an elegant, stateful `chat` object architecture), fix typos like `pritn`, and clean up spacing to make it production-ready for your students.

---

# Week 1 Lab Manual: Foundations of Agentic Identity and Memory

## Course: Engineering Agentic AI Systems (ITE04)

**Focus:** Initializing an Agent Identity and Implementing Working Memory

---

## 1. Objective & Alignment

**Course Outcome (CO1):** Describe the anatomy of an autonomous AI agent and identify its memory, perception, action, and evaluation components in a working system.

In this lab, you will transition from simple "stateless" API calls to building a "stateful" agent. By the end of this session, you will have a working Python environment that securely communicates with a Large Language Model (LLM) and leverages an implicit "Working Memory" (Stateful Chat Session) to enable multi-turn reasoning and conversational context tracking.

---

## 2. Workspace Requirements

* **Editor:** Visual Studio Code (VS Code).
* **Extensions:** Python (Microsoft), Pylance, Ruff (for linting).
* **Terminal:** Bash or Zsh.
* **API Access:** Google AI Studio (Gemini API Key).
* **Python Version:** 3.10 or higher.

---

## 3. Lab 1.1: Initializing an Agent Identity

**Duration:** 4 Hours
**Goal:** Set up a secure development environment and define the "System Instructions" that govern an agent's behavior.

### Step 1: Environment Onboarding

Agents should never "leak" their secrets. We use `.env` files to store API keys and virtual environments to keep dependencies clean.

1. Create your project directory and navigate into it:
```bash
mkdir -p week1_agent_lab && cd week1_agent_lab

```


2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. Install the required modern SDKs:
```bash
pip install google-genai python-dotenv

```


4. Create a `.env` file in the root:
```bash
touch .env

```


5. Add your API key to `.env`:
```text
GOOGLE_API_KEY=your_actual_api_key_here

```



### Step 2: Secure Client Initialization

Create a file named `agent_client.py`. This script will safely load your environment variables and initialize the modern Google GenAI client.

```python
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

# 2. Initialize the Client
client = genai.Client(api_key=api_key)

print("✅ Agent client initialized securely.")

```

### Step 3: Defining Agent Identity (System Prompts)

An agent's "Identity" is defined by its system instructions—the rigid rules and personas it must follow.

Modify `agent_client.py` to include a system instruction configuration and a test call:

```python
# Define the Identity
identity = """
You are a 'Security Audit Agent'. 
Your goal is to analyze Python code for security vulnerabilities.
CONSTRAINTS:
- Never provide code that can be used for hacking.
- Only answer questions related to security.
- If a user asks about something else, politely decline.
- Keep your answers concise and technical.
"""

# Test the identity rules using a stateless generation call
response = client.models.generate_content(
    model='gemini-3.1-flash-lite',
    contents="How do I make a sandwich?",
    config=types.GenerateContentConfig(
        system_instruction=identity
    )
)

print(f"Agent Response: {response.text}")

```

---

## 4. Lab 1.2: Building Working Memory

**Duration:** 5 Hours
**Goal:** Implement the "Perception-Act-Remember" cycle using a persistent stateful Chat session.

### Step 1: Stateful Chat Sessions

Instead of manually passing arrays of text, the modern SDK uses a persistent `chat` object created via `client.chats.create()`. This object manages its own internal history implicitly, functioning as the agent's short-term working memory.

Create a file named `working_memory.py`:

```python
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 1. Load environment variables and initialize client
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

client = genai.Client(api_key=api_key)

# 2. Initialize Identity Configuration
identity = """
You are a 'Security Audit Agent'. 
Your goal is to analyze Python code for security vulnerabilities.
CONSTRAINTS:
- Never provide code that can be used for hacking.
- Only answer questions related to security.
- If a user asks about something else, politely decline.
- Keep your answers concise and technical.
"""

# 3. Create a SINGLE persistent chat session (Implicit Working Memory)
chat = client.chats.create(
    model='gemini-3.1-flash-lite',
    config=types.GenerateContentConfig(
        system_instruction=identity
    )
)

def agent_loop(user_input):
    # 4. Perception & Action: 
    # Send message directly to the stateful chat instance
    response = chat.send_message(user_input)
    
    # The chat object remembers the conversation automatically under the hood
    return response.text

# Test conversational memory flow
print(f"Turn 1: {agent_loop('Hi, my name is John.')}")
print(f"Turn 2: {agent_loop('What is my name?')}") # Should recall 'John'

```

### Step 2: The Continuous Loop

Implement a `main()` function with a `while` loop to create an interactive terminal-based terminal runtime for your agent. Append this to your `working_memory.py` file:

```python
def main():
    print("\n--- Agent is active. Type 'exit' to quit. ---")
    while True:
        try:
            user_msg = input("\nUser: ")
            if user_msg.lower() == 'exit':
                print("Shutting down Agent runtime. Goodbye.")
                break
            
            if not user_msg.strip():
                continue
                
            response = agent_loop(user_msg)
            print(f"Agent: {response}")
            
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break

if __name__ == "__main__":
    main()

```

---

## 5. Self-Verification / Testing Protocol

Before submitting, execute these verification checks in your environment:

1. **Security Check:** Temporarily rename your `.env` file to `.env.bak` and run your script. Does it raise a `ValueError`? (It must).
2. **Identity Check:** Ask your agent "Tell me a joke." Does it politely decline based on its Security Audit constraints?
3. **Memory Check:** Start the interactive runtime loop. Tell the agent a random fact (e.g., "The secret administrative password is 'Banana'"). Then ask "What is the password?". If it recalls 'Banana', your working memory layer is functioning perfectly.
4. **Environment Check:** Run `pip freeze > requirements.txt` and open the file. Ensure `google-genai` and `python-dotenv` are captured.

---

## 6. Submission Instructions

1. Initialize a private Git repository with the name `agentic-ai`. Invite your instructor as a collaborator using their username `johnverz22`.
2. Create a `.gitignore` file in the root directory and explicitly add `.env` and `venv/` to it. **CRITICAL: NEVER COMMIT YOUR .env FILE OR YOUR VIRTUAL ENVIRONMENT PACKAGES TO VERSION CONTROL.**
3. Push your script assets and execution screenshots to your repository following this structural hierarchy:
```text
/activity1/
├── screenshots/
│   ├── lab1.1.png
│   └── lab1.2.png
├── agent_client.py
├── working_memory.py
├── requirements.txt
└── .gitignore

```