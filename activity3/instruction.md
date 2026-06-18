# Activity3: Simulated Tool Use (ReAct)
## Objective
Master the **ReAct** pattern (Reason + Act). You will learn how to manually manage a loop where an agent requests a tool, receives data from the system, and generates a final answer.

---

### Step 1: The ReAct Persona
To use tools, an agent must know they exist and how to "call" them using a specific text format.

**Task:** Create `app.py` and define an identity that includes two simulated tools.
```python
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

react_identity = """
You are a Research Assistant. You have access to the following tools:
1. get_weather(city): Returns current temperature.
2. get_stock_price(symbol): Returns current stock price.

If you need a tool, respond ONLY with: TOOL: [tool_name]([params])
Once you have the info, provide the final answer.
"""
```

---

### Step 2: The Reasoning Phase (Turn 1)
In the first turn, the agent "Perceives" the user request and "Reasons" which tool is needed.

**Task:** Start the `simulated_react_agent` function and capture the first response.
```python
def simulated_react_agent(user_query):
    print(f"\n[USER]: {user_query}")
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config={"system_instruction": react_identity},
        contents=user_query
    )
    # ... logic continues in next step
```

---

### Step 3: The Action & Observation Phase
If the agent requests a tool, we (the system) must provide the "Observation."

**Task:** Add the logic to detect a tool call and provide simulated data.
```python
    if "TOOL:" in response.text:
        tool_call = response.text.split("TOOL:")[1].strip()
        print(f"[ACTION] Agent requested tool: {tool_call}")
        
        # SIMULATION: We pretend the tool returned this data
        observation = "OBSERVATION: The temperature in Manila is 32°C and sunny."
        print(f"[OBSERVE] Providing data: {observation}")
```

---

### Step 4: The Final Response (Turn 2)
Now we send the original query, the agent's thought, and our observation back to the model.

**Task:** Complete the function by generating the final answer.
```python
        final_response = client.models.generate_content(
            model="gemini-2.0-flash",
            config={"system_instruction": react_identity},
            contents=[user_query, response.text, observation]
        )
        print(f"\n[FINAL ANSWER]: {final_response.text}")
    else:
        print(f"\n[FINAL ANSWER]: {response.text}")

# Run it!
simulated_react_agent("What should I wear in Manila today?")
```

---

### Step 5: Verification
Save all files to activity3 including the screenshots of output.
1. Did the agent output `TOOL: get_weather('Manila')`?
2. Did the final answer incorporate the `32°C` data?
3. **Reflection:** Why did we have to send `[user_query, response.text, observation]` as a list in Turn 2?
