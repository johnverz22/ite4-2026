# Activity 2: Goal-Driven Autonomy
## Objective
Learn how to use Python loops to simulate "Planning" and "Autonomy," allowing an agent to break a high-level goal into actionable steps.

---

### Step 1: The Planner Identity
An autonomous agent needs a clear objective. We will create a **Strategic Project Planner**.

**Task:** Create `app.py` and define the following identity.
```python
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

planner_identity = """
You are a Strategic Project Planner. 
Your goal is to take a high-level objective and break it into exactly 3 sequential steps.
Respond ONLY in a numbered list.
"""
```

---

### Step 2: Implementing the Autonomous Loop
Instead of a single prompt, we use a `while` loop. This simulates the agent "working through" a problem step-by-step.

**Task:** Implement the `autonomous_planner` function.
```python
def autonomous_planner(goal):
    print(f"\n[GOAL]: {goal}")
    print("-" * 30)
    
    steps_taken = 0
    max_steps = 3
    plan = []
    
    while steps_taken < max_steps:
        print(f"[REASONING] Planning Step {steps_taken + 1}...")
        
        # We pass the current 'plan' back to the model so it knows what it already decided.
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config={"system_instruction": planner_identity},
            contents=f"Goal: {goal}. Current steps completed: {plan}. Analyze the goal and provide Step {steps_taken + 1}:"
        )
        
        step_description = response.text.strip()
        plan.append(step_description)
        steps_taken += 1
    
    return plan
```

---

### Step 3: Execution
Now, give your agent a complex task.

**Task:** Execute the planner and display the final 3-step plan.
```python
my_goal = "Deploy a secure web application for a small business."
final_plan = autonomous_planner(my_goal)

print("\n--- FINAL AUTONOMOUS PLAN ---")
for i, step in enumerate(final_plan, 1):
    print(f"STEP {i}: {step}")
```

---

### Step 4: Verification & Reflection.
Place all your codes inside activity2 folder inclduing your answer to the following questions (`reflection.md` file). Make sure to take screenshots of your output and save in `activity2/screenshot` folder.
1. Did the agent stay focused on the goal throughout the loop?
2. **Challenge:** Modify the loop to perform 5 steps instead of 3.
3. How does this affect the detail of the plan?
