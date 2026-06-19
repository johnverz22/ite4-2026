# Activity 5: The Structured Medical Intake Form
## Objective
Formulate a nested Pydantic data schema, enforce strict integer range validation, and implement a robust self-correction loop using the Google GenAI SDK to safely parse unstructured patient descriptions into structured clinical records.

## Prerequisites
- Completed Week 2 Labs and Activity 4.
- Active `GOOGLE_API_KEY` in your `.env` file.
- Installed `google-genai`, `pydantic`, and `python-dotenv`.

## Concept Mapping
- **Nested Schema Design:** Constructing complex Python objects containing lists of nested Pydantic models (from `week2_01_pydantic_handout.md` and `pydantic_reference.md`).
- **Constrained Decoding:** Forcing the LLM to adhere to the schema using `response_schema` and `response_mime_type: "application/json"` (from `week2_02_schema_handout.md`).
- **Self-Correction & Exception Handling:** Catching `pydantic.ValidationError` and feeding the traceback details back to the LLM to repair formatting or constraint violations (from `week2_03_adversarial_handout.md`).

---

## Task: Build the "Clinical Intake Agent"
Create a Python script named `medical_intake_agent.py` that processes patient symptoms and outputs a verified medical record.

### 1. Schema Specifications (Pydantic & Enums)
You must define the following schemas in your script:
* **Severity Enum:** Define a `str` Enum named `Severity` with values: `LOW`, `MEDIUM`, and `HIGH`.
* **Symptom Model:** Create a `Symptom` Pydantic model with:
  - `symptom_name` (str)
  - `severity` (Severity Enum)
  - `duration_days` (int, must be greater than or equal to 0: `ge=0`)
* **MedicalIntake Model:** Create a `MedicalIntake` Pydantic model with:
  - `symptoms` (list of `Symptom` objects)
  - `allergies` (list of strings)
  - `urgency_rating` (int, strictly between 1 and 10: `ge=1, le=10`)
  - `clinical_reasoning` (str, acting as a Chain of Thought explaining the triage rating and severity selections)

### 2. Requirements:

#### A. The Self-Correction Loop
* Implement a loop that attempts validation up to `max_retries = 3` times.
* If Pydantic throws a `ValidationError`, print the error locally, append the error feedback to the API history context, and call the Gemini API again to get a corrected response.
* If validation succeeds, return the parsed `MedicalIntake` object. If it fails after 3 attempts, raise a custom exception.

#### B. The Adversarial Check
* Test the agent with an adversarial user input designed to trigger validation range failures. For example:
  *"My stomach is cramping incredibly badly since last night! The pain is unbearable, definitely an urgency of 15 out of 10! I don't think I have allergies."*
* Verify that the Pydantic validator rejects the urgency rating of `15` (due to `le=10`), the self-correction loop catches the error, feeds it back, and the agent corrects it to `10`.

---

## Starter Code Structure
Your `medical_intake_agent.py` should follow this structural blueprint:
```python
import os
from enum import Enum
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class Symptom(BaseModel):
    # TODO: Define Symptom fields
    pass

class MedicalIntake(BaseModel):
    # TODO: Define MedicalIntake fields
    pass

def process_intake(patient_input: str) -> MedicalIntake:
    # TODO: Implement the generation loop with max 3 retries
    # TODO: Catch ValidationError, construct feedback, and append to contents list
    pass

if __name__ == "__main__":
    # Test input
    test_input = "My stomach is cramping incredibly badly since last night! The pain is unbearable, definitely an urgency of 15 out of 10! I don't think I have allergies."
    try:
        record = process_intake(test_input)
        print("\n--- Validated Intake Record ---")
        print(record.model_dump_json(indent=2))
    except Exception as e:
        print(f"Failed: {e}")
```

---

## Validation Checklist
- [ ] Does the script successfully catch a `ValidationError` when the model generates an urgency rating of `15`?
- [ ] Does the terminal log show the retry attempt triggered with the specific Pydantic error details?
- [ ] Is the final output successfully parsed into the `MedicalIntake` model and dumped as a clean JSON structure?
- [ ] Does the `clinical_reasoning` field contain a detailed step-by-step triage thought process?

---
*Self-Reflection:* How does using a strict Enum (`Severity`) prevent model hallucination compared to letting the model output any string for severity level?
