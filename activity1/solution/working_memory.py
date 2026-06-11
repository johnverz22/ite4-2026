import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

# 2. Initialize client
client = genai.Client(api_key=api_key)

# 3. Create a SINGLE stateful chat session (Persistent Working Memory)
chat = client.chats.create(
    model='gemini-3.1-flash-lite'
)

# 4. Agent Loop
def agent_loop(user_input):
    # Simply send the message to the existing chat session.
    # It automatically remembers previous turns.
    response = chat.send_message(user_input)
    return response.text


def main():
    print("\n--- Agent is active. Type 'exit' to quit. ---")
    while True:
        try:
            user_msg = input("\nUser: ")
            if user_msg.lower() == 'exit':
                print("Agent is quitting. Goodbye...")
                break

            if not user_msg.strip():
                continue

            response = agent_loop(user_msg)
            print(f"Agent: {response}")
        except KeyboardInterrupt:
            print("\nSession interruped. Exiting...")
            break;

if __name__ == "__main__":
    main()