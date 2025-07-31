import os
from langchain_openai import OpenAI

# Set your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
# Initialize the OpenAI LLM
llm = OpenAI(temperature=0.7)

def chat():
    print("💬 Hi, this is Natty. Ask me anything! (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        response = llm(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat()

