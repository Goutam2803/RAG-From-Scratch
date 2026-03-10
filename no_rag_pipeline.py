import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_llm(question):

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": question}
        ]
    )

    return response.choices[0].message.content


if __name__ == "__main__":

    question = "Who founded FutureAI Labs?"

    answer = ask_llm(question)

    print("\nQuestion:", question)
    print("\nLLM Answer (No RAG):")
    print(answer)