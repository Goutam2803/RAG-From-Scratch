import os
import faiss
import pickle
from groq import Groq
from dotenv import load_dotenv
from embedder import model

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

index = faiss.read_index("faiss_index.bin")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


def search(query, top_k=2):

    query_embedding = model.encode([query])

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for i in indices[0]:
        results.append(chunks[i])

    return results


def ask_with_rag(question):

    retrieved_chunks = search(question)

    context = "\n".join(retrieved_chunks)

    prompt = f"""
Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content, context