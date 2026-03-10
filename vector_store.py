import faiss
import numpy as np
from embedder import load_document, chunk_text, create_embeddings

def build_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def search(index, query_embedding, chunks, top_k=2):

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for i in indices[0]:
        results.append(chunks[i])

    return results


if __name__ == "__main__":

    document = load_document("data/knowledge.txt")

    chunks = chunk_text(document)

    embeddings = create_embeddings(chunks)

    index = build_faiss_index(embeddings)

    print("\nFAISS index built successfully")

    question = "Who founded FutureAI Labs?"

    print("\nQuestion:", question)

    from embedder import model

    query_embedding = model.encode([question])

    results = search(index, query_embedding, chunks)

    print("\nTop retrieved chunks:\n")

    for r in results:
        print("-", r)
        