import faiss
import pickle
from embedder import load_document, chunk_text, create_embeddings

print("Loading document...")

document = load_document("data/knowledge.txt")

chunks = chunk_text(document)

print("Creating embeddings...")

embeddings = create_embeddings(chunks)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Saving FAISS index...")

faiss.write_index(index, "faiss_index.bin")

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Index built and saved successfully.")