import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from embedder import load_document, chunk_text, create_embeddings, model
from vector_store import build_faiss_index

# Load and chunk document
document = load_document("data/knowledge.txt")
chunks = chunk_text(document)

# Create embeddings
embeddings = create_embeddings(chunks)

# Build FAISS index
index = build_faiss_index(embeddings)

# Question
question = "What is Project Helios?"

# Question embedding
query_embedding = model.encode([question])

# Retrieve nearest chunks
distances, indices = index.search(query_embedding, 2)
retrieved_indices = indices[0]

# Combine embeddings
all_embeddings = np.vstack([embeddings, query_embedding])

# Reduce to 2D
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
reduced = tsne.fit_transform(all_embeddings)

doc_points = reduced[:-1]
query_point = reduced[-1]

plt.figure(figsize=(8,6))

# Plot document chunks
for i in range(len(doc_points)):
    if i in retrieved_indices:
        plt.scatter(doc_points[i,0], doc_points[i,1], s=100)
        plt.annotate(f"Retrieved {i+1}", (doc_points[i,0], doc_points[i,1]))
    else:
        plt.scatter(doc_points[i,0], doc_points[i,1], alpha=0.3)

# Plot question
plt.scatter(query_point[0], query_point[1], marker="*", s=300)
plt.annotate("Question", (query_point[0], query_point[1]))

plt.title("RAG Retrieval Visualization (Embedding Space)")
plt.show()