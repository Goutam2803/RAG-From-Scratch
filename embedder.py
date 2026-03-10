from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_document(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def chunk_text(text, chunk_size=200):
    chunks = []
    
    sentences = text.split(".")
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + "."
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "."

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings


if __name__ == "__main__":

    document = load_document("data/knowledge.txt")

    chunks = chunk_text(document)

    print("\nChunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}\n")

    embeddings = create_embeddings(chunks)

    print("\nEmbedding shape:", embeddings.shape)