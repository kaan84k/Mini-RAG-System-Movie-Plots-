import pandas as pd
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os

DATA_PATH = "data/wiki_movie_plots_deduped.csv"


def chunk_text(text, max_words=300):
    if not isinstance(text, str):
        text = str(text)

    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # We only need the title and plot for ingestion
    print("Selecting needed columns...")
    df = df[["Title", "Plot"]]

    # Limit rows for faster local testing
    print("Limiting to first 500 rows for smaller ingestion...")
    df = df.head(500)
    print(f"Total movies selected: {len(df)}")

    # Turn each plot into one or more chunks with a chunk_id
    print("Chunking plots...")
    rows = []
    for _, row in df.iterrows():
        title = row["Title"]
        plot = row["Plot"]
        chunks = chunk_text(plot, max_words=300)
        for idx, ch in enumerate(chunks):
            rows.append({
                "Title": title,
                "chunk_id": idx,
                "Plot_chunk": ch
            })

    chunked = pd.DataFrame(rows)
    print(f"Total generated chunks: {len(chunked)}")

    # Initialize local ChromaDB and insert chunks
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path="data/chroma_store")

    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name="movie_plots",
        embedding_function=embedding_fn
    )

    print("Inserting into ChromaDB...")
    docs = chunked["Plot_chunk"].tolist()
    metas = [
        {"Title": row["Title"], "chunk_id": int(row["chunk_id"])}
        for _, row in chunked.iterrows()
    ]
    ids = [str(i) for i in range(len(docs))]

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas
    )

    print("Ingestion completed successfully!")


if __name__ == "__main__":
    main()
