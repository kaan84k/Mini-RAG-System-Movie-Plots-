import json
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key from a local .env file (set GEMINI_API_KEY there)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Configure Gemini client
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

# Persistent ChromaDB client pointing to the local store
client = chromadb.PersistentClient(path="data/chroma_store")

# Sentence-transformer model used to embed text for retrieval
embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Get or create a collection for movie plot chunks
collection = client.get_or_create_collection(
    name="movie_plots",
    embedding_function=embedding_fn
)


def clean_json_output(raw_text: str):
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "")
        cleaned = cleaned.replace("```", "")
        cleaned = cleaned.strip()
    return cleaned


def answer_query(query: str, top_k: int = 5):
    # Retrieve top-k similar chunks from the collection
    result = collection.query(query_texts=[query], n_results=top_k)
    docs = result["documents"][0]
    metas = result["metadatas"][0]

    # Build a short context string for the model from the retrieved chunks
    context_blocks = []
    for doc, meta in zip(docs, metas):
        context_blocks.append(
            f"Title: {meta.get('Title', '')}\n"
            f"Chunk ID: {meta.get('chunk_id', '')}\n"
            f"Plot snippet: {doc}"
        )

    context_text = "\n\n---\n\n".join(context_blocks)

    # Prompt the model and ask for strict JSON output
    prompt = f"""
You are a movie plot reasoning assistant.

Use ONLY the following context:

{context_text}

Question: {query}

Respond STRICTLY in JSON only:
{{
  "answer": "...",
  "reasoning": "..."
}}
"""

    # Generate content from Gemini
    response = model.generate_content(prompt)
    raw_text = response.text

    # Try to clean and parse the model output as JSON
    cleaned = clean_json_output(raw_text)
    try:
        data = json.loads(cleaned)
    except Exception:
        # If parsing fails, return raw text to help debugging
        data = {
            "answer": raw_text,
            "reasoning": "Model did not return valid JSON after cleaning."
        }

    return {
        "answer": data.get("answer", ""),
        "contexts": docs,
        "reasoning": data.get("reasoning", "")
    }
