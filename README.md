# Mini RAG System (Movie Plots)

This project is a small, simple **Retrieval-Augmented Generation (RAG)** system built on top of movie plots. It shows how to:

* Read a CSV dataset of movie plots
* Use only a **subset of 500 movies** for a lightweight demo
* Split (chunk) long plot text into smaller pieces
* Create embeddings and store them in **ChromaDB**
* Use **Google Gemini** to answer questions based on retrieved plot chunks
* Interact with the system using a **command-line interface (CLI)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_882_kPb0gP5-OQJXjF9sPCT8ukiqilg?usp=sharing)

[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots?resource=download&utm_source=chatgpt.com)

---

## 1. Project Structure

```bash
mini-rag-movie-plots/
│
├── data/
│   ├── wiki_movie_plots_deduped.csv       # Movie plot dataset (add manually)
│   └── chroma_store/                      # ChromaDB persistent storage
│
├── src/
│   ├── mini_rag_core.py                   # RAG core logic (retrieval + Gemini)
│   ├── cli_rag.py                         # CLI entry point
│   └── ingestion.py                       # Ingestion: first 500 rows → chunks → embeddings                           
│
├── notebooks/
│   └── Mini_RAG_System(Movie_Plots).ipynb # Original Colab notebook
│
├── requirements.txt
├── README.md
└── .env                                   
```

---

## 2. What the System Does

1. **Ingestion** (`ingestion.py`)

   * Loads the CSV dataset from `data/wiki_movie_plots_deduped.csv`.
   * Selects only the **first 500 rows** for a smaller, faster demo.
   * Keeps only the `Title` and `Plot` columns.
   * Splits each plot into chunks of about 300 words.
   * Stores all chunks with metadata (title + chunk id) in **ChromaDB**.

2. **Core RAG Logic** (`mini_rag_core.py`)

   * Loads the persistent ChromaDB collection (`movie_plots`).
   * Uses a SentenceTransformer model (`all-MiniLM-L6-v2`) for embeddings.
   * Configures **Gemini 2.5 Flash** with an API key from `.env`.
   * For a given user query:

     * Retrieves the most relevant chunks from ChromaDB.
     * Builds a context prompt.
     * Sends that context + question to Gemini.
     * Returns a JSON-style result with answer, reasoning and contexts.

3. **CLI Interface** (`cli_rag.py`)

   * Provides a command-line tool to ask questions about movies.
   * Accepts a `--query` argument and optional flags like `--top_k` and `--show_contexts`.

---

## 3. Setup Guide

### 3.1 Clone the Repository

```bash
git clone https://github.com/your-username/mini-rag-movie-plots.git
cd mini-rag-movie-plots
```

### 3.2 Create and Activate Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\\Scripts\\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3.3 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure API Key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

The file is used by `mini_rag_core.py` via `python-dotenv`.

---

## 5. Add Dataset

Download the **Wikipedia Movie Plots** dataset (e.g., from Kaggle).

Place the CSV as:

```bash
data/wiki_movie_plots_deduped.csv
```

The ingestion script will automatically use the **first 500 rows** only.

---

## 6. Run Ingestion (First 500 Rows)

This step loads the CSV, selects the first 500 records, chunks the plots, and writes embeddings to ChromaDB.

```bash
python src/ingestion.py
```

This will create or update a persistent vector store in:

```bash
data/chroma_store/
```

---

## 7. Run the CLI

Once ingestion is complete, you can query the system from the terminal.

### 7.1 Basic Query

```bash
python src/cli_rag.py --query "Which movie features an action and adventure plot?"
```

### 7.2 Show Contexts Used for the Answer

```bash
python src/cli_rag.py --query "A movie about a scientist" --show_contexts
```

### 7.3 Change Number of Retrieved Chunks

```bash
python src/cli_rag.py --query "A movie involving robots" --top_k 10
```

---

## 8. Script Responsibilities

### 8.1 `src/ingestion.py`

* Loads CSV from `data/`.
* Uses only the **first 500 movies**.
* Chunks plot text into ~300-word segments.
* Writes chunks + metadata into ChromaDB.

### 8.2 `src/mini_rag_core.py`

* Sets up ChromaDB persistent client.
* Configures embedding model.
* Configures Gemini via `.env`.
* Exposes `answer_query(query, top_k)` for use by other modules.

### 8.3 `src/cli_rag.py`

* Simple CLI wrapper.
* Uses `answer_query()` from `mini_rag_core.py`.
* Prints answer, reasoning and (optionally) context snippets to the terminal.

---



## 9. Summary

This project is a small, focused example of a RAG pipeline that:

* Uses only a **subset (500 rows)** of a larger dataset to stay fast.
* Splits long text into chunks for better retrieval.
* Leverages ChromaDB for vector search.
* Uses Gemini to generate grounded answers.
* Exposes an easy-to-use CLI for quick experimentation.
