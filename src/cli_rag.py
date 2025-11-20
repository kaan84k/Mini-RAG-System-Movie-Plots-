import argparse
import json
from mini_rag_core import answer_query


def main():
    # CLI args: query text, number of contexts, and whether to include contexts
    parser = argparse.ArgumentParser(description="Mini RAG System CLI")
    parser.add_argument("--query", required=True, type=str,
                        help="Movie-related question")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of retrieved chunks (default: 5)")
    parser.add_argument("--show_contexts", action="store_true",
                        help="Show underlying retrieved text")

    args = parser.parse_args()

    # Call the core pipeline to get an answer
    result = answer_query(args.query, top_k=args.top_k)

    # Prepare JSON output for the user
    output = {
        "answer": result["answer"],
        "contexts": result["contexts"] if args.show_contexts else [],
        "reasoning": result["reasoning"]
    }

    # Print nicely formatted JSON
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
