#该脚本用于测试从本地MedRAG风格的文本块中检索相关信息的流程能否跑通，使用的检索器"Contriever": "facebook/contriever"
"""custom_retrieve.py

Usage:
    python scripts/custom_retrieve.py --input_json path/to/input.json --db_dir path/to/corpus_root --corpus CorpusName --retriever RetrieverName --k 10

This script loads a user-provided JSON (or you can pass a raw JSON string with --input_str),
constructs a Retriever pointed at the local corpus (which should contain a `chunk/` folder of .jsonl files),
runs retrieval using a text field from the input (default: patient_say), and prints the retrieved snippets and scores.
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils import Retriever


def load_input(path=None, input_str=None):
    if path:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif input_str:
        return json.loads(input_str)
    else:
        raise ValueError("Either --input_json or --input_str must be provided")


def main():
    parser = argparse.ArgumentParser(description="Custom retrieval from local MedRAG-style chunks")
    parser.add_argument('--input_json', type=str, help='Path to the input JSON file')
    parser.add_argument('--input_is_jsonl', action='store_true', help='Treat input_json as a JSONL file with multiple JSON objects, one per line')
    parser.add_argument('--input_str', type=str, help='Raw JSON string as input')
    parser.add_argument('--db_dir', type=str, default='./corpus', help='Root folder for corpora')
    parser.add_argument('--corpus', type=str, default='Textbooks', help='Corpus name (folder under db_dir)')
    parser.add_argument('--retriever', type=str, default='MedCPT', help='Retriever name (MedCPT, Contriever, BM25, etc.)')
    parser.add_argument('-k', type=int, default=10, help='Number of snippets to retrieve')
    parser.add_argument('--field', type=str, default='patient_say', help='Field in input JSON to use as query')
    parser.add_argument('--output_jsonl', type=str, default=None, help='Path to output jsonl file. If provided, results for each input object are appended as one JSON object per line.')
    args = parser.parse_args()
    # handle jsonl input (multiple objects) or single json
    inputs = []
    if args.input_is_jsonl and args.input_json:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    inputs.append(json.loads(line))
                except Exception as e:
                    print(f"Warning: skipping invalid JSON line: {e}")
    else:
        # single JSON object via file or string
        data = load_input(args.input_json, args.input_str)
        inputs = [data]

    if len(inputs) == 0:
        print("No valid input objects found.")
        return

    # Map common short names to full retriever ids (keeps backwards compatibility with short names) 构建索引
    short_to_full = {
        "MedCPT": "ncbi/MedCPT-Query-Encoder",
        "Contriever": "facebook/contriever",
        "SPECTER": "allenai/specter",
        "BM25": "bm25",
        "BM25-FAISS": "bm25",
    }
    actual_retriever = short_to_full.get(args.retriever, args.retriever)

    # Initialize retriever once (use mapped name)
    retriever = Retriever(retriever_name=actual_retriever, corpus_name=args.corpus, db_dir=args.db_dir, HNSW=False)

    # prepare output file if requested
    out_fp = None
    if args.output_jsonl:
        out_dir = os.path.dirname(args.output_jsonl)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        out_fp = open(args.output_jsonl, 'a', encoding='utf-8')

    # process each input in order and retrieve
    for idx, data in enumerate(inputs):
        # Build a simple query string from the chosen field
        query = data.get(args.field, None)
        if query is None:
            parts = []
            for key in ("patient_say", "report", "history", "tree", "department"):
                v = data.get(key, None)
                if v:
                    if isinstance(v, dict) or isinstance(v, list):
                        parts.append(json.dumps(v, ensure_ascii=False))
                    else:
                        parts.append(str(v))
            query = "\n".join(parts)

        if not query:
            print(f"[Item {idx}] No query text found in the input. Skipping.")
            continue

        print(f"[Item {idx}] Using query (first 300 chars): {query[:300]!s}\n")

        # perform retrieval
        snippets, scores = retriever.get_relevant_documents(query, k=args.k)

        # Print results
        print(f"[Item {idx}] Retrieved {len(snippets)} snippets:\n")
        for i, (s, sc) in enumerate(zip(snippets, scores)):
            print(f"--- Rank {i+1} | score={sc:.6f} | id={s.get('id', '')}")
            print(f"Title: {s.get('title', '')}")
            content = s.get('content', '')
            preview = content[:1000]
            print(f"Content preview:\n{preview}")
            print()

        # write output record if requested
        if out_fp:
            out_record = {
                'input_index': idx,
                'input': data,
                'query': query,
                'retrieved': snippets,
                'scores': scores
            }
            out_fp.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    if out_fp:
        out_fp.close()


if __name__ == '__main__':
    main()
