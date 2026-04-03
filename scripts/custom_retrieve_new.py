#将检索器换为MedCPT，将输入的信息换为孙哥给的200条测试数据集的格式
"""custom_retrieve_new.py

Usage:
    python scripts/custom_retrieve_xnk.py --input_json path/to/xnk.jsonl --db_dir path/to/corpus_root --corpus CorpusName --retriever RetrieverName --k 10

This script is a thin variant of `scripts/custom_retrieve.py` that expects the xnk-style
JSONL where each object contains an `input` string with markers like:
    <本轮患者发言>
    ...patient utterance...
    </本轮患者发言>

The script extracts the content between those markers (first priority) and uses it as the
retrieval query. If the tag is not present or contains a placeholder like "none" it will
fall back to other fields (patient_say, history, report, department).
"""

import argparse
import json
import os
import re
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


def extract_query_from_xnk(record):
    """Extract text between <本轮患者发言> and </本轮患者发言> in the xnk `input` field.

    Returns a normalized string or None if not found/meaningful.
    """
    raw = None
    # common places where the input text might be stored
    for key in ('input', 'text', 'dialogue', 'content'):
        if key in record and record.get(key) is not None:
            raw = record.get(key)
            break

    if raw is None:
        return None

    if not isinstance(raw, str):
        raw = json.dumps(raw, ensure_ascii=False)

    m = re.search(r"<本轮患者发言>\s*(.*?)\s*</本轮患者发言>", raw, flags=re.S)
    if m:
        q = m.group(1).strip()
        if q == "" or q.lower() in ("none", "null"):
            return None
        return q
    return None


def build_fallback_query(data):
    parts = []
    for key in ("patient_say", "report", "history", "tree", "department"):
        v = data.get(key, None)
        if v:
            if isinstance(v, (dict, list)):
                parts.append(json.dumps(v, ensure_ascii=False))
            else:
                parts.append(str(v))
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Custom retrieval for xnk JSONL inputs")
    parser.add_argument('--input_json', type=str, help='Path to the input JSONL file')
    parser.add_argument('--input_is_jsonl', action='store_true', help='Treat input_json as a JSONL file with multiple JSON objects, one per line')
    parser.add_argument('--input_str', type=str, help='Raw JSON string as input')
    parser.add_argument('--db_dir', type=str, default='./corpus', help='Root folder for corpora')
    parser.add_argument('--corpus', type=str, default='Textbooks', help='Corpus name (folder under db_dir)')
    parser.add_argument('--retriever', type=str, default='MedCPT', help='Retriever name (MedCPT, Contriever, BM25, etc.)')
    parser.add_argument('-k', type=int, default=10, help='Number of snippets to retrieve')
    parser.add_argument('--output_jsonl', type=str, default=None, help='Path to output jsonl file. If provided, results for each input object are appended as one JSON object per line.')
    args = parser.parse_args()

    inputs = []
    if args.input_is_jsonl and args.input_json:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    print(f"Warning: skipping invalid JSON line: {e}")
                    continue

                # try to extract query from xnk input block
                q = extract_query_from_xnk(obj)
                if q is None:
                    # fallback to building from other fields
                    q = build_fallback_query(obj)
                obj['__extracted_query'] = q
                inputs.append(obj)
    else:
        data = load_input(args.input_json, args.input_str)
        # If user passed a single object, try extract similarly
        q = extract_query_from_xnk(data)
        if q is None:
            q = build_fallback_query(data)
        data['__extracted_query'] = q
        inputs = [data]

    if len(inputs) == 0:
        print("No valid input objects found.")
        return

    short_to_full = {
        "MedCPT": "ncbi/MedCPT-Query-Encoder",
        "Contriever": "facebook/contriever",
        "SPECTER": "allenai/specter",
        "BM25": "bm25",
        "BM25-FAISS": "bm25",
    }
    actual_retriever = short_to_full.get(args.retriever, args.retriever)

    try:
        retriever = Retriever(retriever_name=actual_retriever, corpus_name=args.corpus, db_dir=args.db_dir, HNSW=False)
    except Exception as e:
        print(f"Error: failed to initialize Retriever: {e}")
        return

    out_fp = None
    if args.output_jsonl:
        out_dir = os.path.dirname(args.output_jsonl)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        out_fp = open(args.output_jsonl, 'a', encoding='utf-8')

    try:
        for idx, data in enumerate(inputs):
            query = data.get('__extracted_query', None)
            if not query:
                print(f"[Item {idx}] No query found after extraction. Skipping.")
                continue

            # normalize
            if isinstance(query, (dict, list)):
                query = json.dumps(query, ensure_ascii=False)
            else:
                query = str(query)

            if query.strip().lower() in ("none", "null", ""):
                print(f"[Item {idx}] Query is empty/placeholder after normalization. Skipping.")
                continue

            print(f"[Item {idx}] Using query (first 1000 chars): {query[:1000]!s}\n")

            try:
                res = retriever.get_relevant_documents(query, k=args.k)
            except Exception as e:
                print(f"[Item {idx}] Retrieval failed: {e}")
                continue

            if isinstance(res, tuple) and len(res) == 2:
                snippets, scores = res
            else:
                snippets = res
                scores = [None] * len(snippets)

            print(f"[Item {idx}] Retrieved {len(snippets)} snippets:\n")
            for i, (s, sc) in enumerate(zip(snippets, scores)):
                print(f"--- Rank {i+1} | score={sc} | id={s.get('id', '')}")
                print(f"Title: {s.get('title', '')}")
                content = s.get('content', '')
                preview = content[:1000]
                print(f"Content preview:\n{preview}")
                print()

            if out_fp:
                out_record = {
                    'input_index': idx,
                    'input': data,
                    'query': query,
                    'retrieved': snippets,
                    'scores': scores
                }
                out_fp.write(json.dumps(out_record, ensure_ascii=False) + "\n")
    finally:
        if out_fp:
            out_fp.close()


if __name__ == '__main__':
    main()
