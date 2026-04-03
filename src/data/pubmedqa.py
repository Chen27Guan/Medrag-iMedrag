import os
import json
import re
import tqdm
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter


def ends_with_ending_punctuation(s: str) -> bool:
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)


def concat(title: str, content: str) -> str:
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    else:
        return title.strip() + ". " + content.strip()


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_pubmedqa_json(fpath: str) -> dict:
    with open(fpath, 'r', encoding='utf-8') as f:
        return json.load(f)


def join_contexts(contexts: List[str]) -> str:
    # join contexts with double newline, normalize whitespace
    return "\n\n".join([re.sub(r"\s+", " ", c).strip() for c in contexts if c and c.strip()])


if __name__ == "__main__":
    # 可按需修改输入文件路径
    RAW_JSON_PATH = "../../corpus/pubmedqa/ori_pqaa.json"  # 输入 JSON
    OUT_DIR = "../../corpus/pubmedqa/chunk"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_LINES_PER_FILE = 100000  # 每个输出 JSONL 文件最大行数

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    ensure_dir(OUT_DIR)
    if not os.path.exists(RAW_JSON_PATH):
        print(f"输入文件未找到: {RAW_JSON_PATH}")
        raise SystemExit(1)

    data = load_pubmedqa_json(RAW_JSON_PATH)

    part_idx = 0
    line_count = 0
    out_path = os.path.join(OUT_DIR, f"pubmedqa_part{part_idx}.jsonl")
    outf = open(out_path, 'w', encoding='utf-8')

    total_chunks = 0
    for pmid in tqdm.tqdm(sorted(data.keys()), desc="Processing PubMedQA"):
        rec = data[pmid]
        question = rec.get("QUESTION", "").strip()
        contexts = rec.get("CONTEXTS", []) or []
        long_answer = rec.get("LONG_ANSWER", "") or ""

        # 主文本优先使用 CONTEXTS 合并，若为空则使用 LONG_ANSWER 或 QUESTION
        text = join_contexts(contexts)
        if not text:
            text = long_answer.strip()
        if not text:
            text = question

        # split into chunks
        chunks = text_splitter.split_text(text.strip())
        base_title = question if question else f"PubMedQA_{pmid}"

        for i, ch in enumerate(chunks):
            content = re.sub(r"\s+", " ", ch).strip()
            record = {
                "id": f"{pmid}_{i}",
                "pmid": pmid,
                "title": base_title,
                "content": content,
                "contents": concat(base_title, content),
                "labels": rec.get("LABELS", []),
                "meshes": rec.get("MESHES", []),
                "final_decision": rec.get("final_decision", "")
            }

            outf.write(json.dumps(record, ensure_ascii=False) + "\n")
            line_count += 1
            total_chunks += 1

            # 超过最大行数，换新文件
            if line_count >= MAX_LINES_PER_FILE:
                outf.close()
                part_idx += 1
                out_path = os.path.join(OUT_DIR, f"pubmedqa_part{part_idx}.jsonl")
                outf = open(out_path, 'w', encoding='utf-8')
                line_count = 0

    outf.close()
    print(f"总共写入 {total_chunks} 个 chunks，生成了 {part_idx + 1} 个 JSONL 文件，保存在 {OUT_DIR}")
