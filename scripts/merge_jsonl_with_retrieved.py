#!/usr/bin/env python3
"""
merge_jsonl_with_retrieved.py

读取两个行对齐的 JSONL 文件：file_a（主文件）和 file_b（检索结果文件），
将 file_b 中的 `retrieved` 字段（若存在）渲染为文本并插入到 file_a 的
`input` 字符串中的 `<提示>` 标签内（替换内容），输出新的 JSONL 到 stdout 或文件。

用法示例：
python scripts/merge_jsonl_with_retrieved.py --file_a input/a.jsonl --file_b input/b.jsonl --out output/merged.jsonl

假设两文件行数一致且一一对应。
"""
import argparse
import json
import sys
from typing import Any, Dict, List


def load_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON on line {i} of {path}: {e}")


def render_retrieved_to_prompt(retrieved: List[Dict[str, Any]]) -> str:
    """Turn retrieved list into a human-readable prompt block.

    Format:
    - For each snippet include title (if any) and a short content preview.
    - Join with separator lines.
    """
    if not retrieved:
        return ""
    parts = []
    for i, r in enumerate(retrieved, 1):
        title = r.get('title')#取title
        content = r.get('content')#取content
        # keep content short but informative
        preview = content.strip().replace('\n', ' ')[:800]
        parts.append(f"[{i}] {title}: {preview}")
    return "\n\n".join(parts)


def inject_prompt_into_input(original_input: str, prompt_text: str) -> str:
    """Replace the content between <提示>...</提示> with prompt_text.

    If tags not present, append a <提示> section at the end separated by two newlines.
    """
    start_tag = '<提示>'
    end_tag = '</提示>'
    if start_tag in original_input and end_tag in original_input:
        pre, rest = original_input.split(start_tag, 1)
        # rest contains everything after start_tag
        _, suf = rest.split(end_tag, 1)
        return pre + start_tag + '\n' + prompt_text + '\n' + end_tag + suf
    else:
        # append the tag
        if not original_input.endswith('\n'):
            original_input = original_input + '\n\n'
        return original_input + start_tag + '\n' + prompt_text + '\n' + end_tag + '\n'


def merge_line(a_rec: Dict[str, Any], b_rec: Dict[str, Any]) -> Dict[str, Any]:
    # Work on a deep copy of a_rec to avoid mutating inputs
    out = json.loads(json.dumps(a_rec))
    # retrieved may be in b_rec['retrieved'] or b_rec.get('retrieved')
    retrieved = None
    if isinstance(b_rec, dict):
        retrieved = b_rec.get('retrieved') or b_rec.get('retrieved_snippets') or b_rec.get('retrieved_results')
    if not retrieved:
        # try to see if b_rec itself contains fields id/title/content
        if isinstance(b_rec, dict) and any(k in b_rec for k in ('id', 'title', 'content', 'contents', 'text')):
            retrieved = [b_rec]
    prompt_text = render_retrieved_to_prompt(retrieved or [])

    # find the 'input' field inside a_rec; it may be a dict or a string
    if isinstance(out.get('input'), dict):
        # if input is structured, try to place prompt into '提示' subfield if exists, else set '提示'
        inp = out['input']
        if '提示' in inp:
            inp['提示'] = prompt_text
        else:
            inp['提示'] = prompt_text
        out['input'] = inp
    else:
        orig = out.get('input', '') or ''
        out['input'] = inject_prompt_into_input(orig, prompt_text)

    return out


def main():
    p = argparse.ArgumentParser(description='Merge two line-aligned JSONL files: inject retrieved snippets into <提示> of file A')
    p.add_argument('--file_a', required=True, help='主 JSONL 文件（有 <提示> 标签的 input）')
    p.add_argument('--file_b', required=True, help='检索结果 JSONL 文件（含 retrieved 字段）')
    p.add_argument('--out', required=False, help='输出文件路径（默认 stdout）')
    args = p.parse_args()

    gen_a = load_jsonl(args.file_a)
    gen_b = load_jsonl(args.file_b)

    out_f = open(args.out, 'w', encoding='utf-8') if args.out else sys.stdout

    count = 0
    try:
        while True:
            a = next(gen_a)
            b = next(gen_b)
            merged = merge_line(a, b)
            out_f.write(json.dumps(merged, ensure_ascii=False) + '\n')
            count += 1
    except StopIteration:
        pass

    if args.out:
        out_f.close()
    print(f"Merged {count} records", file=sys.stderr)


if __name__ == '__main__':
    main()
