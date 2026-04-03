#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge follow-up retrieval results (file_b) into original prompts (file_a).

For each record in file_b (JSONL) we expect structure like:
{
  "input_index": 0,
  "merged_retrieved": [ {"id":..., "title":..., "content":..., "contents":...}, ... ],
  "followup_queries": [...],
  "qa": [ {"query":..., "retrieved": [...], "answer": ...}, ... ],
  ...
}

file_a is a JSONL where each object's `input` contains a text with tags including `<提示>`.
We will insert the concatenated `contents` from `merged_retrieved` into the `<提示>` tag position
(replacing the tag contents), and append a new tag `<问答>...</问答>` immediately after `<提示>`
containing followup queries and answers from file_b (qa -> query & answer).

Output: writes a merged JSONL to the given output path.

Usage:
  python scripts/merge_followup.py --file_a path/to/a.jsonl --file_b path/to/b.jsonl --out merged.jsonl
"""

import argparse
import json
import os
import re
from typing import List, Dict, Any


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {i+1} of {path}: {e}")
    return data


def safe_join_contents(merged_retrieved: List[Dict[str, Any]]) -> str:
    parts = []
    for idx, item in enumerate(merged_retrieved, start=1):
        text = item.get('contents') or item.get('content') or ''
        if text:
            # prefix each snippet with a numeric label like [1], [2], ...
            prefix = f"[{idx}] "
            # ensure there is no leading/trailing whitespace issues
            snippet = prefix + text.strip()
            parts.append(snippet)
    return "\n\n".join(parts)


def insert_into_prompt(prompt_text: str, insert_text: str) -> str:
    """Find <提示>...</提示> (or just <提示>) and replace/insert the insert_text.

    Behavior:
    - If <提示>...</提示> exists, replace the inner content with insert_text.
    - Else if a self-closing <提示/> or a single tag <提示> exists, replace that tag with
      <提示>insert_text</提示>.
    - If no <提示> tag found, append a new <提示>insert_text</提示> at the end.
    Returns the modified prompt text.
    """
    # try full tag
    full_tag_re = re.compile(r'(<提示\s*>)([\s\S]*?)(</提示>)', flags=re.I)
    m = full_tag_re.search(prompt_text)
    if m:
        before = prompt_text[:m.start()]
        after = prompt_text[m.end():]
        return before + m.group(1) + insert_text + m.group(3) + after

    # try single opening tag <提示> with no closing
    open_tag_re = re.compile(r'(<提示\s*/?>)', flags=re.I)
    m2 = open_tag_re.search(prompt_text)
    if m2:
        # replace the tag with full tag
        start = m2.start()
        end = m2.end()
        before = prompt_text[:start]
        after = prompt_text[end:]
        return before + '<提示>' + insert_text + '</提示>' + after

    # not found -> append
    return prompt_text + '\n\n<提示>' + insert_text + '</提示>\n'


def build_qa_block(qa_list: List[Dict[str, Any]]) -> str:
    """Build a readable <问答> block from qa list. We'll create a simple numbered list.
    Each item will include the follow-up query and the LLM's answer.
    """
    if not qa_list:
        return ''
    parts = ['<问答>']
    for i, qa in enumerate(qa_list, start=1):
        q = qa.get('query', '')
        # prefer qa.answer or qa['answer']
        a = qa.get('answer', '')
        # HTML-escape minimal problematic sequences (keep simple)
        parts.append(f"{i}. 问: {q}\n   答: {a}")
    parts.append('</问答>')
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_a', required=True, help='Path to original file_a (JSONL)')
    parser.add_argument('--file_b', required=True, help='Path to followup file_b (JSONL with merged_retrieved, qa)')
    parser.add_argument('--out', required=True, help='Output JSONL path')
    parser.add_argument('--match_on_index', action='store_true', help='Match records by input_index. If not set, match by order.')
    args = parser.parse_args()

    a = load_jsonl(args.file_a)
    b = load_jsonl(args.file_b)

    # Build map for b by input_index if available
    b_map = {}
    for rec in b:
        idx = rec.get('input_index')
        if idx is None:
            # fallback: use order concatenation (allow multiple per index -> list)
            continue
        b_map[idx] = rec

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, 'w', encoding='utf-8') as outf:
        # if match_on_index, use input_index from a's records; else iterate by order
        for i, rec_a in enumerate(a):
            rec_b = None
            if args.match_on_index:
                idx = rec_a.get('input_index', i)
                rec_b = b_map.get(idx)
            else:
                # try to find b with same input_index, else fallback to same position
                idx = rec_a.get('input_index')
                if idx is not None and idx in b_map:
                    rec_b = b_map.get(idx)
                elif i < len(b):
                    rec_b = b[i]

            merged_text = ''
            qa_block = ''
            if rec_b:
                merged = rec_b.get('merged_retrieved') or rec_b.get('retrieved') or []
                merged_text = safe_join_contents(merged)
                # build QA using rec_b['qa'] if available, otherwise use followup_queries+answer
                qa_list = rec_b.get('qa')
                if not qa_list:
                    # try to construct from followup_queries and answer field
                    fqs = rec_b.get('followup_queries') or []
                    ans = rec_b.get('answer')
                    if fqs:
                        qa_list = []
                        if isinstance(ans, str) and len(fqs) == 1:
                            qa_list.append({'query': fqs[0], 'answer': ans})
                        else:
                            # if multiple followups but no per-query answers, put answers blank
                            for q in fqs:
                                qa_list.append({'query': q, 'answer': ''})
                qa_block = build_qa_block(qa_list or [])

            # now modify rec_a.input which may be a dict containing 'input' string or nested
            a_input = rec_a.get('input')
            if isinstance(a_input, dict):
                # find a textual field to modify: prefer 'input' or 'prompt' keys
                modified = False
                for key in ('input', 'prompt', 'text'):
                    if key in a_input and isinstance(a_input[key], str):
                        new_text = insert_into_prompt(a_input[key], merged_text)
                        # append qa_block immediately after the </提示> if present; else after inserted tag
                        if qa_block:
                            # place qa_block after the first </提示> occurrence
                            if '</提示>' in new_text:
                                new_text = new_text.replace('</提示>', '</提示>\n' + qa_block, 1)
                            else:
                                new_text = new_text + '\n' + qa_block
                        a_input[key] = new_text
                        modified = True
                        break
                if not modified:
                    # fallback: convert whole input dict to string and append tags
                    text_repr = json.dumps(a_input, ensure_ascii=False)
                    new_text = insert_into_prompt(text_repr, merged_text)
                    if qa_block:
                        new_text = new_text + '\n' + qa_block
                    a_input = new_text
                    rec_a['input'] = a_input
                else:
                    rec_a['input'] = a_input
            elif isinstance(a_input, str):
                new_text = insert_into_prompt(a_input, merged_text)
                if qa_block:
                    if '</提示>' in new_text:
                        new_text = new_text.replace('</提示>', '</提示>\n' + qa_block, 1)
                    else:
                        new_text = new_text + '\n' + qa_block
                rec_a['input'] = new_text
            else:
                # unknown structure: append tags at top-level
                extra = '<提示>' + merged_text + '</提示>\n' + qa_block
                rec_a['injected_followup'] = extra

            outf.write(json.dumps(rec_a, ensure_ascii=False) + '\n')

    print(f"Wrote merged output to {args.out}")


if __name__ == '__main__':
    main()
