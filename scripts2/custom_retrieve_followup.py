#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Two-stage retrieval: first retrieve, then use an LLM to generate follow-up queries,
retrieve again with those follow-ups, and merge results into a JSONL output.

Usage example:
  python scripts/custom_retrieve_followup.py --input_json ../evaluate/xnk_8B.jsonl --input_is_jsonl --db_dir ./db --corpus Textbooks --retriever MedCPT -k 5 --num_queries 3 --k_followup 3 --model qwen-plus-2025-04-28 --output_jsonl output/followup_out.jsonl
"""

import argparse
import json
import os
import re
import sys
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils import Retriever
from openai import OpenAI


def create_client():
    """初始化阿里云 DashScope 客户端"""
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return client


def create_chat_completion(messages, model="qwen-plus-2025-04-28", temperature=0.2, max_tokens=512):
    """调用阿里云大模型生成对话回复"""
    client = create_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp


def load_input(path=None, input_str=None):
    if path:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif input_str:
        return json.loads(input_str)
    else:
        raise ValueError("Either --input_json or --input_str must be provided")


def extract_query_from_xnk(record):
    """
    改进后的提取逻辑：
    1. 专门处理嵌套字典结构。
    2. 使用 findall 寻找所有成对的标签。
    3. 取最后一个匹配项，彻底避开指令区干扰。
    """
    raw = None
    # 针对你提供的 JSON 结构进行精准提取
    # record 里的 'input' 字段本身是一个字典，里面又有一个 'input' 键存放字符串
    inner_data = record.get('input')
    if isinstance(inner_data, dict):
        raw = inner_data.get('input') or inner_data.get('content')
    
    # 兜底：如果上面没取到，再从顶层找
    if raw is None:
        for key in ('input', 'text', 'dialogue', 'content'):
            if key in record and record.get(key) is not None:
                raw = record.get(key)
                break

    if raw is None:
        return None

    # 如果 raw 仍然是字典/列表，转为字符串
    if not isinstance(raw, str):
        raw = json.dumps(raw, ensure_ascii=False)

    # 核心修复：findall 会找到文中所有成对出现的标签块
    # 因为指令区只会提到标签名，通常不会有成对的 </本轮患者发言>
    # 即使指令区有，最后一次出现的也必然是真正的患者输入
    matches = re.findall(r"<本轮患者发言>\s*(.*?)\s*</本轮患者发言>", raw, flags=re.S)
    
    if matches:
        # 获取最后一个匹配到的块
        q = matches[-1].strip()
        
        # 排除掉占位符或指令残留（比如匹配到了包含“所属场景”字样的项）
        if "所属的场景" in q or q.lower() in ("none", "null", ""):
            # 如果最后一个依然包含指令，尝试向上找一个，或者说明数据格式不规范
            if len(matches) > 1:
                q = matches[-2].strip()
            else:
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


def parse_queries_from_response(text: str) -> List[str]:
    queries = []
    m = re.search(r"##+\s*Queries\s*(.*?)($|##+)", text, flags=re.S | re.I)
    body = m.group(1).strip() if m else text
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[\d\)\.\-\s]+', '', line)
        if len(line) > 0:
            queries.append(line)
    return queries


def main():
    parser = argparse.ArgumentParser(description="Two-stage retrieval with follow-up question generation")
    parser.add_argument('--input_json', type=str, help='Path to input JSON or JSONL')
    parser.add_argument('--input_is_jsonl', action='store_true', help='Treat input_json as a JSONL file')
    parser.add_argument('--input_str', type=str, help='Raw JSON string as input')
    parser.add_argument('--db_dir', type=str, default='./corpus', help='Root folder for corpora')
    parser.add_argument('--corpus', type=str, default='Textbooks', help='Corpus name')
    parser.add_argument('--retriever', type=str, default='MedCPT', help='Retriever name')
    parser.add_argument('-k', type=int, default=10, help='Number of snippets for first round')
    parser.add_argument('--num_queries', type=int, default=3, help='How many follow-up queries to ask the model to generate')
    parser.add_argument('--k_followup', type=int, default=5, help='Number of snippets per follow-up query')
    parser.add_argument('--model', type=str, default='qwen-plus-2025-04-28', help='Chat model to call')
    parser.add_argument('--output_jsonl', type=str, default=None, help='Path to output jsonl file')
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

                q = extract_query_from_xnk(obj)
                if q is None:
                    q = build_fallback_query(obj)
                obj['__extracted_query'] = q
                inputs.append(obj)
    else:
        data = load_input(args.input_json, args.input_str)
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

    i_medrag_system = '你是一个乐于助人的医学助手，你的任务是按照用户给出的指示进行相应输出。'

    follow_up_instruction_ask = (
        "请根据分析部分中的信息生成 {n} 条简洁、与上下文相关的进一步提问。进一步提问应简单且聚焦，直接关联到答案中的关键词。提问格式的示例：ST 段抬高常见于哪些急性心脏疾病？; 急性心肌梗死的典型临床表现有哪些？; ST 段抬高型心肌梗死与心绞痛的区别是什么？请你直接输出想要追问的问题，不要输出其他内容。\n"
    )

    follow_up_instruction_answer = (
        "请先逐步思考\"分析\"(## 分析)的部分分析所有信息。然后在\"回答\"(## 回答)的部分给出你的答案选择。"
    )

    try:
        for idx, data in enumerate(inputs):
            query = data.get('__extracted_query', None)
            if not query:
                print(f"[Item {idx}] No query found after extraction. Skipping.")
                continue

            if isinstance(query, (dict, list)):
                query = json.dumps(query, ensure_ascii=False)
            else:
                query = str(query)

            if query.strip().lower() in ("none", "null", ""):
                print(f"[Item {idx}] Query is empty/placeholder after normalization. Skipping.")
                continue

            print(f"[Item {idx}] Using query (first 1000 chars): {query[:1000]!s}\n")

            try:
                res1 = retriever.get_relevant_documents(query, k=args.k)
            except Exception as e:
                print(f"[Item {idx}] Retrieval failed (round1): {e}")
                continue

            if isinstance(res1, tuple) and len(res1) == 2:
                snippets1, scores1 = res1
            else:
                snippets1 = res1
                scores1 = [None] * len(snippets1)

            print(f"[Item {idx}] Retrieved {len(snippets1)} snippets (round1).\n")

            context_parts = []
            for s in snippets1[:5]:
                title = s.get('title', '')
                content = s.get('content', '')
                context_parts.append(f"标题: {title}\n内容: {content}")
            context_text = "\n\n".join(context_parts)

            orig_utterance = query
            history = data.get('history') or data.get('H') or data.get('patient_history') or ''
            other_info_parts = []
            for k in ('patient_say', 'report', 'department', 'tree'):
                v = data.get(k)
                if v:
                    if isinstance(v, (dict, list)):
                        other_info_parts.append(f"{k}: " + json.dumps(v, ensure_ascii=False))
                    else:
                        other_info_parts.append(f"{k}: " + str(v))
            other_text = "\n".join(other_info_parts)

            followup_queries = []
            prompt_user = (
                "## 分析\n"
                "原始患者发言:\n" + (orig_utterance or "无") + "\n\n"
                "历史信息:\n" + (history or "无") + "\n\n"
                "其他相关字段:\n" + (other_text or "无") + "\n\n"
                "先前检索到的相关文档片段:\n" + (context_text or "无") + "\n\n"
                + follow_up_instruction_ask.format(n=args.num_queries)
            )
            messages = [
                {"role": "system", "content": i_medrag_system},
                {"role": "user", "content": prompt_user}
            ]

            try:
                resp = create_chat_completion(messages=messages, model=args.model, temperature=0.2, max_tokens=512)
                text_out = resp.choices[0].message.content
                followup_queries = parse_queries_from_response(text_out)
                followup_queries = followup_queries[:args.num_queries]
            except Exception as e:
                print(f"[Item {idx}] Follow-up generation failed: {e}")
                followup_queries = []

            print(f"[Item {idx}] Generated {len(followup_queries)} follow-up queries.\n")

            snippets2 = []
            scores2 = []
            qa_pairs = []
            for q2 in followup_queries:
                try:
                    r = retriever.get_relevant_documents(q2, k=args.k_followup)
                except Exception as e:
                    print(f"[Item {idx}] Retrieval failed for follow-up query '{q2}': {e}")
                    continue

                if isinstance(r, tuple) and len(r) == 2:
                    ss, sc = r
                else:
                    ss = r
                    sc = [None] * len(ss)

                snippets2.extend(ss)
                scores2.extend(sc)

                docs_text = []
                for s in ss[:5]:
                    title = s.get('title', '')
                    content = s.get('content', '')
                    docs_text.append(f"标题: {title}\n内容: {content}")
                docs_context = "\n\n".join(docs_text) if docs_text else "无"

                answer_prompt = (
                    "## 分析\n"
                    "追问:\n" + q2 + "\n\n"
                    "检索到的相关文档:\n" + docs_context + "\n\n"
                    + follow_up_instruction_answer
                )

                answer_text = None
                messages_ans = [
                    {"role": "system", "content": i_medrag_system},
                    {"role": "user", "content": answer_prompt}
                ]
                try:
                    resp_ans = create_chat_completion(messages=messages_ans, model=args.model, temperature=0.0, max_tokens=512)
                    answer_text = resp_ans.choices[0].message.content
                except Exception as e:
                    print(f"[Item {idx}] Answer generation failed for query '{q2}': {e}")
                    answer_text = None

                qa_pairs.append({
                    'query': q2,
                    'retrieved': ss,
                    'answer': answer_text
                })

            merged = []
            seen_ids = set()
            seen_texts = set()
            for s in snippets1 + snippets2:
                sid = s.get('id', None)
                content = (s.get('content') or '')[:1000]
                key = sid if sid is not None else content
                if key in seen_ids or key in seen_texts:
                    continue
                if sid is not None:
                    seen_ids.add(sid)
                else:
                    seen_texts.add(key)
                merged.append(s)

            print(f"[Item {idx}] Merged total {len(merged)} unique snippets.\n")

            if out_fp:
                out_record = {
                    'input_index': idx,
                    'input': data,
                    'query': query,
                    'retrieved_round1': snippets1,
                    'scores_round1': scores1,
                    'followup_queries': followup_queries,
                    'retrieved_round2': snippets2,
                    'scores_round2': scores2,
                    'qa': qa_pairs,
                    'merged_retrieved': merged
                }
                out_fp.write(json.dumps(out_record, ensure_ascii=False) + "\n")
    finally:
        if out_fp:
            out_fp.close()


if __name__ == '__main__':
    main()
