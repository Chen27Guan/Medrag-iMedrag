#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Two-stage retrieval with multi-threading support."""

import argparse
import json
import os
import re
import sys
import threading
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# 注意：如果运行环境没有 src.utils，请确保此处引入正确
try:
    from src.utils import Retriever
except ImportError:
    # 仅作占位，防止无环境时直接报错，实际运行时需要正确的 Retriever
    class Retriever:
        def __init__(self, **kwargs): pass
        def get_relevant_documents(self, q, k=1): return []

from openai import OpenAI

# 本地模型配置
DEFAULT_MODEL_URL = "http://113.59.64.94:8082/v1"
DEFAULT_MODEL_NAME = "Qwen3-8B"

# 初始化全局锁，确保多线程写入同一个文件时不会乱序或损坏
file_lock = threading.Lock()

def create_client():
    """初始化本地部署的模型客户端"""
    return OpenAI(
        api_key="EMPTY",
        base_url=DEFAULT_MODEL_URL,
    )

# 预先创建一个全局客户端供所有线程复用
global_client = create_client()

def create_chat_completion(messages, model=DEFAULT_MODEL_NAME, temperature=0.2, max_tokens=512):
    """调用本地大模型生成对话回复"""
    try:
        resp = global_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp
    except Exception as e:
        print(f"Model API Error: {e}")
        # 返回一个伪造的响应对象以防代码崩溃
        class MockChoice:
            class MockMessage:
                content = ""
            message = MockMessage()
        class MockResp:
            choices = [MockChoice()]
        return MockResp()

def extract_query_from_xnk(record):
    raw = None
    for key in ('input', 'text', 'dialogue', 'content'):
        if key in record and record.get(key) is not None:
            raw = record.get(key)
            break
    if raw is None: return None
    if not isinstance(raw, str):
        raw = json.dumps(raw, ensure_ascii=False)
    
    # 提取 <本轮患者发言>
    m = re.search(r"<本轮患者发言>\s*(.*?)\s*</本轮患者发言>", raw, flags=re.S)
    if m:
        q = m.group(1).strip()
        # 简单清洗：有时候 OCR 或数据会有残留的 '>' 符号
        q = q.lstrip('>').strip() 
        if q == "" or q.lower() in ("none", "null"): return None
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
    """
    从模型响应中解析追问列表。
    修复：增加去除 <think> 标签的逻辑，防止将思考过程误判为 Query。
    """
    if not text:
        return []

    # 1. 核心修复：移除 <think>...</think> 思考过程
    # flags=re.S 让 . 匹配换行符
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    
    # 2. 移除可能的 <answer> 标签（如果只想要内容）
    text = re.sub(r"</?answer>", "", text, flags=re.I)

    queries = []
    # 3. 尝试定位 ## Queries 后的内容，如果没有则使用全部文本
    m = re.search(r"##+\s*Queries\s*(.*?)($|##+)", text, flags=re.S | re.I)
    body = m.group(1).strip() if m else text.strip()
    
    for line in body.splitlines():
        line = line.strip()
        if not line: continue
        
        # 4. 去除列表编号 (如 "1.", "1)", "- ")
        line = re.sub(r'^[\d\)\.\-\s]+', '', line)
        
        # 5. 过滤掉看起来像 XML 标签的残留行 (如 "<think>" 单独一行)
        if line.startswith("<") and line.endswith(">"):
            continue
            
        if len(line) > 0:
            queries.append(line)
            
    return queries

def process_single_item(idx, data, retriever, args, config):
    """
    处理单个数据条目的核心函数，将被多个线程并发调用
    """
    query = data.get('__extracted_query', None)
    if not query: return

    query_str = json.dumps(query, ensure_ascii=False) if isinstance(query, (dict, list)) else str(query)
    if query_str.strip().lower() in ("none", "null", ""): return

    file_tag = os.path.basename(config['input_path'])
    # print(f"[{file_tag}][Item {idx}] Processing...") # 减少日志打印避免刷屏

    try:
        # --- 第一轮检索 ---
        res1 = retriever.get_relevant_documents(query_str, k=args.k)
        snippets1 = res1[0] if isinstance(res1, tuple) else res1
        # 确保 snippets1 是列表
        if not isinstance(snippets1, list): snippets1 = []

        # --- 生成追问 ---
        context_text = "\n\n".join([f"标题: {s.get('title', '')}\n内容: {s.get('content', '')}" for s in snippets1[:5]])
        history = data.get('history') or data.get('H') or ''
        
        prompt_user = (
            "## 分析\n原始患者发言:\n" + query_str + "\n\n"
            "历史信息:\n" + history + "\n\n"
            "检索到的相关文档:\n" + context_text + "\n\n"
            + config['ask_instr'].format(n=args.num_queries)
        )

        resp = create_chat_completion(
            messages=[{"role": "system", "content": config['system_prompt']}, {"role": "user", "content": prompt_user}], 
            model=args.model
        )
        
        raw_content = resp.choices[0].message.content
        followup_queries = parse_queries_from_response(raw_content)[:args.num_queries]

        # --- 第二轮检索与 QA ---
        snippets2, qa_pairs = [], []
        for q2 in followup_queries:
            # 双重保险：跳过包含 <think> 或为空的 query
            if "<think>" in q2 or not q2.strip(): 
                continue

            r = retriever.get_relevant_documents(q2, k=args.k_followup)
            ss = r[0] if isinstance(r, tuple) else r
            if not isinstance(ss, list): ss = []
            
            snippets2.extend(ss)

            # 生成回答
            ans_prompt = f"## 分析\n追问: {q2}\n检索内容:\n" + "\n".join([s.get('content','') for s in ss[:3]]) + "\n" + config['ans_instr']
            resp_ans = create_chat_completion(
                messages=[{"role": "system", "content": config['system_prompt']}, {"role": "user", "content": ans_prompt}], 
                model=args.model, temperature=0.0
            )
            
            # 同样清理回答中的 think 标签（可选，取决于是否想保留思维链在最终结果中，这里选择清理以保持整洁）
            ans_content = resp_ans.choices[0].message.content
            # ans_content = re.sub(r"<think>.*?</think>", "", ans_content, flags=re.S).strip() # 如果需要清洗回答请解开此注释
            
            qa_pairs.append({'query': q2, 'retrieved': ss, 'answer': ans_content})

        # --- 合并去重 ---
        merged = []
        seen = set()
        # 确保 snippets1 和 snippets2 里的元素是字典
        all_snippets = [s for s in snippets1 + snippets2 if isinstance(s, dict)]
        
        for s in all_snippets:
            # 兼容不同检索器返回的数据结构
            key = s.get('id')
            if not key:
                key = s.get('content', '')[:500] # 如果没有ID，用内容哈希
            
            if key and key not in seen:
                merged.append(s); seen.add(key)

        out_record = {
            'input_index': idx, 'input': data, 'query': query_str,
            'retrieved_round1': snippets1, 'followup_queries': followup_queries,
            'qa': qa_pairs, 'merged_retrieved': merged
        }

        # --- 线程安全地写入文件 ---
        with file_lock:
            with open(config['output_path'], 'a', encoding='utf-8') as out_fp:
                out_fp.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"Error at [{file_tag}][Item {idx}]: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Multi-threaded Two-stage retrieval")
    parser.add_argument('--input_json', type=str, help='Path to input JSON/JSONL')
    parser.add_argument('--db_dir', type=str, default='./corpus', help='Root folder for corpora')
    parser.add_argument('--corpus', type=str, default='Textbooks', help='Corpus name')
    parser.add_argument('--retriever', type=str, default='MedCPT', help='Retriever name')
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('--num_queries', type=int, default=3)
    parser.add_argument('--k_followup', type=int, default=5)
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument('--output_jsonl', type=str, default=None)
    parser.add_argument('--threads', type=int, default=32, help='并发线程数')
    args = parser.parse_args()

    file_tasks = [
        ("./input/fsk_test_aligned.jsonl", "./output/fsk_imedrag.jsonl"),
        ("./input/guke_test_aligned.jsonl", "./output/guke_imedrag.jsonl"),
        ("./input/gwk_test_aligned.jsonl", "./output/gwk_imedrag.jsonl"),
        ("./input/mnk_test_aligned.jsonl", "./output/mnk_imedrag.jsonl"),
        ("./input/qk_test_aligned.jsonl", "./output/qk_imedrag.jsonl"),
        ("./input/szk_test_aligned.jsonl", "./output/szk_imedrag.jsonl"),
        ("./input/ttk_test_aligned.jsonl", "./output/ttk_imedrag.jsonl"),
        ("./input/xhk_test_aligned.jsonl", "./output/xhk_imedrag.jsonl"),
    ]

    if args.input_json:
        file_tasks = [(args.input_json, args.output_jsonl or "output/default_out.jsonl")]

    # 1. 初始化检索器 (全局唯一)
    short_to_full = {"MedCPT": "ncbi/MedCPT-Query-Encoder", "BM25": "bm25"}
    actual_retriever = short_to_full.get(args.retriever, args.retriever)
    
    retriever = None
    try:
        print(f"Initializing Retriever: {actual_retriever}...")
        retriever = Retriever(retriever_name=actual_retriever, corpus_name=args.corpus, db_dir=args.db_dir, HNSW=False)
        print("Retriever initialized.")
    except Exception as e:
        print(f"Failed to init Retriever: {e}")
        # 如果是本地测试没有 Retriever 环境，可以选择 return 或者是 mock
        return

    # 配置公共 Prompt
    # 注意：system_prompt 尽量保持简洁，防止模型输出额外的废话
    config_base = {
        'system_prompt': '你是一个乐于助人的医学助手。请严格遵循用户的格式要求。',
        'ask_instr': "请根据上述信息生成 {n} 条简洁的进一步提问。请直接列出问题，不要包含任何XML标签或思考过程。\n",
        'ans_instr': "请先分析信息，然后给出针对追问的回答。"
    }

    # 2. 逐个文件处理
    for input_path, output_path in file_tasks:
        if not os.path.exists(input_path):
            print(f"Skip: {input_path} not found."); continue

        print(f"\n>>> Task Start: {input_path} -> {output_path}")
        
        # 加载数据
        current_items = []
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        obj = json.loads(line)
                        q = extract_query_from_xnk(obj) or build_fallback_query(obj)
                        obj['__extracted_query'] = q
                        current_items.append(obj)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading input file: {e}")
            continue

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 清空或保留旧文件？这里默认 append 模式，如果需要覆盖请修改 mode='w'
        # 建议每次运行前手动清理或此处改为 'w' 并在 process_single_item 中改为 'a' (需注意多进程覆盖问题)
        # 为安全起见，主线程先创建文件
        if not os.path.exists(output_path):
            open(output_path, 'w').close()

        # 针对当前文件的配置
        task_config = config_base.copy()
        task_config.update({'input_path': input_path, 'output_path': output_path})

        # 3. 启动线程池
        print(f"Processing {len(current_items)} items with {args.threads} threads...")
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            future_to_item = {
                executor.submit(process_single_item, i, item, retriever, args, task_config): i 
                for i, item in enumerate(current_items)
            }
            
            count = 0
            for future in as_completed(future_to_item):
                count += 1
                if count % 10 == 0:
                    print(f"Progress: {count}/{len(current_items)}", end='\r')
                pass
        print(f"\nFinished: {input_path}")

    print("\nAll tasks completed.")

if __name__ == '__main__':
    main()