#拆东墙补西墙
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_doctor_concurrent.py

功能:
1. 遍历输入文件夹中的所有 jsonl 文件。
2. 使用多线程并发调用本地 LLM API。
3. 在每次 Input 前自动注入预设的任务指令和场景逻辑。
4. 调用参数增加了 frequency_penalty 以减少复读。

用法示例:
    python generate_doctor_concurrent.py --in ./input_data --out ./output_data --threads 16

作者: Elite Architect
"""

import argparse
import json
import os
import sys
import time
import threading
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- 配置 ---
DEFAULT_MODEL_URL = "http://113.59.64.94:8082/v1"
DEFAULT_MODEL_NAME = "Qwen3-8B"

# --- 任务指令前缀 ---

# --- 全局变量 ---
API_INFO = None
# 使用全局 Session 以复用 TCP 连接
global_session = requests.Session()

def init_session(pool_connections=32, pool_maxsize=32, retries=3):
    """配置全局 Session 的连接池和重试策略"""
    retry_strategy = Retry(
        total=retries,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(
        pool_connections=pool_connections, 
        pool_maxsize=pool_maxsize, 
        max_retries=retry_strategy
    )
    global_session.mount("http://", adapter)
    global_session.mount("https://", adapter)

def detect_api(model_url: str, model_name: str, timeout: int = 5):
    """
    探测可用的 API 端点。
    """
    base = model_url.rstrip('/')
    candidates = [
        {"url": f"{base}/chat/completions", "mode": "chat", "build": lambda p: {"model": model_name, "messages": [{"role": "user", "content": p}], "temperature": 0.0}},
        {"url": f"{base}/v1/chat/completions", "mode": "chat", "build": lambda p: {"model": model_name, "messages": [{"role": "user", "content": p}], "temperature": 0.0}},
        {"url": f"{base}/completions", "mode": "completions", "build": lambda p: {"model": model_name, "prompt": p, "temperature": 0.0}},
        {"url": f"{base}/v1/completions", "mode": "completions", "build": lambda p: {"model": model_name, "prompt": p, "temperature": 0.0}},
        {"url": f"{base}/generate", "mode": "generate", "build": lambda p: {"model": model_name, "prompt": p}},
        {"url": base, "mode": "raw", "build": lambda p: {"model": model_name, "prompt": p}},
    ]

    headers = {"Content-Type": "application/json"}
    probe = "ping"

    print(f"[Init] Detecting API endpoints at {base}...")
    for c in candidates:
        try:
            resp = global_session.post(c["url"], json=c["build"](probe), headers=headers, timeout=timeout)
            if resp.status_code == 200:
                print(f"[Init] Success: Using endpoint {c['url']} (mode: {c['mode']})")
                return {"url": c["url"], "mode": c["mode"]}
        except Exception:
            continue
    
    print("[Warn] Auto-detection failed. Defaulting to raw URL.")
    return None

def call_model_worker(task_data: Dict[str, Any]) -> str:
    """
    线程工作函数：处理单条数据，调用模型。
    """
    instruction = task_data.get("instruction", "")
    input_text = task_data.get("input_text", "")
    model_config = task_data.get("model_config", {})
    
    model_url = model_config.get("url", DEFAULT_MODEL_URL)
    model_name = model_config.get("name", DEFAULT_MODEL_NAME)
    api_endpoint = model_config.get("api_endpoint")
    api_mode = model_config.get("api_mode")

    headers = {"Content-Type": "application/json"}
    target_url = api_endpoint if api_endpoint else model_url
    
    # [MODIFIED] Added frequency_penalty
    common_params = {
        "model": model_name,
        "temperature": 0.2,
        "top_p": 0.95,
        "frequency_penalty": 0.2,
    }

    if api_mode == "chat" or (not api_mode and "chat" in target_url):
        messages = []
        if instruction: messages.append({"role": "system", "content": instruction})
        if input_text: messages.append({"role": "user", "content": input_text})
        if not messages: messages.append({"role": "user", "content": "Hello"})
        
        payload = common_params.copy()
        payload["messages"] = messages
    else:
        prompt = ""
        if instruction: prompt += f"Instruction:\n{instruction}\n\n"
        if input_text: prompt += f"Input:\n{input_text}\n\n"
        prompt += "Please answer concisely and clearly:"
        
        payload = common_params.copy()
        payload["prompt"] = prompt

    try:
        resp = global_session.post(target_url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        
        if isinstance(data, dict):
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
                if "text" in choice:
                    return choice["text"]
            for key in ["output", "generated_text", "response", "content"]:
                if key in data and isinstance(data[key], str):
                    return data[key]
        
        return json.dumps(data, ensure_ascii=False)

    except Exception as e:
        return f"[ERROR_API] {str(e)}"

def process_single_file(file_path: str, out_dir: str, args, api_info):
    """处理单个文件的完整流程"""
    filename = os.path.basename(file_path)
    out_path = os.path.join(out_dir, filename)
    
    processed_count = 0
    if not args.no_resume and os.path.exists(out_path):
        with open(out_path, 'r', encoding='utf-8') as f:
            for _ in f: processed_count += 1
        print(f"[{filename}] Resuming: skipping first {processed_count} lines.")

    lines_to_process = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx < processed_count: continue
            if not line.strip(): continue
            lines_to_process.append((idx, line.strip()))

    if not lines_to_process:
        print(f"[{filename}] Nothing to process.")
        return

    print(f"[{filename}] Processing {len(lines_to_process)} items with {args.threads} threads...")

    model_config = {
        "url": args.url,
        "name": args.model,
        "api_endpoint": api_info.get("url") if api_info else None,
        "api_mode": api_info.get("mode") if api_info else None
    }

    completed_in_this_run = 0
    
    with open(out_path, 'a', encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures_map = {}
            ordered_futures = []

            for original_idx, line_content in lines_to_process:
                try:
                    obj = json.loads(line_content)
                    instruction = obj.get("Instruction") or obj.get("instruction") or obj.get("prompt") or ""
                    input_text = obj.get("Input") or obj.get("input") or obj.get("context") or ""
                    
                    task_data = {
                        "instruction": instruction,
                        "input_text": input_text,
                        "model_config": model_config
                    }
                    
                    future = executor.submit(call_model_worker, task_data)
                    futures_map[future] = obj
                    ordered_futures.append(future)
                except json.JSONDecodeError:
                    print(f"[{filename}] JSON Error at line {original_idx+1}")
                    continue
            
            for future in ordered_futures:
                try:
                    doctor_response = future.result()
                    obj = futures_map[future]
                    obj["doctor"] = doctor_response
                    f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    completed_in_this_run += 1
                    
                    if completed_in_this_run % 100 == 0:
                        f_out.flush()
                        print(f"[{filename}] Progress: {completed_in_this_run}/{len(lines_to_process)}", end='\r')
                        
                except Exception as e:
                    print(f"[{filename}] Error processing item: {e}")

    print(f"\n[{filename}] Completed. Saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Concurrent Batch Doctor Generation")
    parser.add_argument("--in", dest="input_path", required=True, help="Input folder or file path")
    parser.add_argument("--out", dest="output_dir", required=True, help="Output folder path")
    parser.add_argument("--url", dest="url", default=DEFAULT_MODEL_URL, help="Model server base URL")
    parser.add_argument("--model", dest="model", default=DEFAULT_MODEL_NAME, help="Model name")
    parser.add_argument("--threads", type=int, default=32, help="Number of concurrent threads")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing files instead of resuming")
    
    args = parser.parse_args()

    # 1. 初始化 Session 连接池
    init_session(pool_connections=args.threads, pool_maxsize=args.threads)

    # 2. 探测 API
    global API_INFO
    API_INFO = detect_api(args.url, args.model)
    if not API_INFO:
        print("[Warn] Proceeding with raw URL configuration.")

    # 3. 确定输入文件列表
    input_files = []
    if os.path.isfile(args.input_path):
        input_files.append(args.input_path)
    elif os.path.isdir(args.input_path):
        types = ('*.jsonl', '*.json')
        for t in types:
            input_files.extend(glob.glob(os.path.join(args.input_path, t)))
    else:
        print(f"Error: Input path {args.input_path} not found.")
        return

    if not input_files:
        print("No json/jsonl files found in input path.")
        return

    # 4. 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Found {len(input_files)} files. Starting processing with {args.threads} threads...")

    for fpath in sorted(input_files):
        try:
            process_single_file(fpath, args.output_dir, args, API_INFO)
        except Exception as e:
            print(f"CRITICAL ERROR processing file {fpath}: {e}")

    print("\nAll files processed.")

if __name__ == "__main__":
    main()