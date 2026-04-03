#!/usr/bin/env python3
"""
generate_doctor.py

读取一个 jsonl 文件（每行是一个 JSON 对象，包含 `Instruction` 和 `Input` 字段），
调用本地部署的模型（HTTP API），将模型返回的文字保存到 `Doctor` 字段，
并把结果写入新的 jsonl 文件。

默认输入文件: evaluate/merged_xnk.jsonl
默认输出文件: evaluate/merged_xnk_with_doctor.jsonl

示例:
    python scripts/generate_doctor.py --in evaluate/merged_xnk.jsonl --out evaluate/merged_xnk_with_doctor.jsonl

作者: 自动生成
"""
import argparse
import json
import time
import os
from typing import Optional

import requests


DEFAULT_MODEL_URL = "http://172.20.137.216:2033/v1"
DEFAULT_MODEL_NAME = "Qwen3_8B_renji1006_static"

# cached detected API info for this run
API_INFO = None


def detect_api(model_url: str, model_name: str, timeout: int = 5):
    """Try several common endpoints and request formats to discover a working one.

    Returns a dict with keys: 'url' (full endpoint), 'mode' (one of 'chat', 'completions', 'generate', 'response', 'infer', 'raw'),
    and optionally other metadata. Returns None if none worked.
    """
    base = model_url.rstrip('/')
    # candidate endpoints and their body builders
    candidates = []

    # chat completions (OpenAI style)
    candidates.append({
        "url": f"{base}/chat/completions",
        "mode": "chat",
        "build": lambda p: {"model": model_name, "messages": [{"role": "user", "content": p}], "temperature": 0.0},
    })
    candidates.append({
        "url": f"{base}/v1/chat/completions",
        "mode": "chat",
        "build": lambda p: {"model": model_name, "messages": [{"role": "user", "content": p}], "temperature": 0.0},
    })

    # completions with prompt
    candidates.append({
        "url": f"{base}/completions",
        "mode": "completions",
        "build": lambda p: {"model": model_name, "prompt": p, "max_tokens": 1024, "temperature": 0.0},
    })
    candidates.append({
        "url": f"{base}/v1/completions",
        "mode": "completions",
        "build": lambda p: {"model": model_name, "prompt": p, "max_tokens": 1024, "temperature": 0.0},
    })

    # generate / response
    candidates.append({
        "url": f"{base}/generate",
        "mode": "generate",
        "build": lambda p: {"model": model_name, "prompt": p},
    })
    candidates.append({
        "url": f"{base}/response",
        "mode": "response",
        "build": lambda p: {"model": model_name, "input": p},
    })

    # model-specific infer
    candidates.append({
        "url": f"{base}/models/{model_name}/infer",
        "mode": "infer",
        "build": lambda p: {"input": p},
    })

    # root /v1
    candidates.append({
        "url": base,
        "mode": "raw",
        "build": lambda p: {"model": model_name, "prompt": p},
    })

    headers = {"Content-Type": "application/json"}
    probe = "Hello. Please respond with a short acknowledgement like 'ack'."

    for c in candidates:
        try:
            resp = requests.post(c["url"], json=c["build"](probe), headers=headers, timeout=timeout)
        except Exception as e:
            # network error or refused
            # print(f"[DEBUG] probe {c['url']} failed: {e}")
            continue

        if resp.status_code == 200:
            # try to parse to ensure response contains text
            try:
                data = resp.json()
            except Exception:
                return {"url": c["url"], "mode": c["mode"]}

            # simple success
            return {"url": c["url"], "mode": c["mode"]}

    return None


def call_model(prompt: str, model_url: str = DEFAULT_MODEL_URL, model_name: str = DEFAULT_MODEL_NAME, timeout: int = 60, max_retries: int = 3) -> Optional[str]:
    """Call the local model HTTP API and return the generated text.

    This function assumes the model API accepts a JSON body like:
      {"model": "<model_name>", "prompt": "...", "max_tokens": 1024}

    and returns a JSON response containing the generated text in a field that may vary.
    We try to handle common formats.
    """
    global API_INFO
    headers = {"Content-Type": "application/json"}

    # detect API once per run
    if API_INFO is None:
        print(f"[INFO] Detecting available model endpoints under {model_url} ...")
        API_INFO = detect_api(model_url, model_name)
        if API_INFO is None:
            print(f"[WARN] Could not detect a working endpoint under {model_url}. You may need to set the correct URL (e.g. /v1/chat/completions)")

    # build candidate requests based on detected API or fall back to several options
    def build_requests():
        if API_INFO is not None:
            mode = API_INFO.get("mode")
            url = API_INFO.get("url")
            if mode == "chat":
                return [(url, {"model": model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0})]
            if mode == "completions":
                return [(url, {"model": model_name, "prompt": prompt, "max_tokens": 1024, "temperature": 0.0})]
            if mode == "generate":
                return [(url, {"model": model_name, "prompt": prompt})]
            if mode == "response":
                return [(url, {"model": model_name, "input": prompt})]
            if mode == "infer":
                return [(url, {"input": prompt})]
            return [(url, {"model": model_name, "prompt": prompt})]

        # fallback: try a list of reasonable possibilities
        base = model_url.rstrip('/')
        return [
            (f"{base}/chat/completions", {"model": model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}),
            (f"{base}/v1/chat/completions", {"model": model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}),
            (f"{base}/completions", {"model": model_name, "prompt": prompt, "max_tokens": 1024, "temperature": 0.0}),
            (f"{base}/v1/completions", {"model": model_name, "prompt": prompt, "max_tokens": 1024, "temperature": 0.0}),
            (f"{base}/generate", {"model": model_name, "prompt": prompt}),
            (f"{base}/response", {"model": model_name, "input": prompt}),
            (f"{base}/models/{model_name}/infer", {"input": prompt}),
            (model_url, {"model": model_name, "prompt": prompt}),
        ]

    requests_to_try = build_requests()

    for attempt in range(1, max_retries + 1):
        for url, body in requests_to_try:
            try:
                resp = requests.post(url, json=body, headers=headers, timeout=timeout)
            except Exception as e:
                # network or connection error
                print(f"[WARN] Request to {url} failed (attempt {attempt}/{max_retries}): {e}")
                continue

            if resp.status_code != 200:
                # try next candidate
                print(f"[WARN] Non-200 response from {url} (attempt {attempt}/{max_retries}): {resp.status_code} {resp.text}")
                continue

            # success
            try:
                data = resp.json()
            except Exception:
                return resp.text

            # parse response: try common shapes
            if isinstance(data, dict):
                # openai-like chat completion
                if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                    first = data["choices"][0]
                    if isinstance(first, dict):
                        if "message" in first and isinstance(first["message"], dict):
                            content = first["message"].get("content")
                            if isinstance(content, str):
                                return content
                        if "text" in first and isinstance(first["text"], str):
                            return first["text"]

                for key in ("output", "text", "generated_text", "result", "completion"):
                    if key in data and isinstance(data[key], str):
                        return data[key]

                # nested data list
                if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                    for item in data["data"]:
                        if isinstance(item, dict):
                            for key in ("text", "content", "answer"):
                                if key in item and isinstance(item[key], str):
                                    return item[key]

            # fallback: return full JSON
            return json.dumps(data, ensure_ascii=False)

    return None


def call_model_chat(instruction: str, input_text: str, model_url: str = DEFAULT_MODEL_URL, model_name: str = DEFAULT_MODEL_NAME, timeout: int = 60, max_retries: int = 3) -> Optional[str]:
    """Call the model using chat-completions format where instruction -> system, input_text -> user.

    This matches the call in `comprehensive_evaluation.py` which uses:
      messages=[{"role": "system", "content": instruction}, {"role": "user", "content": prompt_input}]
    """
    global API_INFO
    headers = {"Content-Type": "application/json"}

    # detect API once per run
    if API_INFO is None:
        print(f"[INFO] Detecting available model endpoints under {model_url} ...")
        API_INFO = detect_api(model_url, model_name)
        if API_INFO is None:
            print(f"[WARN] Could not detect a working endpoint under {model_url}. You may need to set the correct URL (e.g. /v1/chat/completions)")

    # build chat request (prefer detected chat endpoint)
    base = model_url.rstrip('/')
    if API_INFO and API_INFO.get('mode') == 'chat':
        urls = [API_INFO.get('url')]
    else:
        urls = [f"{base}/chat/completions", f"{base}/v1/chat/completions", f"{base}/completions", f"{base}/v1/completions", model_url]

    messages = []
    if instruction:
        messages.append({"role": "system", "content": instruction})
    if input_text:
        messages.append({"role": "user", "content": input_text})

    for attempt in range(1, max_retries + 1):
        for url in urls:
            body = {"model": model_name, "messages": messages, "temperature": 0}
            try:
                resp = requests.post(url, json=body, headers=headers, timeout=timeout)
            except Exception as e:
                print(f"[WARN] Chat request to {url} failed (attempt {attempt}/{max_retries}): {e}")
                continue

            if resp.status_code != 200:
                print(f"[WARN] Non-200 chat response from {url} (attempt {attempt}/{max_retries}): {resp.status_code} {resp.text}")
                continue

            try:
                data = resp.json()
            except Exception:
                return resp.text

            # prefer OpenAI-like chat response: choices[0].message.content
            if isinstance(data, dict):
                choices = data.get('choices')
                if isinstance(choices, list) and len(choices) > 0:
                    first = choices[0]
                    if isinstance(first, dict):
                        # new style: first['message']['content']
                        if 'message' in first and isinstance(first['message'], dict):
                            content = first['message'].get('content')
                            if isinstance(content, str):
                                return content
                        # older style: first['text']
                        if 'text' in first and isinstance(first['text'], str):
                            return first['text']

                # fallback to other keys
                for key in ("output", "text", "generated_text", "result", "completion"):
                    if key in data and isinstance(data[key], str):
                        return data[key]

            return json.dumps(data, ensure_ascii=False)

    return None


def build_prompt(instruction: str, input_text: Optional[str]) -> str:
    """Construct a prompt combining Instruction and Input for the model."""
    if input_text:
        return f"Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nPlease answer concisely and clearly:"
    else:
        return f"Instruction:\n{instruction}\n\nPlease answer concisely and clearly:"


def process_file(in_path: str, out_path: str, model_url: str, model_name: str, resume: bool = True):
    """Read in_path (jsonl), call model for each item, write to out_path (jsonl).

    If resume=True and out_path exists, resume from first non-processed line.
    """
    seen = 0
    if resume and os.path.exists(out_path):
        # count existing lines to resume
        with open(out_path, "r", encoding="utf-8") as f:
            for _ in f:
                seen += 1
        print(f"Resuming: found {seen} already-processed lines in {out_path}")

    infile = open(in_path, "r", encoding="utf-8")
    outfile = open(out_path, "a", encoding="utf-8")

    for idx, line in enumerate(infile):
        if idx < seen:
            continue

        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except Exception as e:
            print(f"[ERROR] Failed to parse JSON on line {idx+1}: {e}")
            continue

        instruction = obj.get("Instruction") or obj.get("instruction") or obj.get("prompt") or ""
        input_text = obj.get("Input") or obj.get("input") or obj.get("context") or ""

        # Prefer chat-style call (instruction -> system, input_text -> user)
        generated = None
        if instruction or input_text:
            generated = call_model_chat(instruction, input_text, model_url=model_url, model_name=model_name)

        # Fallback to prompt-style completion if chat failed
        if generated is None:
            prompt = build_prompt(instruction, input_text)
            generated = call_model(prompt, model_url=model_url, model_name=model_name)
        if generated is None:
            generated = ""

        # Save under 'doctor' key
        obj["doctor"] = generated

        # Write line
        outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # flush occasionally
        if (idx + 1) % 10 == 0:
            outfile.flush()
            print(f"Processed {idx+1} lines, last Doctor length: {len(generated)}")

    infile.close()
    outfile.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Doctor field by calling local Qwen2.5 model HTTP API")
    parser.add_argument("--in", dest="infile", default="medcite_filtered.jsonl", help="Input jsonl file")
    parser.add_argument("--out", dest="outfile", default="xnk_medcite_8B_renji1006.jsonl", help="Output jsonl file")
    parser.add_argument("--url", dest="url", default=DEFAULT_MODEL_URL, help="Model server base URL")
    parser.add_argument("--model", dest="model", default=DEFAULT_MODEL_NAME, help="Model name to send")
    parser.add_argument("--no-resume", dest="no_resume", action="store_true", help="Do not resume; overwrite output file")

    args = parser.parse_args()

    if args.no_resume and os.path.exists(args.outfile):
        os.remove(args.outfile)

    process_file(args.infile, args.outfile, model_url=args.url, model_name=args.model, resume=not args.no_resume)


if __name__ == "__main__":
    main()
