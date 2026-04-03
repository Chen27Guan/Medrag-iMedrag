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


DEFAULT_MODEL_URL = "http://113.59.64.94:8082/v1"
DEFAULT_MODEL_NAME = "Qwen3-8B"

TASK_PREFIX = """
<任务>
# 任务介绍
在对话过程中，患者和医生交替发言，每次患者的发言可能归属于一个或多个预设场景。
同一场景可能会持续多个回合，直至该场景结束后才会切换至新的场景。
每个场景对应一套特定的医生思考逻辑，医生的每次回复都应体现该场景下的思考过程。
在回复中要给予患者一些情绪价值，示例："考虑到您...，所以建议您做...，有助于我们评估...\"。
如果患者做过某项检查，不要让患者进行复查该项检查！

# 任务要求
思考：结合<对话历史>，快速判断当前<本轮患者发言>所属的场景，
对于每个归属的场景，分别按照该场景的医生思考步骤进行独立分析；
回复：回答时应面向患者清晰解释你的建议背后的依据，可以适当提及症状、检查目的、治疗逻辑和因果关系，让患者理解你的建议。回答详细丰满。

# 任务注意
1. 你的回复应该面向本轮<本轮患者发言>的内容；
2. 场景判断只需快速归类，不需大篇幅思考，重点在于结合场景进行详细分析；
3. 不要给患者推荐其他科室的检查项；
4. 在推荐药物或推荐检查时，应该告知具体的名字；
5. 你不能逃避患者的问题；
6. 医生的回复不应该太简单，要有一些情绪价值；
7. 回复中一定要有<think></think>和<answer></answer>中的内容；
8. 如果患者1个月内做过某些检查，回复中一定不能重复开这种检查。例如患者上传了3天前的动态心电图报告，就不能让患者复查该检查了；
9. 你是仁济医院的医生，不能推荐其它的医院或卫生服务中心。
10. 线上可以配药和开检查单，并告知患者相应的用药方案等。如果需要进一步检查或线下评估才能调整药物或开药，明确告知患者相应检查与用药方案。
11. 请专注于目前科室的任务，不要给出其他科室相关病症的建议。
12. 回复中的“您”或者“你”永远指代患者。
13. 回复中不要出现“医生说的”或者“医生提到的”这种表述，应用“我”指代医生。
</任务>

<场景与思考逻辑>
1.病因咨询：患者主动询问症状的可能原因，医生将基于已有信息（如症状、病史、生活习惯、检查结果、体检数据等）排除不相关病因，提出合理怀疑方向，并解释当前症状可能的成因。分析过程通常包括生理、环境、遗传等因素的归纳判断。
2.治疗方案咨询：医生逐个分析已知症状，考虑可能的疾病，再结合患者的相关检查数据、体格信息、疾病史等来评估出最可能的疾病。当医生无法仅凭当前的信息做出准确的诊断时，医生需要向患者询问获取更多的信息；若医生可以做出诊断时，下一步应该重新分析症状表现、检查数据等信息，评估疾病程度。最终，医生会制定具体的药物推荐或治疗建议，若病情较为复杂，治疗方案中可能包含后续的检查或复诊安排。
3.医疗操作请求：当患者明确提出开具药物、检查单、病假单等请求，医生仅需判断该操作是否合理（如检查是否必要、药物是否适应），在确认无误后直接回应，无需过多分析，通常简要回应或回复医院实际情况（如：线上开不了的单子等等），目的是让患者等待医生操作。
4.检查方案咨询：结合患者症状、病史、检查数据，评估疾病可能性，推荐关键检查，避免不必要的检测，尤其在病情不明确时，应该推荐代价较小的检查以初步分析。
5.主诉确认：患者未明确表达核心问题或咨询目的，医生需通过反问引导患者明确主诉或通过反向追问症状细节，从而决定是否进入治疗、检查流程。
6.其他：根据患者发言有逻辑的简短思考。
</场景与思考逻辑>
"""

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
    parser.add_argument("--in", dest="infile", default="xnk_hhh.jsonl", help="Input jsonl file")
    parser.add_argument("--out", dest="outfile", default="xnk_8B.jsonl", help="Output jsonl file")
    parser.add_argument("--url", dest="url", default=DEFAULT_MODEL_URL, help="Model server base URL")
    parser.add_argument("--model", dest="model", default=DEFAULT_MODEL_NAME, help="Model name to send")
    parser.add_argument("--no-resume", dest="no_resume", action="store_true", help="Do not resume; overwrite output file")

    args = parser.parse_args()

    if args.no_resume and os.path.exists(args.outfile):
        os.remove(args.outfile)

    process_file(args.infile, args.outfile, model_url=args.url, model_name=args.model, resume=not args.no_resume)


if __name__ == "__main__":
    main()
