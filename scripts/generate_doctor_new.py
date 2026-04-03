import json
import os
import glob
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# --- 配置区域 ---
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "sk-fc11d7cb6eb148c09aedfdc4f1026be5"
MODEL_NAME = "qwen3-8b"

INPUT_FOLDER = "./evaluate_imedrag"
OUTPUT_FOLDER = "./result"

# [手动设置并发数]：根据显存大小调整，建议起步 4-8，显存充足可设为 16-32
MAX_CONCURRENT_TASKS = 1

# 强制输出格式的正则表达式
#STRICT_REGEX = r"<think>[\s\S]*?</think><answer>[\s\S]*?</answer>"

# --- 初始化异步客户端 ---
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# 限制并发的信号量
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

async def call_model(data, pbar):
    """
    单个请求的任务函数
    """
    instruction = data.get("instruction", "")
    user_input = data.get("input", "")
    
    async with semaphore:  # 使用信号量控制并发
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": instruction + "\n请严格按照格式输出\n如果格式错误，你的回复将被视为无效。\n思考和回复的内容都不能为空。\n<think>标签内一定要有内容。<answer>标签内一定要有内容。\n"},
                    {"role": "user", "content": user_input}
                ],
                #extra_body={
                    #"guided_regex": STRICT_REGEX
                #},
                extra_body={"enable_thinking": False},
                temperature=0.0,
            )
            model_output = response.choices[0].message.content
            data["doctor"] = model_output

            # --- 附加：实时打印逻辑 ---
            # 使用 tqdm.write 避免破坏进度条结构
            tqdm.write("\n" + "="*50)
            tqdm.write(f"推理完成:")
            tqdm.write(model_output)
            tqdm.write("="*50 + "\n")

        except Exception as e:
            # 记录错误信息，确保脚本不因单条数据报错而停止
            data["doctor"] = f"ERROR: {str(e)}"
        finally:
            pbar.update(1)  # 更新进度条
            return data

async def process_single_file(file_path):
    """
    处理单个文件的异步函数
    """
    file_name = os.path.basename(file_path)
    output_path = os.path.join(OUTPUT_FOLDER, f"{file_name}")
    
    # 读取原始文件所有行
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    if not items:
        return

    print(f"开始并发推理文件: {file_name} (共 {len(items)} 条数据)")
    
    # 使用 tqdm 显示进度
    pbar = tqdm(total=len(items), desc=f"Processing {file_name}")
    
    # 创建所有并发任务
    tasks = [call_model(item, pbar) for item in items]
    
    # 并发执行并等待结果
    results = await asyncio.gather(*tasks)
    pbar.close()

    # 将结果写入新文件（保持原顺序）
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for res in results:
            f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
    
    print(f"文件已保存至: {output_path}")

async def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    jsonl_files = glob.glob(os.path.join(INPUT_FOLDER, "*.jsonl"))
    
    if not jsonl_files:
        print(f"未找到 .jsonl 文件。")
        return

    # 依次处理每个文件（文件间串行，文件内数据行并发）
    for file_path in jsonl_files:
        await process_single_file(file_path)

    print("\n--- 所有任务处理完毕 ---")

if __name__ == "__main__":
    asyncio.run(main())