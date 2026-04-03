import json
import os

# 需要添加到 input 前面的内容
PREFIX_CONTENT = """
<任务>
# 任务介绍
在对话过程中，患者和医生交替发言，每次患者的发言可能归属于一个或多个预设场景。
同一场景可能会持续多个回合，直至该场景结束后才会切换至新的场景。
每个场景对应一套特定的医生思考逻辑，医生的每次回复都应体现该场景下的思考过程。
在回复中要给予患者一些情绪价值，示例："考虑到您...，所以建议您做...，有助于我们评估..."。
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

def process_jsonl_files(directory):
    # 检查路径是否存在
    if not os.path.exists(directory):
        print(f"错误：文件夹路径不存在 -> {directory}")
        return

    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(directory, filename)
            print(f"正在处理文件: {filename} ...")

            processed_lines = []
            file_modified = False

            try:
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            
                            # 检查是否存在 input 字段
                            if "input" in data:
                                original_input = data["input"]
                                
                                # 简单的重复性检查：如果 input 已经包含 <任务> 标签，则跳过添加
                                # 以防止脚本重复运行导致内容堆叠
                                if "<任务>" not in original_input:
                                    # 构造新的 input 内容：换行 + 前缀内容 + 换行 + 原始内容
                                    new_input = "\n" + PREFIX_CONTENT.strip() + "\n\n" + original_input
                                    data["input"] = new_input
                                    file_modified = True
                            
                            # 将处理后的对象转回 json 字符串，ensure_ascii=False 保证中文正常显示
                            processed_lines.append(json.dumps(data, ensure_ascii=False))
                            
                        except json.JSONDecodeError:
                            print(f"警告: 文件 {filename} 第 {line_num} 行 JSON 解析失败，跳过该行。")
                            processed_lines.append(line) # 保持原样

                # 只有在内容发生变化时才写入文件
                if file_modified:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        for line in processed_lines:
                            f.write(line + '\n')
                    print(f"完成: {filename} (已覆盖保存)")
                else:
                    print(f"跳过: {filename} (内容未改变或已包含目标前缀)")

            except Exception as e:
                print(f"处理文件 {filename} 时发生错误: {e}")

if __name__ == "__main__":
    # 请在这里修改你的文件夹路径，例如 r"C:\Data\jsonl_files" 或 "./data"
    folder_path = r"../evaluate_medrag" 
    
    process_jsonl_files(folder_path)
    print("所有任务处理完毕。")