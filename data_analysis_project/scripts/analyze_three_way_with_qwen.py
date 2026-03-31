"""
三方对比分析：通话原文 vs AI提取 vs 经纪人修改

目标：
1. 提取通话中的关键信息
2. 判断经纪人添加的信息是否在通话中出现
3. 区分AI遗漏和经纪人补充
"""
import pandas as pd
import json
import re
import requests
from collections import Counter

# Qwen API配置
QWEN_API = "http://web.datalab.ke.com/proxy/rayserve/10.238.7.8:8000/v1/chat/completions"
MODEL = "Qwen2.5-72B-Instruct-GPTQ-Int4"

def call_qwen(prompt, system_prompt="你是一个数据分析专家"):
    """调用本地Qwen模型"""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(QWEN_API, json=payload, timeout=30)
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"API调用失败: {e}"

# 读取数据（使用python引擎处理复杂CSV）
print("正在读取数据...")
df = pd.read_csv('/Users/lidongliang021/Downloads/2000条diff结果（含原文）.csv',
                 encoding='gbk', engine='python', on_bad_lines='skip')
print(f"成功读取 {len(df)} 条数据\n")

# 解析通话原文，提取对话内容
def extract_conversation(asr_json_str):
    """从ASR JSON中提取对话文本"""
    try:
        asr_data = json.loads(asr_json_str)
        conversation = []
        for item in asr_data:
            role = item.get('role', '未知')
            content = item.get('content', '')
            conversation.append(f"{role}: {content}")
        return "\n".join(conversation)
    except:
        return ""

# 提取人工修改的内容
def extract_human_remark(human_json_str):
    """提取人工修改的otherRemark字段"""
    match = re.search(r'otherRemark:"([^"]*)"', str(human_json_str))
    return match.group(1) if match else ""

print("正在处理数据...")
df['对话文本'] = df['通话原文(ASR)'].apply(extract_conversation)
df['人工备注'] = df['人工最终结果_human_json'].apply(extract_human_remark)

# 过滤有效数据
valid_df = df[(df['对话文本'] != "") & (df['人工备注'] != "")].copy()
print(f"有效数据: {len(valid_df)} 条\n")

# 使用Qwen模型分析前10条样本
print("="*80)
print("使用Qwen模型分析经纪人修改内容")
print("="*80)

analysis_results = []
for i in range(min(10, len(valid_df))):
    conversation = valid_df.iloc[i]['对话文本']
    human_remark = valid_df.iloc[i]['人工备注']

    prompt = f"""
请分析以下房产通话和经纪人的跟进备注：

【通话内容】
{conversation[:500]}

【经纪人备注】
{human_remark}

请判断：经纪人在备注中添加的信息，哪些在通话中提到了，哪些是通话中没有的？

请用以下格式回答：
1. 通话中提到的信息：[列出]
2. 通话中未提到的信息：[列出]
3. 结论：[简短总结]
"""

    print(f"\n分析样本 {i+1}...")
    result = call_qwen(prompt)
    analysis_results.append({
        'index': i,
        'analysis': result
    })
    print(result[:200] + "...")

print(f"\n已分析 {len(analysis_results)} 条样本")

# 生成总结报告
print("\n" + "="*80)
print("总结报告")
print("="*80)

summary_prompt = f"""
基于以上{len(analysis_results)}个样本的分析，请总结：

1. 经纪人最常添加哪些类型的信息？
2. 这些信息中，有多少是通话中提到的（AI应该提取但没提取）？
3. 有多少是通话中没有的（经纪人凭经验补充）？
4. 针对AI生成内容，给出3-5条具体的优化建议。

请简洁明了地回答。
"""

print("\n正在生成总结...")
summary = call_qwen(summary_prompt)
print(summary)

# 保存分析结果
print("\n保存分析结果到文件...")
with open('output/qwen_analysis_results.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("经纪人修改内容分析报告\n")
    f.write("="*80 + "\n\n")
    for result in analysis_results:
        f.write(f"\n样本 {result['index']+1}:\n")
        f.write(result['analysis'] + "\n")
        f.write("-"*80 + "\n")
    f.write("\n\n总结:\n")
    f.write(summary)

print("✅ 分析完成！结果已保存到 output/qwen_analysis_results.txt")



