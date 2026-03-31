"""
利用本地Qwen模型分析经纪人修改内容，提取优化方向

策略：
1. 提取经纪人添加的关键信息
2. 用Qwen模型批量分析修改内容的类型
3. 聚类发现修改模式
4. 生成优化建议
"""
import pandas as pd
import requests
import json
import re
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
    response = requests.post(QWEN_API, json=payload, timeout=30)
    return response.json()['choices'][0]['message']['content']

# 读取diff数据
df = pd.read_csv('data/2000条diff结果.csv', encoding='gbk', on_bad_lines='skip')
df = df.dropna(subset=['ai_json', 'human_json'])
print(f"有效数据: {len(df)} 条\n")

# 提取经纪人修改的内容
human_remarks = []
for i in range(len(df)):
    human_json = str(df.iloc[i]['human_json'])
    match = re.search(r'otherRemark:"([^"]*)"', human_json)
    if match:
        human_remarks.append(match.group(1))

print(f"提取到 {len(human_remarks)} 条修改内容\n")
