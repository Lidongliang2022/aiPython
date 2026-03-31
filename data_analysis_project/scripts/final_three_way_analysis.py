"""
三方对比分析：通话原文 vs AI提取 vs 经纪人修改
目标：区分AI遗漏和经纪人补充
"""
import pandas as pd
import json
import re

# 读取数据
df = pd.read_csv('data/500条diff结果(含原文).tsv', sep='\t', encoding='gbk')
print(f"数据量: {len(df)} 条\n")

# 解析通话内容
def extract_conversation(asr_str):
    try:
        asr_data = json.loads(asr_str)
        texts = [item['content'] for item in asr_data if 'content' in item]
        return ''.join(texts)
    except:
        return ""

# 提取关键信息
def extract_info(json_str, field='otherRemark'):
    match = re.search(f'{field}:"([^"]*)"', str(json_str))
    return match.group(1) if match else ""

df['对话文本'] = df['通话原文(ASR)'].apply(extract_conversation)
df['AI备注'] = df['AI提取结果_ai_json'].apply(extract_info)
df['人工备注'] = df['人工最终结果_human_json'].apply(extract_info)

# 分析前10条样本
print("="*80)
print("样本分析（前10条）")
print("="*80)

for i in range(min(10, len(df))):
    print(f"\n【样本 {i+1}】")
    conv = df.iloc[i]['对话文本']
    ai = df.iloc[i]['AI备注']
    human = df.iloc[i]['人工备注']

    print(f"通话: {conv[:100]}...")
    print(f"AI: {ai[:100]}...")
    print(f"人工: {human[:100]}...")

    # 简单判断：人工备注中的关键词是否在通话中出现
    human_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', human))
    in_conv = sum(1 for kw in human_keywords if kw in conv)
    print(f"分析: 人工添加{len(human_keywords)}个关键词，其中{in_conv}个在通话中出现")

# 统计分析
print("\n" + "="*80)
print("整体统计")
print("="*80)
print(f"AI生成备注的平均长度: {df['AI备注'].str.len().mean():.0f}字")
print(f"人工修改备注的平均长度: {df['人工备注'].str.len().mean():.0f}字")
print(f"人工平均增加: {(df['人工备注'].str.len() - df['AI备注'].str.len()).mean():.0f}字")

