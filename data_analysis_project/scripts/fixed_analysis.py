"""
三方对比分析（修复版）
"""
import pandas as pd
import json

df = pd.read_csv('data/500条diff结果(含原文).tsv', sep='\t', encoding='gbk')
print(f"数据量: {len(df)} 条\n")

# 解析JSON
def parse_json_field(json_str, field='otherRemark'):
    try:
        data = json.loads(json_str)
        return data.get(field, '')
    except:
        return ''

def extract_conversation(asr_str):
    try:
        asr_data = json.loads(asr_str)
        return ''.join([item.get('content', '') for item in asr_data])
    except:
        return ''

df['对话'] = df['通话原文(ASR)'].apply(extract_conversation)
df['AI备注'] = df['AI提取结果_ai_json'].apply(parse_json_field)
df['人工备注'] = df['人工最终结果_human_json'].apply(parse_json_field)

# 过滤有效数据
valid = df[(df['AI备注'] != '') & (df['人工备注'] != '')].copy()
print(f"有效数据: {len(valid)} 条\n")

print("="*80)
print("前5个样本对比")
print("="*80)
for i in range(min(5, len(valid))):
    print(f"\n【样本{i+1}】")
    print(f"通话: {valid.iloc[i]['对话'][:80]}...")
    print(f"AI: {valid.iloc[i]['AI备注'][:80]}...")
    print(f"人工: {valid.iloc[i]['人工备注'][:80]}...")

# 统计分析
print("\n" + "="*80)
print("统计分析")
print("="*80)
print(f"AI备注平均长度: {valid['AI备注'].str.len().mean():.0f}字")
print(f"人工备注平均长度: {valid['人工备注'].str.len().mean():.0f}字")
print(f"人工平均增加: {(valid['人工备注'].str.len() - valid['AI备注'].str.len()).mean():.0f}字")

# 关键词分析
from collections import Counter
import re

ai_words = []
human_words = []
for i in range(len(valid)):
    ai_words.extend(re.findall(r'[\u4e00-\u9fa5]{2,}', valid.iloc[i]['AI备注']))
    human_words.extend(re.findall(r'[\u4e00-\u9fa5]{2,}', valid.iloc[i]['人工备注']))

print(f"\nAI高频词: {', '.join([w for w, c in Counter(ai_words).most_common(10)])}")
print(f"人工高频词: {', '.join([w for w, c in Counter(human_words).most_common(10)])}")

